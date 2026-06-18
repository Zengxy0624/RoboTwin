"""Diffusion-Transformer action policy conditioned on frozen-VFM patch tokens
via cross-attention (vision-only, no language).

Mirrors DP's DiffusionUnetImagePolicy plumbing (set_normalizer / get_optimizer /
compute_loss / predict_action) but:
  - the denoiser is a small DiT (self-attn over the action sequence + cross-attn
    over image tokens) instead of a 1D U-Net,
  - conditioning is the raw patch tokens (B, To*N, C), NOT a pooled global vector,
  - the full action horizon is always predicted (no inpainting of obs into the
    action trajectory).

Obs at train/eval:
  obs["head_cam_tokens"] : (B, To, N, C)   frozen-VFM patch tokens
  obs["agent_pos"]       : (B, To, 14)     proprioception
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

from .model import DiT


class DiffusionDiTPolicy(BaseImagePolicy):

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        # DiT hyperparams
        hidden_size=384,
        depth=6,
        num_heads=6,
        mlp_ratio=1.0,
        normalize_tokens=False,
        **kwargs,
    ):
        super().__init__()

        action_dim = shape_meta["action"]["shape"][0]
        state_dim = shape_meta["obs"]["agent_pos"]["shape"][0]
        # head_cam_tokens shape is [N, C]; we only need C (per-token width).
        tok_shape = shape_meta["obs"]["head_cam_tokens"]["shape"]
        img_dim = tok_shape[-1]
        n_tokens = tok_shape[0]

        self.model = DiT(
            action_dim=action_dim,
            state_dim=state_dim,
            img_dim=img_dim,
            horizon=horizon,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            max_img_tokens=n_tokens * n_obs_steps,
        )
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.normalize_tokens = normalize_tokens

        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ---------- conditioning ----------
    def _build_cond(self, nobs):
        """nobs: normalized obs dict. Returns (img_tokens (B, To*N, C), state (B, Sdim)).
        Only the first n_obs_steps frames condition the policy (the dataset serves the
        full horizon; the trailing frames are future and must not leak)."""
        To = self.n_obs_steps
        tokens = nobs["head_cam_tokens"][:, :To]    # (B, To, N, C)
        B, _, N, C = tokens.shape
        img_tokens = tokens.reshape(B, To * N, C)   # flatten obs-frames of tokens
        state = nobs["agent_pos"][:, To - 1]        # last obs step (B, Sdim)
        return img_tokens, state

    def _normalize_obs(self, obs_dict):
        # action+agent_pos go through the LinearNormalizer; tokens optionally per-C.
        nobs = {"agent_pos": self.normalizer["agent_pos"].normalize(obs_dict["agent_pos"])}
        tok = obs_dict["head_cam_tokens"]
        if self.normalize_tokens:
            tok = self.normalizer["head_cam_tokens"].normalize(tok)
        nobs["head_cam_tokens"] = tok
        return nobs

    # ---------- training ----------
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(self, weight_decay, learning_rate, betas):
        return torch.optim.AdamW(
            self.parameters(), lr=learning_rate, betas=tuple(betas), weight_decay=weight_decay
        )

    def compute_loss(self, batch):
        nobs = self._normalize_obs(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])   # (B, H, A)

        img_tokens, state = self._build_cond(nobs)

        noise = torch.randn(nactions.shape, device=nactions.device)
        bsz = nactions.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=nactions.device
        ).long()
        noisy = self.noise_scheduler.add_noise(nactions, noise, timesteps)

        pred = self.model(noisy, timesteps, img_tokens, state)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = nactions
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        return F.mse_loss(pred, target)

    # ---------- inference ----------
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self._normalize_obs(obs_dict)
        img_tokens, state = self._build_cond(nobs)
        B = img_tokens.shape[0]
        device, dtype = self.device, self.dtype

        trajectory = torch.randn(
            (B, self.horizon, self.action_dim), device=device, dtype=dtype
        )
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            model_output = self.model(
                trajectory, t.to(device).expand(B), img_tokens, state
            )
            trajectory = scheduler.step(model_output, t, trajectory, **self.kwargs).prev_sample

        action_pred = self.normalizer["action"].unnormalize(trajectory)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {"action": action, "action_pred": action_pred}
