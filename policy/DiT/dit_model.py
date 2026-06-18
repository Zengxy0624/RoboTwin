"""Deploy-time wrapper for the DiT policy. Mirrors policy/DP/dp_model.py (DP) but
featurizes the live head_cam into patch TOKENS (N, C) via the same frozen encoder
used at precompute, and serves them as obs["head_cam_tokens"] through DP's runner
(the runner is key-agnostic: it just stacks the last n_obs_steps of each obs key).
"""
import numpy as np
import torch
import hydra
import dill
import sys, os

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
# DP package first so `import diffusion_policy` resolves to DP's, then DiT for dit.*
sys.path.insert(0, os.path.join(parent_dir, "../DP"))
sys.path.insert(0, parent_dir)

from diffusion_policy.workspace.robotworkspace import RobotWorkspace  # noqa: E402
from diffusion_policy.env_runner.dp_runner import DPRunner  # noqa: E402


class DiT_Model:

    def __init__(self, ckpt_file: str, n_obs_steps, n_action_steps, encoder):
        self.policy = self.get_policy(ckpt_file, None, "cuda:0")
        self.runner = DPRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)
        # frozen encoder that turns the live head_cam image into patch tokens
        # (identical forward to precompute_tokens.py).
        self.encoder = encoder.to("cuda:0").eval()

    def _featurize(self, observation):
        if observation is None:
            return observation
        img = observation["head_cam"]  # (3, H, W) in [0,1]
        x = torch.from_numpy(np.asarray(img)).float().unsqueeze(0).to("cuda:0")
        with torch.no_grad():
            tok = self.encoder(x).cpu().numpy()[0]  # (N, C)
        obs = {k: v for k, v in observation.items() if k not in ("head_cam", "left_cam", "right_cam")}
        obs["head_cam_tokens"] = tok
        return obs

    def update_obs(self, observation):
        self.runner.update_obs(self._featurize(observation))

    def reset_obs(self):
        self.runner.reset_obs()

    def get_action(self, observation=None):
        return self.runner.get_action(self.policy, self._featurize(observation))

    def get_last_obs(self):
        return self.runner.obs[-1]

    def get_policy(self, checkpoint, output_dir, device):
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: RobotWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device(device)
        policy.to(device)
        policy.eval()
        return policy
