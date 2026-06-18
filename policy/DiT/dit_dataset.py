"""Dataset over precomputed frozen-encoder PATCH TOKENS (head_cam_tokens) for the
cross-attention DiT policy. Mirrors RobotFeatureDataset but the camera obs is a
(N, C) token grid per frame instead of a pooled (D,) vector. The numba batched
sampler handles the extra trailing dim transparently (it slices on axis 0).
"""
from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.robot_image_dataset import batch_sample_sequence


class RobotTokenDataset(BaseImageDataset):

    def __init__(self, zarr_path, horizon=1, pad_before=0, pad_after=0, seed=42,
                 val_ratio=0.0, batch_size=128, max_train_episodes=None,
                 normalize_tokens=False):
        super().__init__()
        self.normalize_tokens = normalize_tokens
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=["head_cam_tokens", "state", "action"])

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, sequence_length=horizon,
            pad_before=pad_before, pad_after=pad_after, episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()}
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, sequence_length=self.horizon,
            pad_before=self.pad_before, pad_after=self.pad_after, episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        if self.normalize_tokens:
            # per-C normalization over (T*N) rows: fit on the flattened token dim.
            from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
            tok = self.replay_buffer["head_cam_tokens"][:]   # (T, N, C)
            sf = SingleFieldLinearNormalizer()
            sf.fit(tok.reshape(-1, tok.shape[-1]), last_n_dims=1, mode=mode, **kwargs)
            normalizer["head_cam_tokens"] = sf
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            return {k: torch.from_numpy(v) for k, v in sample.items()}
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(self.buffers[k], v, self.sampler.indices, idx,
                                      self.sampler.sequence_length)
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        return {
            "obs": {
                "head_cam_tokens": samples["head_cam_tokens"].to(device, non_blocking=True),  # B,T,N,C
                "agent_pos": samples["state"].to(device, non_blocking=True),                   # B,T,14
            },
            "action": samples["action"].to(device, non_blocking=True),                         # B,T,A
        }
