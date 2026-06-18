import numpy as np
import torch
import hydra
import dill
import sys, os

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)

from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.env_runner.dp_runner import DPRunner

class DP:

    def __init__(self, ckpt_file: str, n_obs_steps, n_action_steps, encoder=None):
        self.policy = self.get_policy(ckpt_file, None, "cuda:0")
        self.runner = DPRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)
        # For precomputed-feature policies: a frozen rgb_model that turns the live
        # head_cam image into head_cam_feat on the fly (identical to the precompute).
        self.encoder = encoder.to("cuda:0").eval() if encoder is not None else None

    def _featurize(self, observation):
        if self.encoder is None or observation is None:
            return observation
        img = observation["head_cam"]  # (3, H, W) in [0,1]
        x = torch.from_numpy(np.asarray(img)).float().unsqueeze(0).to("cuda:0")
        with torch.no_grad():
            feat = self.encoder(x).cpu().numpy()[0]  # (D,)
        obs = {k: v for k, v in observation.items() if k not in ("head_cam", "left_cam", "right_cam")}
        obs["head_cam_feat"] = feat
        return obs

    def update_obs(self, observation):
        self.runner.update_obs(self._featurize(observation))

    def reset_obs(self):
        self.runner.reset_obs()

    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, self._featurize(observation))
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

    def get_policy(self, checkpoint, output_dir, device):
        # load checkpoint
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: RobotWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device(device)
        policy.to(device)
        policy.eval()

        return policy
