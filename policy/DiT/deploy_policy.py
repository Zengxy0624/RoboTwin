import os
import sys

import numpy as np
import yaml

# dit_model inserts policy/DP and policy/DiT onto sys.path so `diffusion_policy`
# (DP's) and `dit.*` both import. Importing it here makes those paths available.
from .dit_model import DiT_Model


def encode_obs(observation):
    # DiT is vision-only and uses head_camera only.
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    obs = dict(head_cam=head_cam)
    obs["agent_pos"] = observation["joint_action"]["vector"]
    return obs


def get_model(usr_args):
    encoder_tag = usr_args.get("encoder_tag", "dinov3")
    feat_encoder = usr_args.get("feat_encoder", "dinov3")
    ckpt_root = "./policy/DiT" if encoder_tag.endswith("_dit") else "./policy/DP"
    ckpt_file = (f"{ckpt_root}/checkpoints/{encoder_tag}/"
                 f"{usr_args['task_name']}-{usr_args['ckpt_setting']}-"
                 f"{usr_args['expert_data_num']}-{usr_args['seed']}/"
                 f"{usr_args['checkpoint_num']}.ckpt")
    action_dim = usr_args["left_arm_dim"] + usr_args["right_arm_dim"] + 2  # 2 grippers

    cfg_path = f"./policy/DiT/diffusion_policy/config/robot_dit_feat_{action_dim}.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    n_obs_steps = train_cfg["n_obs_steps"]
    n_action_steps = train_cfg["n_action_steps"]

    from dit_encoder import build_dit_encoder
    encoder = build_dit_encoder(feat_encoder)

    return DiT_Model(ckpt_file, n_obs_steps=n_obs_steps,
                     n_action_steps=n_action_steps, encoder=encoder)


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    TASK_ENV.get_instruction()  # vision-only; instruction unused

    actions = model.get_action(obs)
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model):
    model.reset_obs()
