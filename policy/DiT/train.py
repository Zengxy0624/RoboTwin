"""Train entrypoint for the DiT policy. Reuses DP's RobotWorkspace + Hydra.

The DiT code (dit.*, dit_dataset) lives under policy/DiT, but the `diffusion_policy`
package, the data/, and the checkpoints/ tree all live under policy/DP. The
workspace saves checkpoints with a RELATIVE path resolved against the cwd, so we
chdir into policy/DP to make DiT checkpoints land at
policy/DP/checkpoints/{encoder_tag}/... -- the exact path the eval driver reads.
"""
import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import os
import pathlib
import hydra
import yaml
from omegaconf import OmegaConf

_DIT_DIR = pathlib.Path(__file__).parent.resolve()
_DP_DIR = (_DIT_DIR / ".." / "DP").resolve()
# DP first so `import diffusion_policy` -> DP's package; DiT for dit.* / dit_dataset.
sys.path.insert(0, str(_DP_DIR))
sys.path.insert(0, str(_DIT_DIR))
os.chdir(_DP_DIR)  # checkpoints + zarr data paths resolve relative to policy/DP

from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402


def get_camera_config(camera_type):
    camera_config_path = _DP_DIR / "../../task_config/_camera_config.yml"
    assert camera_config_path.is_file(), "task config file is missing"
    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(_DIT_DIR / "diffusion_policy" / "config"),
)
def main(cfg: OmegaConf):
    head_camera_cfg = get_camera_config(cfg.head_camera_type)
    cfg.task.image_shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]
    # DiT reads obs['head_cam_tokens'] directly; no head_cam shape to patch.
    OmegaConf.resolve(cfg)
    cfg.task.image_shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    print(cfg.task.dataset.zarr_path, cfg.task_name)
    workspace.run()


if __name__ == "__main__":
    main()
