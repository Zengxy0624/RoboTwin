"""DiT's own frozen-encoder builder: bare encoder names -> raw patch tokens.

The cross-attention DiT reads PRE-POOL patch tokens (return_tokens=True), so it does
NOT use the DP campaign's spatial-softmax "_ss" readout variants -- hence bare names
(dinov3, clip, depth_v2, sam3, vjepa) and no "_ss" here.

The encoder *implementations* (DINOv3Encoder, ...) are reused from DP's vfm_encoder
so the live-eval features stay bit-identical to precompute_tokens.py. Only this thin
dispatch is DiT-local; duplicating the model classes would risk train/eval drift.
"""
from diffusion_policy.model.vision.vfm_encoder import (
    DINOv3Encoder,
    CLIPEncoder,
    DepthAnythingEncoder,
    SAMEncoder,
    SAM3Encoder,
    VJEPA2Encoder,
    _DINOV3_PATH,
)


def build_dit_encoder(name):
    """Frozen VFM -> (B, N, C) patch tokens for the cross-attention DiT."""
    if name == "dinov3":
        return DINOv3Encoder(model_path=_DINOV3_PATH, self_preproc=True, return_tokens=True)
    if name == "clip":
        return CLIPEncoder(return_tokens=True)
    if name in {"depth", "depth_v2"}:
        return DepthAnythingEncoder(return_tokens=True)
    if name == "sam":
        return SAMEncoder(return_tokens=True)
    if name == "sam3":
        return SAM3Encoder(return_tokens=True)
    if name == "vjepa":
        return VJEPA2Encoder(return_tokens=True)
    raise ValueError(f"unknown DiT encoder: {name!r}")
