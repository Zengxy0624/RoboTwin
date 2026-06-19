"""Vision encoders as drop-in rgb_model for MultiImageObsEncoder.

Contract required by MultiImageObsEncoder (non-shared path):
    input : (B, 3, H, W)   image tensor (already resized + normalized by the
                           encoder's `key_transform_map`)
    output: (B, D)         flat feature vector, concatenated with low_dim obs.

For the cross-encoder comparison we read out *all* backbones with the SAME
spatial-aware head (parameter-free spatial-softmax / soft-argmax), so the
comparison reflects feature quality rather than which encoder best survives
global-average pooling. Frozen VFMs (DINOv3/SAM/...) keep their dense spatial
features; the from-scratch ResNet18 + spatial-softmax is the original Diffusion
Policy recipe.
"""
import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.vision.model_getter import get_resnet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
SAM3_MEAN = [0.5, 0.5, 0.5]
SAM3_STD = [0.5, 0.5, 0.5]

# Absolute paths to local VFM weights (gated / large HF repos, downloaded once).
_CHECKPOINT_ROOT = "/home/xzeng28/projects/visual-prior-interface/checkpoints"
_DINOV3_PATH = os.path.join(_CHECKPOINT_ROOT, "dinov3-vitl16")
_CLIP_PATH = os.path.join(_CHECKPOINT_ROOT, "clip-vit-large-patch14")
_DEPTH_V2_PATH = os.path.join(_CHECKPOINT_ROOT, "depth-anything-v2-large-hf")
_VJEPA2_PATH = os.path.join(_CHECKPOINT_ROOT, "vjepa2-vitl-fpc64-256")
_SAM3_PATH = os.path.join(_CHECKPOINT_ROOT, "sam3")


def build_frozen_encoder(tag, return_tokens=False):
    """Single source of truth for the 5 cacheable frozen encoders, so feature
    precompute and live eval construct the EXACT same rgb_model -> bit-identical
    features (train/eval consistency). Each maps raw [0,1] (B,3,H,W) -> (B,D).

    return_tokens=True instead returns the pre-pool patch tokens (B, N, C) for
    cross-attention policies (the DiT). For DINOv3 the prenorm LayerNorm is applied
    to the tokens when set on the encoder; for fmap backbones the (B,C,h,w) map is
    flattened to (B, h*w, C)."""
    rt = return_tokens
    if tag == "dinov3_ss":
        return DINOv3Encoder(model_path=_DINOV3_PATH, self_preproc=True, return_tokens=rt)
    if tag == "dinov3_ln":  # LayerNorm patches + mean-pool (1024-d) -- weak, deprecated
        return DINOv3Encoder(model_path=_DINOV3_PATH, self_preproc=True, prenorm=True, pool="mean", return_tokens=rt)
    if tag == "dinov3_sf":  # LayerNorm + spatial-flat 4x4 (16384-d): keeps spatial, the right readout
        return DINOv3Encoder(model_path=_DINOV3_PATH, self_preproc=True, prenorm=True, pool="spatial_flat", return_tokens=rt)
    if tag == "dinov3_cls":  # CLS token (1024-d) -- the standard DINOv3 global readout (valid once weights fixed)
        return DINOv3Encoder(model_path=_DINOV3_PATH, self_preproc=True, pool="cls", return_tokens=rt)
    if tag == "clip_ss":
        return CLIPEncoder(return_tokens=rt)
    if tag == "depth_ss":
        return DepthAnythingEncoder(return_tokens=rt)
    if tag == "depth_v2_ss":  # Depth-Anything V2 (better DINOv2 backbone), official 518 input
        return DepthAnythingEncoder(model_name=_DEPTH_V2_PATH, return_tokens=rt)
    if tag == "sam_ss":
        return SAMEncoder(return_tokens=rt)
    if tag == "sam3_ss":
        return SAM3Encoder(return_tokens=rt)
    if tag == "vjepa_ss":
        return VJEPA2Encoder(return_tokens=rt)
    raise ValueError(f"not a cacheable frozen encoder: {tag}")


class SpatialSoftmax(nn.Module):
    """Parameter-free spatial softmax (soft-argmax).

    (B, C, H, W) -> (B, 2C): for each channel, softmax over the H*W spatial
    positions, then the expected (x, y) coordinate (normalized to [-1, 1]).
    Encodes *where* each feature channel fires, preserving spatial layout.
    """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat):  # (B, C, H, W)
        B, C, H, W = feat.shape
        ys = torch.linspace(-1.0, 1.0, H, device=feat.device, dtype=feat.dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=feat.device, dtype=feat.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        gx = gx.reshape(-1)  # (H*W,)
        gy = gy.reshape(-1)
        attn = torch.softmax(feat.reshape(B, C, H * W) / self.temperature, dim=-1)  # (B,C,HW)
        ex = (attn * gx).sum(dim=-1)  # (B, C)
        ey = (attn * gy).sum(dim=-1)  # (B, C)
        return torch.cat([ex, ey], dim=1)  # (B, 2C)


def _reduce(feat_map, tokens, pool, ssm, prenorm=False):
    """feat_map: (B,C,H,W) spatial map; tokens: (B,N,C) or None for cls/mean.
    prenorm: parameter-free LayerNorm over channels per spatial location before
    pooling. Dense-ViT features (DINOv3/SAM/...) have a large location-shared
    component that dominates global pooling and makes the pooled vector ~constant
    across images; LayerNorm strips it so the discriminative part survives."""
    if prenorm:
        fm = feat_map.permute(0, 2, 3, 1)                 # B,H,W,C
        fm = F.layer_norm(fm, (fm.shape[-1],))
        feat_map = fm.permute(0, 3, 1, 2)                 # B,C,H,W
    if pool == "spatial_softmax":
        return ssm(feat_map)
    elif pool == "mean":
        return feat_map.mean(dim=(2, 3))
    elif pool == "spatial_flat":
        # keep spatial layout: downsample to 4x4 and flatten -> (B, C*16). Preserves
        # "where" info that global pooling destroys (needed for DINOv3's dense features).
        return F.adaptive_avg_pool2d(feat_map, 4).flatten(1)
    else:
        raise ValueError(f"unknown pool: {pool}")


class DINOv3Encoder(nn.Module):
    """Frozen DINOv3 ViT. Reads out patch tokens with spatial-softmax (default)
    or global mean / cls. Expects 224x224 ImageNet-normalized input.
    """

    def __init__(self, model_path, num_prefix_tokens=5, freeze=True,
                 pool="spatial_softmax", img_size=224, self_preproc=False,
                 prenorm=False, temperature=1.0, return_tokens=False):
        super().__init__()
        from transformers import AutoModel
        self.num_prefix_tokens = num_prefix_tokens  # 1 CLS + 4 register
        self.freeze = freeze
        self.pool = pool
        self.prenorm = prenorm
        self.return_tokens = return_tokens
        # self_preproc=True: take a raw [0,1] image and do DINOv3's resize+ImageNet-norm
        # internally, so precompute/train/eval all share one forward (bit-identical features).
        self.self_preproc = self_preproc
        self.pre = _Preproc(img_size, IMAGENET_MEAN, IMAGENET_STD) if self_preproc else None
        self.model = AutoModel.from_pretrained(model_path)
        self.ssm = SpatialSoftmax(temperature)
        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, x):
        ctx = torch.no_grad() if self.freeze else nullcontext()
        with ctx:
            if self.pre is not None:
                x = self.pre(x)
            tokens = self.model(x).last_hidden_state            # (B, P+N, C)
            if self.pool == "cls" and not self.return_tokens:
                return tokens[:, 0]
            patches = tokens[:, self.num_prefix_tokens:, :]     # (B, N, C)
            B, N, C = patches.shape
            if self.return_tokens:
                if self.prenorm:
                    patches = F.layer_norm(patches, (patches.shape[-1],))
                return patches                                  # (B, N, C)
            h = w = int(round(N ** 0.5))
            fmap = patches.transpose(1, 2).reshape(B, C, h, w)  # (B, C, h, w)
        return _reduce(fmap, patches, self.pool, self.ssm, self.prenorm)


class ResNetSpatialEncoder(nn.Module):
    """ResNet (from scratch by default) read out with spatial-softmax instead of
    the usual global-avg-pool. This is the original Diffusion Policy recipe.
    Trainable (not frozen) so it serves as the scratch-trained baseline.
    """

    def __init__(self, name="resnet18", weights=None, pool="spatial_softmax",
                 temperature=1.0):
        super().__init__()
        # get_resnet sets fc=Identity; we also drop avgpool to keep the (B,C,h,w) map.
        backbone = get_resnet(name=name, weights=weights)
        self.stem = nn.Sequential(*list(backbone.children())[:-2])  # up to layer4 -> (B,512,h,w)
        self.pool = pool
        self.ssm = SpatialSoftmax(temperature)

    def forward(self, x):
        fmap = self.stem(x)  # (B, C, h, w)
        return _reduce(fmap, None, self.pool, self.ssm)


class _Preproc(nn.Module):
    """Resize [0,1] image to the backbone's expected size and normalize.
    Used by the VFM adapters so each owns its own preprocessing (their config
    sets imagenet_norm=False, resize_shape=null)."""

    def __init__(self, size, mean, std):
        super().__init__()
        self.size = size
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):  # x: (B,3,H,W) in [0,1]
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std


class _SamPreproc(nn.Module):
    """Official SAM preprocessing: resize the LONGEST edge to `size` preserving
    aspect ratio, normalize, then zero-pad bottom/right to size x size. (The plain
    square resize distorts aspect ratio, which SAM was not trained for.)"""

    def __init__(self, size, mean, std):
        super().__init__()
        self.size = size
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):  # x: (B,3,H,W) in [0,1]
        H, W = x.shape[-2:]
        scale = self.size / max(H, W)
        nh, nw = int(H * scale + 0.5), int(W * scale + 0.5)
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return F.pad(x, (0, self.size - nw, 0, self.size - nh))  # pad right/bottom with 0


class _FrozenVFM(nn.Module):
    """Shared frozen-backbone plumbing + spatial-softmax readout."""

    def __init__(self, freeze, pool, temperature, return_tokens=False):
        super().__init__()
        self.freeze = freeze
        self.pool = pool
        self.ssm = SpatialSoftmax(temperature)
        self.return_tokens = return_tokens

    @staticmethod
    def _map_to_tokens(fmap):
        # (B, C, h, w) -> (B, h*w, C) for cross-attention conditioning
        B, C, h, w = fmap.shape
        return fmap.flatten(2).transpose(1, 2)

    def _finish_freeze(self):
        if self.freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    @staticmethod
    def _tokens_to_map(tokens, drop_prefix):
        patches = tokens[:, drop_prefix:, :]          # (B, N, C)
        B, N, C = patches.shape
        h = w = int(round(N ** 0.5))
        return patches.transpose(1, 2).reshape(B, C, h, w)


def _load_prefixed_safetensors(module, path, prefix):
    """Load only one submodule from a safetensors checkpoint."""
    from safetensors import safe_open

    state_dict = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefix):
                state_dict[key[len(prefix):]] = f.get_tensor(key)
    module.load_state_dict(state_dict, strict=True)


class CLIPEncoder(_FrozenVFM):
    """Frozen CLIP ViT vision tower -> spatial-softmax. 224 input, CLIP norm."""

    def __init__(self, model_name=_CLIP_PATH, freeze=True,
                 pool="spatial_softmax", img_size=224, temperature=1.0, return_tokens=False):
        super().__init__(freeze, pool, temperature, return_tokens)
        from transformers import CLIPVisionModel
        self.pre = _Preproc(img_size, CLIP_MEAN, CLIP_STD)
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self._finish_freeze()

    def forward(self, x):
        ctx = torch.no_grad() if self.freeze else nullcontext()
        with ctx:
            tokens = self.model(self.pre(x)).last_hidden_state  # (B, 1+N, C)
            fmap = self._tokens_to_map(tokens, drop_prefix=1)   # drop CLS
            if self.return_tokens:
                return self._map_to_tokens(fmap)
        return _reduce(fmap, None, self.pool, self.ssm)


class SAMEncoder(_FrozenVFM):
    """Frozen SAM vision encoder -> (B,256,64,64) map -> spatial-softmax.
    1024 input, ImageNet norm. Heaviest (64x64)."""

    def __init__(self, model_name="facebook/sam-vit-large", freeze=True,
                 pool="spatial_softmax", img_size=1024, chunk_size=4, temperature=1.0, return_tokens=False):
        super().__init__(freeze, pool, temperature, return_tokens)
        from transformers import SamModel
        self.pre = _SamPreproc(img_size, IMAGENET_MEAN, IMAGENET_STD)  # aspect-preserve + pad
        self.chunk_size = chunk_size  # SAM @1024 attention is O((HW)^2); chunk the
        self.model = SamModel.from_pretrained(model_name)  # batch to cap peak memory.
        self._finish_freeze()

    def forward(self, x):
        ctx = torch.no_grad() if self.freeze else nullcontext()
        with ctx:
            img = self.pre(x)  # (B,3,1024,1024)
            maps = [self.model.get_image_embeddings(img[i:i + self.chunk_size])
                    for i in range(0, img.shape[0], self.chunk_size)]
            fmap = torch.cat(maps, dim=0)  # (B, 256, 64, 64)
            if self.return_tokens:
                return self._map_to_tokens(fmap)  # (B, 4096, 256)
        return _reduce(fmap, None, self.pool, self.ssm)


class DepthAnythingEncoder(_FrozenVFM):
    """Frozen Depth-Anything backbone (DINOv2) deepest feature map ->
    spatial-softmax. 224 input, ImageNet norm."""

    def __init__(self, model_name=_DEPTH_V2_PATH, freeze=True,
                 pool="spatial_softmax", img_size=518, temperature=1.0, return_tokens=False):  # 518 = official (14x37)
        super().__init__(freeze, pool, temperature, return_tokens)
        from transformers import AutoModelForDepthEstimation
        self.pre = _Preproc(img_size, IMAGENET_MEAN, IMAGENET_STD)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self._finish_freeze()

    def forward(self, x):
        ctx = torch.no_grad() if self.freeze else nullcontext()
        with ctx:
            out = self.model.backbone.forward_with_filtered_kwargs(self.pre(x))
            tokens = out.feature_maps[-1]                        # (B, 1+N, C)
            fmap = self._tokens_to_map(tokens, drop_prefix=1)    # drop CLS
            if self.return_tokens:
                return self._map_to_tokens(fmap)
        return _reduce(fmap, None, self.pool, self.ssm)


class VJEPA2Encoder(_FrozenVFM):
    """Frozen V-JEPA2 (video model) applied to a single frame by tiling it into
    a short clip, then averaging the temporal tubelets -> spatial-softmax.
    256 input, ImageNet norm. Expensive (runs the video backbone)."""

    def __init__(self, model_name=_VJEPA2_PATH, freeze=True,
                 pool="spatial_softmax", img_size=256, n_frames=16, temperature=1.0, return_tokens=False):  # 16 = official single-image
        super().__init__(freeze, pool, temperature, return_tokens)
        from transformers import AutoModel
        self.pre = _Preproc(img_size, IMAGENET_MEAN, IMAGENET_STD)
        self.n_frames = n_frames
        self.model = AutoModel.from_pretrained(model_name)
        self._finish_freeze()

    def forward(self, x):
        ctx = torch.no_grad() if self.freeze else nullcontext()
        with ctx:
            img = self.pre(x)                                   # (B,3,H,W)
            vid = img.unsqueeze(1).repeat(1, self.n_frames, 1, 1, 1)  # (B,T,3,H,W)
            feats = self.model.get_vision_features(vid)         # (B, T'*Npatch, C)
            B, N, C = feats.shape
            spatial = (img.shape[-1] // 16) ** 2                # patches per frame
            T = N // spatial
            tokens = feats.reshape(B, T, spatial, C).mean(dim=1)  # (B, Npatch, C)
            if self.return_tokens:
                return tokens                                   # (B, 256, C)
            fmap = self._tokens_to_map(tokens, drop_prefix=0)
        return _reduce(fmap, None, self.pool, self.ssm)


class SAM3Encoder(_FrozenVFM):
    """Frozen SAM3 / Perception Encoder ViT tokens.

    The HF repo is a full SAM3 video/detector checkpoint, but the DiT only needs
    the vision encoder. We load the `detector_model.vision_encoder.*` weights
    into `Sam3VisionModel` and read the raw ViT patch tokens. A 224 input with
    SAM3's patch size 14 gives a 16x16 grid = 256 tokens.
    """

    def __init__(self, model_path=_SAM3_PATH, freeze=True,
                 pool="spatial_softmax", img_size=224, temperature=1.0, return_tokens=False):
        super().__init__(freeze, pool, temperature, return_tokens)
        from transformers import AutoConfig, Sam3VisionModel

        cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
        vision_config = cfg.detector_config.vision_config
        # SAM3 ViT RoPE is created for config.backbone_config.image_size at init.
        # Match it to our token-budget input size (224 -> 16x16 tokens).
        vision_config.backbone_config.image_size = img_size
        self.pre = _Preproc(img_size, SAM3_MEAN, SAM3_STD)
        self.model = Sam3VisionModel(vision_config)
        _load_prefixed_safetensors(
            self.model,
            os.path.join(model_path, "model.safetensors"),
            "detector_model.vision_encoder.",
        )
        self._finish_freeze()

    def forward(self, x):
        ctx = torch.no_grad() if self.freeze else nullcontext()
        with ctx:
            tokens = self.model(self.pre(x)).last_hidden_state  # (B, 256, 1024) at 224
            if self.return_tokens:
                return tokens
            B, N, C = tokens.shape
            h = w = int(round(N ** 0.5))
            if h * w != N:
                raise RuntimeError(f"SAM3 token count is not square: {N}")
            fmap = tokens.transpose(1, 2).reshape(B, C, h, w)
        return _reduce(fmap, tokens, self.pool, self.ssm)

