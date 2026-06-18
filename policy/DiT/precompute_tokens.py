"""Precompute frozen-encoder PATCH TOKENS for a task's zarr, once, for the
cross-attention DiT policy.

Like scripts/campaign/precompute_features.py but caches the pre-pool tokens
(T, N, C) instead of the pooled (T, D). The SAME build_frozen_encoder(enc,
return_tokens=True) is used here and at eval time -> bit-identical features.

Usage (run from the robotwin2 root, in the RoboTwin conda env):
    python policy/DiT/precompute_tokens.py <task> <setting> <num> <enc> [--bs 32] [--limit N]
"""
import sys
import os
import shutil
import argparse

sys.path.append("policy/DP")
import zarr
import numpy as np
import torch
from diffusion_policy.model.vision.vfm_encoder import build_frozen_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("setting")
    ap.add_argument("num")
    ap.add_argument("enc")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="only process first N frames (0 = all)")
    a = ap.parse_args()

    src = f"policy/DP/data/{a.task}-{a.setting}-{a.num}.zarr"
    suffix = f"-first{a.limit}" if a.limit else ""
    dst = f"policy/DP/data/feat_tok/{a.enc}/{a.task}-{a.setting}-{a.num}{suffix}.tokzarr"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(os.path.join(dst, ".zgroup")):
        print(f"[skip] {dst} exists")
        return

    z = zarr.open(src, "r")
    imgs = z["data"]["head_camera"]      # (T, 3, H, W) uint8 CHW
    T = imgs.shape[0]
    if a.limit:
        T = min(T, a.limit)
    enc = build_frozen_encoder(a.enc, return_tokens=True).cuda().eval()

    # Stream tokens straight to disk: keep only one batch in RAM, never the whole
    # (T, N, C) tensor. depth tokens are 1369x1024, so accumulating all T frames +
    # the np.concatenate would peak at ~2x ~50GB and stall under memory pressure.
    # Write to a .tmp dir and rename at the end so a crash never leaves a cache that
    # looks complete (the campaign's .zgroup check would otherwise skip the rerun).
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    out = zarr.open(tmp, "w")
    g = out.create_group("data")
    ds = None
    for i in range(0, T, a.bs):
        x = torch.from_numpy(imgs[i:i + a.bs][:]).float() / 255.0   # (b,3,H,W) in [0,1]
        with torch.no_grad():
            t = enc(x.cuda()).cpu().numpy().astype(np.float32)      # (b, N, C)
        if ds is None:
            N, C = t.shape[1], t.shape[2]
            ds = g.create_dataset("head_cam_tokens", shape=(T, N, C), chunks=(32, N, C), dtype="float32")
        ds[i:i + t.shape[0]] = t
        print(f"  {min(i + a.bs, T)}/{T}", end="\r")
    print(f"\ntokens shape {(T, ds.shape[1], ds.shape[2])}")

    state = z["data"]["state"][:T]
    action = z["data"]["action"][:T]
    g.create_dataset("state", data=state, chunks=(256,) + state.shape[1:])
    g.create_dataset("action", data=action, chunks=(256,) + action.shape[1:])
    m = out.create_group("meta")
    ee = z["meta"]["episode_ends"][:]
    if a.limit:
        # keep only complete episodes that fit in the first T frames; if none, make
        # one synthetic episode of length T so the dataset still builds.
        ee = ee[ee <= T]
        if len(ee) == 0:
            ee = np.array([T], dtype=ee.dtype)
    m.create_dataset("episode_ends", data=ee)
    os.rename(tmp, dst)   # only a fully-written cache gets the real name
    print(f"[ok] wrote {dst}  (episodes={len(ee)})")


if __name__ == "__main__":
    main()
