"""Small DiT building blocks (vision-only, no language, no adaLN).

Simplified from policy/RDT/models/rdt/blocks.py but with NO timm dependency
(the RoboTwin conda env has no timm). RmsNorm / Attention / Mlp are reimplemented
in plain torch. The block is pre-norm: self-attn -> cross-attn (over frozen-VFM
image tokens) -> FFN, all RmsNorm, no adaLN modulation.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RmsNorm(nn.Module):
    """Root-mean-square layer norm (timm-compatible signature)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TimestepEmbedder(nn.Module):
    """Embeds scalar diffusion timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class SelfAttention(nn.Module):
    """Multi-head self-attention with qk RmsNorm, torch SDPA."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RmsNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RmsNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class CrossAttention(nn.Module):
    """Cross-attention: query = action/state tokens, key/value = image tokens."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = RmsNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RmsNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, c):
        B, N, C = x.shape
        _, L, _ = c.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)


class DiTBlock(nn.Module):
    """Pre-norm DiT block: self-attn -> cross-attn over img tokens -> FFN."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RmsNorm(hidden_size)
        self.attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True)
        self.norm2 = RmsNorm(hidden_size)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True)
        self.norm3 = RmsNorm(hidden_size)
        self.ffn = Mlp(hidden_size, hidden_features=int(hidden_size * mlp_ratio))

    def forward(self, x, c):
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), c)
        x = x + self.ffn(self.norm3(x))
        return x


class FinalLayer(nn.Module):
    """Final projection to action dimension."""

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size)
        self.ffn_final = Mlp(hidden_size, hidden_features=hidden_size, out_features=out_channels)

    def forward(self, x):
        return self.ffn_final(self.norm_final(x))


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """embed_dim: per-position dim; pos: (M,) -> (M, embed_dim)."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    if not isinstance(pos, np.ndarray):
        pos = np.array(pos, dtype=np.float64)
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)
