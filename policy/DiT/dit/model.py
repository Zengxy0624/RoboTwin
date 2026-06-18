"""Small vision-only DiT action denoiser with cross-attention over frozen-VFM
patch tokens. No language. Mirrors RDT's structure but tiny.

forward(noisy_action (B,H,A), timestep (B,), img_tokens (B,L,Cimg), state (B,Sdim))
-> predicted noise (B,H,A).

Sequence fed through the transformer: [t, state, a_1..a_H] (length H+2), with a
learnable position embedding. Every block cross-attends the SAME image tokens
(projected once). The last H tokens are read out and projected to action dim.
"""
import torch
import torch.nn as nn

from .blocks import (
    DiTBlock,
    FinalLayer,
    TimestepEmbedder,
    get_1d_sincos_pos_embed_from_grid,
)


class DiT(nn.Module):

    def __init__(
        self,
        action_dim,
        state_dim,
        img_dim,
        horizon,
        hidden_size=384,
        depth=6,
        num_heads=6,
        mlp_ratio=1.0,
        max_img_tokens=1024,
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.action_dim = action_dim

        self.action_adaptor = nn.Linear(action_dim, hidden_size)
        self.state_adaptor = nn.Linear(state_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.img_adaptor = nn.Linear(img_dim, hidden_size)

        # learnable position embeddings
        seq_len = horizon + 2  # [t, state, a_1..a_H]
        self.x_pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_size))
        self.img_pos_embed = nn.Parameter(torch.zeros(1, max_img_tokens, hidden_size))

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, action_dim)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_basic_init)

        # sincos init for the action-sequence position embedding
        pos = get_1d_sincos_pos_embed_from_grid(self.hidden_size, torch.arange(self.x_pos_embed.shape[1]))
        self.x_pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        nn.init.normal_(self.img_pos_embed, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(self, noisy_action, timestep, img_tokens, state):
        # build the query sequence
        a = self.action_adaptor(noisy_action)              # (B, H, hidden)
        s = self.state_adaptor(state).unsqueeze(1)         # (B, 1, hidden)
        t = self.t_embedder(timestep).unsqueeze(1)         # (B, 1, hidden)
        x = torch.cat([t, s, a], dim=1)                    # (B, H+2, hidden)
        x = x + self.x_pos_embed

        # image conditioning tokens (projected once, cross-attended every block)
        c = self.img_adaptor(img_tokens)                   # (B, L, hidden)
        c = c + self.img_pos_embed[:, : c.shape[1]]

        for block in self.blocks:
            x = block(x, c)

        x = x[:, -self.horizon:]                           # keep action tokens
        return self.final_layer(x)                         # (B, H, action_dim)
