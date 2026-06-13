"""Cross-modal attention (P6-M2-T1) — proprio query attends to depth tokens.

BLOCKED to run (torch). Architecture per roadmap §10.4.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """Single-head-ish cross attention: a proprio query attends over depth tokens."""

    def __init__(self, proprio_dim: int, token_dim: int, embed_dim: int = 128,
                 num_heads: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(proprio_dim, embed_dim)
        self.kv_proj = nn.Linear(token_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, proprio: torch.Tensor, depth_tokens: torch.Tensor) -> torch.Tensor:
        # proprio: (B, proprio_dim) -> query (B, 1, embed)
        q = self.q_proj(proprio).unsqueeze(1)
        kv = self.kv_proj(depth_tokens)            # (B, tokens, embed)
        attended, _ = self.attn(q, kv, kv)         # (B, 1, embed)
        return self.out(attended.squeeze(1))       # (B, embed)
