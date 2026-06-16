"""Depth encoder (P6-M1-T2) — CNN patch embedding -> depth tokens. BLOCKED to run (torch)."""
from __future__ import annotations

import torch
import torch.nn as nn


class DepthEncoder(nn.Module):
    """Encode a (B, 1, H, W) depth crop into (B, num_tokens, token_dim) depth tokens."""

    def __init__(self, token_dim: int = 64, patch: int = 8):
        super().__init__()
        self.token_dim = token_dim
        # strided conv patch embedding -> tokens
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(64, token_dim, kernel_size=patch, stride=patch),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        x = self.stem(depth)                       # (B, token_dim, h, w)
        b, c, h, w = x.shape
        return x.reshape(b, c, h * w).permute(0, 2, 1)  # (B, tokens, token_dim)
