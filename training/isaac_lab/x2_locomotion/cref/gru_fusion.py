"""GRU recurrent fusion + actor head (P6-M2-T2). BLOCKED to run (torch).

Combines the proprio feature and the depth-aware (cross-attended) feature, integrates over
time with a GRU, and produces the joint action (roadmap §10.4).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GruFusionActor(nn.Module):
    def __init__(self, proprio_dim: int, fused_dim: int, hidden: int = 256,
                 action_dim: int = 12):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(proprio_dim + fused_dim, hidden), nn.ELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.actor = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(),
                                   nn.Linear(hidden, action_dim))

    def forward(self, proprio_feat: torch.Tensor, depth_feat: torch.Tensor,
                hidden_state=None):
        x = torch.cat([proprio_feat, depth_feat], dim=-1)
        x = self.pre(x).unsqueeze(1)               # (B, 1, hidden)
        out, h = self.gru(x, hidden_state)
        action = self.actor(out.squeeze(1))
        return action, h
