"""Auxiliary heads (P6-M2-T3) to stabilise raw-depth learning (roadmap §10.7).

Terrain-type classification, height-sample reconstruction, safe-foothold probability, and
depth-validity-mask prediction. Removed/disabled for deployment if not needed.
BLOCKED to run (torch).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AuxiliaryHeads(nn.Module):
    def __init__(self, feat_dim: int, num_terrain_classes: int = 10,
                 height_dim: int = 121, num_feet: int = 2):
        super().__init__()
        self.terrain_type = nn.Linear(feat_dim, num_terrain_classes)
        self.height_recon = nn.Linear(feat_dim, height_dim)
        self.foothold_prob = nn.Linear(feat_dim, num_feet)
        self.depth_valid = nn.Linear(feat_dim, height_dim)

    def forward(self, feat: torch.Tensor) -> dict:
        return {
            "terrain_type_logits": self.terrain_type(feat),
            "height_reconstruction": self.height_recon(feat),
            "foothold_prob": torch.sigmoid(self.foothold_prob(feat)),
            "depth_validity": torch.sigmoid(self.depth_valid(feat)),
        }
