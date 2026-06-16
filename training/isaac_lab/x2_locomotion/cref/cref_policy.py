"""CReF policy (P6-M2) — assembles depth encoder + cross-modal attention + GRU fusion +
auxiliary heads into the full raw-depth recurrent policy (roadmap §10.4). BLOCKED to run (torch).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .depth_encoder import DepthEncoder
from .cross_modal_attention import CrossModalAttention
from .gru_fusion import GruFusionActor
from .aux_heads import AuxiliaryHeads


class CReFPolicy(nn.Module):
    def __init__(self, proprio_dim: int = 47, token_dim: int = 64, embed_dim: int = 128,
                 action_dim: int = 12, use_aux: bool = True):
        super().__init__()
        self.depth_encoder = DepthEncoder(token_dim=token_dim)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU())
        self.cross_attn = CrossModalAttention(proprio_dim=128, token_dim=token_dim,
                                              embed_dim=embed_dim)
        self.fusion = GruFusionActor(proprio_dim=128, fused_dim=embed_dim,
                                     action_dim=action_dim)
        self.aux = AuxiliaryHeads(feat_dim=128 + embed_dim) if use_aux else None

    def forward(self, depth: torch.Tensor, proprio: torch.Tensor, hidden=None):
        tokens = self.depth_encoder(depth)
        p = self.proprio_encoder(proprio)
        depth_feat = self.cross_attn(p, tokens)
        action, hidden = self.fusion(p, depth_feat, hidden)
        aux_out = self.aux(torch.cat([p, depth_feat], dim=-1)) if self.aux is not None else {}
        return action, hidden, aux_out
