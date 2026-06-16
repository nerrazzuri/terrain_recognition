"""Policy / value networks (P4-M1-T3) per roadmap §8.6.

Asymmetric actor-critic: a height-map encoder + proprio encoder feed the actor; the critic
takes privileged observations. Dims come from configs/training_default.yaml.

BLOCKED to run: requires PyTorch. The module imports torch; it is exported to ONNX in
scripts/export_onnx.py and numerically checked against this reference (P4-M4-T4).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .observations import OBSERVATION_DIM


def _mlp(sizes: list[int], activation=nn.ELU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Height-map actor-critic. height_dim defaults to the 121 height samples."""

    def __init__(self, action_dim: int = 12, height_dim: int = 121,
                 privileged_dim: int | None = None):
        super().__init__()
        proprio_dim = OBSERVATION_DIM - height_dim          # 47
        self.height_dim = height_dim
        self.proprio_dim = proprio_dim

        self.height_encoder = _mlp([height_dim, 128, 64])
        self.proprio_encoder = _mlp([proprio_dim, 256, 128])
        self.actor = _mlp([64 + 128, 256, 256, action_dim])
        crit_in = privileged_dim if privileged_dim is not None else OBSERVATION_DIM
        self.critic = _mlp([crit_in, 512, 256, 1])

    def _split(self, obs: torch.Tensor):
        return obs[..., : self.proprio_dim], obs[..., self.proprio_dim:]

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        proprio, height = self._split(obs)
        latent = torch.cat([self.height_encoder(height), self.proprio_encoder(proprio)], dim=-1)
        return self.actor(latent)

    def value(self, privileged_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(privileged_obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # deployment forward: observation -> action (what export_onnx traces)
        return self.act(obs)
