"""rsl_rl PPO runner config for the X2 standing task (Isaac Lab 2.3).

Mirrors the ppo block in configs/training_default.yaml. Asymmetric actor-critic is added once
privileged observations are introduced (rough/stairs stages); Stage A uses a shared MLP.
"""
from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class X2StandingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "x2_standing"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class X2FlatWalkPPORunnerCfg(X2StandingPPORunnerCfg):
    """Stage B flat walking — same PPO setup, more iterations (walking is harder)."""
    experiment_name = "x2_flat_walk"
    max_iterations = 3000
