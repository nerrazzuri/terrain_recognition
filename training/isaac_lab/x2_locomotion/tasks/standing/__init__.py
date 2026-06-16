"""Register the X2 standing Gym task (requires gymnasium + Isaac Lab)."""
try:
    import gymnasium as gym

    from . import x2_standing_env_cfg
    from ...agents.rsl_rl_ppo_cfg import X2StandingPPORunnerCfg

    gym.register(
        id="Isaac-X2-Standing-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": x2_standing_env_cfg.X2StandingEnvCfg,
            "rsl_rl_cfg_entry_point": X2StandingPPORunnerCfg,
        },
    )
except ModuleNotFoundError:
    pass  # gymnasium/isaaclab not installed — pure-logic imports still work
