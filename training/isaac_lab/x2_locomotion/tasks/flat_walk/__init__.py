"""Register the X2 flat-walking Gym task."""
import gymnasium as gym

from . import x2_flat_walk_env_cfg
from ...agents.rsl_rl_ppo_cfg import X2FlatWalkPPORunnerCfg

gym.register(
    id="Isaac-X2-FlatWalk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": x2_flat_walk_env_cfg.X2FlatWalkEnvCfg,
        "rsl_rl_cfg_entry_point": X2FlatWalkPPORunnerCfg,
    },
)
