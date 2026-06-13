"""x2_actuator_cfg (P3-M2-T4) — actuator / PD parameters for the X2 legs.

Builds Isaac Lab ImplicitActuatorCfg objects for the 12 leg joints from
configs/joint_limits_x2_ultra.yaml (limits/effort/velocity loaded from config, never
hardcoded — AGENTS.md §3). PD stiffness/damping are first estimates and MUST be validated
in sim (robot stands under gravity with stable PD) before training.

BLOCKED: requires Isaac Lab installed. Importing this module needs ``isaaclab``.
"""
from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402  (Isaac Lab)

from x2_common import config_loader
from .x2_joint_map import aimdk_leg_order

# First-estimate PD gains by joint group (validate in sim before trusting).
_STIFFNESS = {"hip": 150.0, "knee": 200.0, "ankle": 80.0}
_DAMPING = {"hip": 5.0, "knee": 6.0, "ankle": 3.0}


def _group(joint_name: str) -> str:
    if "knee" in joint_name:
        return "knee"
    if "ankle" in joint_name:
        return "ankle"
    return "hip"


def build_actuator_cfgs() -> dict[str, ImplicitActuatorCfg]:
    """Return one actuator cfg per leg joint, with limits from the config file."""
    limits = config_loader.load_config("joint_limits_x2_ultra")
    joints = limits["joints"]
    cfgs: dict[str, ImplicitActuatorCfg] = {}
    for name in aimdk_leg_order():
        jl = joints[name]
        g = _group(name)
        cfgs[name] = ImplicitActuatorCfg(
            joint_names_expr=[name],
            effort_limit=float(jl["effort_max"]),
            velocity_limit=float(jl["vel_max"]),
            stiffness=_STIFFNESS[g],
            damping=_DAMPING[g],
        )
    return cfgs
