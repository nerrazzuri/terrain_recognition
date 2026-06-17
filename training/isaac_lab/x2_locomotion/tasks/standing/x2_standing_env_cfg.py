"""X2 Stage-A standing environment (Isaac Lab 2.3, manager-based).

Flat ground, 12-DoF leg position actions, learn to stand/balance (the passive stance tips
without a controller — the policy's job is to hold it upright). Built from the stable
``isaaclab.envs.mdp`` term API so it does not depend on isaaclab_tasks internals.

Reward shape (roadmap §8.9, standing slice): stay alive + at target height + upright; penalise
base motion, torques/accel/action-rate, joint-limit proximity. Terminate on fall / bad tilt /
torso-or-pelvis ground contact.
"""
from __future__ import annotations

import math

import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ...robots.x2_robot_cfg import build_robot_cfg, BASE_HEIGHT_M, TERMINATION_BODY_NAMES
from ...robots.x2_joint_map import aimdk_leg_order


def _gait_phase_zeros(env) -> torch.Tensor:
    """Placeholder gait phase (sin/cos) — zeros for Stage A standing. Shape: (N, 2)."""
    return torch.zeros(env.num_envs, 2, device=env.device)


def _height_samples_zeros(env) -> torch.Tensor:
    """Placeholder height samples (11×11 grid) — zeros for Stage A flat ground. Shape: (N, 121)."""
    return torch.zeros(env.num_envs, 121, device=env.device)

LEG_JOINTS = aimdk_leg_order()


@configclass
class X2SceneCfg(InteractiveSceneCfg):
    """Flat ground + X2 + contact sensor + light."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0),
    )
    robot: ArticulationCfg = build_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


@configclass
class CommandsCfg:
    """Zero velocity command (stand still); keeps the 3 command obs for later walking stages."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(10.0, 10.0), rel_standing_envs=1.0,
        heading_command=False, debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
    )


@configclass
class ActionsCfg:
    """12-DoF leg joint-position offsets from the default pose (roadmap §8.3)."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=LEG_JOINTS, scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Order MUST match OBSERVATION_LAYOUT in tasks/common/observations.py (168-dim contract).
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        # Stage A: gait phase and height samples are zeros (standing on flat ground).
        # Replace with real clock/raycaster terms when the walking curriculum starts.
        gait_phase = ObsTerm(func=_gait_phase_zeros)
        height_samples = ObsTerm(func=_height_samples_zeros)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.6, 1.2), "dynamic_friction_range": (0.4, 1.0),
                "restitution_range": (0.0, 0.0), "num_buckets": 64})
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.1, 0.1)},
                "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot")})
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale, mode="reset",
        params={"position_range": (0.95, 1.05), "velocity_range": (0.0, 0.0)})
    push = EventTerm(
        func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}})


@configclass
class RewardsCfg:
    """Locomotion reward set. Zero command ⇒ stand planted; non-zero ⇒ walk.

    Velocity tracking + a hip roll/yaw deviation penalty replace the bare alive bonus, which
    is what caused the wide-splay / drifting stance (the policy gamed "stay upright" by
    spreading its legs). Tracking the (zero) command keeps it planted; the deviation penalty
    keeps feet under the hips.
    """

    # --- task: track the commanded base velocity ---
    # std tightened 0.5 -> 0.25 so velocity error actually bites: at std=0.5 a robot standing
    # still scored ~exp(-cmd^2/0.5^2) ~ 0.9 of this reward for small commands, letting the
    # policy "succeed" by standing (Stage B walked nowhere on video). With 0.25 + Stage-B's
    # 0.3-0.8 m/s commands, standing scores near zero, so it must actually move.
    track_lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0,
                               params={"command_name": "base_velocity", "std": 0.25})
    track_ang_vel_z = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.5,
                              params={"command_name": "base_velocity", "std": 0.25})
    # --- posture / stability ---
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-1.0,
                          params={"target_height": BASE_HEIGHT_M})
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # keep a natural, narrow stance (anti-splay)
    hip_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.5,
                            params={"asset_cfg": SceneEntityCfg(
                                "robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])})
    # NOTE: a feet_air_time stepping reward (encourages clean steps) lives in the locomotion
    # task mdp (isaaclab_tasks...velocity.mdp), not core isaaclab.envs.mdp — add it once base
    # walking works. Velocity tracking already drives forward motion.
    # --- effort / smoothness ---
    dof_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-5)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # --- fall penalty ---
    terminating = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg(
            "contact_forces", body_names=TERMINATION_BODY_NAMES),
            "threshold": 1.0})


@configclass
class X2StandingEnvCfg(ManagerBasedRLEnvCfg):
    scene: X2SceneCfg = X2SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # timing: 200 Hz physics, 50 Hz policy (decimation 4) — configs/training_default.yaml
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
