from __future__ import annotations

import copy
import math
import os

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.amp.amp_constants import AMP_ISAAC_BODY_IDS, AMP_ISAAC_ANCHOR_ID
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as BASE_G1_CFG
from unitree_rl_lab.assets.robots.unitree import (
    ARMATURE_5020,
    ARMATURE_7520_14,
    ARMATURE_7520_22,
    STIFFNESS_5020,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
    DAMPING_5020,
    DAMPING_7520_14,
    DAMPING_7520_22,
)
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp import amp_observations as amp_mdp
from unitree_rl_lab.tasks.locomotion.mdp import amp_events as amp_event_mdp
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_amp_cfg import AMP_MOTION_DIR

from .velocity_env_cfg import RobotEnvCfg as G1VelocityEnvCfg
from .velocity_env_cfg import ObservationsCfg as VelocityObservationsCfg
from .velocity_env_cfg import RobotSceneCfg as VelocityRobotSceneCfg

# USD file path (relative to the unitree_rl_lab package root).
_AMP_USD = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..", "..", "..",
    "g1_29dof_rev_1_0.usd",
))



# -----------------------------------------------------------------------------
# Hardcoded Isaac body index helpers (no name-based resolution).
# -----------------------------------------------------------------------------
def _amp_body_cfg():
    return SceneEntityCfg("robot", body_ids=AMP_ISAAC_BODY_IDS.tolist())

def _amp_anchor_cfg():
    return SceneEntityCfg("robot", body_ids=[int(AMP_ISAAC_ANCHOR_ID)])


# -----------------------------------------------------------------------------
# AMP_MJLab-compatible G1 actuator parameters
# -----------------------------------------------------------------------------
# These reproduce AMP_MJLab's motor-derived PD gains:
#   stiffness = armature * (2*pi*10)^2
#   damping   = 2 * damping_ratio * armature * (2*pi*10)
#   damping_ratio = 2.0
# The action scale follows AMP_MJLab:
#   action_scale = 0.25 * effort_limit / stiffness
# -----------------------------------------------------------------------------

# ARMATURE_5020, 7520-14, 7520-22 imported from unitree.py.
# 5010-16 is not defined in unitree.py — define here.
ARMATURE_5010_16 = 0.0021812

STIFFNESS_5010_16 = ARMATURE_5010_16 * (10.0 * 2.0 * math.pi) ** 2
DAMPING_5010_16 = 2.0 * 2.0 * ARMATURE_5010_16 * (10.0 * 2.0 * math.pi)

# Parallel 5020 for waist and ankles (4-bar linkage, 2 actuators).
STIFFNESS_5020_PARALLEL = STIFFNESS_5020 * 2.0
DAMPING_5020_PARALLEL = DAMPING_5020 * 2.0
ARMATURE_5020_PARALLEL = ARMATURE_5020 * 2.0


def _amp_action_scale(effort_limit: float, stiffness: float) -> float:
    return 0.25 * effort_limit / stiffness


AMP_ACTION_SCALE = {
    # N5020-16: shoulders, elbows, wrist roll
    ".*_elbow_joint": _amp_action_scale(25.0, STIFFNESS_5020),
    ".*_shoulder_pitch_joint": _amp_action_scale(25.0, STIFFNESS_5020),
    ".*_shoulder_roll_joint": _amp_action_scale(25.0, STIFFNESS_5020),
    ".*_shoulder_yaw_joint": _amp_action_scale(25.0, STIFFNESS_5020),
    ".*_wrist_roll_joint": _amp_action_scale(25.0, STIFFNESS_5020),
    # N7520-14.3: hip yaw and waist yaw
    ".*_hip_yaw_joint": _amp_action_scale(88.0, STIFFNESS_7520_14),
    "waist_yaw_joint": _amp_action_scale(88.0, STIFFNESS_7520_14),
    # N7520-22.5: hip pitch, hip roll, knee
    ".*_hip_pitch_joint": _amp_action_scale(139.0, STIFFNESS_7520_22),
    ".*_hip_roll_joint": _amp_action_scale(139.0, STIFFNESS_7520_22),
    ".*_knee_joint": _amp_action_scale(139.0, STIFFNESS_7520_22),
    # N5010-16: wrist pitch and wrist yaw
    ".*_wrist_pitch_joint": _amp_action_scale(10.0, STIFFNESS_5010_16),
    ".*_wrist_yaw_joint": _amp_action_scale(10.0, STIFFNESS_5010_16),
    # parallel N5020: waist pitch/roll and ankles
    "waist_pitch_joint": _amp_action_scale(50.0, STIFFNESS_5020_PARALLEL),
    "waist_roll_joint": _amp_action_scale(50.0, STIFFNESS_5020_PARALLEL),
    ".*_ankle_pitch_joint": _amp_action_scale(50.0, STIFFNESS_5020_PARALLEL),
    ".*_ankle_roll_joint": _amp_action_scale(50.0, STIFFNESS_5020_PARALLEL),
}

# AMP_MJLab robot cfg with motor-derived PD gains and KNEES_BENT_KEYFRAME.
AMP_G1_CFG = copy.deepcopy(BASE_G1_CFG)

AMP_G1_CFG.init_state.pos = (0.0, 0.0, 0.78)
AMP_G1_CFG.init_state.joint_pos = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}
AMP_G1_CFG.init_state.joint_vel = {".*": 0.0}

AMP_G1_CFG.spawn.usd_path = _AMP_USD

AMP_G1_CFG.actuators = {
    "AMP_N5020_16": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_elbow_joint",
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_wrist_roll_joint",
        ],
        effort_limit_sim=25.0,
        velocity_limit_sim=37.0,
        stiffness=STIFFNESS_5020,
        damping=DAMPING_5020,
        armature=ARMATURE_5020,
    ),
    "AMP_N7520_14": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint",
            "waist_yaw_joint",
        ],
        effort_limit_sim=88.0,
        velocity_limit_sim=32.0,
        stiffness=STIFFNESS_7520_14,
        damping=DAMPING_7520_14,
        armature=ARMATURE_7520_14,
    ),
    "AMP_N7520_22": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_pitch_joint",
            ".*_hip_roll_joint",
            ".*_knee_joint",
        ],
        effort_limit_sim=139.0,
        velocity_limit_sim=20.0,
        stiffness=STIFFNESS_7520_22,
        damping=DAMPING_7520_22,
        armature=ARMATURE_7520_22,
    ),
    "AMP_N5010_16": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
        ],
        effort_limit_sim=10.0,
        velocity_limit_sim=22.0,
        stiffness=STIFFNESS_5010_16,
        damping=DAMPING_5010_16,
        armature=ARMATURE_5010_16,
    ),
    "AMP_N5020_16_WAIST_PARALLEL": ImplicitActuatorCfg(
        joint_names_expr=[
            "waist_pitch_joint",
            "waist_roll_joint",
        ],
        effort_limit_sim=50.0,
        velocity_limit_sim=37.0,
        stiffness=STIFFNESS_5020_PARALLEL,
        damping=DAMPING_5020_PARALLEL,
        armature=ARMATURE_5020_PARALLEL,
    ),
    "AMP_N5020_16_ANKLE_PARALLEL": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
        ],
        effort_limit_sim=50.0,
        velocity_limit_sim=37.0,
        stiffness=STIFFNESS_5020_PARALLEL,
        damping=DAMPING_5020_PARALLEL,
        armature=ARMATURE_5020_PARALLEL,
    ),
}


@configclass
class AmpRewardsCfg:
    track_anchor_linear_velocity = RewTerm(
        func=mdp.track_anchor_linear_velocity,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": 1.0,
            "anchor_cfg": _amp_anchor_cfg(),
        },
    )
    track_anchor_angular_velocity = RewTerm(
        func=mdp.track_anchor_angular_velocity,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": 1.0,
            "anchor_cfg": _amp_anchor_cfg(),
        },
    )
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=(".*ankle_roll.*",)),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=(".*ankle_roll.*",)),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class AmpTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
    )
    bad_base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.5},
    )


@configclass
class AmpRobotSceneCfg(VelocityRobotSceneCfg):
    robot: ArticulationCfg = AMP_G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class AmpActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=AMP_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class AmpObservationsCfg(VelocityObservationsCfg):
    @configclass
    class PolicyCfg(VelocityObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 4
            self.concatenate_terms = True
            self.enable_corruption = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(VelocityObservationsCfg.CriticCfg):
        body_pos_b = ObsTerm(
            func=amp_mdp.robot_body_pos_b,
            params={
                "anchor_cfg": _amp_anchor_cfg(),
                "body_cfg": _amp_body_cfg(),
            },
        )
        body_ori_b = ObsTerm(
            func=amp_mdp.robot_body_ori_b,
            params={
                "anchor_cfg": _amp_anchor_cfg(),
                "body_cfg": _amp_body_cfg(),
            },
        )

        def __post_init__(self):
            super().__post_init__()
            self.history_length = 4
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()

    @configclass
    class AmpCfg(ObsGroup):
        body_pos_b = ObsTerm(
            func=amp_mdp.robot_body_pos_b,
            params={
                "anchor_cfg": _amp_anchor_cfg(),
                "body_cfg": _amp_body_cfg(),
            },
        )
        body_ori_b = ObsTerm(
            func=amp_mdp.robot_body_ori_b,
            params={
                "anchor_cfg": _amp_anchor_cfg(),
                "body_cfg": _amp_body_cfg(),
            },
        )
        body_lin_vel_b = ObsTerm(
            func=amp_mdp.robot_body_lin_vel_b,
            params={
                "body_cfg": _amp_body_cfg(),
            },
        )
        body_ang_vel_b = ObsTerm(
            func=amp_mdp.robot_body_ang_vel_b,
            params={
                "body_cfg": _amp_body_cfg(),
            },
        )

        def __post_init__(self):
            self.history_length = 1
            self.concatenate_terms = True
            self.enable_corruption = False

    amp: AmpCfg = AmpCfg()


@configclass
class RobotEnvCfg(G1VelocityEnvCfg):
    scene: AmpRobotSceneCfg = AmpRobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: AmpObservationsCfg = AmpObservationsCfg()
    actions: AmpActionsCfg = AmpActionsCfg()
    rewards: AmpRewardsCfg = AmpRewardsCfg()
    terminations: AmpTerminationsCfg = AmpTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "flat":   terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
                "wave":   terrain_gen.HfWaveTerrainCfg(proportion=0.3, amplitude_range=(0.05, 0.2), num_waves=4),
                "rough":  terrain_gen.HfRandomUniformTerrainCfg(proportion=0.4, noise_range=(0.02, 0.14), noise_step=0.01, downsampled_scale=0.2),
            },
        )
        self.scene.terrain.max_init_terrain_level = 9
        self.curriculum = {}

        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        self.commands.base_velocity.resampling_time_range = (3.0, 8.0)
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.rel_heading_envs = 0.25
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.ranges.heading = (-math.pi / 2, math.pi / 2)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 2, math.pi / 2)

        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-0.4, 0.4),
                    "roll": (-0.72, 0.72),
                    "pitch": (-0.72, 0.72),
                    "yaw": (-0.78, 0.78),
                },
            },
        )

        # AMP motion reset events.
        self.events.init_motion_loader = EventTerm(
            func=amp_event_mdp.init_motion_loader,
            mode="startup",
            params={"motion_dir": AMP_MOTION_DIR},
        )
        self.events.reset_from_motion = EventTerm(
            func=amp_event_mdp.reset_from_motion_data,
            mode="reset",
            params={
                "motion_dir": AMP_MOTION_DIR,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            },
        )


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
