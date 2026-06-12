import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt  # noqa: F401

from isaaclab.app import AppLauncher

# ============================================================
# Fixed settings
# ============================================================
TASK = "Unitree-G1-29dof-Velocity"

ONNX_PATH = (
    "/home/wufy/projects/mjlab/AMP_mjlab/logs/rsl_rl/g1_amp_locomotion/2026-06-11_09-48-21/export/Unitree-G1-AMP-Flat_model_29300.onnx"
)

NUM_ENVS = 1
MAX_STEPS = 800
VIDEO = True

VX = 3.0
VY = 0.0
WZ = 0.0

OBS_CLIP = 100.0
ACT_CLIP = 100.0

NUM_DOF = 29
FRAME_DIM = 96

# Fixed world-side third-person camera.
# eye = root + CAM_EYE_OFFSET
# target = root + CAM_TARGET_OFFSET
CAM_EYE_OFFSET = np.array([-1.5, 5.5, 1.8], dtype=np.float32)
CAM_TARGET_OFFSET = np.array([0.5, 0.0, 0.75], dtype=np.float32)

# ============================================================
# Launch Isaac Sim
# ============================================================
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = VIDEO

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================
# Imports after SimulationApp
# ============================================================
import gymnasium as gym
import onnxruntime
import torch

repo_root = Path(__file__).resolve().parents[2]
unitree_src = repo_root / "source" / "unitree_rl_lab"
if unitree_src.exists():
    sys.path.insert(0, str(unitree_src))

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401

from isaaclab.actuators import ImplicitActuatorCfg
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.assets.robots.unitree import (
    UNITREE_G1_29DOF_MIMIC_CFG,
    UnitreeUrdfFileCfg,
)

AMP_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def get_action_cfg(env_cfg):
    if hasattr(env_cfg.actions, "JointPositionAction"):
        return env_cfg.actions.JointPositionAction
    if hasattr(env_cfg.actions, "joint_pos"):
        return env_cfg.actions.joint_pos
    raise RuntimeError("Cannot find JointPositionAction config.")


def apply_amp_mjlab_actuator_cfg(robot_cfg, action_cfg):
    """
    Match AMP_mjlab action interpretation as closely as possible.

    AMP_mjlab action:
        target_q = default_q + action * action_scale

    Approx action_scale:
        0.25 * tau_max / kp
    """

    action_cfg.scale = {
        # 7520_14 group
        ".*_hip_yaw_joint": 0.443,
        "waist_yaw_joint": 0.443,

        # 7520_22 group
        ".*_hip_pitch_joint": 0.274,
        ".*_hip_roll_joint": 0.274,
        ".*_knee_joint": 0.274,

        # 2 x 5020 equivalent group
        ".*_ankle_pitch_joint": 0.365,
        ".*_ankle_roll_joint": 0.365,
        "waist_pitch_joint": 0.365,
        "waist_roll_joint": 0.365,

        # 5020 arm group
        ".*_shoulder_pitch_joint": 0.365,
        ".*_shoulder_roll_joint": 0.365,
        ".*_shoulder_yaw_joint": 0.365,
        ".*_elbow_joint": 0.365,
        ".*_wrist_roll_joint": 0.365,

        # small wrist group
        ".*_wrist_pitch_joint": 0.290,
        ".*_wrist_yaw_joint": 0.290,
    }
    action_cfg.use_default_offset = True

    robot_cfg.actuators = {
        "amp_7520_14": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", "waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness=49.64,
            damping=3.16,
            armature=0.01257,
        ),
        "amp_7520_22": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=139.0,
            velocity_limit_sim=20.0,
            stiffness=126.80,
            damping=8.07,
            armature=0.03212,
        ),
        "amp_5020_pair_ankle": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=34.23,
            damping=2.18,
            armature=0.00867,
        ),
        "amp_5020_pair_waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_pitch_joint",
                "waist_roll_joint",
            ],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=34.23,
            damping=2.18,
            armature=0.00867,
        ),
        "amp_5020_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=25.0,
            velocity_limit_sim=37.0,
            stiffness=17.12,
            damping=1.09,
            armature=0.00434,
        ),
        "amp_wrist_small": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=10.0,
            velocity_limit_sim=22.0,
            stiffness=8.61,
            damping=0.55,
            armature=0.00218,
        ),
    }


def disable_terms(env_cfg):
    if hasattr(env_cfg, "events") and env_cfg.events is not None:
        for name in [
            "physics_material",
            "add_base_mass",
            "base_external_force_torque",
            "push_robot",
        ]:
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)

        if hasattr(env_cfg.events, "reset_base") and env_cfg.events.reset_base is not None:
            env_cfg.events.reset_base.params["pose_range"] = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
            env_cfg.events.reset_base.params["velocity_range"] = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }

        if hasattr(env_cfg.events, "reset_robot_joints") and env_cfg.events.reset_robot_joints is not None:
            env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
            env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    if hasattr(env_cfg, "terminations") and env_cfg.terminations is not None:
        for name in list(vars(env_cfg.terminations).keys()):
            if not name.startswith("_"):
                try:
                    setattr(env_cfg.terminations, name, None)
                except Exception:
                    pass

    if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
        for name in list(vars(env_cfg.curriculum).keys()):
            if not name.startswith("_"):
                try:
                    setattr(env_cfg.curriculum, name, None)
                except Exception:
                    pass


def make_env_cfg():
    env_cfg = parse_env_cfg(
        TASK,
        device=args_cli.device,
        num_envs=NUM_ENVS,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )

    old_robot_cfg = env_cfg.scene.robot

    robot_cfg = copy.deepcopy(UNITREE_G1_29DOF_MIMIC_CFG)
    robot_cfg.prim_path = old_robot_cfg.prim_path
    robot_cfg.collision_group = old_robot_cfg.collision_group
    robot_cfg.debug_vis = old_robot_cfg.debug_vis

    default_urdf = (
        repo_root
        / "unitree_ros"
        / "robots"
        / "g1_description"
        / "g1_29dof_rev_1_0.urdf"
    )
    print(f"[INFO] Loading URDF: {default_urdf}")
    urdf_spawn = UnitreeUrdfFileCfg(asset_path=str(default_urdf))
    urdf_spawn.replace_asset(
        meshes_dir=str(default_urdf.parent / "meshes"),
        urdf_path=str(default_urdf),
    )
    robot_cfg.spawn = urdf_spawn

    env_cfg.scene.robot = robot_cfg

    action_cfg = get_action_cfg(env_cfg)
    apply_amp_mjlab_actuator_cfg(robot_cfg, action_cfg)

    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.episode_length_s = 1.0e9

    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.ranges.lin_vel_x = (VX, VX)
    cmd_cfg.ranges.lin_vel_y = (VY, VY)
    cmd_cfg.ranges.ang_vel_z = (WZ, WZ)
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)

    if hasattr(cmd_cfg, "heading_command"):
        cmd_cfg.heading_command = False
    if hasattr(cmd_cfg, "debug_vis"):
        cmd_cfg.debug_vis = False
    if hasattr(cmd_cfg, "rel_standing_envs"):
        cmd_cfg.rel_standing_envs = 0.0
    if hasattr(cmd_cfg, "rel_heading_envs"):
        cmd_cfg.rel_heading_envs = 0.0

    disable_terms(env_cfg)
    return env_cfg


def get_joint_names(robot):
    if hasattr(robot, "joint_names"):
        return list(robot.joint_names)
    if hasattr(robot.data, "joint_names"):
        return list(robot.data.joint_names)
    if hasattr(robot, "find_joints"):
        out = robot.find_joints(".*", preserve_order=True)
        if isinstance(out, tuple) and len(out) == 2:
            ids, names = out
            return [name for _, name in sorted(zip(ids, names), key=lambda x: x[0])]
    raise RuntimeError("Cannot read robot joint names.")


def build_mappings(robot, device):
    isaac_names = get_joint_names(robot)

    missing = [name for name in AMP_JOINT_NAMES if name not in isaac_names]
    if missing:
        raise RuntimeError(f"Missing joints in Isaac model: {missing}\nIsaac joints: {isaac_names}")

    amp_to_isaac = torch.tensor(
        [isaac_names.index(name) for name in AMP_JOINT_NAMES],
        device=device,
        dtype=torch.long,
    )
    isaac_to_amp = torch.tensor(
        [AMP_JOINT_NAMES.index(name) for name in isaac_names],
        device=device,
        dtype=torch.long,
    )

    print("[INFO] Isaac joint order:")
    for i, name in enumerate(isaac_names):
        print(f"  {i:02d}: {name}")

    print("[INFO] AMP -> Isaac:", amp_to_isaac.detach().cpu().tolist())
    print("[INFO] Isaac -> AMP:", isaac_to_amp.detach().cpu().tolist())
    return amp_to_isaac, isaac_to_amp


def root_ang_vel_b(robot):
    if hasattr(robot.data, "root_link_ang_vel_b"):
        return robot.data.root_link_ang_vel_b
    return robot.data.root_ang_vel_b


def projected_gravity_b(robot):
    if hasattr(robot.data, "projected_gravity_b"):
        return robot.data.projected_gravity_b
    return robot.data.root_projected_gravity_b


def command(env):
    cmd = env.unwrapped.command_manager.get_command("base_velocity")
    target = torch.tensor([VX, VY, WZ], device=cmd.device, dtype=cmd.dtype).view(1, 3)
    cmd[:, :3] = target
    return cmd[:, :3]


def make_amp_frame(env, robot, amp_to_isaac, prev_action_amp):
    joint_pos_amp = robot.data.joint_pos[:, amp_to_isaac]
    joint_vel_amp = robot.data.joint_vel[:, amp_to_isaac]
    default_pos_amp = robot.data.default_joint_pos[:, amp_to_isaac]

    frame = torch.cat(
        [
            root_ang_vel_b(robot),
            projected_gravity_b(robot),
            command(env),
            joint_pos_amp - default_pos_amp,
            joint_vel_amp,
            prev_action_amp,
        ],
        dim=-1,
    )

    if frame.shape[-1] != FRAME_DIM:
        raise RuntimeError(f"AMP frame dim should be {FRAME_DIM}, got {frame.shape[-1]}")

    return frame


def set_third_person_camera(env, robot):
    root = robot.data.root_pos_w[0].detach().cpu().numpy()
    eye = root + CAM_EYE_OFFSET
    target = root + CAM_TARGET_OFFSET

    try:
        env.unwrapped.sim.set_camera_view(eye=eye.tolist(), target=target.tolist())
    except TypeError:
        env.unwrapped.sim.set_camera_view(eye.tolist(), target.tolist())


def make_onnx_session():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = onnxruntime.InferenceSession(ONNX_PATH, providers=providers)

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print(f"[INFO] ONNX input : {input_info.name}, shape={input_info.shape}")
    print(f"[INFO] ONNX output: {output_info.name}, shape={output_info.shape}")
    print(f"[INFO] ONNX providers: {session.get_providers()}")

    return session, input_info, output_info


def main():
    session, input_info, output_info = make_onnx_session()

    input_dim = input_info.shape[-1]
    history = input_dim // FRAME_DIM if isinstance(input_dim, int) else 4
    print(f"[INFO] history={history}, obs_dim={history * FRAME_DIM}")
    print(f"[INFO] command=({VX}, {VY}, {WZ})")

    env = gym.make(
        TASK,
        cfg=make_env_cfg(),
        render_mode="rgb_array" if VIDEO else None,
    )

    if VIDEO:
        video_dir = os.path.join(os.path.dirname(ONNX_PATH), "isaacsim_videos")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=MAX_STEPS,
            disable_logger=True,
        )
        print("[INFO] Recording video to:", video_dir)

    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]

    env.reset()
    amp_to_isaac, isaac_to_amp = build_mappings(robot, device)

    prev_action_amp = torch.zeros(NUM_ENVS, NUM_DOF, device=device)
    obs_buf = torch.zeros(NUM_ENVS, history, FRAME_DIM, device=device)

    first_frame = make_amp_frame(env, robot, amp_to_isaac, prev_action_amp)
    obs_buf[:] = first_frame.unsqueeze(1).repeat(1, history, 1)

    set_third_person_camera(env, robot)

    with torch.inference_mode():
        for step in range(MAX_STEPS):
            frame = make_amp_frame(env, robot, amp_to_isaac, prev_action_amp)
            obs_buf = torch.roll(obs_buf, shifts=-1, dims=1)
            obs_buf[:, -1, :] = frame

            obs = obs_buf.reshape(NUM_ENVS, -1)
            obs = torch.clamp(obs, -OBS_CLIP, OBS_CLIP)
            obs_np = obs.detach().cpu().numpy().astype(np.float32)

            action_np = session.run([output_info.name], {input_info.name: obs_np})[0]
            action_amp = torch.as_tensor(action_np, device=device, dtype=torch.float32)

            if action_amp.ndim == 1:
                action_amp = action_amp.unsqueeze(0)

            if action_amp.shape[-1] != NUM_DOF:
                raise RuntimeError(f"ONNX action dim should be {NUM_DOF}, got {action_amp.shape}")

            action_amp = torch.nan_to_num(action_amp, nan=0.0, posinf=0.0, neginf=0.0)
            action_amp = torch.clamp(action_amp, -ACT_CLIP, ACT_CLIP)

            action_isaac = action_amp[:, isaac_to_amp]

            set_third_person_camera(env, robot)
            env.step(action_isaac)

            prev_action_amp = action_amp.clone()

            if step % 50 == 0:
                h = robot.data.root_pos_w[0, 2].item()
                act_abs = action_amp.abs().mean().item()
                print(f"[{step:04d}] h={h:.3f}, act_abs={act_abs:.3f}, cmd=({VX:.2f},{VY:.2f},{WZ:.2f})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()