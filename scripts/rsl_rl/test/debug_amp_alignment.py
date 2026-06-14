"""Replay NPZ frame through Isaac, compare AMP obs vs NPZ expected.

Usage:
    python scripts/rsl_rl/debug_amp_alignment.py \
        --task Unitree-G1-29dof-AMP-Flat \
        --npz path/to/motion.npz \
        --frame 0 \
        --headless
"""

import argparse
import pathlib
import sys

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-AMP-Flat")
parser.add_argument("--npz", type=str, required=True)
parser.add_argument("--frame", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401
sys.path.pop(0)

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import (
    subtract_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
)

from unitree_rl_lab.amp.amp_constants import (
    NPZ_BODY_NAMES, AMP_BODY_NAMES, AMP_ANCHOR_NAME,
    NPZ_BY_ISAAC_JOINT, AMP_NPZ_BODY_IDS, AMP_NPZ_ANCHOR_ID,
)

NPZ_TO_ISAAC_JOINT = NPZ_BY_ISAAC_JOINT.tolist()


def amp_obs_from_arrays(body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, body_ids, anchor_id):
    num = len(body_ids)
    anchor_pos = body_pos_w[anchor_id].unsqueeze(0).repeat(num, 1)
    anchor_quat = body_quat_w[anchor_id].unsqueeze(0).repeat(num, 1)
    pos_b, quat_b = subtract_frame_transforms(anchor_pos, anchor_quat, body_pos_w[body_ids], body_quat_w[body_ids])
    ori_b = matrix_from_quat(quat_b)[..., :, :2].reshape(1, -1)
    lin_b = quat_apply_inverse(body_quat_w[body_ids], body_lin_vel_w[body_ids])
    ang_b = quat_apply_inverse(body_quat_w[body_ids], body_ang_vel_w[body_ids])
    return torch.cat([pos_b.reshape(-1), ori_b.reshape(-1), lin_b.reshape(-1), ang_b.reshape(-1)], dim=0)


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]

    data = np.load(args_cli.npz)
    f = args_cli.frame

    # ----- NPZ expert AMP obs -----
    npz_pos = torch.tensor(data["body_pos_w"][f], dtype=torch.float32, device=device)
    npz_quat = torch.tensor(data["body_quat_w"][f], dtype=torch.float32, device=device)
    npz_lin = torch.tensor(data["body_lin_vel_w"][f], dtype=torch.float32, device=device)
    npz_ang = torch.tensor(data["body_ang_vel_w"][f], dtype=torch.float32, device=device)
    npz_amp_expected = amp_obs_from_arrays(npz_pos, npz_quat, npz_lin, npz_ang, AMP_NPZ_BODY_IDS, AMP_NPZ_ANCHOR_ID)

    # ----- Isaac replay: write NPZ root + joints -----
    rs = robot.data.default_root_state.clone()
    rs[:, 0:3] = torch.tensor(data["body_pos_w"][f, 0], dtype=torch.float32, device=device)
    rs[:, 3:7] = torch.tensor(data["body_quat_w"][f, 0], dtype=torch.float32, device=device)
    rs[:, 7:10] = torch.tensor(data["body_lin_vel_w"][f, 0], dtype=torch.float32, device=device)
    rs[:, 10:13] = torch.tensor(data["body_ang_vel_w"][f, 0], dtype=torch.float32, device=device)
    if hasattr(robot, "write_root_state_to_sim"):
        robot.write_root_state_to_sim(rs)
    else:
        robot.write_root_link_pose_to_sim(rs[:, :7])
        robot.write_root_link_velocity_to_sim(rs[:, 7:13])

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    npz_jp = torch.tensor(data["joint_pos"][f], dtype=torch.float32, device=device)
    npz_jv = torch.tensor(data["joint_vel"][f], dtype=torch.float32, device=device)
    for npz_i, isaac_i in enumerate(NPZ_TO_ISAAC_JOINT):
        joint_pos[:, isaac_i] = npz_jp[npz_i]
        joint_vel[:, isaac_i] = npz_jv[npz_i]
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    env.unwrapped.sim.forward()
    env.unwrapped.scene.update(env.unwrapped.physics_dt)

    # ----- Isaac AMP obs: index Isaac body array with Isaac body IDs -----
    isaac_body_names = list(robot.body_names)
    isaac_body_ids = [isaac_body_names.index(n) for n in AMP_BODY_NAMES]
    isaac_anchor_id = isaac_body_names.index(AMP_ANCHOR_NAME)
    isaac_amp = amp_obs_from_arrays(
        robot.data.body_link_pos_w[0], robot.data.body_link_quat_w[0],
        robot.data.body_link_lin_vel_w[0], robot.data.body_link_ang_vel_w[0],
        isaac_body_ids, isaac_anchor_id,
    )

    d = (isaac_amp - npz_amp_expected).abs()
    print(f"\n[isaac_replay_vs_npz] mean_abs={d.mean().item():.8e}  max_abs={d.max().item():.8e}")

    print("\n===== Segment diff =====")
    parts = [("pos     ", 39), ("ori6d   ", 78), ("lin_vel ", 39), ("ang_vel ", 39)]
    off = 0
    for name, dim in parts:
        seg = isaac_amp[off:off+dim] - npz_amp_expected[off:off+dim]
        print(f"  {name} mean_abs={seg.abs().mean().item():.8e}  max_abs={seg.abs().max().item():.8e}")
        off += dim

    print()
    if d.mean().item() < 1e-2 and d.max().item() < 1e-1:
        print("PASS: AMP obs from Isaac replay matches NPZ expected.")
        print("      Body order, joint order, and link frames are aligned.")
    else:
        print("FAIL: Large diff. Likely joint axis sign or link frame mismatch.")
        print("      Check NPZ_TO_ISAAC_JOINT mapping and USD vs MuJoCo link frames.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
