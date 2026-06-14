"""AMP event callbacks: motion reset from motion capture data."""

from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING

import numpy as np
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

from unitree_rl_lab.amp.amp_constants import NPZ_BY_ISAAC_JOINT as NPZ_BY_ISAAC


class _MotionResetManager:
    """Singleton that loads and caches motion data for resetting envs."""

    _instance: _MotionResetManager | None = None

    def __init__(self) -> None:
        self.walk_run_frames: dict[str, dict[str, torch.Tensor]] = {}

    @classmethod
    def get(cls) -> _MotionResetManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def init(self, env: ManagerBasedRLEnv, motion_dir: str) -> None:
        if motion_dir in self.walk_run_frames:
            return

        assert os.path.isdir(motion_dir), f"Motion directory not found: {motion_dir}"
        motion_files = sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(motion_dir)
            for f in files if f.endswith(".npz")
        ])
        assert motion_files, f"No npz files in {motion_dir}"

        all_root_pos, all_root_quat = [], []
        all_root_lin_vel, all_root_ang_vel = [], []
        all_joint_pos, all_joint_vel = [], []

        for path in motion_files:
            data = np.load(path)
            body_pos_w = data["body_pos_w"]   # (T, N, 3)
            body_quat_w = data["body_quat_w"] # (T, N, 4)
            body_lin_vel_w = data["body_lin_vel_w"]
            body_ang_vel_w = data["body_ang_vel_w"]
            dof_pos = data["joint_pos"]       # (T, J)  in NPZ order
            dof_vel = data["joint_vel"]       # (T, J)

            all_root_pos.append(torch.from_numpy(body_pos_w[:, 0, :]))
            all_root_quat.append(torch.from_numpy(body_quat_w[:, 0, :]))
            all_root_lin_vel.append(torch.from_numpy(body_lin_vel_w[:, 0, :]))
            all_root_ang_vel.append(torch.from_numpy(body_ang_vel_w[:, 0, :]))
            all_joint_pos.append(torch.from_numpy(dof_pos))
            all_joint_vel.append(torch.from_numpy(dof_vel))

        self.walk_run_frames[motion_dir] = {
            "root_pos": torch.cat(all_root_pos, dim=0).to(env.device),
            "root_quat": torch.cat(all_root_quat, dim=0).to(env.device),
            "root_lin_vel": torch.cat(all_root_lin_vel, dim=0).to(env.device),
            "root_ang_vel": torch.cat(all_root_ang_vel, dim=0).to(env.device),
            "joint_pos": torch.cat(all_joint_pos, dim=0).to(env.device),
            "joint_vel": torch.cat(all_joint_vel, dim=0).to(env.device),
        }
        count = self.walk_run_frames[motion_dir]["root_pos"].shape[0]
        print(f"[MotionResetManager] Loaded {len(motion_files)} clips, {count} frames from {motion_dir}")

    def reset(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        motion_dir: str,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        if len(env_ids) == 0:
            return

        frames = self.walk_run_frames[motion_dir]
        total = frames["root_pos"].shape[0]
        idx = torch.randint(0, total, (len(env_ids),), device=env.device)

        robot: Articulation = env.scene[asset_cfg.name]

        # Root pose.
        root_pos = frames["root_pos"][idx]
        root_quat = frames["root_quat"][idx]
        positions = env.scene.env_origins[env_ids].clone()
        positions[:, 2] = root_pos[:, 2]
        root_pose = torch.cat([positions, root_quat], dim=-1)
        robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

        # Root velocity.
        root_vel = torch.cat(
            [frames["root_lin_vel"][idx], frames["root_ang_vel"][idx]], dim=-1
        )
        robot.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

        # Joint state: NPZ order -> Isaac/USD order via hardcoded index map.
        idx_map = NPZ_BY_ISAAC.to(env.device)
        joint_pos = frames["joint_pos"][idx][:, idx_map]
        joint_vel = frames["joint_vel"][idx][:, idx_map]

        joint_ids = asset_cfg.joint_ids
        if joint_ids is None:
            joint_ids = torch.arange(robot.num_joints, device=env.device, dtype=torch.long)
        elif isinstance(joint_ids, slice):
            joint_ids = torch.arange(robot.num_joints, device=env.device, dtype=torch.long)[joint_ids]
        elif isinstance(joint_ids, list):
            joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
        else:
            joint_ids = joint_ids.to(device=env.device, dtype=torch.long)

        joint_pos = joint_pos[:, joint_ids]
        joint_vel = joint_vel[:, joint_ids]

        limits = robot.data.soft_joint_pos_limits
        if limits is not None:
            cur_limits = limits[env_ids][:, joint_ids]
            joint_pos = joint_pos.clamp(cur_limits[..., 0], cur_limits[..., 1])

        robot.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            env_ids=env_ids,
            joint_ids=joint_ids,
        )


# ------------------------------------------------------------------
# Event callback wrappers
# ------------------------------------------------------------------


def init_motion_loader(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
) -> None:
    """Startup event: load motion data for reset."""
    _MotionResetManager.get().init(env=env, motion_dir=motion_dir)


def reset_from_motion_data(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset event: reset envs from random motion frames."""
    _MotionResetManager.get().reset(
        env=env, env_ids=env_ids, motion_dir=motion_dir, asset_cfg=asset_cfg
    )
