from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_body_pos_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg,
    body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[anchor_cfg.name]

    anchor_pos_w = robot.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]
    anchor_quat_w = robot.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]

    body_pos_w = robot.data.body_link_pos_w[:, body_cfg.body_ids]
    body_quat_w = robot.data.body_link_quat_w[:, body_cfg.body_ids]

    num_bodies = body_pos_w.shape[1]
    pos_b, _ = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg,
    body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[anchor_cfg.name]

    anchor_pos_w = robot.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]
    anchor_quat_w = robot.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]

    body_pos_w = robot.data.body_link_pos_w[:, body_cfg.body_ids]
    body_quat_w = robot.data.body_link_quat_w[:, body_cfg.body_ids]

    num_bodies = body_pos_w.shape[1]
    _, quat_b = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    mat = matrix_from_quat(quat_b)
    return mat[..., :, :2].reshape(env.num_envs, -1)


def robot_body_lin_vel_b(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[body_cfg.name]

    body_lin_vel_w = robot.data.body_link_lin_vel_w[:, body_cfg.body_ids]
    body_quat_w = robot.data.body_link_quat_w[:, body_cfg.body_ids]

    num_bodies = body_lin_vel_w.shape[1]
    body_lin_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_lin_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)

    return body_lin_vel_b.reshape(env.num_envs, -1)


def robot_body_ang_vel_b(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[body_cfg.name]

    body_ang_vel_w = robot.data.body_link_ang_vel_w[:, body_cfg.body_ids]
    body_quat_w = robot.data.body_link_quat_w[:, body_cfg.body_ids]

    num_bodies = body_ang_vel_w.shape[1]
    body_ang_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_ang_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)

    return body_ang_vel_b.reshape(env.num_envs, -1)
