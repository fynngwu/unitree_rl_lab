"""Minimal AMP alignment checker for Unitree RL Lab + AMP_mjlab data."""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import pathlib
import sys

# -----------------------------------------------------------------------------
# Same rsl_rl bootstrap as amp_train.py. Must happen before Isaac/RSL imports.
# -----------------------------------------------------------------------------
AMP_MJLAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
AMP_RSL_RL_DIR = os.path.join(AMP_MJLAB_ROOT, "rsl_rl")

if not os.path.isfile(os.path.join(AMP_RSL_RL_DIR, "__init__.py")):
    raise RuntimeError(f"Cannot find bundled AMP_mjlab rsl_rl: {AMP_RSL_RL_DIR}")

sys.path.insert(0, AMP_MJLAB_ROOT)
importlib.invalidate_caches()

import rsl_rl  # noqa: E402

loaded_rsl_rl = os.path.abspath(getattr(rsl_rl, "__file__", ""))
if not loaded_rsl_rl.startswith(os.path.abspath(AMP_RSL_RL_DIR)):
    raise RuntimeError(
        "Wrong rsl_rl loaded.\n"
        f"  loaded   : {loaded_rsl_rl}\n"
        f"  expected : {AMP_RSL_RL_DIR}"
    )

print(f"[PASS] rsl_rl = {loaded_rsl_rl}")

# -----------------------------------------------------------------------------

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401,E402

sys.path.pop(0)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-AMP-Flat")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--motion_file", type=str, default=None)
parser.add_argument("--debug_frames", type=int, default=1024)
parser.add_argument("--strict_names", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401,E402
import unitree_rl_lab.tasks  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def unpack_obs(result):
    if isinstance(result, tuple):
        return result
    return result, {"observations": {}}


def first_npz(path: str) -> str:
    if os.path.isfile(path):
        return path
    files = sorted(glob.glob(os.path.join(path, "**", "*.npz"), recursive=True))
    if not files:
        raise RuntimeError(f"No npz found in {path}")
    return files[0]


def decode_names(x) -> list[str]:
    out = []
    for v in list(x):
        if isinstance(v, bytes):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return out


def get_names(data, key: str) -> list[str] | None:
    if key not in data.files:
        return None
    return decode_names(data[key])


def print_names(title: str, names: list[str], max_lines: int = 80):
    print(f"\n===== {title} ({len(names)}) =====")
    for i, n in enumerate(names[:max_lines]):
        print(f"{i:03d}: {n}")
    if len(names) > max_lines:
        print(f"... {len(names) - max_lines} more")


def part_stats(name: str, x: torch.Tensor, num_bodies: int):
    parts = {
        "pos     ": (0, 3 * num_bodies),
        "ori6d   ": (3 * num_bodies, 9 * num_bodies),
        "lin_vel ": (9 * num_bodies, 12 * num_bodies),
        "ang_vel ": (12 * num_bodies, 15 * num_bodies),
    }
    print(f"\n===== {name}: shape={tuple(x.shape)} =====")
    for k, (a, b) in parts.items():
        y = x[:, a:b]
        print(
            f"{k} mean_abs={y.abs().mean().item():9.5f} "
            f"std={y.std().item():9.5f} "
            f"min={y.min().item():9.5f} "
            f"max={y.max().item():9.5f}"
        )


def compare_stats(policy: torch.Tensor, expert: torch.Tensor, num_bodies: int):
    parts = {
        "pos     ": (0, 3 * num_bodies),
        "ori6d   ": (3 * num_bodies, 9 * num_bodies),
        "lin_vel ": (9 * num_bodies, 12 * num_bodies),
        "ang_vel ": (12 * num_bodies, 15 * num_bodies),
    }
    print("\n===== policy vs expert segment mismatch =====")
    for k, (a, b) in parts.items():
        p = policy[:, a:b]
        e = expert[:, a:b]
        mean_diff = (p.mean(0) - e.mean(0)).abs().mean().item()
        std_ratio = (p.std(0).mean() / (e.std(0).mean() + 1.0e-6)).item()
        print(f"{k} mean_diff={mean_diff:9.5f}  policy_std/expert_std={std_ratio:9.5f}")


def make_expert_amp_obs(
    data,
    target_body_names: list[str],
    anchor_name: str,
    isaac_body_names: list[str],
    device: str,
    max_frames: int,
    strict_names: bool,
):
    required = [
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ]
    for k in required:
        if k not in data.files:
            raise RuntimeError(f"motion npz missing key: {k}")

    npz_body_names = get_names(data, "body_names")
    if npz_body_names is None:
        msg = (
            "[FAIL] motion npz has no body_names. Current AMPLoader cannot prove "
            "MuJoCo expert body order == Isaac USD body order."
        )
        print("\n" + msg)
        if strict_names:
            raise RuntimeError(msg)
        print("[WARN] Continue with current unsafe AMPLoader assumption for statistics only.")
        src_body_names = isaac_body_names
    else:
        src_body_names = npz_body_names

    missing = [n for n in list(target_body_names) + [anchor_name] if n not in src_body_names]
    if missing:
        raise RuntimeError(f"motion body_names missing target bodies: {missing}")

    body_ids = [src_body_names.index(n) for n in target_body_names]
    anchor_id = src_body_names.index(anchor_name)

    print("\n===== expert target body mapping =====")
    for n, i in zip(target_body_names, body_ids):
        print(f"{n:32s} -> motion_body[{i}]")
    print(f"{anchor_name:32s} -> motion_body[{anchor_id}]  [anchor]")

    T = int(data["body_pos_w"].shape[0])
    N = min(max_frames, max(1, T - 1))
    ids = torch.linspace(0, T - 2, N, device=device).long()

    body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)[ids]
    body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)[ids]
    body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)[ids]
    body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)[ids]

    anchor_pos = body_pos_w[:, anchor_id, :][:, None, :].expand(-1, len(body_ids), -1)
    anchor_quat = body_quat_w[:, anchor_id, :][:, None, :].expand(-1, len(body_ids), -1)

    bpos = body_pos_w[:, body_ids, :]
    bquat = body_quat_w[:, body_ids, :]
    blin = body_lin_vel_w[:, body_ids, :]
    bang = body_ang_vel_w[:, body_ids, :]

    pos_b, quat_b = subtract_frame_transforms(anchor_pos, anchor_quat, bpos, bquat)
    ori6d = matrix_from_quat(quat_b)[..., :, :2].reshape(N, -1)

    lin_b = quat_apply_inverse(bquat.reshape(-1, 4), blin.reshape(-1, 3)).reshape(N, len(body_ids), 3)
    ang_b = quat_apply_inverse(bquat.reshape(-1, 4), bang.reshape(-1, 3)).reshape(N, len(body_ids), 3)

    expert = torch.cat(
        [
            pos_b.reshape(N, -1),
            ori6d,
            lin_b.reshape(N, -1),
            ang_b.reshape(N, -1),
        ],
        dim=-1,
    )
    return expert


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    device = args_cli.device if args_cli.device is not None else agent_cfg.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    agent_cfg.device = device

    train_cfg = agent_cfg.to_dict()
    amp_body_names = list(train_cfg["amp_body_names"])
    amp_anchor_name = train_cfg["amp_anchor_name"]
    motion_path = args_cli.motion_file or train_cfg["amp_motion_files"]
    motion_npz = first_npz(motion_path)

    print("\n===== AMP config =====")
    print(f"task              : {args_cli.task}")
    print(f"num_envs          : {args_cli.num_envs}")
    print(f"device            : {device}")
    print(f"motion_path       : {motion_path}")
    print(f"first_npz         : {motion_npz}")
    print(f"amp_body_num      : {len(amp_body_names)}")
    print(f"amp_anchor_name   : {amp_anchor_name}")
    print(f"amp_reward_coef   : {train_cfg.get('amp_reward_coef')}")
    print(f"amp_task_lerp     : {train_cfg.get('amp_task_reward_lerp')}")
    print(f"amp_hidden_dims   : {train_cfg.get('amp_discr_hidden_dims')}")
    print(f"expected amp dim  : {15 * len(amp_body_names)}")
    print(f"expected D input  : {30 * len(amp_body_names)}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    robot = env.unwrapped.scene["robot"]
    isaac_body_names = list(robot.body_names)
    isaac_joint_names = list(robot.joint_names)

    print_names("Isaac USD body_names", isaac_body_names)
    print_names("Isaac USD joint_names", isaac_joint_names)

    print("\n===== AMP target body mapping in Isaac USD =====")
    missing_isaac = [n for n in amp_body_names + [amp_anchor_name] if n not in isaac_body_names]
    if missing_isaac:
        print(f"[FAIL] target bodies missing in Isaac USD: {missing_isaac}")
    else:
        for n in amp_body_names:
            print(f"{n:32s} -> isaac_body[{isaac_body_names.index(n)}]")
        print(f"{amp_anchor_name:32s} -> isaac_body[{isaac_body_names.index(amp_anchor_name)}]  [anchor]")

    data = np.load(motion_npz, allow_pickle=True)
    print("\n===== motion npz keys/shapes =====")
    for k in data.files:
        arr = data[k]
        shape = getattr(arr, "shape", None)
        print(f"{k:20s} shape={shape} dtype={getattr(arr, 'dtype', None)}")

    npz_body_names = get_names(data, "body_names")
    npz_joint_names = get_names(data, "joint_names")

    if npz_body_names is not None:
        print_names("motion npz body_names", npz_body_names)
        body_same_order = npz_body_names == isaac_body_names[: len(npz_body_names)]
        print(f"\nbody_names same index order as Isaac prefix: {body_same_order}")
    else:
        print("\n[FAIL] motion npz has no body_names. Body-order alignment is unverifiable.")

    if npz_joint_names is not None:
        print_names("motion npz joint_names", npz_joint_names)
        same_joint_order = npz_joint_names == isaac_joint_names[: len(npz_joint_names)]
        print(f"\njoint_names same index order as Isaac prefix: {same_joint_order}")
        missing_joints = [n for n in npz_joint_names if n not in isaac_joint_names]
        if missing_joints:
            print(f"[FAIL] motion joints missing in Isaac USD: {missing_joints[:20]}")
    else:
        print("\n[WARN] motion npz has no joint_names. Joint reset/order alignment is unverifiable.")

    # Step once to populate all observation groups (AMP group is updated at step).
    actions = torch.zeros(env.num_envs, env.num_actions, device=device)
    obs, rew, dones, infos = env.step(actions)
    obs, extras = unpack_obs((obs, infos))
    if "amp" not in extras.get("observations", {}):
        _, extras2 = unpack_obs(env.get_observations())
        if "amp" in extras2.get("observations", {}):
            extras = extras2
        else:
            raise RuntimeError("extras['observations']['amp'] not found. AMP observation group is not active.")

    policy_amp = extras["observations"]["amp"].to(device)
    expert_amp = make_expert_amp_obs(
        data=data,
        target_body_names=amp_body_names,
        anchor_name=amp_anchor_name,
        isaac_body_names=isaac_body_names,
        device=device,
        max_frames=args_cli.debug_frames,
        strict_names=args_cli.strict_names,
    )

    expected_dim = 15 * len(amp_body_names)

    print("\n===== dimension check =====")
    print(f"policy_amp_dim : {policy_amp.shape[1]}")
    print(f"expert_amp_dim : {expert_amp.shape[1]}")
    print(f"expected_dim   : {expected_dim}")
    print("[PASS]" if policy_amp.shape[1] == expert_amp.shape[1] == expected_dim else "[FAIL] AMP dim mismatch")

    n = len(amp_body_names)
    part_stats("policy AMP obs from Isaac env", policy_amp, n)
    part_stats("expert AMP obs from motion npz", expert_amp, n)
    compare_stats(policy_amp[: min(policy_amp.shape[0], expert_amp.shape[0])], expert_amp[: policy_amp.shape[0]], n)

    print("\n===== final verdict =====")
    if npz_body_names is None:
        print("[FAIL] Biggest issue: motion npz has no body_names. Fix this first.")
        print("       Otherwise AMPLoader is probably indexing MuJoCo expert arrays with Isaac USD body indices.")
    elif missing_isaac:
        print("[FAIL] AMP target bodies are missing from Isaac USD.")
    elif policy_amp.shape[1] != expert_amp.shape[1]:
        print("[FAIL] AMP observation dimension mismatch.")
    else:
        print("[PASS] Basic AMP metadata/dimension checks passed.")
        print("       Next suspicious item, if D is still too strong: USD link-frame offset vs MuJoCo link-frame offset.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
