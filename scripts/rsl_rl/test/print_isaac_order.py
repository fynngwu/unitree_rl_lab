"""Print Isaac USD body/joint order for G1 29DOF."""
import argparse, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-AMP-Flat")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym, isaaclab_tasks, unitree_rl_lab.tasks
from isaaclab_tasks.utils import parse_env_cfg

env = gym.make(args.task, cfg=parse_env_cfg(args.task, device=args.device, num_envs=1))
env.reset()
robot = env.unwrapped.scene["robot"]

with open("/tmp/isaac_body_joint_order.txt", "w") as f:
    f.write("body_names:\n")
    for i, n in enumerate(robot.body_names):
        f.write(f"  body[{i:03d}] = '{n}'\n")
    f.write("\njoint_names:\n")
    for i, n in enumerate(robot.joint_names):
        f.write(f"  joint[{i:03d}] = '{n}'\n")

env.close()
simulation_app.close()
