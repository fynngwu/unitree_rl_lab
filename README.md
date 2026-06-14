# Unitree RL Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/ZwcVwxv5rq)


## Overview

This project provides a set of reinforcement learning environments for Unitree robots, built on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab).

Currently supports Unitree **Go2**, **H1** and **G1-29dof** robots.

<div align="center">

| <div align="center"> Isaac Lab </div> | <div align="center">  Mujoco </div> |  <div align="center"> Physical </div> |
|--- | --- | --- |
| [<img src="https://oss-global-cdn.unitree.com/static/d879adac250648c587d3681e90658b49_480x397.gif" width="240px">](g1_sim.gif) | [<img src="https://oss-global-cdn.unitree.com/static/3c88e045ab124c3ab9c761a99cb5e71f_480x397.gif" width="240px">](g1_mujoco.gif) | [<img src="https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif" width="240px">](g1_real.gif) |

</div>

## Installation

Prerequisites: [Isaac Lab](https://isaac-sim.github.io/IsaacLab) installed and its Python environment activated.

```bash
# Activate IsaacLab environment (conda or venv)
conda activate env_isaaclab
# or: source /path/to/IsaacLab/env_isaaclab/bin/activate

# Clone this repository
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
cd unitree_rl_lab

# Install the package in editable mode
pip install -e source/unitree_rl_lab/

# Download AMP reference motion data
# Place .npz motion files into source/assets/motions/g1/amp/WalkandRun/
# (see AMP Motion Data section below)
```

### Unitree Robot Description Files

**Method 1: USD Files**
- Download from [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model)
  ```bash
  git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
  ```
- Set `UNITREE_MODEL_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`:
  ```python
  UNITREE_MODEL_DIR = "/path/to/unitree_model"
  ```

**Method 2: URDF Files** (recommended, Isaac Sim >= 5.0)
- Download from [unitree_ros](https://github.com/unitreerobotics/unitree_ros)
  ```bash
  git clone https://github.com/unitreerobotics/unitree_ros.git
  ```
- Set `UNITREE_ROS_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`:
  ```python
  UNITREE_ROS_DIR = "/path/to/unitree_ros/unitree_ros"
  ```

### AMP Motion Data

AMP tasks require reference motion data (`.npz` files) for training the discriminator. Copy the motion clips to the expected location:

```bash
# Copy motion data from existing assets directory (if present)
cp -r assets/motions source/assets/motions
```

If you don't have the data locally, download it from [link-to-motion-data] and place it at:
```
source/assets/motions/g1/amp/WalkandRun/
```

### AMP Training

```bash
# Activate environment (if not already active)
conda activate env_isaaclab

# Train AMP policy
python scripts/rsl_rl/amp_train.py --task Unitree-G1-29dof-AMP-Flat --headless

# Play a trained AMP policy
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-AMP-Flat

# List available tasks
./unitree_rl_lab.sh -l
```

