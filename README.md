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

# Robot USD model is included in the repository.
```



### AMP Training

Default AMP motion data (walking and running) is already included in the repository. To train:

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

### Custom AMP Motion Data

If you want to use your own motion data, convert CSV clips from [GMR](https://github.com/chengqiang0103/GMR) (exported via `scripts/batch_gmr_pkl_to_csv.py`) to NPZ format:

**Install mjlab dependencies (in a separate virtual environment):**

```bash
pip install mjlab
```

**Convert CSV to NPZ:**

```bash
python scripts/csv_to_npz.py \
  --input-file motion_data.csv \
  --output-name motion.npz \
  --input-fps 30 \
  --output-fps 50
```

Place the converted `.npz` files into:
```
source/assets/motions/g1/amp/WalkandRun/
```

