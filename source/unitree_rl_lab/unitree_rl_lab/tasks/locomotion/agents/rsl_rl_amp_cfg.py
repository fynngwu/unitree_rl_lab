import os

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from unitree_rl_lab.amp.amp_constants import AMP_BODY_NAMES, AMP_ANCHOR_NAME

# Resolve to unitree_rl_lab project root (up 4 levels from this file's package path).
# This file lives at: source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/
# Project root is:    source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/../../../../../
# After pip -e install, the actual path is resolved relative to this file on disk.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up to unitree_rl_lab repo root: agents/ -> locomotion/ -> tasks/ -> unitree_rl_lab/ -> source/ -> unitree_rl_lab/
AMP_MOTION_DIR = os.path.abspath(os.path.normpath(os.path.join(
    _THIS_DIR, "..", "..", "..", "..", "..", "assets", "motions", "g1", "amp", "WalkandRun"
)))
assert os.path.exists(AMP_MOTION_DIR), f"AMP motion dir not found: {AMP_MOTION_DIR}"


@configclass
class AmpPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "AMPPPO"


@configclass
class G1AmpFlatRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "g1_amp_flat_unitree_rl_lab"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = AmpPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    amp_reward_coef = 0.1
    amp_motion_files = AMP_MOTION_DIR
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.75
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 29

    amp_body_names = AMP_BODY_NAMES
    amp_anchor_name = AMP_ANCHOR_NAME
