"""Shared AMP constants for G1 29DOF: NPZ/MuJoCo ↔ Isaac/USD conversion tables."""

import torch

# =============================================================================
# NPZ (MuJoCo MJCF depth-first) body order
#    pelvis -> left leg 6 -> right leg 6 -> waist 3 -> left arm 7 -> right arm 7
# =============================================================================
NPZ_BODY_NAMES = [
    "pelvis",                    # 00
    "left_hip_pitch_link",        # 01
    "left_hip_roll_link",         # 02
    "left_hip_yaw_link",          # 03
    "left_knee_link",             # 04
    "left_ankle_pitch_link",      # 05
    "left_ankle_roll_link",       # 06
    "right_hip_pitch_link",       # 07
    "right_hip_roll_link",        # 08
    "right_hip_yaw_link",         # 09
    "right_knee_link",            # 10
    "right_ankle_pitch_link",     # 11
    "right_ankle_roll_link",      # 12
    "waist_yaw_link",             # 13
    "waist_roll_link",            # 14
    "torso_link",                 # 15
    "left_shoulder_pitch_link",   # 16
    "left_shoulder_roll_link",    # 17
    "left_shoulder_yaw_link",     # 18
    "left_elbow_link",            # 19
    "left_wrist_roll_link",       # 20
    "left_wrist_pitch_link",      # 21
    "left_wrist_yaw_link",        # 22
    "right_shoulder_pitch_link",  # 23
    "right_shoulder_roll_link",   # 24
    "right_shoulder_yaw_link",    # 25
    "right_elbow_link",           # 26
    "right_wrist_roll_link",      # 27
    "right_wrist_pitch_link",     # 28
    "right_wrist_yaw_link",       # 29
]

# =============================================================================
# NPZ (MuJoCo) joint order
# =============================================================================
NPZ_JOINT_NAMES = [
    "left_hip_pitch_joint",       # 00
    "left_hip_roll_joint",        # 01
    "left_hip_yaw_joint",         # 02
    "left_knee_joint",            # 03
    "left_ankle_pitch_joint",     # 04
    "left_ankle_roll_joint",      # 05
    "right_hip_pitch_joint",      # 06
    "right_hip_roll_joint",       # 07
    "right_hip_yaw_joint",        # 08
    "right_knee_joint",           # 09
    "right_ankle_pitch_joint",    # 10
    "right_ankle_roll_joint",     # 11
    "waist_yaw_joint",            # 12
    "waist_roll_joint",           # 13
    "waist_pitch_joint",          # 14
    "left_shoulder_pitch_joint",  # 15
    "left_shoulder_roll_joint",   # 16
    "left_shoulder_yaw_joint",    # 17
    "left_elbow_joint",           # 18
    "left_wrist_roll_joint",      # 19
    "left_wrist_pitch_joint",     # 20
    "left_wrist_yaw_joint",       # 21
    "right_shoulder_pitch_joint", # 22
    "right_shoulder_roll_joint",  # 23
    "right_shoulder_yaw_joint",   # 24
    "right_elbow_joint",          # 25
    "right_wrist_roll_joint",     # 26
    "right_wrist_pitch_joint",    # 27
    "right_wrist_yaw_joint",      # 28
]

# =============================================================================
# Joint conversion: NPZ ↔ Isaac
#   NPZ_BY_ISAAC_JOINT[isaac_idx] = npz_idx
#   ISAAC_BY_NPZ_JOINT[npz_idx]   = isaac_idx
# =============================================================================
NPZ_BY_ISAAC_JOINT = torch.tensor([
    0,   # isaac 00 left_hip_pitch_joint      <- npz 00
    6,   # isaac 01 right_hip_pitch_joint     <- npz 06
    12,  # isaac 02 waist_yaw_joint           <- npz 12
    1,   # isaac 03 left_hip_roll_joint       <- npz 01
    7,   # isaac 04 right_hip_roll_joint      <- npz 07
    13,  # isaac 05 waist_roll_joint          <- npz 13
    2,   # isaac 06 left_hip_yaw_joint        <- npz 02
    8,   # isaac 07 right_hip_yaw_joint       <- npz 08
    14,  # isaac 08 waist_pitch_joint         <- npz 14
    3,   # isaac 09 left_knee_joint           <- npz 03
    9,   # isaac 10 right_knee_joint          <- npz 09
    15,  # isaac 11 left_shoulder_pitch_joint <- npz 15
    22,  # isaac 12 right_shoulder_pitch_joint<- npz 22
    4,   # isaac 13 left_ankle_pitch_joint    <- npz 04
    10,  # isaac 14 right_ankle_pitch_joint   <- npz 10
    16,  # isaac 15 left_shoulder_roll_joint  <- npz 16
    23,  # isaac 16 right_shoulder_roll_joint <- npz 23
    5,   # isaac 17 left_ankle_roll_joint     <- npz 05
    11,  # isaac 18 right_ankle_roll_joint    <- npz 11
    17,  # isaac 19 left_shoulder_yaw_joint   <- npz 17
    24,  # isaac 20 right_shoulder_yaw_joint  <- npz 24
    18,  # isaac 21 left_elbow_joint          <- npz 18
    25,  # isaac 22 right_elbow_joint         <- npz 25
    19,  # isaac 23 left_wrist_roll_joint     <- npz 19
    26,  # isaac 24 right_wrist_roll_joint    <- npz 26
    20,  # isaac 25 left_wrist_pitch_joint    <- npz 20
    27,  # isaac 26 right_wrist_pitch_joint   <- npz 27
    21,  # isaac 27 left_wrist_yaw_joint      <- npz 21
    28,  # isaac 28 right_wrist_yaw_joint     <- npz 28
], dtype=torch.long)

ISAAC_BY_NPZ_JOINT = torch.argsort(NPZ_BY_ISAAC_JOINT)

# =============================================================================
# Body conversion: Isaac full 30 ↔ NPZ full 30
#   NPZ_BY_ISAAC_BODY[isaac_idx] = npz_idx
#   ISAAC_BY_NPZ_BODY[npz_idx]   = isaac_idx
# =============================================================================
NPZ_BY_ISAAC_BODY = torch.tensor([
    0,   1,  7, 13,  2,  8, 14,  3,  9, 15,
    4,  10, 16, 23,  5, 11, 17, 24,  6, 12,
    18, 25, 19, 26, 20, 27, 21, 28, 22, 29,
], dtype=torch.long)

ISAAC_BY_NPZ_BODY = torch.argsort(NPZ_BY_ISAAC_BODY)

# =============================================================================
# AMP 13-link body IDs (in NPZ order)
# =============================================================================
AMP_BODY_NAMES = (
    "pelvis",                    # npz 00
    "left_hip_roll_link",        # npz 02
    "left_knee_link",            # npz 04
    "right_hip_roll_link",       # npz 08
    "right_knee_link",           # npz 10
    "left_shoulder_roll_link",   # npz 17
    "left_elbow_link",           # npz 19
    "left_wrist_yaw_link",       # npz 22
    "right_shoulder_roll_link",  # npz 24
    "right_elbow_link",          # npz 26
    "right_wrist_yaw_link",      # npz 29
)
AMP_ANCHOR_NAME = "torso_link"  # npz 15

AMP_NPZ_BODY_IDS = torch.tensor([
    0,   # pelvis
    2,   # left_hip_roll_link
    4,   # left_knee_link
    8,   # right_hip_roll_link
    10,  # right_knee_link
    17,  # left_shoulder_roll_link
    19,  # left_elbow_link
    22,  # left_wrist_yaw_link
    24,  # right_shoulder_roll_link
    26,  # right_elbow_link
    29,  # right_wrist_yaw_link
], dtype=torch.long)

AMP_NPZ_ANCHOR_ID = torch.tensor(15, dtype=torch.long)  # torso_link

AMP_ISAAC_BODY_IDS = torch.tensor([
    0,   # pelvis                  <- npz 0
    4,   # left_hip_roll_link       <- npz 2
    10,  # left_knee_link           <- npz 4
    5,   # right_hip_roll_link      <- npz 8
    11,  # right_knee_link          <- npz 10
    16,  # left_shoulder_roll_link  <- npz 17
    22,  # left_elbow_link          <- npz 19
    28,  # left_wrist_yaw_link      <- npz 22
    17,  # right_shoulder_roll_link <- npz 24
    23,  # right_elbow_link         <- npz 26
    29,  # right_wrist_yaw_link     <- npz 29
], dtype=torch.long)

AMP_ISAAC_ANCHOR_ID = torch.tensor(9, dtype=torch.long)  # torso_link in Isaac

# =============================================================================
# AMP observation dimension
# =============================================================================
AMP_OBS_DIM = 15 * len(AMP_BODY_NAMES)  # 195
