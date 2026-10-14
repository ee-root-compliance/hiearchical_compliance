import os
import copy
import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
import active_adaptation.utils.symmetry as symmetry_utils

from isaaclab.assets import ArticulationCfg as _ArticulationCfg
from isaaclab.utils import configclass

from typing import Mapping

@configclass
class ArticulationCfg(_ArticulationCfg):
    joint_symmetry_mapping: Mapping[str, list[int | tuple[int, str]]] = None
    spatial_symmetry_mapping: Mapping[str, str] = None

ASSET_PATH = os.path.dirname(__file__)

G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/G1/g1_29dof_rev_1_0_flat.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": -0.28,
            ".*_knee_joint": 0.5,
            ".*_ankle_pitch_joint": -0.23,
            # ".*_elbow_pitch_joint": 0.87,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            ".*wrist_roll_joint": 0.0,
            ".*wrist_pitch_joint": 0.0,
            ".*wrist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=".*",
            effort_limit_sim={
                # legs
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
                # waist
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
                # feet
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
                # arms
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                # legs
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
                # waist
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
                # feet
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint": 37.0,
                # arms
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                # legs (7520_14 / 7520_22)
                ".*_hip_yaw_joint": 40.17923847137318,
                ".*_hip_pitch_joint": 40.17923847137318,
                ".*_hip_roll_joint": 99.09842777666113,
                ".*_knee_joint": 99.09842777666113,
                # waist
                "waist_yaw_joint": 40.17923847137318,         # 7520_14
                "waist_roll_joint": 28.50124619574858,        # 2 * 5020
                "waist_pitch_joint": 28.50124619574858,       # 2 * 5020
                # feet
                ".*_ankle_pitch_joint": 28.50124619574858,    # 2 * 5020
                ".*_ankle_roll_joint": 28.50124619574858,     # 2 * 5020
                # arms
                ".*_shoulder_pitch_joint": 14.25062309787429, # 5020
                ".*_shoulder_roll_joint": 14.25062309787429,  # 5020
                ".*_shoulder_yaw_joint": 14.25062309787429,   # 5020
                ".*_elbow_joint": 14.25062309787429,          # 5020
                ".*_wrist_roll_joint": 14.25062309787429,     # 5020
                ".*_wrist_pitch_joint": 16.77832748089279,    # 4010
                ".*_wrist_yaw_joint": 16.77832748089279,      # 4010
            },
            damping={
                # legs (7520_14 / 7520_22)
                ".*_hip_yaw_joint": 2.5578897650279457,
                ".*_hip_pitch_joint": 2.5578897650279457,
                ".*_hip_roll_joint": 6.3088018534966395,
                ".*_knee_joint": 6.3088018534966395,
                # waist
                "waist_yaw_joint": 2.5578897650279457,        # 7520_14
                "waist_roll_joint": 1.814445686584846,        # 2 * 5020
                "waist_pitch_joint": 1.814445686584846,       # 2 * 5020
                # feet
                ".*_ankle_pitch_joint": 1.814445686584846,    # 2 * 5020
                ".*_ankle_roll_joint": 1.814445686584846,     # 2 * 5020
                # arms
                ".*_shoulder_pitch_joint": 0.907222843292423, # 5020
                ".*_shoulder_roll_joint": 0.907222843292423,  # 5020
                ".*_shoulder_yaw_joint": 0.907222843292423,   # 5020
                ".*_elbow_joint": 0.907222843292423,          # 5020
                ".*_wrist_roll_joint": 0.907222843292423,     # 5020
                ".*_wrist_pitch_joint": 1.06814150219,        # 4010
                ".*_wrist_yaw_joint": 1.06814150219,          # 4010
            },
            armature={
                # legs
                ".*_hip_yaw_joint": 0.010177520,              # 7520_14
                ".*_hip_pitch_joint": 0.010177520,            # 7520_14
                ".*_hip_roll_joint": 0.025101925,             # 7520_22
                ".*_knee_joint": 0.025101925,                 # 7520_22
                # waist
                "waist_yaw_joint": 0.010177520,               # 7520_14
                "waist_roll_joint": 0.00721945,               # 2 * 5020
                "waist_pitch_joint": 0.00721945,              # 2 * 5020
                # feet
                ".*_ankle_pitch_joint": 0.00721945,           # 2 * 5020
                ".*_ankle_roll_joint": 0.00721945,            # 2 * 5020
                # arms
                ".*_shoulder_pitch_joint": 0.003609725,       # 5020
                ".*_shoulder_roll_joint": 0.003609725,        # 5020
                ".*_shoulder_yaw_joint": 0.003609725,         # 5020
                ".*_elbow_joint": 0.003609725,                # 5020
                ".*_wrist_roll_joint": 0.003609725,           # 5020
                ".*_wrist_pitch_joint": 0.00425,              # 4010
                ".*_wrist_yaw_joint": 0.00425,                # 4010
            }
        ),
    },
    joint_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_joint": (1, "right_hip_pitch_joint"),
        "left_hip_roll_joint": (-1, "right_hip_roll_joint"),
        "left_hip_yaw_joint": (-1, "right_hip_yaw_joint"),
        "left_knee_joint": (1, "right_knee_joint"),
        "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
        "left_ankle_roll_joint": (-1, "right_ankle_roll_joint"),
        "waist_yaw_joint": (-1, "waist_yaw_joint"),
        "waist_roll_joint": (-1, "waist_roll_joint"),
        "waist_pitch_joint": (1, "waist_pitch_joint"),
        "left_shoulder_pitch_joint": (1, "right_shoulder_pitch_joint"),
        "left_shoulder_roll_joint": (-1, "right_shoulder_roll_joint"),
        "left_shoulder_yaw_joint": (-1, "right_shoulder_yaw_joint"),
        "left_elbow_joint": (1, "right_elbow_joint"),
        "left_wrist_yaw_joint": (-1, "right_wrist_yaw_joint"),
        "left_wrist_roll_joint": (-1, "right_wrist_roll_joint"),
        "left_wrist_pitch_joint": (1, "right_wrist_pitch_joint"),
    }),
    spatial_symmetry_mapping=symmetry_utils.mirrored({
        "left_hip_pitch_link": "right_hip_pitch_link",
        "left_hip_roll_link": "right_hip_roll_link",
        "left_hip_yaw_link": "right_hip_yaw_link",
        "left_knee_link": "right_knee_link",
        "left_ankle_pitch_link": "right_ankle_pitch_link",
        "left_ankle_roll_link": "right_ankle_roll_link",
        "pelvis": "pelvis",
        "torso_link": "torso_link",
        "waist_yaw_link": "waist_yaw_link",
        "waist_roll_link": "waist_roll_link",
        "left_shoulder_pitch_link": "right_shoulder_pitch_link",
        "left_shoulder_roll_link": "right_shoulder_roll_link",
        "left_shoulder_yaw_link": "right_shoulder_yaw_link",
        "left_elbow_link": "right_elbow_link",
        "left_wrist_yaw_link": "right_wrist_yaw_link",
        "left_wrist_roll_link": "right_wrist_roll_link",
        "left_wrist_pitch_link": "right_wrist_pitch_link",
        "right_hand_mimic": "left_hand_mimic",
        "head_mimic": "head_mimic",
    })
)

G1_COL_FULL = copy.deepcopy(G1_CFG)
G1_COL_FULL.spawn.usd_path = f"{ASSET_PATH}/G1/g1_flat_fullcol.usd"