# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# encos_robots_cfg.py
# Copyright (c) 2025-2026, Master Jia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Motor cfgs (Unitree-style T–N)
from .encos_actuators import (
    MotorCfg_A4310,
    MotorCfg_A6408,
    MotorCfg_A8112,
    MotorCfg_A10020,
)


def rpm_to_rads(rpm: float) -> float:
    """Convert rpm to rad/s."""
    return rpm * (2.0 * math.pi / 60.0)


# 刚度 (Kp) - 大幅提升
_MVR_10DOF_STIFFNESS = {
    "hip_roll": 40.0,  # 增加侧向刚度
    "hip_yaw": 40.0,
    "hip_pitch": 150.0,  # 🚨 主力关节：大幅增加
    "knee": 150.0,  # 🚨 主力关节：大幅增加
    "ankle_pitch": 80.0,  # 🚨 平衡关节：增加以锁住脚掌
}

# 阻尼 (Kd) - 配合 Kp 增加，通常 Kp 的 5% - 10% 是安全区
_MVR_10DOF_DAMPING = {
    "hip_roll": 2.0,
    "hip_yaw": 2.0,
    "hip_pitch": 8.0,
    "knee": 8.0,
    "ankle_pitch": 3.0,
}

_MVR_10DOF_DEFAULT_JOINT_POS = {
    # 直立姿态优化版
    "left_hip_roll_joint": 0.00,
    "left_hip_yaw_joint": 0.00,
    "left_hip_pitch_joint": 0.25,  # 更直
    "left_knee_joint": -0.60,  # 更直 (从 -0.955 改为 -0.6)
    "left_ankle_pitch_joint": 0.35,  # 配合膝盖调整，保证脚掌水平
    "right_hip_roll_joint": 0.00,
    "right_hip_yaw_joint": 0.00,
    "right_hip_pitch_joint": -0.25,  # 注意负号
    "right_knee_joint": 0.60,  # 注意正号
    "right_ankle_pitch_joint": 0.35,
}

# _MVR_10DOF_DEFAULT_JOINT_POS = {
#     # -------- 左腿 (Left Leg) --------
#     "left_hip_roll_joint": 0.0,
#     "left_hip_yaw_joint": 0.0,
#     "left_hip_pitch_joint": 0.0,  # 屈髋
#     "left_knee_joint": 0.0,  # 屈膝
#     "left_ankle_pitch_joint": 0.0,  # 踝关节配平
#     # -------- 右腿 (Right Leg) --------
#     "right_hip_roll_joint": 0.0,
#     "right_hip_yaw_joint": 0.0,
#     "right_hip_pitch_joint": 0.0,  # 屈髋 (镜像)
#     "right_knee_joint": 0.0,  # 屈膝 (镜像)
#     "right_ankle_pitch_joint": 0.0,  # 踝关节配平
# }

MVR_10DOF_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/jhl/projects/Project-RL/robots/Mvr_10dof/usd/Mvr_10dof_no_col_inf/Mvr_10dof_no_col_inf.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=[0, 0, 0.60],
        joint_pos=_MVR_10DOF_DEFAULT_JOINT_POS,
    ),
    # ==========================================================
    # 核心修改：通过拆分左右腿的定义，强制锁定顺序
    # 顺序：左腿(Roll, Yaw, Pitch, Knee, Ankle) -> 右腿(...)
    # ==========================================================
    actuators={
        # ------------------------------------------------------
        # 第一阶段：左腿 (Left Leg Chain)
        # ------------------------------------------------------
        # 1. 左腿 Roll & Yaw (物理参数相同，可以合并，确保顺序)
        # 对应 Index: 0, 1
        "left_hip_roll_yaw": ImplicitActuatorCfg(
            joint_names_expr=["left_hip_roll_joint", "left_hip_yaw_joint"],
            effort_limit_sim=36,
            velocity_limit_sim=rpm_to_rads(87.0),
            stiffness={
                "left_hip_roll_joint": _MVR_10DOF_STIFFNESS["hip_roll"],
                "left_hip_yaw_joint": _MVR_10DOF_STIFFNESS["hip_yaw"],
            },
            damping={
                "left_hip_roll_joint": _MVR_10DOF_DAMPING["hip_roll"],
                "left_hip_yaw_joint": _MVR_10DOF_DAMPING["hip_yaw"],
            },
            armature=0.0236,
        ),
        # 2. 左腿 Pitch & Knee (物理参数相同，合并)
        # 对应 Index: 2, 3
        "left_hip_pitch_knee": ImplicitActuatorCfg(
            joint_names_expr=["left_hip_pitch_joint", "left_knee_joint"],
            effort_limit_sim=90,  # 注意：这是大电机
            velocity_limit_sim=rpm_to_rads(157.0),
            stiffness={
                "left_hip_pitch_joint": _MVR_10DOF_STIFFNESS["hip_pitch"],
                "left_knee_joint": _MVR_10DOF_STIFFNESS["knee"],
            },
            damping={
                "left_hip_pitch_joint": _MVR_10DOF_DAMPING["hip_pitch"],
                "left_knee_joint": _MVR_10DOF_DAMPING["knee"],
            },
            armature=0.0475,
        ),
        # 3. 左腿 Ankle (单独定义，因为参数特殊)
        # 对应 Index: 4
        "left_ankle": ImplicitActuatorCfg(
            joint_names_expr=["left_ankle_pitch_joint"],
            effort_limit_sim=36,
            velocity_limit_sim=rpm_to_rads(87.0),
            stiffness=_MVR_10DOF_STIFFNESS["ankle_pitch"],
            damping=_MVR_10DOF_DAMPING["ankle_pitch"],
            armature=0.0236,
        ),
        # ------------------------------------------------------
        # 第二阶段：右腿 (Right Leg Chain)
        # ------------------------------------------------------
        # 4. 右腿 Roll & Yaw
        # 对应 Index: 5, 6
        "right_hip_roll_yaw": ImplicitActuatorCfg(
            joint_names_expr=["right_hip_roll_joint", "right_hip_yaw_joint"],
            effort_limit_sim=36,
            velocity_limit_sim=rpm_to_rads(87.0),
            stiffness={
                "right_hip_roll_joint": _MVR_10DOF_STIFFNESS["hip_roll"],
                "right_hip_yaw_joint": _MVR_10DOF_STIFFNESS["hip_yaw"],
            },
            damping={
                "right_hip_roll_joint": _MVR_10DOF_DAMPING["hip_roll"],
                "right_hip_yaw_joint": _MVR_10DOF_DAMPING["hip_yaw"],
            },
            armature=0.0236,
        ),
        # 5. 右腿 Pitch & Knee
        # 对应 Index: 7, 8
        "right_hip_pitch_knee": ImplicitActuatorCfg(
            joint_names_expr=["right_hip_pitch_joint", "right_knee_joint"],
            effort_limit_sim=90,  # 大电机
            velocity_limit_sim=rpm_to_rads(157.0),
            stiffness={
                "right_hip_pitch_joint": _MVR_10DOF_STIFFNESS["hip_pitch"],
                "right_knee_joint": _MVR_10DOF_STIFFNESS["knee"],
            },
            damping={
                "right_hip_pitch_joint": _MVR_10DOF_DAMPING["hip_pitch"],
                "right_knee_joint": _MVR_10DOF_DAMPING["knee"],
            },
            armature=0.0475,
        ),
        # 6. 右腿 Ankle
        # 对应 Index: 9
        "right_ankle": ImplicitActuatorCfg(
            joint_names_expr=["right_ankle_pitch_joint"],
            effort_limit_sim=36,
            velocity_limit_sim=rpm_to_rads(87.0),
            stiffness=_MVR_10DOF_STIFFNESS["ankle_pitch"],
            damping=_MVR_10DOF_DAMPING["ankle_pitch"],
            armature=0.0236,
        ),
    },
)

# -----------------------------------------------------------------------------
# Robot: MVR_22DOF_CONFIG
# -----------------------------------------------------------------------------

_MVR_22DOF_STIFFNESS = {
    "hip_pitch": 250.0,
    "hip_roll": 150.0,
    "hip_yaw": 150.0,
    "knee": 250.0,
    "ankle_pitch": 200.0,
    "ankle_roll": 100.0,
    "waist": 10.0,
    "head": 10.0,
    "arm_pitch": 5.0,
    "arm_roll": 5.0,
    "arm_yaw": 5.0,
    "arm_elbow": 5.0,
}

_MVR_22DOF_DAMPING = {
    "hip_pitch": 10.0,
    "hip_roll": 10.0,
    "hip_yaw": 10.0,
    "knee": 10.0,
    "ankle_pitch": 0.1,
    "ankle_roll": 0.1,
    "waist": 5.0,
    "head": 5.0,
    "arm_pitch": 5.0,
    "arm_roll": 5.0,
    "arm_yaw": 5.0,
    "arm_elbow": 5.0,
}

_MVR_22DOF_DEFAULT_JOINT_POS = {
    # Left Leg
    "left_hip_pitch_joint": -0.25,
    "left_hip_roll_joint": -0.03,
    "left_hip_yaw_joint": 0.01,
    "left_knee_joint": -0.50,
    "left_ankle_pitch_joint": 0.15,
    "left_ankle_roll_joint": -0.01,
    # Right Leg
    "right_hip_pitch_joint": 0.25,
    "right_hip_roll_joint": 0.03,
    "right_hip_yaw_joint": -0.01,
    "right_knee_joint": 0.50,
    "right_ankle_pitch_joint": -0.15,
    "right_ankle_roll_joint": 0.01,
    # Torso & Head
    "waist_joint": 0.0,
    "head_joint": 0.0,
    # Left Arm
    "left_arm_pitch_joint": 0.0,
    "left_arm_roll_joint": 0.0,
    "left_arm_yaw_joint": 0.0,
    "left_elbow_joint": -1.0,
    # Right Arm
    "right_arm_pitch_joint": 0.0,
    "right_arm_roll_joint": 0.0,
    "right_arm_yaw_joint": 0.0,
    "right_elbow_joint": 1.0,
}

MVR_22DOF_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/jhl/projects/Project-RL/robots/Mvr_22dof/usd/Mvr_22dof.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,  # align to Unitree-style (non-zero)
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=[0, 0, 0.70],
        joint_pos=_MVR_22DOF_DEFAULT_JOINT_POS,
    ),
    actuators={
        # heavy -> A10020 : hip_pitch + knee
        "heavy_motors": MotorCfg_A10020(
            joint_names_expr=[".*_hip_pitch_joint", ".*_knee_joint"],
            stiffness={
                ".*_hip_pitch_joint": _MVR_22DOF_STIFFNESS["hip_pitch"],
                ".*_knee_joint": _MVR_22DOF_STIFFNESS["knee"],
            },
            damping={
                ".*_hip_pitch_joint": _MVR_22DOF_DAMPING["hip_pitch"],
                ".*_knee_joint": _MVR_22DOF_DAMPING["knee"],
            },
            effort_limit_sim=150,
            velocity_limit_sim=rpm_to_rads(143.0),
            armature=0.0703,
        ),
        # medium -> A8112 : hip_roll + hip_yaw + waist
        "medium_motors": MotorCfg_A8112(
            joint_names_expr=[".*_hip_roll_joint", ".*_hip_yaw_joint", "waist_joint"],
            stiffness={
                ".*_hip_roll_joint": _MVR_22DOF_STIFFNESS["hip_roll"],
                ".*_hip_yaw_joint": _MVR_22DOF_STIFFNESS["hip_yaw"],
                "waist_joint": _MVR_22DOF_STIFFNESS["waist"],
            },
            damping={
                ".*_hip_roll_joint": _MVR_22DOF_DAMPING["hip_roll"],
                ".*_hip_yaw_joint": _MVR_22DOF_DAMPING["hip_yaw"],
                "waist_joint": _MVR_22DOF_DAMPING["waist"],
            },
            effort_limit_sim=90,
            velocity_limit_sim=rpm_to_rads(157.0),
            armature=0.0475,
        ),
        # light -> A6408 : arm_pitch + arm_roll
        "arm_upper_motors": MotorCfg_A6408(
            joint_names_expr=[".*_arm_pitch_joint", ".*_arm_roll_joint"],
            stiffness={
                ".*_arm_pitch_joint": _MVR_22DOF_STIFFNESS["arm_pitch"],
                ".*_arm_roll_joint": _MVR_22DOF_STIFFNESS["arm_roll"],
            },
            damping={
                ".*_arm_pitch_joint": _MVR_22DOF_DAMPING["arm_pitch"],
                ".*_arm_roll_joint": _MVR_22DOF_DAMPING["arm_roll"],
            },
            effort_limit_sim=60,
            velocity_limit_sim=rpm_to_rads(149.0),
            armature=0.0389,
        ),
        # micro -> A4310 : ankles + head + arm_yaw + elbow
        "micro_motors": MotorCfg_A4310(
            joint_names_expr=[
                ".*_ankle_.*",
                "head_joint",
                ".*_arm_yaw_joint",
                ".*_elbow_joint",
            ],
            stiffness={
                ".*_ankle_pitch_joint": _MVR_22DOF_STIFFNESS["ankle_pitch"],
                ".*_ankle_roll_joint": _MVR_22DOF_STIFFNESS["ankle_roll"],
                "head_joint": _MVR_22DOF_STIFFNESS["head"],
                ".*_arm_yaw_joint": _MVR_22DOF_STIFFNESS["arm_yaw"],
                ".*_elbow_joint": _MVR_22DOF_STIFFNESS["arm_elbow"],
            },
            damping={
                ".*_ankle_pitch_joint": _MVR_22DOF_DAMPING["ankle_pitch"],
                ".*_ankle_roll_joint": _MVR_22DOF_DAMPING["ankle_roll"],
                "head_joint": _MVR_22DOF_DAMPING["head"],
                ".*_arm_yaw_joint": _MVR_22DOF_DAMPING["arm_yaw"],
                ".*_elbow_joint": _MVR_22DOF_DAMPING["arm_elbow"],
            },
            effort_limit_sim=36,
            velocity_limit_sim=rpm_to_rads(87.0),
            armature=0.0236,
        ),
    },
)
