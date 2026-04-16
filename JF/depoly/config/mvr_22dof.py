# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field

from config.base_config import BaseRobotConfig


@dataclass
class Mvr22DofConfig(BaseRobotConfig):
    """Mvr_22dof 的 MuJoCo sim2sim 配置。"""

    # ==========================================
    # 1. 身份与路径
    # ==========================================
    name: str = "Mvr_22dof"

    xml_path: str = "/home/jhl/projects/Project-RL/robots/mjcf/Mvr_22dof_inertia_new_mujoco.xml"
    policy_path: str = "/home/jhl/projects/Project-RL/logs/rsl_rl/Mvr_22dof_six/2026-04-02_10-26-53/exported/policy.pt"

    # ==========================================
    # 2. 仿真参数
    # ==========================================
    dt: float = 0.001  # 物理步长
    decimation: int = 5  # 控制分频
    base_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.65)
    command: tuple[float, float, float] = (0.2, 0.0, 0.0)
    log_path: str = "depoly/logs/sim2sim_log.csv"
    lin_vel_scale: float = 1.0
    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    gait_period: float = 0.8

    # Isaac Lab 当前主任务是 Mvr_22dof_six
    num_observations: int = 55
    num_actions: int = 6

    # ==========================================
    # 3. 关节定义 (The Truth)
    # ==========================================
    # 必须与 Isaac Lab 训练顺序严格一致
    joint_names: list[str] = field(
        default_factory=lambda: [
            # Left Leg
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            # Right Leg
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            # Torso & Head
            "waist_joint",
            "head_joint",
            # Left Arm
            "left_arm_pitch_joint",
            "left_arm_roll_joint",
            "left_arm_yaw_joint",
            "left_elbow_joint",
            # Right Arm
            "right_arm_pitch_joint",
            "right_arm_roll_joint",
            "right_arm_yaw_joint",
            "right_elbow_joint",
        ]
    )
    policy_joint_names: list[str] = field(
        default_factory=lambda: [
            "left_hip_pitch_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "right_hip_pitch_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
        ]
    )
    observation_joint_names: list[str] = field(
        default_factory=lambda: [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_joint",
            "head_joint",
            "left_arm_pitch_joint",
            "left_arm_roll_joint",
            "left_arm_yaw_joint",
            "left_elbow_joint",
            "right_arm_pitch_joint",
            "right_arm_roll_joint",
            "right_arm_yaw_joint",
            "right_elbow_joint",
        ]
    )
    include_last_action_obs: bool = False

    # ==========================================
    # 4. 物理参数 (全名显式定义)
    # ==========================================

    # --- A. 初始位置 (Nominal Pose) ---
    # 严格保留了你之前代码中左右不对称的设定
    default_dof_pos: dict[str, float] = field(
        default_factory=lambda: {
            # Left Leg
            "left_hip_pitch_joint": -0.25,
            "left_hip_roll_joint": -0.03,
            "left_hip_yaw_joint": -0.01,
            "left_knee_joint": -0.50,
            "left_ankle_pitch_joint": 0.23,
            "left_ankle_roll_joint": -0.01,
            # Right Leg (注意：部分符号翻转，部分未翻转，源自你的实际调试)
            "right_hip_pitch_joint": 0.25,
            "right_hip_roll_joint": 0.03,
            "right_hip_yaw_joint": -0.01,  # Yaw 保持负号
            "right_knee_joint": 0.50,
            "right_ankle_pitch_joint": -0.23,
            "right_ankle_roll_joint": 0.01,
            # Torso
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
    )

    # --- B. 刚度 (Kp) ---
    kps: dict[str, float] = field(
        default_factory=lambda: {
            # Hip: Pitch=200, Roll=150, Yaw=100
            "left_hip_pitch_joint": 400.0,
            "left_hip_roll_joint": 150.0,
            "left_hip_yaw_joint": 150.0,
            "right_hip_pitch_joint": 400.0,
            "right_hip_roll_joint": 150.0,
            "right_hip_yaw_joint": 150.0,
            # Knee: 400
            "left_knee_joint": 200.0,
            "right_knee_joint": 200.0,
            # Ankle: Pitch=40, Roll=30
            "left_ankle_pitch_joint": 200.0,
            "left_ankle_roll_joint": 100.0,
            "right_ankle_pitch_joint": 200.0,
            "right_ankle_roll_joint": 100.0,
            # Torso: Waist=100, Head=20
            "waist_joint": 10.0,
            "head_joint": 10.0,
            # Arm: Pitch=40, Roll=40, Yaw=20, Elbow=20
            "left_arm_pitch_joint": 5.0,
            "left_arm_roll_joint": 5.0,
            "left_arm_yaw_joint": 5.0,
            "left_elbow_joint": 5.0,
            "right_arm_pitch_joint": 5.0,
            "right_arm_roll_joint": 5.0,
            "right_arm_yaw_joint": 5.0,
            "right_elbow_joint": 5.0,
        }
    )

    # --- C. 阻尼 (Kd) ---
    kds: dict[str, float] = field(
        default_factory=lambda: {
            # Hip: Pitch=10, Roll=5, Yaw=5
            "left_hip_pitch_joint": 10.0,
            "left_hip_roll_joint": 10.0,
            "left_hip_yaw_joint": 10.0,
            "right_hip_pitch_joint": 10.0,
            "right_hip_roll_joint": 10.0,
            "right_hip_yaw_joint": 10.0,
            # Knee: 10
            "left_knee_joint": 10.0,
            "right_knee_joint": 10.0,
            # Ankle: Pitch=2.5, Roll=2.0
            "left_ankle_pitch_joint": 0.1,
            "left_ankle_roll_joint": 0.1,
            "right_ankle_pitch_joint": 0.1,
            "right_ankle_roll_joint": 0.1,
            # Torso: Waist=5, Head=1
            "waist_joint": 5.0,
            "head_joint": 5.0,
            # Arm: Pitch=2, Roll=2, Yaw=1, Elbow=1
            "left_arm_pitch_joint": 5.0,
            "left_arm_roll_joint": 5.0,
            "left_arm_yaw_joint": 5.0,
            "left_elbow_joint": 5.0,
            "right_arm_pitch_joint": 5.0,
            "right_arm_roll_joint": 5.0,
            "right_arm_yaw_joint": 5.0,
            "right_elbow_joint": 5.0,
        }
    )

    # --- D. 动作缩放 (Action Scale) ---
    action_scales: dict[str, float] = field(
        default_factory=lambda: {
            name: 0.25
            for name in [
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_joint",
                "head_joint",
                "left_arm_pitch_joint",
                "left_arm_roll_joint",
                "left_arm_yaw_joint",
                "left_elbow_joint",
                "right_arm_pitch_joint",
                "right_arm_roll_joint",
                "right_arm_yaw_joint",
                "right_elbow_joint",
            ]
        }
    )

    # --- E. 扭矩限制 (Torque Limits) ---
    torque_limits: dict[str, float] = field(
        default_factory=lambda: {
            # A10020 (150 Nm)
            "left_hip_pitch_joint": 150,
            "right_hip_pitch_joint": 150,
            "left_knee_joint": 150,
            "right_knee_joint": 150,
            # A8112 (90 Nm)
            "left_hip_roll_joint": 90,
            "right_hip_roll_joint": 90,
            "left_hip_yaw_joint": 90,
            "right_hip_yaw_joint": 90,
            "waist_joint": 90,
            # A6408 (60 Nm)
            "left_arm_pitch_joint": 60,
            "right_arm_pitch_joint": 60,
            "left_arm_roll_joint": 60,
            "right_arm_roll_joint": 60,
            # A4310 (36 Nm)
            "left_ankle_pitch_joint": 36,
            "right_ankle_pitch_joint": 36,
            "left_ankle_roll_joint": 36,
            "right_ankle_roll_joint": 36,
            "head_joint": 36,
            "left_arm_yaw_joint": 36,
            "right_arm_yaw_joint": 36,
            "left_elbow_joint": 36,
            "right_elbow_joint": 36,
        }
    )

    # --- F. 动作截断 ---
    clip_actions: dict[str, float] = field(
        default_factory=lambda: {
            name: 100.0
            for name in [
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_joint",
                "head_joint",
                "left_arm_pitch_joint",
                "left_arm_roll_joint",
                "left_arm_yaw_joint",
                "left_elbow_joint",
                "right_arm_pitch_joint",
                "right_arm_roll_joint",
                "right_arm_yaw_joint",
                "right_elbow_joint",
            ]
        }
    )
