# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field

from config.base_config import BaseRobotConfig


@dataclass
class Mvr10DofConfig(BaseRobotConfig):
    """Mvr_10dof 的 MuJoCo sim2sim 配置。"""

    # ==========================================
    # 1. 身份与路径
    # ==========================================
    name: str = "Mvr_10dof"

    xml_path: str = "/home/jhl/projects/Project-RL/robots/Mvr_10dof/mjcf/Mvr_10dof_no_col_inf.xml"
    policy_path: str = "/home/jhl/projects/Project-RL/logs/rsl_rl/Mvr_10dof/2026-04-08_15-17-57/exported/policy.pt"

    # ==========================================
    # 2. 仿真参数
    # ==========================================
    dt: float = 0.001        # 物理步长 (1000Hz)
    decimation: int = 10     # 控制分频 (100Hz), 与训练一致
    base_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.60)
    command: tuple[float, float, float] = (0.4, 0.0, 0.0)
    log_path: str = "depoly/logs/sim2sim_log.csv"

    # 观测缩放系数 — 与 walk_env_cfg.py 中 ObsTerm scale 一致
    lin_vel_scale: float = 1.0
    ang_vel_scale: float = 0.2   # base_ang_vel scale=0.2
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05  # joint_vel_rel scale=0.05

    gait_period: float = 1.6     # gait_phase period=1.6

    # 维度自检
    # obs = ang_vel(3) + gravity(3) + cmd(3) + joint_pos(10) + joint_vel(10) + last_action(10) + gait_phase(2) = 41
    num_observations: int = 41
    num_actions: int = 10

    # ==========================================
    # 3. 关节定义 (The Truth)
    # ==========================================
    # 顺序与 walk_env_cfg.py 中 DESIRED_JOINT_ORDER 严格一致
    joint_names: list[str] = field(
        default_factory=lambda: [
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
        ]
    )
    # 10dof 所有关节均被策略控制，无掩码
    policy_joint_names: list[str] = field(
        default_factory=lambda: [
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
        ]
    )
    observation_joint_names: list[str] = field(
        default_factory=lambda: [
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
        ]
    )
    # 训练时观测包含 last_action
    include_last_action_obs: bool = True

    # ==========================================
    # 4. 物理参数 (全名显式定义)
    # ==========================================

    # --- A. 初始位置 (与 _MVR_10DOF_DEFAULT_JOINT_POS 一致) ---
    default_dof_pos: dict[str, float] = field(
        default_factory=lambda: {
            "left_hip_roll_joint": 0.00,
            "left_hip_yaw_joint": 0.00,
            "left_hip_pitch_joint": 0.25,
            "left_knee_joint": -0.60,
            "left_ankle_pitch_joint": 0.35,
            "right_hip_roll_joint": 0.00,
            "right_hip_yaw_joint": 0.00,
            "right_hip_pitch_joint": -0.25,
            "right_knee_joint": 0.60,
            "right_ankle_pitch_joint": 0.35,
        }
    )

    # --- B. 刚度 (Kp) — 与 _MVR_10DOF_STIFFNESS 一致 ---
    kps: dict[str, float] = field(
        default_factory=lambda: {
            "left_hip_roll_joint": 40.0,
            "left_hip_yaw_joint": 40.0,
            "left_hip_pitch_joint": 150.0,
            "left_knee_joint": 150.0,
            "left_ankle_pitch_joint": 80.0,
            "right_hip_roll_joint": 40.0,
            "right_hip_yaw_joint": 40.0,
            "right_hip_pitch_joint": 150.0,
            "right_knee_joint": 150.0,
            "right_ankle_pitch_joint": 80.0,
        }
    )

    # --- C. 阻尼 (Kd) — 与 _MVR_10DOF_DAMPING 一致 ---
    kds: dict[str, float] = field(
        default_factory=lambda: {
            "left_hip_roll_joint": 2.0,
            "left_hip_yaw_joint": 2.0,
            "left_hip_pitch_joint": 8.0,
            "left_knee_joint": 8.0,
            "left_ankle_pitch_joint": 3.0,
            "right_hip_roll_joint": 2.0,
            "right_hip_yaw_joint": 2.0,
            "right_hip_pitch_joint": 8.0,
            "right_knee_joint": 8.0,
            "right_ankle_pitch_joint": 3.0,
        }
    )

    # --- D. 动作缩放 (与训练 scale=0.25 一致) ---
    action_scales: dict[str, float] = field(
        default_factory=lambda: {name: 0.25 for name in [
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
        ]}
    )

    # --- E. 扭矩限制 (来自 Isaac Lab effort_limit_sim) ---
    torque_limits: dict[str, float] = field(
        default_factory=lambda: {
            # A4310 (36 Nm): hip_roll, hip_yaw, ankle_pitch
            "left_hip_roll_joint": 36.0,
            "left_hip_yaw_joint": 36.0,
            "left_ankle_pitch_joint": 36.0,
            "right_hip_roll_joint": 36.0,
            "right_hip_yaw_joint": 36.0,
            "right_ankle_pitch_joint": 36.0,
            # A8112/A10020 (90 Nm): hip_pitch, knee
            "left_hip_pitch_joint": 90.0,
            "left_knee_joint": 90.0,
            "right_hip_pitch_joint": 90.0,
            "right_knee_joint": 90.0,
        }
    )

    # --- F. 动作截断 ---
    clip_actions: dict[str, float] = field(
        default_factory=lambda: {name: 100.0 for name in [
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
        ]}
    )
