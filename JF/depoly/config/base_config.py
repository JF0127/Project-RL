# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BaseRobotConfig:
    """
    机器人配置基类
    职责：定义必要参数，并提供将字典参数按 joint_names 顺序转换为数组的功能。
    """

    # ==========================================
    # 1. 基础信息
    # ==========================================
    name: str = "robot_base"
    xml_path: str = ""  # 必须是绝对路径
    policy_path: str = ""  # 必须是绝对路径

    # ==========================================
    # 2. 仿真与控制
    # ==========================================
    dt: float = 0.001
    decimation: int = 10
    base_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.6)
    command: tuple[float, float, float] = (0.4, 0.0, 0.0)
    log_path: str = "depoly/logs/sim2sim_log.csv"

    lin_vel_scale: float = 1.0
    ang_vel_scale: float = 1.0
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 1.0
    gait_period: float = 0.0

    # 维度自检
    num_observations: int = 0
    num_actions: int = 0

    # ==========================================
    # 3. 关节定义 (核心真理)
    # ==========================================
    # 必须是完整全名 (e.g. "Left_knee_joint")
    # 且顺序必须与 Isaac Lab 训练时的顺序严格一致
    joint_names: list[str] = field(default_factory=list)
    policy_joint_names: list[str] = field(default_factory=list)
    observation_joint_names: list[str] = field(default_factory=list)
    include_last_action_obs: bool = False
    # ==========================================
    # 4. 物理参数 (字典存储)
    # ==========================================
    # 必须使用与 joint_names 一致的全名作为 Key
    default_dof_pos: dict[str, float] = field(default_factory=dict)
    kps: dict[str, float] = field(default_factory=dict)
    kds: dict[str, float] = field(default_factory=dict)

    action_scales: dict[str, float] = field(default_factory=dict)
    clip_actions: dict[str, float] = field(default_factory=dict)
    torque_limits: dict[str, float] = field(default_factory=dict)

    # ==========================================
    # 5. 核心工具
    # ==========================================
    def __post_init__(self):
        if self.xml_path and not os.path.isabs(self.xml_path):
            print(f"[Warning] xml_path 建议使用绝对路径: {self.xml_path}")
        if self.policy_path and not os.path.isabs(self.policy_path):
            print(f"[Warning] policy_path 建议使用绝对路径: {self.policy_path}")
        self.validate_definition()

    def parse_params_to_array(self, param_dict: dict[str, float], default_value: float = 0.0) -> np.ndarray:
        """
        [严格排序]
        依据 self.joint_names 的顺序，从 param_dict 中提取对应的值。
        不进行任何模糊匹配，Key 必须完全相等。
        """
        ordered_list = []
        missing_keys = []

        for name in self.joint_names:
            if name in param_dict:
                ordered_list.append(param_dict[name])
            else:
                # 记录缺失，稍后报错，保证鲁棒性
                ordered_list.append(default_value)
                missing_keys.append(name)

        if missing_keys:
            # 这里抛出异常比打印警告更好，强制用户去 Config 里把名字写对
            raise ValueError(f"[Config Error] 参数字典中缺少以下关节定义: {missing_keys}")

        return np.array(ordered_list, dtype=np.float32)

    def validate_paths(self):
        if not self.xml_path or not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"[Config Error] XML 不存在: {self.xml_path}")
        if not self.policy_path or not os.path.exists(self.policy_path):
            raise FileNotFoundError(f"[Config Error] Policy 不存在: {self.policy_path}")

    @property
    def controlled_joint_names(self) -> list[str]:
        return self.policy_joint_names or self.joint_names

    @property
    def observed_joint_names(self) -> list[str]:
        return self.observation_joint_names or self.joint_names

    @property
    def expected_num_actions(self) -> int:
        return len(self.controlled_joint_names)

    @property
    def expected_num_observations(self) -> int:
        obs_dim = 3 + 3 + 3  # ang vel + gravity + command
        obs_dim += len(self.observed_joint_names) * 2  # joint pos + joint vel
        if self.include_last_action_obs:
            obs_dim += len(self.controlled_joint_names)
        if self.gait_period > 0:
            obs_dim += 2
        return obs_dim

    def validate_definition(self):
        if len(self.base_init_pos) != 3:
            raise ValueError(f"[Config Error] base_init_pos 必须是 3 维: {self.base_init_pos}")
        if len(self.command) != 3:
            raise ValueError(f"[Config Error] command 必须是 3 维: {self.command}")
        if not self.joint_names:
            raise ValueError("[Config Error] joint_names 不能为空")

        joint_name_set = set(self.joint_names)
        unknown_policy_joints = [name for name in self.controlled_joint_names if name not in joint_name_set]
        if unknown_policy_joints:
            raise ValueError(f"[Config Error] policy_joint_names 中存在未知关节: {unknown_policy_joints}")

        unknown_observation_joints = [name for name in self.observed_joint_names if name not in joint_name_set]
        if unknown_observation_joints:
            raise ValueError(f"[Config Error] observation_joint_names 中存在未知关节: {unknown_observation_joints}")

        if self.num_actions != self.expected_num_actions:
            raise ValueError(f"[Config Error] num_actions={self.num_actions}, 但按配置应为 {self.expected_num_actions}")
        if self.num_observations != self.expected_num_observations:
            raise ValueError(
                "[Config Error] "
                f"num_observations={self.num_observations}, 但按配置应为 {self.expected_num_observations}"
            )

        parameter_dicts = {
            "default_dof_pos": self.default_dof_pos,
            "kps": self.kps,
            "kds": self.kds,
            "action_scales": self.action_scales,
            "clip_actions": self.clip_actions,
            "torque_limits": self.torque_limits,
        }
        for name, param_dict in parameter_dicts.items():
            missing = [joint_name for joint_name in self.joint_names if joint_name not in param_dict]
            if missing:
                raise ValueError(f"[Config Error] {name} 缺少以下关节定义: {missing}")
