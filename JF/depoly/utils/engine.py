# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause


import mujoco
import numpy as np


class MujocoEngine:
    """MuJoCo 硬件抽象层。"""

    def __init__(
        self,
        xml_path: str,
        dt: float,
        joint_names: list[str],
    ):
        print(f"[Engine] Loading MuJoCo model from: {xml_path}")
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise ValueError(f"Failed to load MuJoCo XML: {e}")

        # 设置物理步长
        self.model.opt.timestep = dt

        # ---------------------------------------------------------
        # 建立索引映射
        # ---------------------------------------------------------
        self.joint_names = joint_names
        self.joint_indices = []
        self.dof_vel_indices = []
        self.actuator_indices = []

        print(f"[Engine] Building index mapping for {len(joint_names)} joints...")
        for name in joint_names:
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id == -1:
                raise ValueError(f"[Engine Error] XML 中找不到关节: '{name}'")

            self.joint_indices.append(self.model.jnt_qposadr[j_id])
            self.dof_vel_indices.append(self.model.jnt_dofadr[j_id])

            a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if a_id == -1:
                raise ValueError(f"[Engine Error] XML 中找不到执行器: '{name}'")
            self.actuator_indices.append(a_id)

        self.joint_indices = np.array(self.joint_indices, dtype=np.int32)
        self.dof_vel_indices = np.array(self.dof_vel_indices, dtype=np.int32)
        self.actuator_indices = np.array(self.actuator_indices, dtype=np.int32)

        # 传感器
        self.imu_quat_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation")
        self.imu_gyro_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")

    def reset(self, initial_dof_pos: np.ndarray, base_pos=None):
        mujoco.mj_resetData(self.model, self.data)
        if base_pos is None:
            base_pos = np.array([0.0, 0.0, 0.6])
        if len(base_pos) != 3:
            raise ValueError(f"[Engine Error] base_pos 必须是 3 维: {base_pos}")
        if len(initial_dof_pos) != len(self.joint_names):
            raise ValueError(
                f"[Engine Error] initial_dof_pos 维度错误: got {len(initial_dof_pos)}, expected {len(self.joint_names)}"
            )

        self.data.qpos[0:3] = base_pos
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.joint_indices] = initial_dof_pos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_sensors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.imu_quat_idx != -1:
            q_adr = self.model.sensor_adr[self.imu_quat_idx]
            g_adr = self.model.sensor_adr[self.imu_gyro_idx]
            base_quat = self.data.sensordata[q_adr : q_adr + 4]
            base_ang_vel = self.data.sensordata[g_adr : g_adr + 3]
        else:
            base_quat = self.data.qpos[3:7].copy()
            base_ang_vel = self.data.qvel[3:6].copy()

        dof_pos = self.data.qpos[self.joint_indices]
        dof_vel = self.data.qvel[self.dof_vel_indices]
        return base_quat.copy(), base_ang_vel.copy(), dof_pos.copy(), dof_vel.copy()

    def step(
        self,
        target_dof_pos: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        torque_limits: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        expected_dim = len(self.joint_names)
        for name, value in {
            "target_dof_pos": target_dof_pos,
            "kp": kp,
            "kd": kd,
            "torque_limits": torque_limits,
        }.items():
            if len(value) != expected_dim:
                raise ValueError(f"[Engine Error] {name} 维度错误: got {len(value)}, expected {expected_dim}")

        # 计算 PD
        current_pos = self.data.qpos[self.joint_indices]
        current_vel = self.data.qvel[self.dof_vel_indices]
        kp_torque = kp * (target_dof_pos - current_pos)
        kd_torque = -kd * current_vel
        raw_torque = kp_torque + kd_torque
        applied_torque = np.clip(raw_torque, -torque_limits, torque_limits)

        self.data.ctrl[self.actuator_indices] = applied_torque

        # 物理步进
        mujoco.mj_step(self.model, self.data)
        return applied_torque, kp_torque, kd_torque
