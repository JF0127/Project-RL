# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

import os
import time
from datetime import datetime

import mujoco
import mujoco.viewer
import numpy as np
import torch
import utils.observation as obs_utils
from config.base_config import BaseRobotConfig
from utils.engine import MujocoEngine
from utils.logger import DataLogger


class Sim2SimRunner:
    """MuJoCo sim2sim 运行器，支持所有 BaseRobotConfig 子类。"""

    def __init__(
        self,
        config: BaseRobotConfig,
        record: bool = False,
        record_path: str | None = None,
        record_camera: str | None = None,
        record_width: int = 1280,
        record_height: int = 720,
    ):
        self.cfg = config
        self.cfg.validate_paths()
        self.device = torch.device("cpu")
        self.step_counter = 0

        self.record = record
        self.record_camera = record_camera
        self.record_width = record_width
        self.record_height = record_height
        if record_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_dir = os.path.join(os.path.dirname(__file__), "..", "videos")
            os.makedirs(video_dir, exist_ok=True)
            self.record_path = os.path.join(video_dir, f"sim2sim_{ts}.mp4")
        else:
            self.record_path = record_path
        self._frames: list[np.ndarray] = []

        self._init_runtime()

    def _init_runtime(self):
        print(f"[Runner] Initializing MuJoCo engine for {self.cfg.name}...")
        self.engine = MujocoEngine(
            xml_path=self.cfg.xml_path,
            dt=self.cfg.dt,
            joint_names=self.cfg.joint_names,
        )
        self.logger = DataLogger(self.cfg.log_path)
        self._init_policy()
        self._init_mappings()
        self._init_buffers()

    def _init_policy(self):
        print(f"[Sim2Sim] Loading policy from: {self.cfg.policy_path}")
        try:
            self.policy = torch.jit.load(self.cfg.policy_path, map_location=self.device)
            self.policy.eval()
        except Exception as e:
            raise RuntimeError(f"无法加载模型文件: {e}")

    def _init_mappings(self):
        policy_names = self.cfg.controlled_joint_names
        obs_names = self.cfg.observed_joint_names

        self.policy_indices = [self.cfg.joint_names.index(name) for name in policy_names]
        self.obs_indices = [self.cfg.joint_names.index(name) for name in obs_names]

        self.policy_indices_tensor = torch.tensor(self.policy_indices, dtype=torch.long, device=self.device)
        self.obs_indices_tensor = torch.tensor(self.obs_indices, dtype=torch.long, device=self.device)

        print(f"[Mapping] Policy controls {len(self.policy_indices)} / {len(self.cfg.joint_names)} joints.")
        print(f"[Mapping] Policy observes {len(self.obs_indices)} joints.")

    def _init_buffers(self):
        self.kp_base_np = self.cfg.parse_params_to_array(self.cfg.kps)
        self.kd_base_np = self.cfg.parse_params_to_array(self.cfg.kds)
        self.torque_limits_np = self.cfg.parse_params_to_array(self.cfg.torque_limits)
        self.default_pos_np = self.cfg.parse_params_to_array(self.cfg.default_dof_pos)

        full_default_pos_tensor = torch.tensor(self.default_pos_np, dtype=torch.float32, device=self.device)
        self.policy_default_pos_tensor = full_default_pos_tensor[self.policy_indices_tensor].unsqueeze(0)
        self.obs_default_pos_tensor = full_default_pos_tensor[self.obs_indices_tensor].unsqueeze(0)
        self.full_target_pos_tensor = full_default_pos_tensor.clone().unsqueeze(0)

        full_action_scales = torch.tensor(
            self.cfg.parse_params_to_array(self.cfg.action_scales),
            dtype=torch.float32,
            device=self.device,
        )
        self.policy_action_scale = full_action_scales[self.policy_indices_tensor].unsqueeze(0)

        full_clip = torch.tensor(
            self.cfg.parse_params_to_array(self.cfg.clip_actions),
            dtype=torch.float32,
            device=self.device,
        )
        self.policy_clip = full_clip[self.policy_indices_tensor].unsqueeze(0)

        self.last_action = torch.zeros((1, self.cfg.expected_num_actions), dtype=torch.float32, device=self.device)
        self.command = torch.tensor(self.cfg.command, dtype=torch.float32, device=self.device).unsqueeze(0)
        self._validate_runtime_layout()

    def _validate_runtime_layout(self):
        if self.policy_action_scale.shape[-1] != self.cfg.num_actions:
            raise ValueError(
                "[Sim2Sim Error] policy_action_scale 维度与 num_actions 不一致: "
                f"{self.policy_action_scale.shape[-1]} vs {self.cfg.num_actions}"
            )
        if self.policy_clip.shape[-1] != self.cfg.num_actions:
            raise ValueError(
                "[Sim2Sim Error] policy_clip 维度与 num_actions 不一致: "
                f"{self.policy_clip.shape[-1]} vs {self.cfg.num_actions}"
            )

    def run(self):
        print("\n==================================================")
        print("Starting Runner: Sim2SimRunner")
        print(f"Robot: {self.cfg.name}")
        if self.record:
            print(f"Recording: {self.record_path}")
        print("Press Ctrl+C to stop.")
        print("==================================================\n")

        self.reset()

        renderer = None
        if self.record:
            renderer = mujoco.Renderer(self.engine.model, height=self.record_height, width=self.record_width)

        try:
            with mujoco.viewer.launch_passive(self.engine.model, self.engine.data) as viewer:
                self._run_loop(viewer, renderer)
        except KeyboardInterrupt:
            print("\n[Runner] Stopped by user (Ctrl+C).")
        finally:
            if renderer is not None:
                renderer.close()
            self.logger.save()
            if self.record and self._frames:
                self._save_video()
            print("[Runner] Cleanup done.")

    def reset(self):
        init_pos = self.cfg.parse_params_to_array(self.cfg.default_dof_pos)
        base_pos = np.array(self.cfg.base_init_pos, dtype=np.float32)

        self.engine.reset(initial_dof_pos=init_pos, base_pos=base_pos)

        self.last_action.fill_(0.0)
        self.step_counter = 0

        mujoco.mj_forward(self.engine.model, self.engine.data)
        print("[Sim2Sim] Reset complete.")

    def sync_viewer(self, viewer, step_start_time):
        viewer.sync()
        step_dt = self.cfg.decimation * self.cfg.dt
        elapsed = time.time() - step_start_time
        if elapsed < step_dt:
            time.sleep(step_dt - elapsed)

    def _run_loop(self, viewer, renderer=None):
        print("\n=== Sim2Sim Inference Started ===")

        while viewer.is_running():
            step_start = time.time()
            sim_time = self.step_counter * self.cfg.decimation * self.cfg.dt

            obs, phase = self._build_observation(sim_time)
            raw_actions, scaled_action = self._infer_actions(obs)
            target_pos_np = self._build_target_positions(scaled_action)
            for _ in range(self.cfg.decimation):
                self.engine.step(
                    target_dof_pos=target_pos_np,
                    kp=self.kp_base_np,
                    kd=self.kd_base_np,
                    torque_limits=self.torque_limits_np,
                )

            if renderer is not None:
                camera = self.record_camera if self.record_camera is not None else -1
                renderer.update_scene(self.engine.data, camera=camera)
                self._frames.append(renderer.render().copy())

            self._log_step(sim_time, obs, raw_actions, scaled_action, phase)

            self.step_counter += 1
            self.sync_viewer(viewer, step_start)

            if self.step_counter % 50 == 0:
                record_info = f" | Frames: {len(self._frames)}" if self.record else ""
                print(
                    f"\rTime: {sim_time:.2f}s | Phase: {phase:.2f}{record_info}",
                    end="",
                )

    def _save_video(self):
        try:
            import imageio
        except ImportError:
            print("[Record] 缺少 imageio，请运行: pip install imageio imageio-ffmpeg")
            return

        step_dt = self.cfg.decimation * self.cfg.dt
        fps = max(1, round(1.0 / step_dt))
        print(f"\n[Record] 正在保存视频 ({len(self._frames)} 帧, {fps} fps) -> {self.record_path}")
        os.makedirs(os.path.dirname(os.path.abspath(self.record_path)), exist_ok=True)
        imageio.mimwrite(self.record_path, self._frames, fps=fps)
        print(f"[Record] 视频已保存: {self.record_path}")

    def _build_observation(self, sim_time: float) -> tuple[torch.Tensor, float]:
        base_quat_np, base_ang_vel_np, dof_pos_np, dof_vel_np = self.engine.get_sensors()
        base_quat = torch.from_numpy(base_quat_np).float().to(self.device).unsqueeze(0)
        base_ang_vel = torch.from_numpy(base_ang_vel_np).float().to(self.device).unsqueeze(0)
        full_dof_pos = torch.from_numpy(dof_pos_np).float().to(self.device).unsqueeze(0)
        full_dof_vel = torch.from_numpy(dof_vel_np).float().to(self.device).unsqueeze(0)

        obs_dof_pos = full_dof_pos[:, self.obs_indices_tensor]
        obs_dof_vel = full_dof_vel[:, self.obs_indices_tensor]

        obs_list = [
            obs_utils.get_obs_ang_vel(base_ang_vel, self.cfg.ang_vel_scale),
            obs_utils.get_obs_gravity(base_quat, self.device),
            obs_utils.get_obs_cmd(self.command, self.cfg.lin_vel_scale),
            obs_utils.get_obs_dof_pos(obs_dof_pos, self.obs_default_pos_tensor, self.cfg.dof_pos_scale),
            obs_utils.get_obs_dof_vel(obs_dof_vel, self.cfg.dof_vel_scale),
        ]

        if self.cfg.include_last_action_obs:
            obs_list.append(obs_utils.get_obs_last_action(self.last_action))

        phase = 0.0
        if self.cfg.gait_period > 0:
            phase = sim_time % self.cfg.gait_period
            obs_list.append(obs_utils.get_obs_gait_phase(sim_time, self.cfg.gait_period, self.device))

        obs = torch.cat(obs_list, dim=-1)
        if obs.shape[-1] != self.cfg.num_observations:
            raise ValueError(
                f"[Sim2Sim Error] 观测维度不匹配: got {obs.shape[-1]}, expected {self.cfg.num_observations}"
            )
        return obs, phase

    def _infer_actions(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            raw_actions = self.policy(obs)
        if raw_actions.shape[-1] != self.cfg.num_actions:
            raise ValueError(
                f"[Sim2Sim Error] 动作维度不匹配: got {raw_actions.shape[-1]}, expected {self.cfg.num_actions}"
            )

        clipped_action = torch.clamp(raw_actions, -self.policy_clip, self.policy_clip)
        scaled_action = clipped_action * self.policy_action_scale
        self.last_action = raw_actions.clone()
        return raw_actions, scaled_action

    def _build_target_positions(self, scaled_action: torch.Tensor) -> np.ndarray:
        full_target = self.full_target_pos_tensor.clone()
        policy_target = self.policy_default_pos_tensor + scaled_action
        full_target[:, self.policy_indices_tensor] = policy_target
        return full_target.squeeze(0).cpu().numpy()

    def _log_step(
        self,
        sim_time: float,
        obs: torch.Tensor,
        raw_actions: torch.Tensor,
        scaled_action: torch.Tensor,
        phase: float,
    ):
        self.logger.log({
            "time": sim_time,
            "obs": obs,
            "act_raw": raw_actions,
            "act_scaled": scaled_action,
            "phase": phase,
        })
