# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import os
import time

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import RL.tasks  # noqa: F401
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    # ========================== 设置相机跟随逻辑 ==========================
    if args_cli.video:
        # 注意：这里不需要再从 ViewerCfg 导入 OriginType 了，直接用字符串即可

        # 1. 设置相机跟随对象为 Asset (机器人)
        env_cfg.viewer.origin_type = "asset_root"

        # 2. 指定要跟随的 Asset 名称 (必须与你场景中定义的机器人名称一致)
        # 你的代码后面用了 isaac_env.scene["robot"]，所以这里填 "robot"
        env_cfg.viewer.asset_name = "robot"

        # 3. 指定观测哪个环境索引 (通常录制视频时我们只看第 0 个环境)
        env_cfg.viewer.env_index = 0

        # 4. 设置相机相对于机器人的位置 (eye) 和 观察点 (lookat)
        # 例如：从后上方观察 (x=-2, z=2 表示在机器人后方 2 米，上方 2 米)
        env_cfg.viewer.eye = (-2.0, 0.0, 2.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
    # ====================================================================
    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # ========================== Debug Joint Order ==========================
    print("\n[INFO] >>> Checking Joint Order for Sim2Sim Alignment <<<")

    # 1. Get the underlying ManagerBasedRLEnv
    # Depending on wrappers, you might need to go deeper.
    # Usually env.unwrapped is enough for ManagerBasedRLEnv.
    isaac_env = env.unwrapped

    # 2. Access the Robot Asset
    # Note: 'robot' is the default name in most configs.
    # If your config uses a different name (e.g. 'anymal'), change it here.
    robot_asset = isaac_env.scene["robot"]

    # 3. Print all available joints in the USD (Physics simulation order)
    print(f"Total Joints in Asset: {robot_asset.num_joints}")
    print(f"All Joint Names (Sim Order): {robot_asset.joint_names}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if version.parse(installed_version) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        policy_nn = None
    else:
        # version 2.3 onwards
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            # version 2.2 and below
            policy_nn = runner.alg.actor_critic

        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # ========================== CSV 初始化 ==========================
    csv_filename = "data.csv"
    csv_path = os.path.join(export_model_dir, csv_filename)
    print(f"[INFO] Recording Obs and Actions to: {csv_path}")

    csv_file = open(csv_path, mode="w", newline="")  # noqa: SIM115
    csv_writer = csv.writer(csv_file)  # noqa: SIM115
    header_written = False
    # ==============================================================

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # ========================== 数据记录逻辑 ==========================
            # 1. 解析 OBS
            if isinstance(obs, dict) or hasattr(obs, "keys"):
                # 如果是 dict，尝试取 'policy' 键 (通常是 Critic 或 Actor 观察)
                obs_data = obs["policy"][0].cpu().numpy().tolist()
            else:
                obs_data = obs[0].cpu().numpy().tolist()

            # 2. 解析 ACTIONS
            act_data = actions[0].cpu().numpy().tolist()

            # 3. 写入 CSV
            if not header_written:
                header = [f"Obs_{i}" for i in range(len(obs_data))] + [f"Act_{i}" for i in range(len(act_data))]
                csv_writer.writerow(header)
                header_written = True

            csv_writer.writerow(obs_data + act_data)
            # ==============================================================

            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
