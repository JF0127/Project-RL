# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))

from config.mvr_22dof import Mvr22DofConfig


def get_latest_log(log_dir: str) -> str:
    log_path = Path(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"找不到日志目录: {log_dir}")

    csv_files = sorted(log_path.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"日志目录下没有 CSV 文件: {log_dir}")
    return str(csv_files[0])


def get_joint_index(joint_name: str, joint_names: list[str]) -> int:
    try:
        return joint_names.index(joint_name)
    except ValueError as exc:
        raise ValueError(f"未知关节名: {joint_name}") from exc


def get_action_index(joint_name: str, controlled_joint_names: list[str]) -> int:
    try:
        return controlled_joint_names.index(joint_name)
    except ValueError as exc:
        raise ValueError(f"关节 {joint_name} 不在策略控制关节中: {controlled_joint_names}") from exc


def resolve_obs_joint_column(joint_name: str, cfg: Mvr22DofConfig) -> int:
    obs_joint_index = get_joint_index(joint_name, cfg.observed_joint_names)
    # obs layout:
    # 0..2 ang_vel, 3..5 gravity, 6..8 cmd
    # 9..(9+n-1) dof_pos, then dof_vel, then optional last_action, then phase
    return 9 + obs_joint_index


def validate_columns(df: pd.DataFrame, columns: list[str]):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"日志缺少必要列: {missing}")


def build_joint_column_map(joint_name: str, cfg: Mvr22DofConfig) -> dict[str, str]:
    obs_column = f"obs_{resolve_obs_joint_column(joint_name, cfg)}"
    action_index = get_action_index(joint_name, cfg.controlled_joint_names)
    act_raw_column = f"act_raw_{action_index}"
    act_scaled_column = f"act_scaled_{action_index}"
    return {
        "obs": obs_column,
        "act_raw": act_raw_column,
        "act_scaled": act_scaled_column,
    }


def plot_joint_log(
    csv_file: str,
    cfg: Mvr22DofConfig,
    joint_name: str,
    save_fig: bool = False,
    output_dir: str = "depoly/logs/plots",
):
    df = pd.read_csv(csv_file)

    joint_columns = build_joint_column_map(joint_name, cfg)
    obs_column = joint_columns["obs"]
    act_raw_column = joint_columns["act_raw"]
    act_scaled_column = joint_columns["act_scaled"]
    required_columns = ["time", obs_column, act_raw_column, act_scaled_column, "phase"]
    validate_columns(df, required_columns)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    ax1, ax2, ax3, ax4 = axes

    time = df["time"]
    obs = df[obs_column]
    act_raw = df[act_raw_column]
    act_scaled = df[act_scaled_column]
    phase = df["phase"]

    ax1.plot(time, obs, color="blue", linewidth=1.4, label=f"{joint_name} obs")
    ax1.set_ylabel("Obs")
    ax1.set_title(f"{joint_name} Observation")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    ax2.plot(time, act_raw, color="orange", linewidth=1.4, label=f"{joint_name} act_raw")
    ax2.set_ylabel("Raw")
    ax2.set_title(f"{joint_name} Raw Action")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    ax3.plot(time, act_scaled, color="green", linewidth=1.4, label=f"{joint_name} act_scaled")
    ax3.set_ylabel("Scaled")
    ax3.set_title(f"{joint_name} Scaled Action")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right")

    ax4.plot(time, obs, color="blue", linewidth=1.1, label="obs")
    ax4.plot(time, act_raw, color="orange", linewidth=1.1, label="act_raw")
    ax4.plot(time, act_scaled, color="green", linewidth=1.1, label="act_scaled")
    ax4.plot(time, phase, color="red", linewidth=1.1, label="phase")
    ax4.set_ylabel("Value")
    ax4.set_xlabel("Time (s)")
    ax4.set_title(f"{joint_name} Overlay")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right")

    plt.suptitle(f"Sim2Sim Log Analysis: {joint_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(csv_file).stem
        save_path = os.path.join(output_dir, f"{base_name}_{joint_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图像已保存: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_multi_joint_actions(
    csv_file: str,
    cfg: Mvr22DofConfig,
    joint_names: list[str],
    save_fig: bool = False,
    output_dir: str = "depoly/logs/plots",
):
    df = pd.read_csv(csv_file)
    required_columns = ["time", "phase"]
    for joint_name in joint_names:
        required_columns.append(build_joint_column_map(joint_name, cfg)["act_scaled"])
    validate_columns(df, required_columns)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1, ax2 = axes
    time = df["time"]

    for joint_name in joint_names:
        joint_columns = build_joint_column_map(joint_name, cfg)
        ax1.plot(time, df[joint_columns["act_scaled"]], linewidth=1.3, label=joint_name)

    ax1.set_ylabel("Scaled Action")
    ax1.set_title("Multi-Joint Scaled Actions")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    ax2.plot(time, df["phase"], color="red", linewidth=1.3, label="phase")
    ax2.set_ylabel("Phase")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Phase Alignment")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.suptitle("Sim2Sim Multi-Joint Action Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(csv_file).stem
        save_path = os.path.join(output_dir, f"{base_name}_multi_joint_actions.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图像已保存: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_left_right_comparison(
    csv_file: str,
    cfg: Mvr22DofConfig,
    left_joint: str,
    right_joint: str,
    save_fig: bool = False,
    output_dir: str = "depoly/logs/plots",
):
    df = pd.read_csv(csv_file)
    left_columns = build_joint_column_map(left_joint, cfg)
    right_columns = build_joint_column_map(right_joint, cfg)
    required_columns = [
        "time",
        "phase",
        left_columns["obs"],
        left_columns["act_scaled"],
        right_columns["obs"],
        right_columns["act_scaled"],
    ]
    validate_columns(df, required_columns)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    ax1, ax2, ax3 = axes
    time = df["time"]

    ax1.plot(time, df[left_columns["obs"]], linewidth=1.3, label=left_joint)
    ax1.plot(time, df[right_columns["obs"]], linewidth=1.3, label=right_joint)
    ax1.set_ylabel("Obs")
    ax1.set_title("Left vs Right Observation")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    ax2.plot(time, df[left_columns["act_scaled"]], linewidth=1.3, label=left_joint)
    ax2.plot(time, df[right_columns["act_scaled"]], linewidth=1.3, label=right_joint)
    ax2.set_ylabel("Scaled Action")
    ax2.set_title("Left vs Right Scaled Action")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    ax3.plot(time, df["phase"], color="red", linewidth=1.3, label="phase")
    ax3.set_ylabel("Phase")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Phase")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right")

    plt.suptitle(f"Left/Right Comparison: {left_joint} vs {right_joint}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(csv_file).stem
        save_path = os.path.join(output_dir, f"{base_name}_{left_joint}_vs_{right_joint}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图像已保存: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_phase_overview(
    csv_file: str,
    save_fig: bool = False,
    output_dir: str = "depoly/logs/plots",
):
    df = pd.read_csv(csv_file)
    validate_columns(df, ["time", "phase"])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["time"], df["phase"], color="red", linewidth=1.4, label="phase")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phase")
    ax.set_title("Phase Overview")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(csv_file).stem
        save_path = os.path.join(output_dir, f"{base_name}_phase.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图像已保存: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize latest deploy sim2sim log by joint name")
    parser.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="指定日志文件；默认自动读取 depoly/logs 下最新 CSV",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="depoly/logs",
        help="自动查找最新日志时使用的目录",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "multi", "compare", "phase"],
        help="可视化模式: single=单关节, multi=多关节, compare=左右对比, phase=仅相位",
    )
    parser.add_argument(
        "--joint",
        type=str,
        default="left_knee_joint",
        help="single 模式下要可视化的策略控制关节名",
    )
    parser.add_argument(
        "--joints",
        type=str,
        nargs="+",
        default=["left_hip_pitch_joint", "left_knee_joint", "right_hip_pitch_joint", "right_knee_joint"],
        help="multi 模式下要可视化的多个策略控制关节名",
    )
    parser.add_argument(
        "--left-joint",
        type=str,
        default="left_knee_joint",
        help="compare 模式下左侧关节名",
    )
    parser.add_argument(
        "--right-joint",
        type=str,
        default="right_knee_joint",
        help="compare 模式下右侧关节名",
    )
    parser.add_argument(
        "--list-joints",
        action="store_true",
        help="打印当前可用的策略控制关节名并退出",
    )
    parser.add_argument(
        "--save-fig",
        action="store_true",
        help="保存图像到 depoly/logs/plots，而不是弹窗显示",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="depoly/logs/plots",
        help="保存图像目录",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    cfg = Mvr22DofConfig()
    if args.list_joints:
        print("Available controlled joints:")
        for joint_name in cfg.controlled_joint_names:
            print(joint_name)
        return

    csv_file = args.csv_file or get_latest_log(args.log_dir)
    print(f"Using log file: {csv_file}")
    if args.mode == "single":
        plot_joint_log(
            csv_file=csv_file,
            cfg=cfg,
            joint_name=args.joint,
            save_fig=args.save_fig,
            output_dir=args.output_dir,
        )
    elif args.mode == "multi":
        plot_multi_joint_actions(
            csv_file=csv_file,
            cfg=cfg,
            joint_names=args.joints,
            save_fig=args.save_fig,
            output_dir=args.output_dir,
        )
    elif args.mode == "compare":
        plot_left_right_comparison(
            csv_file=csv_file,
            cfg=cfg,
            left_joint=args.left_joint,
            right_joint=args.right_joint,
            save_fig=args.save_fig,
            output_dir=args.output_dir,
        )
    elif args.mode == "phase":
        plot_phase_overview(
            csv_file=csv_file,
            save_fig=args.save_fig,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
