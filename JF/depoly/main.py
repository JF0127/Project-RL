# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

import argparse

from config.mvr_10dof import Mvr10DofConfig
from config.mvr_22dof import Mvr22DofConfig
from runners.sim2sim_runner import Sim2SimRunner

ROBOT_CONFIGS = {
    "Mvr_10dof": Mvr10DofConfig,
    "Mvr_22dof": Mvr22DofConfig,
}


def main():
    parser = argparse.ArgumentParser(description="Mvr MuJoCo sim2sim entry point")
    parser.add_argument(
        "--robot",
        type=str,
        default="Mvr_22dof",
        choices=list(ROBOT_CONFIGS.keys()),
        help="选择机器人配置 (默认: Mvr_22dof)",
    )
    parser.add_argument("--policy-path", type=str, default=None, help="覆盖默认 policy.pt 路径")
    parser.add_argument("--xml-path", type=str, default=None, help="覆盖默认 MuJoCo XML 路径")

    # 录制参数
    parser.add_argument("--record", action="store_true", help="录制行走视频")
    parser.add_argument("--record-path", type=str, default=None, help="视频保存路径 (默认自动生成)")
    parser.add_argument("--record-camera", type=str, default=None, help="录制用相机名称 (默认: 自由视角)")
    parser.add_argument("--record-width", type=int, default=1280, help="视频宽度 (默认: 1280)")
    parser.add_argument("--record-height", type=int, default=720, help="视频高度 (默认: 720)")

    args = parser.parse_args()

    print(f"[*] Loading Configuration for: {args.robot}")
    cfg = ROBOT_CONFIGS[args.robot]()
    if args.policy_path:
        cfg.policy_path = args.policy_path
    if args.xml_path:
        cfg.xml_path = args.xml_path

    runner = Sim2SimRunner(
        config=cfg,
        record=args.record,
        record_path=args.record_path,
        record_camera=args.record_camera,
        record_width=args.record_width,
        record_height=args.record_height,
    )
    runner.run()


if __name__ == "__main__":
    main()
