# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym  # type: ignore

gym.register(
    id="Mvr_22dof_six",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.walk_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "RL.tasks.locomotion.base.agents.rsl_rl_ppo_cfg:MVR_22DOF_SIXPPORunnerCfg",
    },
)
