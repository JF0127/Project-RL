# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.utils.string as string_utils
import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass


class MaskedJointPositionAction(JointPositionAction):
    """Joint position action that overrides selected joints with fixed position targets."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        fixed_joint_ids, _, fixed_values = string_utils.resolve_matching_names_values(
            cfg.fixed_joint_positions, self._asset.joint_names
        )

        self._fixed_joint_ids = fixed_joint_ids
        self._fixed_targets = None

        if fixed_joint_ids:
            self._fixed_targets = torch.tensor(fixed_values, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

    def apply_actions(self):
        if self._fixed_targets is not None:
            self._asset.set_joint_position_target(self._fixed_targets, joint_ids=self._fixed_joint_ids)
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


@configclass
class MaskedJointPositionActionCfg(ActionTermCfg):
    """Joint position action cfg with absolute fixed targets for selected joints."""

    class_type: type[ActionTerm] = MaskedJointPositionAction

    joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    preserve_order: bool = False
    use_default_offset: bool = True
    fixed_joint_positions: dict[str, float] = MISSING
