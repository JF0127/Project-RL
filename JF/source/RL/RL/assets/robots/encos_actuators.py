# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# encos_actuators.py
# Copyright (c) 2025-2026, Master Jia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


def rpm_to_rads(rpm: float) -> float:
    """Convert rpm to rad/s."""
    return rpm * (2.0 * math.pi / 60.0)


class EncosTNActuator(DelayedPDActuator):
    """Unitree-style T–N actuator: PD -> torque, then T–N clipping + smooth friction.

    T–N curve:
      - |v| < X1: |tau| <= Y (flat)
      - X1 <= |v| <= X2: linear drop from (X1, Y) to (X2, 0)
      - |v| > X2: limit 0

    Directional peak torque:
      - same_direction (v * tau > 0): Y1
      - opposite direction: Y2 (defaults to Y1)

    Smooth friction:
      - tau_fric = Fs * tanh(v/Va) + Fd * v
    """

    cfg: EncosTNActuatorCfg

    def __init__(self, cfg: EncosTNActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._effort_y1 = self._parse_joint_parameter(cfg.Y1, 1e9)
        self._effort_y2 = self._parse_joint_parameter(cfg.Y2, cfg.Y1)
        self._velocity_x1 = self._parse_joint_parameter(cfg.X1, 1e9)
        self._velocity_x2 = self._parse_joint_parameter(cfg.X2, 1e9)

        self._friction_static = self._parse_joint_parameter(cfg.Fs, 0.0)
        self._friction_dynamic = self._parse_joint_parameter(cfg.Fd, 0.0)
        self._activation_vel = self._parse_joint_parameter(cfg.Va, 0.01)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # save current joint vel (used by _clip_effort)
        self._joint_vel[:] = joint_vel

        # Delayed PD compute (internally calls _clip_effort)
        control_action = super().compute(control_action, joint_pos, joint_vel)

        # smooth friction (Unitree style)
        self.applied_effort -= (
            self._friction_static * torch.tanh(joint_vel / self._activation_vel) + self._friction_dynamic * joint_vel
        )

        # effort-only output
        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort
        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # same direction check
        same_direction = (self._joint_vel * effort) > 0
        max_effort = torch.where(same_direction, self._effort_y1, self._effort_y2)

        # flat region vs linear drop
        max_effort = torch.where(
            self._joint_vel.abs() < self._velocity_x1, max_effort, self._compute_effort_limit(max_effort)
        )
        return torch.clip(effort, -max_effort, max_effort)

    def _compute_effort_limit(self, max_effort: torch.Tensor) -> torch.Tensor:
        # linear drop: (X1, max_effort) -> (X2, 0)
        k = -max_effort / (self._velocity_x2 - self._velocity_x1)
        limit = k * (self._joint_vel.abs() - self._velocity_x1) + max_effort
        return limit.clip(min=0.0)


@configclass
class EncosTNActuatorCfg(DelayedPDActuatorCfg):
    """Generic cfg for ENCOS motors (Unitree-style)."""

    class_type: type = EncosTNActuator

    # T–N curve
    X1: float = 1e9  # rad/s
    X2: float = 1e9  # rad/s
    Y1: float = MISSING  # N·m
    Y2: float | None = None  # N·m (defaults to Y1)

    # friction (optional)
    Fs: float = 0.0
    Fd: float = 0.0
    Va: float = 0.01


# -----------------------------
# Motor library (single source of truth)
# Mapping confirmed by you:
#   heavy  -> A10020
#   medium -> A8112
#   light  -> A6408
#   micro  -> A4310
# armature values are OUTPUT-SIDE equivalent and are given by you.
# -----------------------------


@configclass
class MotorCfg_A10020(EncosTNActuatorCfg):
    # EC-A10020-P1-12
    X1 = rpm_to_rads(115.0)  # rated speed
    X2 = rpm_to_rads(143.0)  # peak speed (approx no-load)
    Y1 = 150
    Y2 = None
    armature = 0.0703


@configclass
class MotorCfg_A8112(EncosTNActuatorCfg):
    # EC-A8112-P1-18
    X1 = rpm_to_rads(130.0)
    X2 = rpm_to_rads(157.0)
    Y1 = 90
    Y2 = None
    armature = 0.0475


@configclass
class MotorCfg_A6408(EncosTNActuatorCfg):
    # EC-A6408-P2-25
    X1 = rpm_to_rads(105.0)
    X2 = rpm_to_rads(149.0)
    Y1 = 60
    Y2 = None
    armature = 0.0389


@configclass
class MotorCfg_A4310(EncosTNActuatorCfg):
    # EC-A4310-P2-36
    X1 = rpm_to_rads(41.0)
    X2 = rpm_to_rads(87.0)
    Y1 = 36
    Y2 = None
    armature = 0.0236
