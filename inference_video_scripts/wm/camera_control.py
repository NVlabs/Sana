# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# Licensed under the Apache License, Version 2.0
# SPDX-License-Identifier: Apache-2.0

"""Shared camera-control core.

One implementation of the pose math + velocity model, used by BOTH the
action-string inference rollout (``action_string_to_c2w``) and the interactive
realtime demo, so the two can never drift. Only the *input surface* differs:
the inference path maps DSL letters, the demo maps physical keys — both produce
the same canonical controls and run through the same smoother + integrator.

Coordinate convention: OpenCV (``+X right, +Y down, +Z forward``); poses are
camera-to-world, starting from the identity.

Control scheme (unified):
    W / S            forward / back translation (along heading)
    A / D            yaw left / right rotation        (DSL: a/d ; demo: A/D)
    Up / Down        pitch up / down rotation         (DSL: i/k ; demo: ArrowUp/Down)
    Left / Right     strafe left / right translation  (DSL: j/l ; demo: ArrowLeft/Right)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FPS = 16

# Gentle default motion magnitudes (per frame), shared by demo + inference.
DEFAULT_TRANSLATION_SPEED = 0.025
DEFAULT_ROTATION_SPEED_DEG = 0.6
DEFAULT_PITCH_LIMIT_DEG = 60.0

# Exponential-smoothing time constants (seconds): faster ramp on press, slower
# coast on release so motion eases to a stop instead of snapping.
TAU_PRESS = 0.45
TAU_COAST = 1.0

# Canonical control tokens.
CONTROL_TOKENS = frozenset(
    {"forward", "back", "strafe_left", "strafe_right", "yaw_left", "yaw_right", "pitch_up", "pitch_down"}
)

# Input-surface mappings -> canonical controls (the ONLY thing that differs
# between the two entry points).
DSL_KEY_TO_CONTROL: dict[str, str] = {
    "w": "forward",
    "s": "back",
    "a": "yaw_left",
    "d": "yaw_right",
    "i": "pitch_up",
    "k": "pitch_down",
    "j": "strafe_left",
    "l": "strafe_right",
}
DEMO_KEY_TO_CONTROL: dict[str, str] = {
    "w": "forward",
    "s": "back",
    "a": "yaw_left",
    "d": "yaw_right",
    "up": "pitch_up",
    "down": "pitch_down",
    "left": "strafe_left",
    "right": "strafe_right",
}


def rot_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rot_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


@dataclass
class VelocityState:
    """Per-frame velocity: translation (tx forward, sx strafe-right) + rotation
    (yaw right+, pitch up+), in the same units as the motion magnitudes."""

    tx: float = 0.0
    sx: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0

    def snap_to(self, target: VelocityState) -> None:
        self.tx, self.sx, self.yaw, self.pitch = target.tx, target.sx, target.yaw, target.pitch

    def step_toward(self, target: VelocityState, dt: float) -> None:
        for attr in ("tx", "sx", "yaw", "pitch"):
            cur = getattr(self, attr)
            tgt = getattr(target, attr)
            tau = TAU_PRESS if abs(tgt) > 1e-12 else TAU_COAST
            alpha = 1.0 - math.exp(-dt / tau)
            setattr(self, attr, cur + alpha * (tgt - cur))


def controls_to_target_velocity(
    controls: set[str],
    *,
    translation_speed: float = DEFAULT_TRANSLATION_SPEED,
    rotation_speed_rad: float | None = None,
) -> VelocityState:
    """Map a set of canonical control tokens to a target velocity."""
    if rotation_speed_rad is None:
        rotation_speed_rad = math.radians(DEFAULT_ROTATION_SPEED_DEG)
    fwd = (1.0 if "forward" in controls else 0.0) - (1.0 if "back" in controls else 0.0)
    strafe = (1.0 if "strafe_right" in controls else 0.0) - (1.0 if "strafe_left" in controls else 0.0)
    yaw = (1.0 if "yaw_right" in controls else 0.0) - (1.0 if "yaw_left" in controls else 0.0)
    pit = (1.0 if "pitch_up" in controls else 0.0) - (1.0 if "pitch_down" in controls else 0.0)
    return VelocityState(
        tx=fwd * translation_speed,
        sx=strafe * translation_speed,
        yaw=yaw * rotation_speed_rad,
        pitch=pit * rotation_speed_rad,
    )


class CameraPoseIntegrator:
    """Integrate per-frame velocity into a camera-to-world pose (the shared core).

    ``rot_y(yaw) @ R @ rot_x(pitch)`` for rotation; translation is on the
    horizontal (y=0) plane along the projected forward / right axes. Pitch
    saturates at ``±pitch_limit``.
    """

    def __init__(self, pitch_limit_rad: float = math.radians(DEFAULT_PITCH_LIMIT_DEG)) -> None:
        self.pose = np.eye(4, dtype=np.float64)
        self.pitch = 0.0
        self.pitch_limit = float(pitch_limit_rad)

    def step(self, v: VelocityState) -> np.ndarray:
        new_pitch = max(-self.pitch_limit, min(self.pitch_limit, self.pitch + v.pitch))
        pitch_step = new_pitch - self.pitch
        self.pitch = new_pitch

        R = self.pose[:3, :3]
        R_new = rot_y(v.yaw) @ R @ rot_x(pitch_step)

        fwd = R_new[:, 2].copy()
        fwd[1] = 0.0
        rgt = R_new[:, 0].copy()
        rgt[1] = 0.0
        fn = float(np.linalg.norm(fwd))
        rn = float(np.linalg.norm(rgt))
        if fn > 0:
            fwd /= fn + 1e-6
        if rn > 0:
            rgt /= rn + 1e-6
        T_ = self.pose[:3, 3] + fwd * v.tx + rgt * v.sx

        self.pose = np.eye(4, dtype=np.float64)
        self.pose[:3, :3] = R_new
        self.pose[:3, 3] = T_
        return self.pose.copy()
