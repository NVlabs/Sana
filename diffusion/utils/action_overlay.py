# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Sana-WM action overlay (Genie-3 style) for camera-controlled videos.

Renders a small WASD key cluster + rotation joystick on top of each frame,
driven by the per-frame relative pose extracted from the camera trajectory.

Conventions (OpenCV camera frame):
    +X right, +Y down, +Z forward.
    W/S → translate along +Z / -Z (forward / back).
    D/A → translate along +X / -X (right / left).
    yaw > 0 (pan right)  → joystick deflects right.
    pitch > 0 (tilt down) → joystick deflects down.

The renderer pre-builds static layers (key tiles, joystick base) once;
per-frame work is just compositing them onto the input frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Pose → per-frame keys + normalized rotation
# ---------------------------------------------------------------------------


def _pose_inverse(p: np.ndarray) -> np.ndarray:
    R = p[:3, :3]
    t = p[:3, 3]
    inv = np.eye(4, dtype=p.dtype)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def _per_frame_deltas(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(N-1, 3)`` per-frame translation and YXZ-euler rotation (deg)."""
    n = c2w.shape[0]
    trans = np.zeros((n - 1, 3), dtype=np.float64)
    rots = np.zeros((n - 1, 3), dtype=np.float64)
    for i in range(n - 1):
        rel = _pose_inverse(c2w[i]) @ c2w[i + 1]
        trans[i] = rel[:3, 3]
        rots[i] = Rotation.from_matrix(rel[:3, :3]).as_euler("YXZ", degrees=True)
    return trans, rots


def _translation_keys(
    trans: np.ndarray,
    *,
    floor_dx: float = 0.005,
    floor_dz: float = 0.005,
    frac_dx: float = 0.30,
    frac_dz: float = 0.30,
) -> list[list[str]]:
    """Discretise per-frame translation into WASD key lists.

    Thresholds are adaptive: ``thresh = max(floor, frac * p95(|delta|))``.
    """
    p95 = np.percentile(np.abs(trans), 95.0, axis=0) if trans.size else np.zeros(3)
    thr_dx = max(floor_dx, frac_dx * p95[0])
    thr_dz = max(floor_dz, frac_dz * p95[2])

    per_frame: list[list[str]] = []
    for dx, _dy, dz in trans:
        keys: list[str] = []
        if abs(dz) > thr_dz:
            keys.append("W" if dz > 0 else "S")
        if abs(dx) > thr_dx:
            keys.append("D" if dx > 0 else "A")
        per_frame.append(keys)
    per_frame.append(list(per_frame[-1]) if per_frame else [])
    return per_frame


def _normalised_rotation(
    rots: np.ndarray, *, floor_deg: float = 0.5, ema_alpha: float = 0.35
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame ``(yaw, pitch)`` in ``[-1, 1]`` with EMA smoothing."""
    p95 = np.percentile(np.abs(rots), 95.0, axis=0) if rots.size else np.zeros(3)
    yaw_scale = max(floor_deg, p95[0])
    pitch_scale = max(floor_deg, p95[1])
    n = rots.shape[0]
    yaw = np.zeros(n + 1, dtype=np.float64)
    pitch = np.zeros(n + 1, dtype=np.float64)
    yaw_ema = pitch_ema = 0.0
    for i in range(n):
        y = float(np.clip(rots[i, 0] / yaw_scale, -1.0, 1.0))
        p = float(np.clip(rots[i, 1] / pitch_scale, -1.0, 1.0))
        yaw_ema = ema_alpha * y + (1.0 - ema_alpha) * yaw_ema
        pitch_ema = ema_alpha * p + (1.0 - ema_alpha) * pitch_ema
        yaw[i] = yaw_ema
        pitch[i] = pitch_ema
    if n > 0:
        yaw[-1] = yaw[-2]
        pitch[-1] = pitch[-2]
    return yaw, pitch


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


_FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


@dataclass(frozen=True)
class _Layout:
    width: int
    height: int

    @property
    def key_size(self) -> int:
        return max(32, int(self.height * 0.08))

    @property
    def key_gap(self) -> int:
        return max(4, int(self.key_size * 0.15))

    @property
    def key_radius(self) -> int:
        return max(4, int(self.key_size * 0.2))


class ActionOverlayRenderer:
    """Renders the WASD-cluster + rotation-joystick overlay onto video frames."""

    CORNER_CHOICES = ("bottom-left", "bottom-right", "top-left", "top-right")

    def __init__(self, width: int, height: int):
        self.layout = _Layout(int(width), int(height))
        self.width, self.height = self.layout.width, self.layout.height
        self.font = _load_font(int(self.layout.key_size * 0.5))
        self._key_tiles = self._build_key_tiles()

    def _build_key_tiles(self) -> dict[tuple[str, bool], Image.Image]:
        sz, r = self.layout.key_size, self.layout.key_radius
        tiles: dict[tuple[str, bool], Image.Image] = {}
        for key in ("W", "A", "S", "D"):
            for pressed in (False, True):
                fill = (255, 255, 255, 200) if pressed else (0, 0, 0, 100)
                outline = (255, 255, 255, 255) if pressed else (255, 255, 255, 60)
                text_color = (0, 0, 0, 220) if pressed else (255, 255, 255, 180)
                tile = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
                d = ImageDraw.Draw(tile)
                d.rounded_rectangle(
                    [0, 0, sz - 1, sz - 1],
                    radius=r,
                    fill=fill,
                    outline=outline,
                    width=max(1, int(sz * 0.03)),
                )
                bbox = d.textbbox((0, 0), key, font=self.font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                d.text((sz / 2 - tw / 2, sz / 2 - th / 2 - 2), key, fill=text_color, font=self.font)
                tiles[(key, pressed)] = tile
        return tiles

    def _draw_joystick(self, canvas: Image.Image, cx: int, cy: int, yaw: float, pitch: float) -> None:
        yaw = float(np.clip(yaw, -1.0, 1.0))
        pitch = float(np.clip(pitch, -1.0, 1.0))
        radius = max(30, int((self.layout.key_size * 2 + self.layout.key_gap) * 0.47))

        # Outer shadow / glassy plate.
        shadow = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).ellipse(
            [cx - radius - 14, cy - radius - 14, cx + radius + 14, cy + radius + 14],
            fill=(0, 0, 0, 88),
        )
        canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(max(8, int(radius * 0.16)))))

        d = ImageDraw.Draw(canvas)
        d.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(7, 9, 13, 104),
            outline=(255, 255, 255, 95),
            width=max(1, int(radius * 0.035)),
        )
        d.ellipse(
            [cx - radius - 7, cy - radius - 7, cx + radius + 7, cy + radius + 7],
            outline=(255, 255, 255, 42),
            width=max(1, int(radius * 0.025)),
        )
        d.line([cx - radius * 0.63, cy, cx + radius * 0.63, cy], fill=(255, 255, 255, 56), width=1)
        d.line([cx, cy - radius * 0.63, cx, cy + radius * 0.63], fill=(255, 255, 255, 56), width=1)

        # Direction triangles + active arcs.
        marker_offset = int(radius * 0.78)
        marker_size = max(7, int(radius * 0.16))
        self._draw_arrow(d, cx + marker_offset, cy, "right", yaw > 0.08, marker_size)
        self._draw_arrow(d, cx - marker_offset, cy, "left", yaw < -0.08, marker_size)
        self._draw_arrow(d, cx, cy - marker_offset, "up", pitch < -0.08, marker_size)
        self._draw_arrow(d, cx, cy + marker_offset, "down", pitch > 0.08, marker_size)

        # Joystick knob + travel line.
        max_offset = radius * 0.48
        kx = int(cx + yaw * max_offset)
        ky = int(cy + pitch * max_offset)
        glow = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.line([cx, cy, kx, ky], fill=(255, 255, 255, 78), width=max(3, int(radius * 0.055)))
        gd.ellipse(
            [kx - radius * 0.22, ky - radius * 0.22, kx + radius * 0.22, ky + radius * 0.22], fill=(255, 255, 255, 110)
        )
        canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(5, int(radius * 0.09)))))

        d = ImageDraw.Draw(canvas)
        d.line([cx, cy, kx, ky], fill=(255, 255, 255, 120), width=max(1, int(radius * 0.025)))
        kr = max(7, int(radius * 0.13))
        d.ellipse(
            [kx - kr, ky - kr, kx + kr, ky + kr], fill=(255, 255, 255, 230), outline=(255, 255, 255, 255), width=1
        )
        ir = max(3, int(kr * 0.36))
        d.ellipse([kx - ir, ky - ir, kx + ir, ky + ir], fill=(20, 24, 30, 170))

    @staticmethod
    def _draw_arrow(d: ImageDraw.ImageDraw, cx: int, cy: int, direction: str, active: bool, size: int) -> None:
        if direction == "right":
            pts = [(cx - size * 0.55, cy - size), (cx + size * 0.65, cy), (cx - size * 0.55, cy + size)]
        elif direction == "left":
            pts = [(cx + size * 0.55, cy - size), (cx - size * 0.65, cy), (cx + size * 0.55, cy + size)]
        elif direction == "up":
            pts = [(cx - size, cy + size * 0.55), (cx, cy - size * 0.65), (cx + size, cy + size * 0.55)]
        else:
            pts = [(cx - size, cy - size * 0.55), (cx, cy + size * 0.65), (cx + size, cy - size * 0.55)]
        d.polygon(pts, fill=(255, 255, 255, 210) if active else (255, 255, 255, 72))

    def render_panel(
        self,
        pressed_keys: Sequence[str],
        yaw: float,
        pitch: float,
        corner: str = "bottom-left",
    ) -> Image.Image:
        """Return an RGBA overlay of size ``(width, height)``."""
        canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        sz, gap = self.layout.key_size, self.layout.key_gap
        margin = int(self.height * 0.05)
        cluster_w = sz * 3 + gap * 2
        cluster_h = sz * 2 + gap
        joy_radius = max(30, int(cluster_h * 0.47))
        joy_gap = int(sz * 1.0)

        if corner == "bottom-right":
            sx = self.width - margin - cluster_w
            sy = self.height - margin - cluster_h
            jcx = sx - joy_gap - joy_radius
        elif corner == "top-left":
            sx, sy = margin, margin
            jcx = sx + cluster_w + joy_gap + joy_radius
        elif corner == "top-right":
            sx = self.width - margin - cluster_w
            sy = margin
            jcx = sx - joy_gap - joy_radius
        else:  # bottom-left default
            sx, sy = margin, self.height - margin - cluster_h
            jcx = sx + cluster_w + joy_gap + joy_radius
        jcy = sy + cluster_h // 2

        # Soft panel shadow.
        shadow = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rounded_rectangle(
            [sx - 8, sy - 8, sx + cluster_w + 8, sy + cluster_h + 8],
            radius=max(10, int(self.layout.key_radius * 1.35)),
            fill=(0, 0, 0, 74),
        )
        canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(max(7, int(sz * 0.18)))))

        positions = {
            "W": (sx + sz + gap, sy),
            "A": (sx, sy + sz + gap),
            "S": (sx + sz + gap, sy + sz + gap),
            "D": (sx + (sz + gap) * 2, sy + sz + gap),
        }
        for key in ("W", "A", "S", "D"):
            canvas.alpha_composite(self._key_tiles[(key, key in pressed_keys)], dest=positions[key])
        self._draw_joystick(canvas, jcx, jcy, yaw, pitch)
        return canvas


def apply_overlay(
    video_hwc: np.ndarray,
    c2w: np.ndarray,
    *,
    corner: str = "bottom-left",
) -> np.ndarray:
    """Composite the action overlay onto each frame.

    Args:
        video_hwc: ``(T, H, W, 3)`` uint8 video.
        c2w: ``(T_pose, 4, 4)`` camera-to-world poses driving the overlay.
            Truncated to ``video_hwc.shape[0]`` frames if longer.
        corner: Panel placement.

    Returns:
        ``(T, H, W, 3)`` uint8 array with the overlay composited.
    """
    T, H, W = video_hwc.shape[:3]
    n_poses = min(int(c2w.shape[0]), T)
    poses = c2w[:n_poses].astype(np.float32)

    trans, rots = _per_frame_deltas(poses)
    keys = _translation_keys(trans)
    yaw, pitch = _normalised_rotation(rots)

    renderer = ActionOverlayRenderer(width=W, height=H)
    out = np.empty_like(video_hwc)
    for t in range(T):
        i = min(t, len(keys) - 1)
        panel = renderer.render_panel(
            pressed_keys=keys[i],
            yaw=float(yaw[i]),
            pitch=float(pitch[i]),
            corner=corner,
        )
        frame = Image.fromarray(video_hwc[t]).convert("RGBA")
        frame.alpha_composite(panel)
        out[t] = np.asarray(frame.convert("RGB"), dtype=np.uint8)
    return out
