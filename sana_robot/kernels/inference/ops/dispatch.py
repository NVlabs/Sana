# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

SM90 = "sm90"
SM110 = "sm110"
SM120 = "sm120"
ARCHITECTURES = {(9, 0): SM90, (11, 0): SM110, (12, 0): SM120}
TUNED_MARGIN = 64


def device_architecture(device) -> str | None:
    """Return the kernel architecture of a CUDA device: ``sm90`` (H100), ``sm110`` (Jetson Thor), ``sm120`` (RTX 5090, RTX PRO 6000), or None for any other device."""
    device = torch.device(device)
    if device.type != "cuda":
        return None
    return ARCHITECTURES.get(tuple(torch.cuda.get_device_capability(device)))


class TunedTable(dict):
    """Launch constants keyed by a token count, or by a tuple that starts with one, that also serve the counts near a measured one.

    The constants were measured at a few token counts and change slowly with the count. ``get`` returns the entry of
    the key when the table holds it; otherwise the entry of the nearest of the ``tuned`` counts, with the rest of the
    key equal, that lies within ``margin`` tokens (the larger count on a tie); otherwise ``default``.

    Args:
        entries: The constants by key.
        tuned: The token counts whose constants were measured on the device.
        margin: The largest distance in tokens at which a measured count serves another.
    """

    def __init__(self, entries: dict, tuned, margin: int = TUNED_MARGIN) -> None:
        super().__init__(entries)
        self.tuned = frozenset(int(count) for count in tuned)
        self.margin = margin

    def get(self, key, default=None):
        if key in self:
            return self[key]
        (tokens, rest) = (key[0], key[1:]) if isinstance(key, tuple) else (key, None)
        nearby = [
            (abs(count - tokens), -count, (count, *rest) if rest is not None else count)
            for count in self.tuned
            if abs(count - tokens) <= self.margin
        ]
        nearby = sorted(candidate for candidate in nearby if candidate[2] in self)
        return self[nearby[0][2]] if nearby else default
