"""Refine sigma grids (Euler, CFG 1, model time 1 - sigma)."""

REFINE_SHIFT = 6.0


def shifted_grid(start, steps, shift=REFINE_SHIFT):
    """Euler grid from `start` to 0: even spacing in unshifted sigma, then the shift map s*x / (1 + (s - 1)*x).
    `shifted_grid(0.909375, 4)` is the refiner's original 4-step grid (0.909375, 0.841446, 0.732075, 0.526697, 0)."""
    raw = start / (shift - (shift - 1) * start)
    return [start] + [shift * x / (1 + (shift - 1) * x) for x in (raw * (1 - i / steps) for i in range(1, steps))] + [0.0]


# 3 steps (default, the measured release setting): start 0.909375, then the shift-6 points 6/7 and 2/3.
# 2 steps: the same shift-6 rule from a lower start noise of 0.7 -> [0.7, 0.494118, 0].
SCHEDULES = {3: [0.909375, 0.8571428571428571, 0.6666666666666666, 0.0], 2: shifted_grid(0.7, 2)}
