"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import math
from typing import List, Optional

from .warmup import WarmupLRScheduler


class CosineLRScheduler(WarmupLRScheduler):
    """Linear warmup, base lr until ``t_start``, then cosine decay to min_lr.

    ``t_max`` is the step at which the decay completes (set it to the expected
    real step count, e.g. an early-stop horizon, not an upper bound like
    training_steps); lr stays clamped at min_lr afterwards. ``t_start`` defaults
    to the end of warmup, so decay begins immediately as it always has.
    """
    def __init__(
        self,
        param_groups: List[dict],
        t_max: int,
        min_lr: float = 0.0,
        t_start: Optional[int] = None,
        warmup_t: Optional[int] = None,
        warmup_init_lr: float = 0.0,
        **kwargs
    ):
        super().__init__(param_groups, warmup_t, warmup_init_lr)
        self.t_max = t_max
        self.min_lr = min_lr
        self.t_start = (warmup_t or 0) if t_start is None else t_start
        assert self.t_start < self.t_max, f"t_start={self.t_start} must precede t_max={self.t_max}"

    def step(self, t):
        if self.warmup_t is not None and t < self.warmup_t:
            super().step(t)
            return
        if t < self.t_start:
            for param_group, lr in zip(self.param_groups, self.base_lrs):
                param_group["lr"] = lr
            return
        progress = min(1.0, (t - self.t_start) / (self.t_max - self.t_start))
        factor = 0.5 * (1 + math.cos(math.pi * progress))
        for param_group, lr in zip(self.param_groups, self.base_lrs):
            param_group["lr"] = self.min_lr + (lr - self.min_lr) * factor
