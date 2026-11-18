"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import bisect
from typing import List, Optional

from .warmup import WarmupLRScheduler


class MultiStepLRScheduler(WarmupLRScheduler):
    def __init__(
        self,
        param_groups: List[dict],
        milestones: List[int],
        gamma: float = 0.1,
        warmup_t: Optional[int] = None,
        warmup_init_lr: float = 0.0,
        **kwargs
    ):
        super().__init__(param_groups, warmup_t, warmup_init_lr)
        self.milestones = milestones
        self.gamma = gamma

    def step(self, t):
        if self.warmup_t is not None and t < self.warmup_t:
            super().step(t)
        else:
            exp = bisect.bisect_right(self.milestones, t)
            for param_group, lr in zip(self.param_groups, self.base_lrs):
                param_group["lr"] = lr * self.gamma ** exp
