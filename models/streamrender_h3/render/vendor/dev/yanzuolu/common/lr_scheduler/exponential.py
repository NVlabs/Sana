"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from typing import List, Optional

from .warmup import WarmupLRScheduler


class ExponentialLRScheduler(WarmupLRScheduler):
    def __init__(
        self,
        param_groups: List[dict],
        gamma: float = 0.1,
        min_lr: float = 0.0,
        warmup_t: Optional[int] = None,
        warmup_init_lr: float = 0.0,
        **kwargs
    ):
        super().__init__(param_groups, warmup_t, warmup_init_lr)
        self.gamma = gamma
        self.min_lr = min_lr

    def step(self, t):
        if self.warmup_t is not None and t < self.warmup_t:
            super().step(t)
        else:
            for param_group, lr in zip(self.param_groups, self.base_lrs):
                param_group["lr"] = max(self.min_lr, lr * self.gamma ** (t - self.warmup_t))
