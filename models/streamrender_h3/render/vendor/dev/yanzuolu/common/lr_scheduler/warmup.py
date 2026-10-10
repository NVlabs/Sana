"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from typing import Any, List, Optional, Tuple

from . import BaseLRScheduler


class WarmupLRScheduler(BaseLRScheduler):
    """ Simple lr scheduler with only warmup.
    """
    def __init__(
        self,
        targets: List[Tuple[Any, int]],
        warmup_t: Optional[int] = None,
        warmup_init_lr: float = 0.0,
        **kwargs
    ):
        # ``targets`` holds (optimizer, group_index) pairs, not the param_group
        # dicts themselves. Both Optimizer.load_state_dict and DCP's
        # set_optimizer_state_dict REPLACE optimizer.param_groups with the dicts
        # carried in the checkpoint, so a dict captured here is silently orphaned
        # the moment a run resumes: every lr written to it is discarded and the
        # optimizer keeps the checkpoint's lr forever. Resolving through the
        # optimizer on each access makes the write land on whatever param_group
        # the optimizer currently owns.
        self.targets = targets
        self.warmup_t = warmup_t
        self.warmup_init_lr = warmup_init_lr
        # Read before any resume, so this is the configured lr and the schedule
        # stays replayable from step 0 rather than from the checkpoint's value.
        self.base_lrs = [group["lr"] for group in self.param_groups]

    @property
    def param_groups(self) -> List[dict]:
        return [optimizer.param_groups[index] for optimizer, index in self.targets]

    def step(self, t):
        if self.warmup_t is not None and t < self.warmup_t:
            warmup_factor = (t + 1) / self.warmup_t
            for param_group, lr in zip(self.param_groups, self.base_lrs):
                param_group["lr"] = self.warmup_init_lr + warmup_factor * (lr - self.warmup_init_lr)
        else:
            for param_group, lr in zip(self.param_groups, self.base_lrs):
                param_group["lr"] = lr
