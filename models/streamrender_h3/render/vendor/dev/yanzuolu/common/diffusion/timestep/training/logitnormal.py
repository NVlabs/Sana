"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from typing import Sequence

import torch
from torch.distributions import LogisticNormal

from . import BaseTrainingTimesteps


class LogitNormalTrainingTimesteps(BaseTrainingTimesteps):
    def __init__(
        self,
        loc: float,
        scale: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dist = LogisticNormal(loc, scale)

    def sample(
        self,
        size: Sequence[int],
        seqlens: Sequence[int],
        device: torch.device,
    ):
        t = self.dist.sample(size)[..., 0].to(torch.float64)
        return self.postprocess_sample(t, seqlens, device)
