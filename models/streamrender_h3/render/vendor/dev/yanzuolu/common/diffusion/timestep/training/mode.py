"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from typing import Sequence

import torch

from . import BaseTrainingTimesteps


class ModeTrainingTimesteps(BaseTrainingTimesteps):
    def __init__(
        self,
        scale: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.scale = scale

    def sample(
        self,
        size: Sequence[int],
        seqlens: Sequence[int],
        device: torch.device,
    ):
        t = torch.rand(size, dtype=torch.float64)
        t = 1 - t - self.scale * (torch.cos(torch.pi / 2 * t) ** 2 - 1 + t)
        return self.postprocess_sample(t, seqlens, device)
