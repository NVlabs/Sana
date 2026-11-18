"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from typing import Sequence

import torch

from . import BaseTrainingTimesteps


class UniformTrainingTimesteps(BaseTrainingTimesteps):
    def sample(
        self,
        size: Sequence[int],
        seqlens: Sequence[int],
        device: torch.device,
    ):
        t = torch.rand(size, dtype=torch.float64)
        return self.postprocess_sample(t, seqlens, device)
