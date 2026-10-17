"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from typing import Optional, Union, List

import torch

from .. import BaseTimesteps


class BaseTrainingTimesteps(BaseTimesteps):
    def __init__(
        self,
        T: Union[int, float],
        scaling_min: float = 0.0,
        scaling_max: float = 1.0,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
        shift: Optional[float] = None,
        dynamic_shift: bool = False,
        shift_list: Optional[List[float]] = None,
        seqlen_list: Optional[List[float]] = None,
        **kwargs
    ):
        self.scaling_min = scaling_min
        self.scaling_max = scaling_max
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        super().__init__(
            T=T,
            shift=shift,
            dynamic_shift=dynamic_shift,
            shift_list=shift_list,
            seqlen_list=seqlen_list,
            **kwargs
        )

    def postprocess_sample(self, t, seqlens, device):
        shift = self.shift
        if self.dynamic_shift:
            assert isinstance(seqlens, torch.Tensor) and len(seqlens) == len(t), "seqlens must be a tensor and have the same length as t"
            shift = torch.tensor([self.get_shift(s) for s in seqlens])

        if shift is not None:
            t = shift * t / (1 + (shift - 1) * t)
        t = t * (self.scaling_max - self.scaling_min) + self.scaling_min
        if self.clamp_min is not None:
            t = t.clamp_min(self.clamp_min)
        if self.clamp_max is not None:
            t = t.clamp_max(self.clamp_max)
        t = t.mul(self.T).to(device)
        return t if self.is_continuous() else t.round().long().clamp(0, self.T)
