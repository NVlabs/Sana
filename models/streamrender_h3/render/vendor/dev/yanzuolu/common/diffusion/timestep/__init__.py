"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import math
from bisect import bisect_right
from typing import List, Optional, Sequence, Union

import torch


class BaseTimesteps:
    """ General class for both discrete and continuous timestep. """

    timesteps: torch.Tensor

    def __init__(
        self,
        T: Union[int, float],
        shift: Optional[float] = None,
        dynamic_shift: bool = False,
        shift_list: Optional[List[float]] = None,
        seqlen_list: Optional[List[float]] = None,
        shift_exp: bool = False,
        **kwargs
    ):
        self.T = T
        self.shift = shift
        self.dynamic_shift = dynamic_shift
        self.shift_list = shift_list
        self.seqlen_list = seqlen_list
        self.shift_exp = shift_exp

        if self.shift is not None and self.shift_exp:
            self.shift = math.exp(self.shift)

        if self.dynamic_shift:
            # assert self.shift is None, "shift should be None when dynamic_shift is True"
            assert self.shift_list is not None, "shift_list should be provided when dynamic_shift is True"
            assert self.seqlen_list is not None, "seqlen_list should be provided when dynamic_shift is True"
            assert len(self.shift_list) == len(self.seqlen_list), "shift_list and seqlen_list should have the same length"

    def get_shift(self, seqlen: torch.Tensor) -> torch.Tensor:
        index = bisect_right(self.seqlen_list, seqlen)

        if index == 0:  # seqlen is smaller than the smallest seqlen in seqlen_list
            x1, y1 = self.seqlen_list[0], self.shift_list[0]
            x2, y2 = self.seqlen_list[1], self.shift_list[1]
        elif index == len(self.shift_list):  # seqlen is larger than the largest seqlen in seqlen_list
            x1, y1 = self.seqlen_list[-2], self.shift_list[-2]
            x2, y2 = self.seqlen_list[-1], self.shift_list[-1]
        else:
            x1, y1 = self.seqlen_list[index - 1], self.shift_list[index - 1]
            x2, y2 = self.seqlen_list[index], self.shift_list[index]

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        s = m * seqlen + b

        s = s.item()
        if self.shift_exp:
            s = math.exp(s)
        return s

    def is_continuous(self) -> bool:
        return isinstance(self.T, float)

    def sample(
        self,
        size: Sequence[int],
        seqlens: Sequence[int],
        device: torch.device,
    ):
        """
        This function will be called inside local_seed()
        so that there is no need to specify generator
        """
        raise NotImplementedError
