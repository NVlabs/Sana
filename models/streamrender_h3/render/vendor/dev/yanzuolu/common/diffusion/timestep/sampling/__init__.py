"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from typing import List, Optional, Sequence, Union

import torch

from ....seed import RandomState
from ....distributed.ops import get_device
from .. import BaseTimesteps


class BaseSamplingTimesteps(BaseTimesteps):
    def __init__(
        self,
        T: Union[int, float],
        shift: Optional[float] = None,
        num_sampling_steps: Optional[int] = None,
        sampling_skip_min: Optional[int] = None,
        sampling_skip_max: Optional[int] = None,
        dynamic_shift: bool = False,
        shift_list: Optional[List[float]] = None,
        seqlen_list: Optional[List[float]] = None,
        **kwargs
    ):
        super().__init__(
            T=T,
            shift=shift,
            dynamic_shift=dynamic_shift,
            shift_list=shift_list,
            seqlen_list=seqlen_list,
            **kwargs
        )
        self.num_sampling_steps = num_sampling_steps
        self.sampling_skip_min = sampling_skip_min
        self.sampling_skip_max = sampling_skip_max
        self.kwargs = kwargs

        self.timesteps = None
        if not self.dynamic_shift and self.num_sampling_steps is not None:
            self.set_timesteps(self.num_sampling_steps, get_device(), **self.kwargs)

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
        min_index = self.sampling_skip_min if self.sampling_skip_min is not None else 0
        max_index = self.num_sampling_steps - (self.sampling_skip_max if self.sampling_skip_max is not None else 0)
        assert max_index > min_index, \
            f"max_index {max_index} must be greater than min_index {min_index}"
        random_index = torch.randint(min_index, max_index, size, device=device)
        if self.dynamic_shift:
            seqlens = torch.as_tensor(seqlens, device=device)
            assert len(seqlens) == len(random_index), "seqlens must have the same batch size as sampled timesteps"
            self.set_timesteps(self.num_sampling_steps, get_device(), seqlen=seqlens, **self.kwargs)

        if self.timesteps.ndim == 1:
            timesteps = self.timesteps[random_index]
        else:
            assert len(self.timesteps) == len(random_index), "dynamic timesteps must match sampled batch size"
            timesteps = self.timesteps[torch.arange(len(random_index), device=device), random_index]
        return timesteps

    def set_timesteps(
        self,
        num_sampling_steps: int,
        device: torch.device,
        **kwargs
    ):
        raise NotImplementedError

    def get_next_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        """
        Return next timesteps by t.
        Will return bound if index(t)+1 is reaching the end.
        """
        assert self.timesteps is not None, "Timesteps must be set before calling get_next_timesteps"
        curr_idx = self.index(t)
        next_idx = curr_idx + 1
        bound = 0.0 if self.is_continuous() else 0  # last step

        if self.timesteps.ndim == 1:
            s = self.timesteps[next_idx.clamp_max(self.num_sampling_steps - 1)]
        else:
            s = self.timesteps[torch.arange(len(t)), next_idx.clamp_max(self.num_sampling_steps - 1)]
        s = s.where(next_idx < self.num_sampling_steps, bound)
        return s

    def get_timesteps_by_index(self, index: torch.Tensor) -> torch.Tensor:
        """
        Return timesteps by index.
        Will return bound if index is reaching the end.
        """
        assert self.timesteps is not None, "Timesteps must be set before calling get_timesteps_by_index"
        bound = 0.0 if self.is_continuous() else 0  # last step
        t = self.timesteps[index.clamp_max(self.num_sampling_steps - 1)]
        t = t.where(index < self.num_sampling_steps, bound)
        return t

    def index(self, t: torch.Tensor) -> torch.Tensor:
        """
        Find index by t.
        Return index of the same shape as t.
        Index is -1 if t not found in timesteps.
        """
        i, j = t.reshape(-1, 1).eq(self.timesteps).nonzero(as_tuple=True)
        idx = torch.full_like(t, fill_value=-1, dtype=torch.int)
        idx.view(-1)[i] = j.int()
        return idx

    def lerp_random(self, min_t: torch.Tensor, max_t: torch.Tensor, rng: RandomState) -> torch.Tensor:
        """
        Return a random timestep between min_t (exclusive) and max_t (inclusive).
        Will handle continuous and discrete timesteps.
        """
        bsz = len(min_t)
        diff_t = max_t - min_t
        timepoints = 1. - torch.rand((bsz,), device=get_device(), generator=rng.torch_cuda_generator)  # (0,1]
        timesteps = min_t + timepoints * diff_t
        if not self.is_continuous():
            timesteps = timesteps.round().long()
        return timesteps

    def lerp_multistep(self, min_t: torch.Tensor, max_t: torch.Tensor, multistep: int) -> torch.Tensor:
        """
        Return multiple timesteps between min_t (exclusive) and max_t (inclusive) with equal distance.
        Will handle continuous and discrete timesteps.
        """
        diff_t = max_t - min_t
        timesteps = torch.stack([
            min_t + i * diff_t / multistep for i in range(multistep, 0, -1)
        ])
        if not self.is_continuous():
            timesteps = timesteps.round().long()
        return timesteps
