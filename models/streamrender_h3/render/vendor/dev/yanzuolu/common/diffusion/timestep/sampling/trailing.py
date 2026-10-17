"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from typing import Optional

import numpy as np
import torch

from . import BaseSamplingTimesteps


class TrailingSamplingTimesteps(BaseSamplingTimesteps):
    def __init__(self, **kwargs):
        self.final_linear_steps = kwargs.get("final_linear_steps", 0)
        super().__init__(**kwargs)

    def set_timesteps(
        self,
        num_sampling_steps: int = None,
        device: torch.device = "cpu",
        *,
        final_linear_steps: Optional[int] = None,
        seqlen: Optional[torch.Tensor] = None,
        **kwargs
    ):
        shift = self.shift
        if self.dynamic_shift:
            assert seqlen is not None, "seqlen should be provided when dynamic_shift is True"
            if len(seqlen) > 1:  # if hybrid then set different shift for each sample
                assert isinstance(seqlen, torch.Tensor)
                timesteps = []
                for s in seqlen:
                    self.shift = self.get_shift(s)
                    self.set_timesteps(num_sampling_steps, device="cpu", final_linear_steps=final_linear_steps, seqlen=[s])
                    timesteps.append(self.timesteps)
                self.timesteps = torch.stack(timesteps).to(device)
                return
            else:
                shift = self.get_shift(seqlen[0])

        if num_sampling_steps is None:
            num_sampling_steps = self.num_sampling_steps
        if final_linear_steps is None:
            final_linear_steps = self.final_linear_steps

        t = np.arange(1.0, 0, -1.0 / (num_sampling_steps - final_linear_steps))
        if shift is not None:
            t = shift * t / (1 + (shift - 1) * t)
        if final_linear_steps > 0:
            linear_t = np.arange(t[-1], 0, -t[-1] / (final_linear_steps + 1))[1:]
            t = np.concatenate([t, linear_t])

        if isinstance(self.T, float):
            timesteps = torch.from_numpy(t) * self.T
        else:
            timesteps = t * self.T
            timesteps = timesteps.round().astype(np.int64)
            timesteps = torch.from_numpy(timesteps)

        self.num_sampling_steps = num_sampling_steps
        self.timesteps = timesteps.to(device)
