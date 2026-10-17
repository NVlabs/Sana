"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from enum import Enum

import torch

from . import BaseSchedule


class BetaScheduleType(str, Enum):
    """
    Mainly refer to diffusers DDIMScheduler.
    """
    from diffusers import DDIMScheduler

    linear = "linear"
    scaled_linear = "scaled_linear"


class DiscreteVariancePreservingSchedule(BaseSchedule):
    """
    Discrete variance preserving schedule (vp) is originally proposed in DDPM.
    It is also widely used by Stable Diffusion.

        x_t = sqrt(alphas_cumprod[t]) * x_0 + sqrt(1 - alphas_cumprod[t]) * x_T

    Can be used as discrete only.
    """

    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        beta_schedule: BetaScheduleType,
        betas: torch.Tensor = None,
        set_alpha_to_one: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        if betas is not None:
            self.betas = torch.tensor(betas, dtype=torch.float32)
        elif beta_schedule == BetaScheduleType.linear:
            self.betas = torch.linspace(beta_start, beta_end, self.T, dtype=torch.float32)
        elif beta_schedule == BetaScheduleType.scaled_linear:
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, self.T, dtype=torch.float32) ** 2
        else:
            raise NotImplementedError

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        if set_alpha_to_one:
            self.alphas_cumprod[0] = 1.0

    def A(self, t: torch.Tensor) -> torch.Tensor:
        self.alphas_cumprod = self.alphas_cumprod.to(device=t.device)
        return self.alphas_cumprod[t] ** 0.5

    def B(self, t: torch.Tensor) -> torch.Tensor:
        self.alphas_cumprod = self.alphas_cumprod.to(device=t.device)
        return (1 - self.alphas_cumprod[t]) ** 0.5
