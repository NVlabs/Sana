"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch

from ...seed import RandomState
from ..schedule import BaseSchedule


class BaseSampler(ABC):
    schedule: BaseSchedule

    def __init__(
        self,
        schedule: BaseSchedule,
        autocast: dict = {},
        eps: float = 1e-6,
        sigma_max: float = 1.0,
        sigma_min: float = 0.0,
    ):
        self.schedule = schedule
        self.eps = eps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self._autocast = dict(autocast)

    def autocast(self) -> torch.autocast:
        """Autocast context for sampler math, honoring the ``autocast`` config node."""
        return torch.autocast(
            device_type="cuda",
            dtype=getattr(torch, self._autocast.get("dtype", "bfloat16")),
            enabled=self._autocast.get("enabled", False),
            cache_enabled=self._autocast.get("cache_enabled", True),
        )

    def _sigma_denominator(self) -> float:
        if isinstance(self.schedule.T, float):
            return float(self.schedule.T)
        return float(max(self.schedule.T, 1))

    def _sigma_to_schedule_value(self, sigma: float, ref: torch.Tensor) -> torch.Tensor:
        denom = self._sigma_denominator()
        value = sigma * denom
        if isinstance(self.schedule.T, float):
            return torch.full_like(ref, fill_value=value, dtype=ref.dtype, device=ref.device)
        return torch.full_like(ref, fill_value=int(round(value)), dtype=ref.dtype, device=ref.device)

    def _normalized_sigma(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(device=t.device, dtype=torch.float32) / self._sigma_denominator()

    def _replace_upper_boundary(self, t: torch.Tensor) -> torch.Tensor:
        mask = self._normalized_sigma(t) >= (1.0 - self.eps)
        replacement = self._sigma_to_schedule_value(self.sigma_max, t)
        return torch.where(mask, replacement, t)

    def _replace_lower_boundary(self, t: torch.Tensor) -> torch.Tensor:
        mask = self._normalized_sigma(t) <= self.eps
        replacement = self._sigma_to_schedule_value(self.sigma_min, t)
        return torch.where(mask, replacement, t)

    def _effective_upper_boundary(
        self,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        effective_t = self._replace_upper_boundary(t)
        # Regularize the singular formula at the upper boundary without
        # changing the actual transition direction.
        return torch.maximum(effective_t, s)

    def _effective_lower_boundary(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        effective_s = self._replace_lower_boundary(s)
        # Regularize the singular formula at the lower boundary without
        # changing the actual transition direction.
        return torch.minimum(effective_s, t)

    @abstractmethod
    def step_to(
        self,
        pred: Union[List[torch.Tensor], torch.Tensor],
        x_t: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        s: torch.Tensor,
        rng: Union[List[RandomState], RandomState],
        seqlens: torch.Tensor
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Steps from x_t at timestep t to x_s at timestep s. Returns x_s.
        """
        raise NotImplementedError

    @abstractmethod
    def transition_kernel(
        self,
        pred: Union[List[torch.Tensor], torch.Tensor],
        x_t: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Union[List[torch.Tensor], torch.Tensor]]:
        """
        Returns the transition kernel p(x_s | x_t, pred) as Gaussian mean/std for
        stochastic samplers. Deterministic samplers should raise NotImplementedError.
        """
        raise NotImplementedError

    @abstractmethod
    def transition_score_grad_coeff(
        self,
        pred: Union[List[torch.Tensor], torch.Tensor],
        x_t: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        raise NotImplementedError
