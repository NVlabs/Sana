"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from typing import List, Tuple, Union

import torch

from ...seed import RandomState
from ..schedule import expand_dims
from . import BaseSampler


class ConsistencySampler(BaseSampler):
    def transition_kernel(
        self,
        pred: Union[List[torch.Tensor], torch.Tensor],
        x_t: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Union[List[torch.Tensor], torch.Tensor]]:
        if isinstance(pred, list):
            mean, std = zip(*[
                self.transition_kernel(pred[i], x_t[i], t[i:i+1], s[i:i+1])
                for i in range(len(pred))
            ])
            return list(mean), list(std)

        effective_s = self._effective_lower_boundary(s, t)

        with self.autocast():
            pred_x_0, _ = self.schedule.convert_from_pred(pred, x_t, t)
            mean = self.schedule.forward(pred_x_0, torch.zeros_like(pred_x_0), s)
            std = self.schedule.B(expand_dims(effective_s, x_t.ndim))
        return mean, std

    def step_to(
        self,
        pred: Union[List[torch.Tensor], torch.Tensor],
        x_t: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        s: torch.Tensor,
        rng: Union[List[RandomState], RandomState],
        *args,
        **kwargs
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Steps from x_t at timestep t to x_s at timestep s. Returns x_s.
        """
        if isinstance(pred, list):
            return [
                self.step_to(
                    pred[i], x_t[i], t[i:i+1], s[i:i+1],
                    rng[i] if isinstance(rng, list) else rng
                ) for i in range(len(pred))
            ]

        # Step from x_t to x_s.
        with self.autocast():
            pred_x_0, _ = self.schedule.convert_from_pred(pred, x_t, t)
            if isinstance(rng, list):
                bsz = pred.shape[0]
                assert len(rng) == bsz
                noises = torch.stack([
                    torch.randn(size=pred_x_0.shape[1:], device=pred_x_0.device, generator=rng[i].torch_cuda_generator)
                    for i in range(bsz)
                ])
            else:
                noises = torch.randn(size=pred_x_0.shape, device=pred_x_0.device, generator=rng.torch_cuda_generator)
            pred_x_s = self.schedule.forward(pred_x_0, noises, s).to(x_t.dtype)
        return pred_x_s

    def transition_score_grad_coeff(
        self,
        pred: Union[List[torch.Tensor], torch.Tensor],
        x_t: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        if isinstance(pred, list):
            return [
                self.transition_score_grad_coeff(pred[i], x_t[i], t[i:i+1], s[i:i+1])
                for i in range(len(pred))
            ]

        with self.autocast():
            s = expand_dims(s, x_t.ndim)
            coeff = self.schedule.A(s)
        return coeff.to(x_t.dtype)
