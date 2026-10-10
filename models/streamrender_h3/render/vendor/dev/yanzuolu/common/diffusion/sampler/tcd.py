"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from typing import List, Tuple, Union

import torch

from ...seed import RandomState
from ..schedule import expand_dims
from . import BaseSampler


class TCDSampler(BaseSampler):
    def __init__(
        self,
        eta: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.eta = eta

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

        e = (1 - self.eta) * s
        e = e.to(dtype=s.dtype)

        with self.autocast():
            pred_x_0, pred_x_T = self.schedule.convert_from_pred(pred, x_t, t)
            pred_x_e = self.schedule.forward(pred_x_0, pred_x_T, e)

            s_expanded = expand_dims(s, x_t.ndim)
            e_expanded = expand_dims(e, x_t.ndim)
            A_s = self.schedule.A(s_expanded)
            A_e = self.schedule.A(e_expanded)
            B_s = self.schedule.B(s_expanded)
            B_e = self.schedule.B(e_expanded)
            scale = A_s / A_e
            std = (B_s.pow(2) - scale.pow(2) * B_e.pow(2)).clamp(min=0).sqrt()
            mean = scale * pred_x_e
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

        e = (1 - self.eta) * s
        e = e.to(dtype=s.dtype)

        # Step from x_t to x_s.
        with self.autocast():
            pred_x_0, pred_x_T = self.schedule.convert_from_pred(pred, x_t, t)
            pred_x_e = self.schedule.forward(pred_x_0, pred_x_T, e)
            if isinstance(rng, list):
                bsz = pred.shape[0]
                assert len(rng) == bsz
                noises = torch.stack([
                    torch.randn(size=pred_x_0.shape[1:], device=pred_x_0.device, generator=rng[i].torch_cuda_generator)
                    for i in range(bsz)
                ])
            else:
                noises = torch.randn(size=pred_x_0.shape, device=pred_x_0.device, generator=rng.torch_cuda_generator)
            pred_x_s = self.schedule.forward_from_prev(pred_x_e, noises, t=s, s=e).to(x_t.dtype)
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

        effective_t = self._replace_lower_boundary(t)
        e = (1 - self.eta) * s
        e = e.to(dtype=s.dtype)

        with self.autocast():
            t = expand_dims(effective_t, x_t.ndim)
            s = expand_dims(s, x_t.ndim)
            e = expand_dims(e, x_t.ndim)
            coeff = self.schedule.A(s) * (
                1 - self.schedule.B(e) * self.schedule.A(t)
                / (
                    self.schedule.A(e).clamp_min(self.eps)
                    * self.schedule.B(t).clamp_min(self.eps)
                )
            )
        return coeff.to(x_t.dtype)
