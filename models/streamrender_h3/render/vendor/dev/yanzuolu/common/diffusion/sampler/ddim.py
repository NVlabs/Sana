"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from typing import List, Tuple, Union

import torch

from . import BaseSampler


class DDIMSampler(BaseSampler):
    def step_to(
        self,
        pred: Union[List[torch.Tensor], torch.Tensor],
        x_t: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        s: torch.Tensor,
        *args,
        **kwargs
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Steps from x_t at timestep t to x_s at timestep s. Returns x_s.
        """
        if isinstance(pred, list):
            return [self.step_to(pred[i], x_t[i], t[i:i+1], s[i:i+1]) for i in range(len(pred))]

        # Step from x_t to x_s.
        with self.autocast():
            pred_x_0, pred_x_T = self.schedule.convert_from_pred(pred, x_t, t)
            pred_x_s = self.schedule.forward(pred_x_0, pred_x_T, s).to(x_t.dtype)
        return pred_x_s

    def transition_kernel(
        self,
        pred: Union[List[torch.Tensor], torch.Tensor],
        x_t: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Union[List[torch.Tensor], torch.Tensor]]:
        raise NotImplementedError("DDIMSampler is deterministic and does not define a stochastic transition kernel.")

    def transition_score_grad_coeff(
        self,
        pred: Union[List[torch.Tensor], torch.Tensor],
        x_t: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        raise NotImplementedError("DDIMSampler is deterministic and does not define a stochastic transition score gradient coefficient.")
