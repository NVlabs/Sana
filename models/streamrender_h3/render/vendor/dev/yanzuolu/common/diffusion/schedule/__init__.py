"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from enum import Enum
from typing import List, Tuple, Union

import torch


def expand_dims(tensor: torch.Tensor, ndim: int):
    """
    Expand tensor to target ndim. New dims are added to the right.
    For example, if the tensor shape was (8,), target ndim is 4, return (8, 1, 1, 1).
    """
    shape = tensor.shape + (1,) * (ndim - tensor.ndim)
    return tensor.reshape(shape)


class PredictionType(str, Enum):
    """
    x_0:
        Predict data sample.
    x_T:
        Predict noise sample.
        Proposed by DDPM (https://arxiv.org/abs/2006.11239)
        Proved problematic by zsnr paper (https://arxiv.org/abs/2305.08891)
    v_cos:
        Predict velocity dx/dt based on the cosine schedule (A_t * x_T - B_t * x_0).
        Proposed by progressive distillation (https://arxiv.org/abs/2202.00512)
    v_lerp:
        Predict velocity dx/dt based on the lerp schedule (x_T - x_0).
        Proposed by rectified flow (https://arxiv.org/abs/2209.03003)
    score:
        Predict score d log p(x_t) / d x_t under the conditional Gaussian induced by x_0 and x_T.
    """

    x_0 = "x_0"
    x_T = "x_T"
    v_cos = "v_cos"
    v_lerp = "v_lerp"
    score = "score"


class BaseSchedule:
    """
    Diffusion schedules are uniquely defined by T, A, B:

        x_t = A(t) * x_0 + B(t) * x_T, where t in [0, T]

    Schedules can be continuous or discrete.
    """

    def __init__(
        self,
        T: Union[int, float],
        pred_type: PredictionType,
        **kwargs
    ):
        self.T = T
        self.pred_type = pred_type

    def A(self, t: torch.Tensor) -> torch.Tensor:
        """
        Interpolation coefficient A.
        Returns tensor with the same shape as t.
        """
        raise NotImplementedError

    def B(self, t: torch.Tensor) -> torch.Tensor:
        """
        Interpolation coefficient B.
        Returns tensor with the same shape as t.
        """
        raise NotImplementedError

    def forward(
        self,
        x_0: Union[List[torch.Tensor], torch.Tensor],
        x_T: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Diffusion forward function.
        """
        if isinstance(x_0, list):
            return [self.forward(x_0[i], x_T[i], t[i:i+1]) for i in range(len(x_0))]
        else:
            t = expand_dims(t, x_0.ndim)
            return self.A(t) * x_0 + self.B(t) * x_T

    def forward_from_prev(
        self,
        x_prev: Union[List[torch.Tensor], torch.Tensor],
        noise: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        s: torch.Tensor
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Forward function from previous step. For example, step from x_s to x_t.
        """
        if isinstance(x_prev, list):
            return [self.forward_from_prev(x_prev[i], noise[i], t[i:i+1], s[i:i+1]) for i in range(len(x_prev))]
        else:
            t = expand_dims(t, x_prev.ndim)
            s = expand_dims(s, x_prev.ndim)
            A_t = self.A(t)
            A_s = self.A(s)
            B_t = self.B(t)
            B_s = self.B(s)
            # λ = A(t) / A(s)
            scale = A_t / A_s

            # σ = sqrt(B(t)^2 - λ^2 * B(s)^2)
            noise_scale = (B_t.pow(2) - scale.pow(2) * B_s.pow(2)).clamp(min=0).sqrt()
            return scale * x_prev + noise_scale * noise

    def convert_from_pred(
        self,
        pred: Union[List[torch.Tensor], torch.Tensor],
        x_t: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        pred_type: PredictionType = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert from prediction. Return predicted x_0 and x_T.
        """
        if isinstance(x_t, list):
            pred_x0s, pred_xTs = [], []
            for i in range(len(x_t)):
                pred_x_0, pred_x_T = self.convert_from_pred(pred[i], x_t[i], t[i:i+1], pred_type)
                pred_x0s.append(pred_x_0)
                pred_xTs.append(pred_x_T)
            return pred_x0s, pred_xTs

        if pred_type is None:
            pred_type = self.pred_type

        t = expand_dims(t, x_t.ndim)
        A_t = self.A(t)
        B_t = self.B(t)

        if pred_type == PredictionType.x_T:
            pred_x_T = pred
            pred_x_0 = (x_t - B_t * pred_x_T) / A_t
        elif pred_type == PredictionType.x_0:
            pred_x_0 = pred
            pred_x_T = (x_t - A_t * pred_x_0) / B_t
        elif pred_type == PredictionType.v_cos:
            assert torch.allclose(A_t**2 + B_t**2, torch.ones_like(A_t), atol=1e-4), \
                "PredictionType.v_cos requires a Variance Preserving schedule (A^2 + B^2 == 1)"
            pred_x_0 = A_t * x_t - B_t * pred
            pred_x_T = A_t * pred + B_t * x_t
        elif pred_type == PredictionType.v_lerp:
            pred_x_0 = (x_t - B_t * pred) / (A_t + B_t)
            pred_x_T = (x_t + A_t * pred) / (A_t + B_t)
        else:
            raise NotImplementedError

        return pred_x_0, pred_x_T

    def convert_to_pred(
        self,
        x_0: Union[List[torch.Tensor], torch.Tensor],
        x_T: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        pred_type: PredictionType = None
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Convert to prediction target given x_0 and x_T.
        """
        if isinstance(x_0, list):
            return [self.convert_to_pred(x_0[i], x_T[i], t[i:i+1], pred_type) for i in range(len(x_0))]

        if pred_type is None:
            pred_type = self.pred_type

        if pred_type == PredictionType.x_T:
            return x_T
        if pred_type == PredictionType.x_0:
            return x_0
        if pred_type == PredictionType.v_cos:
            t = expand_dims(t, x_0.ndim)
            A_t = self.A(t)
            B_t = self.B(t)
            assert torch.allclose(A_t**2 + B_t**2, torch.ones_like(A_t), atol=1e-4), \
                "PredictionType.v_cos target requires a Variance Preserving schedule (A^2 + B^2 == 1)"
            return self.A(t) * x_T - self.B(t) * x_0
        if pred_type == PredictionType.v_lerp:
            return x_T - x_0
        if pred_type == PredictionType.score:
            t = expand_dims(t, x_0.ndim)
            B_t = self.B(t)
            return -x_T / B_t
        raise NotImplementedError
