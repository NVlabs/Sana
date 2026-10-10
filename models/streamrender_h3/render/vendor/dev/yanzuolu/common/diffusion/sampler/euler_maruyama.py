"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from typing import List, Tuple, Union
import math

import torch

from ...seed import RandomState
from ..schedule import expand_dims
from . import BaseSampler


class EulerMaruyamaSampler(BaseSampler):
    def __init__(
        self,
        noise_level: float = math.sqrt(2.0),
        **kwargs
    ):
        super().__init__(**kwargs)
        # This sampler is written in a schedule-agnostic bridge form using A(t)
        # and B(t), while Flow-GRPO writes the same kernel as a v_lerp-specific
        # closed form. Under lerp + v_lerp, the two kernels should agree exactly.
        #
        # Externally, we keep noise_level on the same scale as Flow-GRPO. In the
        # generic bridge formulation below,
        #   diffusion_sq = 2 B (B' - (A'/A) B) * internal_noise_level^2.
        # For lerp this becomes
        #   diffusion_sq = 2 sigma / (1 - sigma) * internal_noise_level^2,
        # while Flow-GRPO uses
        #   diffusion_sq = sigma / (1 - sigma) * noise_level^2.
        # Therefore matching Flow-GRPO's external noise_level requires
        #   internal_noise_level = noise_level / sqrt(2).
        #
        # In the internal bridge scale, noise_level = 1 is the bridge-preserving
        # strength. Therefore the bridge-preserving default on the external
        # Flow-GRPO scale is noise_level = sqrt(2).
        self.noise_level = noise_level / math.sqrt(2.0)

    def _compute_reverse_kernel(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        effective_t = self._effective_upper_boundary(t, s)
        pred_x_0, pred_x_T = self.schedule.convert_from_pred(pred, x_t, t)

        # Let x_t = A(t) * x_0 + B(t) * x_T with x_T ~ N(0, I). The target bridge
        # marginals satisfy
        #   x_t | x_0 ~ N(A(t) * x_0, B(t)^2 I).
        #
        # A schedule-agnostic way to recover these marginals is to start from the
        # probability-flow ODE
        #   dx = (A'(t) * x_0 + B'(t) * x_T) dt
        # and then build a family of marginal-preserving SDEs that share the same
        # PF-ODE but differ in path-wise stochasticity. Writing the base bridge SDE
        # as
        #   dx = f(t) * x * dt + g_base(t) * dW_t,
        # matching the conditional mean and variance gives
        #   f(t) = A'(t) / A(t),
        #   g_base(t)^2 = d(B(t)^2)/dt - 2 f(t) B(t)^2
        #               = 2 B(t) (B'(t) - f(t) B(t)).
        #
        # For any scalar noise_level, we then define
        #   g(t)^2 = noise_level^2 * g_base(t)^2
        # and the corresponding forward SDE drift as
        #   u(t, x) = v_pf(t, x) + 0.5 * g(t)^2 * score_t(x),
        # where v_pf is the PF-ODE drift. This keeps the same marginals in the
        # continuous-time exact-score limit; changing noise_level only changes the
        # path measure, not the target marginal flow.
        #
        # The corresponding probability-flow ODE is
        #   dx = (A'(t) * x_0 + B'(t) * x_T) dt
        #      = (f(t) * x - 0.5 * g(t)^2 * score_t(x)) dt,
        # while the reverse-time SDE is
        #   dx = (A'(t) * x_0 + B'(t) * x_T - 0.5 * g(t)^2 * score_t(x)) dt
        #      + g(t) * dW_bar_t.
        # Here the conditional score induced by the bridge is
        #   score_t(x_t | x_0) = -(x_t - A(t) * x_0) / B(t)^2 = -x_T / B(t).
        #
        # We approximate A'(t) and B'(t) by finite differences over the current
        # interval [s, t], then apply Euler-Maruyama to obtain the Gaussian step
        # p(x_s | x_t, pred). Under lerp + v_lerp this generic form reduces to the
        # same mean/std update used by Flow-GRPO for the same external
        # noise_level. For nonlinear schedules this remains a generic EM
        # discretization of the corresponding reverse SDE rather than a
        # schedule-specific closed form.
        #
        # For lerp + v_lerp:
        #   A(t) = 1 - t, B(t) = t, v = x_T - x_0.
        # The schedule-induced PF-ODE is exactly dx/dt = v, while this sampler follows
        # a stochastic reverse SDE whose diffusion strength can be scaled by noise_level.
        # We expose noise_level using the Flow-GRPO convention; the default
        # noise_level = sqrt(2) maps to the bridge-preserving internal scale 1.

        t_expanded = expand_dims(t, x_t.ndim)
        s_expanded = expand_dims(s, x_t.ndim)

        A_t = self.schedule.A(t_expanded)
        A_s = self.schedule.A(s_expanded)
        B_t = self.schedule.B(t_expanded)
        B_s = self.schedule.B(s_expanded)
        A_t_regularized = self.schedule.A(expand_dims(effective_t, x_t.ndim))

        t_numeric = t_expanded.to(dtype=x_t.dtype)
        s_numeric = s_expanded.to(dtype=x_t.dtype)

        dt = (t_numeric - s_numeric).clamp_min(self.eps)

        # Finite-difference local derivatives over the current interval.
        dA_dt = (A_t - A_s) / dt
        dB_dt = (B_t - B_s) / dt
        drift_coeff = dA_dt / A_t_regularized.clamp_min(self.eps)

        # The conditional score induced by x_t = A_t x_0 + B_t x_T.
        score = -pred_x_T / B_t.clamp_min(self.eps)

        # Base bridge diffusion implied by A/B, rescaled by noise_level to pick
        # a member of the marginal-preserving SDE family described above.
        diffusion_sq = 2.0 * B_t * (dB_dt - drift_coeff * B_t)
        diffusion_sq = (diffusion_sq * (self.noise_level ** 2)).clamp(min=0.0)

        pf_ode_drift = dA_dt * pred_x_0 + dB_dt * pred_x_T
        reverse_mean = x_t + (s_numeric - t_numeric) * (
            pf_ode_drift - 0.5 * diffusion_sq * score
        )
        reverse_std = (diffusion_sq * dt).clamp(min=0.0).sqrt()
        return reverse_mean, reverse_std

    def transition_kernel(
        self,
        pred: Union[List[torch.Tensor], torch.Tensor],
        x_t: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Union[List[torch.Tensor], torch.Tensor]]:
        if isinstance(pred, list):
            mean, std = zip(*[
                self.transition_kernel(pred[i], x_t[i], t[i:i + 1], s[i:i + 1])
                for i in range(len(pred))
            ])
            return list(mean), list(std)

        with self.autocast():
            mean, std = self._compute_reverse_kernel(pred, x_t, t, s)
        return mean.to(x_t.dtype), std.to(x_t.dtype)

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
        Steps from x_t at timestep t to x_s at timestep s using Euler-Maruyama.
        """
        if isinstance(pred, list):
            return [
                self.step_to(
                    pred[i], x_t[i], t[i:i + 1], s[i:i + 1],
                    rng[i] if isinstance(rng, list) else rng
                ) for i in range(len(pred))
            ]

        with self.autocast():
            mean, std = self._compute_reverse_kernel(pred, x_t, t, s)
            if isinstance(rng, list):
                bsz = pred.shape[0]
                assert len(rng) == bsz
                noises = torch.stack([
                    torch.randn(
                        size=mean.shape[1:],
                        device=mean.device,
                        generator=rng[i].torch_cuda_generator,
                    )
                    for i in range(bsz)
                ])
            else:
                noises = torch.randn(
                    size=mean.shape,
                    device=mean.device,
                    generator=rng.torch_cuda_generator,
                )
            x_s = mean + std * noises
        return x_s.to(x_t.dtype)

    def transition_score_grad_coeff(
        self,
        pred: Union[List[torch.Tensor], torch.Tensor],
        x_t: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        if isinstance(pred, list):
            return [
                self.transition_score_grad_coeff(pred[i], x_t[i], t[i:i + 1], s[i:i + 1])
                for i in range(len(pred))
            ]

        effective_t = self._effective_upper_boundary(t, s)

        with self.autocast():
            t_expanded = expand_dims(t, x_t.ndim)
            s_expanded = expand_dims(s, x_t.ndim)

            A_t = self.schedule.A(t_expanded)
            A_s = self.schedule.A(s_expanded)
            B_t = self.schedule.B(t_expanded)
            B_s = self.schedule.B(s_expanded)
            A_t_regularized = self.schedule.A(expand_dims(effective_t, x_t.ndim))

            t_numeric = t_expanded.to(dtype=x_t.dtype)
            s_numeric = s_expanded.to(dtype=x_t.dtype)
            dt = (t_numeric - s_numeric).clamp_min(self.eps)

            dA_dt = (A_t - A_s) / dt
            dB_dt = (B_t - B_s) / dt
            drift_coeff = dA_dt / A_t_regularized.clamp_min(self.eps)
            diffusion_sq = 2.0 * B_t * (dB_dt - drift_coeff * B_t)
            diffusion_sq = (diffusion_sq * (self.noise_level ** 2)).clamp(min=0.0)
            reverse_dt = s_numeric - t_numeric

            coeff = reverse_dt * (
                dA_dt
                - A_t / B_t.clamp_min(self.eps)
                * (dB_dt + 0.5 * diffusion_sq / B_t.clamp_min(self.eps))
            )
        return coeff.to(x_t.dtype)
