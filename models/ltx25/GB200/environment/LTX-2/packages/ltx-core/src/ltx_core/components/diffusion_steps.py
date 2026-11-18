import torch

from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.utils import to_velocity


def _get_ancestral_step(
    sigma_from: torch.Tensor,
    sigma_to: torch.Tensor,
    eta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(sigma_down, sigma_up)`` for one DDIM ancestral sampling step.
    Both inputs are in the rescaled parameterization ``sigma / alpha``.
    Returns ``sigma_down`` (deterministic component) and ``sigma_up``
    (stochastic component) in the same rescaled space.
    """
    if not eta:
        return sigma_to, torch.zeros_like(sigma_to)
    variance = sigma_to**2 * (sigma_from**2 - sigma_to**2).clamp(min=0) / sigma_from**2
    sigma_up = (eta * variance**0.5).clamp(max=sigma_to)
    sigma_down = (sigma_to**2 - sigma_up**2).clamp(min=0) ** 0.5
    return sigma_down, sigma_up


class EulerDiffusionStep(DiffusionStepProtocol):
    """
    First-order Euler method for diffusion sampling.
    Takes a single step from the current noise level (sigma) to the next by
    computing velocity from the denoised prediction and applying: sample + velocity * dt.
    """

    def step(
        self, sample: torch.Tensor, denoised_sample: torch.Tensor, sigmas: torch.Tensor, step_index: int, **_kwargs
    ) -> torch.Tensor:
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma
        velocity = to_velocity(sample, sigma, denoised_sample)

        return (sample.to(torch.float32) + velocity.to(torch.float32) * dt).to(sample.dtype)


class Res2sDiffusionStep(DiffusionStepProtocol):
    """
    Second-order diffusion step for res_2s sampling with SDE noise injection.
    Used by the res_2s denoising loop. Advances the sample from the current
    sigma to the next by mixing a deterministic update (from the denoised
    prediction) with injected noise via ``get_sde_coeff``, producing
    variance-preserving transitions.
    """

    @staticmethod
    def get_sde_coeff(
        sigma_next: torch.Tensor,
        sigma_up: torch.Tensor | None = None,
        sigma_down: torch.Tensor | None = None,
        sigma_max: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute SDE coefficients (alpha_ratio, sigma_down, sigma_up) for the step.
        Given either ``sigma_down`` or ``sigma_up``, returns the mixing
        coefficients used for variance-preserving noise injection. If
        ``sigma_up`` is provided, ``sigma_down`` and ``alpha_ratio`` are
        derived; if ``sigma_down`` is provided, ``sigma_up`` and
        ``alpha_ratio`` are derived.
        """
        if sigma_down is not None:
            alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
            sigma_up = (sigma_next**2 - sigma_down**2 * alpha_ratio**2).clamp(min=0) ** 0.5
        elif sigma_up is not None:
            # Fallback to avoid sqrt(neg_num)
            sigma_up.clamp_(max=sigma_next * 0.9999)
            sigmax = sigma_max if sigma_max is not None else torch.ones_like(sigma_next)
            sigma_signal = sigmax - sigma_next
            sigma_residual = (sigma_next**2 - sigma_up**2).clamp(min=0) ** 0.5
            alpha_ratio = sigma_signal + sigma_residual
            sigma_down = sigma_residual / alpha_ratio
        else:
            alpha_ratio = torch.ones_like(sigma_next)
            sigma_down = sigma_next
            sigma_up = torch.zeros_like(sigma_next)

        sigma_up = torch.nan_to_num(sigma_up if sigma_up is not None else torch.zeros_like(sigma_next), 0.0)
        # Replace NaNs in sigma_down with corresponding sigma_next elements (float32)
        nan_mask = torch.isnan(sigma_down)
        sigma_down[nan_mask] = sigma_next[nan_mask].to(sigma_down.dtype)
        alpha_ratio = torch.nan_to_num(alpha_ratio, 1.0)

        return alpha_ratio, sigma_down, sigma_up

    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
        noise: torch.Tensor,
        eta: float = 0.5,
    ) -> torch.Tensor:
        """Advance one step with SDE noise injection via get_sde_coeff.
        Args:
            sample: Current noisy sample.
            denoised_sample: Denoised prediction from the model.
            sigmas: Noise schedule tensor.
            step_index: Current step index in the schedule.
            noise: Random noise tensor for stochastic injection.
            eta: Controls stochastic noise injection strength (0=deterministic, 1=maximum). Default 0.5.
        Returns:
            Next sample with SDE noise injection applied.
        """
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        alpha_ratio, sigma_down, sigma_up = self.get_sde_coeff(sigma_next, sigma_up=sigma_next * eta)
        output_dtype = denoised_sample.dtype
        if torch.any(sigma_up == 0) or torch.any(sigma_next == 0):
            return denoised_sample

        # Extract epsilon prediction
        eps_next = (sample - denoised_sample) / (sigma - sigma_next)
        denoised_next = sample - sigma * eps_next

        # Mix deterministic and stochastic components
        x_noised = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise
        return x_noised.to(output_dtype)


class EulerCfgPpDiffusionStep(DiffusionStepProtocol):
    """Euler step using the CFG++ correction for the ODE derivative.
    Instead of the standard velocity formula, the ODE derivative is computed
    from the unconditioned prediction, keeping the conditioned prediction as
    the target denoised state.  Ancestral (DDIM) noise injection is applied
    in the rescaled sigma parameterization (sigma / alpha).
    All diffusion quantities (alpha, ODE derivative, ancestral coefficients)
    are computed internally from ``sigmas`` and ``uncond_denoised``.
    Reference: CFG++ (https://arxiv.org/abs/2406.08070).
    """

    def __init__(self, eta: float = 1.0, s_noise: float = 1.0) -> None:
        self.eta = eta
        self.s_noise = s_noise

    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
        uncond_denoised: torch.Tensor,
        noise: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Advance one CFG++ Euler step.
        Args:
            sample: Current noisy latent x_t.
            denoised_sample: Conditioned denoised prediction x_0^cond.
            sigmas: Full sigma schedule tensor.
            step_index: Current step index.
            uncond_denoised: Unconditioned denoised prediction x_0^uncond,
                used to compute the ODE derivative direction.
            noise: Noise tensor for stochastic injection; ignored when
                ``eta=0`` or ``s_noise=0``.
        Returns:
            Updated latent x_{t-1}.
        """
        sigma_s = sigmas[step_index].to(torch.float32)
        sigma_t = sigmas[step_index + 1].to(torch.float32)
        _eps = torch.finfo(torch.float32).eps
        # Clamp to avoid division by zero when sigma == 1.0 exactly.
        alpha_s = (1.0 - sigma_s).clamp(min=_eps)
        alpha_t = (1.0 - sigma_t).clamp(min=_eps)

        x = sample.to(torch.float32)
        denoised = denoised_sample.to(torch.float32)
        uncond = uncond_denoised.to(torch.float32)

        # ODE derivative: direction toward noise using uncond prediction (CFG++ correction)
        d = (x - alpha_s * uncond) / sigma_s

        # Ancestral step in rescaled sigma space (sigma / alpha)
        sigma_down, sigma_up = _get_ancestral_step(sigma_s / alpha_s, sigma_t / alpha_t, eta=self.eta)
        sigma_down = alpha_t * sigma_down

        x_next = alpha_t * denoised + sigma_down * d
        if noise is not None and self.eta > 0 and self.s_noise > 0:
            x_next = x_next + alpha_t * noise.to(torch.float32) * self.s_noise * sigma_up
        return x_next.to(sample.dtype)
