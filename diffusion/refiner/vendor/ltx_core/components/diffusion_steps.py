import torch

from diffusion.refiner.vendor.ltx_core.components.protocols import DiffusionStepProtocol
from diffusion.refiner.vendor.ltx_core.utils import to_velocity


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
