import logging
from dataclasses import dataclass, replace
from functools import partial
from typing import Callable

import torch
from tqdm import tqdm

from ltx_core.components.diffusion_steps import (
    EulerAncestralDiffusionStep,
    EulerCfgPpDiffusionStep,
    Res2sDiffusionStep,
)
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.devices import highest_precision_float
from ltx_core.model.transformer import X0Model
from ltx_core.utils import to_denoised, to_velocity
from ltx_pipelines.utils.helpers import post_process_latent, timesteps_from_mask
from ltx_pipelines.utils.res2s import get_res2s_coefficients
from ltx_pipelines.utils.types import DenoisedLatentResult, Denoiser, LatentState

logger = logging.getLogger(__name__)


def _step_state(
    state: LatentState | None,
    denoised: torch.Tensor | None,
    stepper: DiffusionStepProtocol,
    sigmas: torch.Tensor,
    step_idx: int,
) -> LatentState | None:
    """Advance one diffusion step for a single modality, or return ``None`` if absent."""
    if state is None or denoised is None:
        return state
    denoised = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
    return replace(state, latent=stepper.step(state.latent, denoised, sigmas, step_idx))


def euler_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    stepper: DiffusionStepProtocol,
    transformer: X0Model,
    denoiser: Denoiser,
) -> tuple[LatentState | None, LatentState | None]:
    """
    Perform the joint audio-video denoising loop over a diffusion schedule.
    Either ``video_state`` or ``audio_state`` may be ``None`` for absent
    modalities; the absent modality is passed through unchanged.
    ### Parameters
    sigmas:
        A 1D tensor of noise levels (diffusion sigmas) defining the sampling
        schedule. All steps except the last element are iterated over.
    video_state:
        The current video :class:`LatentState`, or ``None`` if video is absent.
    audio_state:
        The current audio :class:`LatentState`, or ``None`` if audio is absent.
    stepper:
        An implementation of :class:`DiffusionStepProtocol` that updates a
        latent given the current latent, its denoised estimate, the full
        ``sigmas`` schedule, and the current step index.
    transformer:
        The diffusion model passed to the denoiser at each step.
    denoiser:
        A callable implementing :class:`Denoiser`. It is invoked as
        ``denoiser(transformer, video_state, audio_state, sigmas, step_index)``
        and must return a :class:`~ltx_pipelines.utils.types.DenoisedLatentResult`.
    ### Returns
    tuple[LatentState | None, LatentState | None]
        Final ``(video_state, audio_state)`` after the denoising loop.
    """
    for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
        video_result, audio_result = denoiser(transformer, video_state, audio_state, sigmas, step_idx)
        denoised_video = video_result.denoised if video_result is not None else None
        denoised_audio = audio_result.denoised if audio_result is not None else None

        video_state = _step_state(video_state, denoised_video, stepper, sigmas, step_idx)
        audio_state = _step_state(audio_state, denoised_audio, stepper, sigmas, step_idx)

    return (video_state, audio_state)


def gradient_estimating_euler_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    stepper: DiffusionStepProtocol,
    transformer: X0Model,
    denoiser: Denoiser,
    ge_gamma: float = 2.0,
) -> tuple[LatentState | None, LatentState | None]:
    """
    Perform the joint audio-video denoising loop using gradient-estimation sampling.
    Same interface as :func:`euler_denoising_loop` with an additional
    ``ge_gamma`` parameter for velocity correction.
    ### Parameters
    ge_gamma:
        Gradient estimation coefficient controlling the velocity correction term.
        Default is 2.0. Paper: https://openreview.net/pdf?id=o2ND9v0CeK
    ### Returns
    tuple[LatentState | None, LatentState | None]
        See :func:`euler_denoising_loop` for return value description.
    """

    previous_audio_velocity = None
    previous_video_velocity = None

    def update_velocity_and_sample(
        noisy_sample: torch.Tensor, denoised_sample: torch.Tensor, sigma: float, previous_velocity: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_velocity = to_velocity(noisy_sample, sigma, denoised_sample)
        if previous_velocity is not None:
            delta_v = current_velocity - previous_velocity
            total_velocity = ge_gamma * delta_v + previous_velocity
            denoised_sample = to_denoised(noisy_sample, total_velocity, sigma)
        return current_velocity, denoised_sample

    for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
        video_result, audio_result = denoiser(transformer, video_state, audio_state, sigmas, step_idx)
        denoised_video = video_result.denoised if video_result is not None else None
        denoised_audio = audio_result.denoised if audio_result is not None else None

        if video_state is not None and denoised_video is not None:
            denoised_video = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
        if audio_state is not None and denoised_audio is not None:
            denoised_audio = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)

        if sigmas[step_idx + 1] == 0:
            if video_state is not None and denoised_video is not None:
                video_state = replace(video_state, latent=denoised_video)
            if audio_state is not None and denoised_audio is not None:
                audio_state = replace(audio_state, latent=denoised_audio)
            return video_state, audio_state

        if video_state is not None and denoised_video is not None:
            previous_video_velocity, denoised_video = update_velocity_and_sample(
                video_state.latent, denoised_video, sigmas[step_idx], previous_video_velocity
            )
            video_state = replace(
                video_state, latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx)
            )

        if audio_state is not None and denoised_audio is not None:
            previous_audio_velocity, denoised_audio = update_velocity_and_sample(
                audio_state.latent, denoised_audio, sigmas[step_idx], previous_audio_velocity
            )
            audio_state = replace(
                audio_state, latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx)
            )

    return (video_state, audio_state)


def _get_plain_noise(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Draw standard Gaussian noise matching the shape, dtype, and device of ``x``."""
    return torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)


def _channelwise_normalize(x: torch.Tensor) -> torch.Tensor:
    return x.sub_(x.mean(dim=(-2, -1), keepdim=True)).div_(x.std(dim=(-2, -1), keepdim=True))


def _get_new_noise(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    # float64 on CUDA/CPU for numerical stability; MPS has no float64, so degrade to float32.
    noise = torch.randn(
        x.shape, generator=generator, dtype=highest_precision_float(generator.device), device=generator.device
    )
    noise = (noise - noise.mean()) / noise.std()
    return _channelwise_normalize(noise)


def _inject_sde_noise(
    state: LatentState,
    sample: torch.Tensor,
    denoised_sample: torch.Tensor,
    step_noise_generator: torch.Generator,
    new_noise_fn: Callable[[torch.Tensor, torch.Generator], torch.Tensor],
    stepper: DiffusionStepProtocol,
    sigmas: torch.Tensor,
    step_idx: int,
    legacy_mode: bool = False,
    eta: float = 0.5,
) -> torch.Tensor:
    sigmas_copy = sigmas.clone()
    hp = highest_precision_float(state.denoise_mask.device)
    new_noise = new_noise_fn(state.latent, step_noise_generator)
    if not legacy_mode:
        timesteps = timesteps_from_mask(state.denoise_mask.to(hp), sigmas_copy[step_idx].to(hp))
        next_timesteps = timesteps_from_mask(state.denoise_mask.to(hp), sigmas_copy[step_idx + 1].to(hp))
        sigmas = torch.stack([timesteps, next_timesteps])
        step_idx = 0
    x_next = stepper.step(
        sample=sample,
        denoised_sample=denoised_sample,
        sigmas=sigmas,
        step_index=step_idx,
        noise=new_noise,
        eta=eta,
    )

    if legacy_mode:
        x_next = post_process_latent(x_next, state.denoise_mask, state.clean_latent)

    return x_next


def res2s_audio_video_denoising_loop(  # noqa: PLR0913,PLR0915,PLR0912
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    stepper: DiffusionStepProtocol,
    transformer: X0Model,
    denoiser: Denoiser,
    noise_seed: int = -1,
    noise_seed_substep: int | None = None,
    eta: float = 0.5,
    bongmath: bool = True,
    bongmath_max_iter: int = 100,
    new_noise_fn: Callable[[torch.Tensor, torch.Generator], torch.Tensor] = _get_new_noise,
    model_dtype: torch.dtype = torch.bfloat16,
    legacy_mode: bool = True,
) -> tuple[LatentState | None, LatentState | None]:
    """
    Joint audio-video denoising loop using the res_2s second-order sampler.
    Iterates over the diffusion schedule with a two-stage Runge-Kutta step:
    evaluates the denoiser at the current point and at a midpoint (with SDE
    noise), then combines both with RK coefficients. Supports anchor-point
    refinement (bong iteration) and optional SDE noise injection. Requires
    :class:`Res2sDiffusionStep` as ``stepper``.
    Either modality may be ``None`` (absent).
    ### Parameters
    transformer:
        The diffusion model passed to the denoiser at each step.
    denoiser:
        Callable implementing :class:`Denoiser`.
    noise_seed:
        Seed for step-level SDE noise; substep seed defaults to ``noise_seed + 10000``.
    noise_seed_substep:
        Optional seed for substep SDE noise; if None, derived from ``noise_seed``.
    eta:
        Controls stochastic noise injection strength (0=deterministic, 1=maximum).
        Applies to main diffusion steps; substeps always use 0.5. Default 0.5.
    bongmath:
        Whether to run iterative anchor refinement (bong iteration) when step size is small.
    bongmath_max_iter:
        Max iterations for bong refinement when enabled.
    new_noise_fn:
        Callable ``(latent, generator) -> noise`` for SDE injection.
    model_dtype:
        Dtype for latent state updates (e.g. bfloat16).
    ### Returns
    tuple[LatentState | None, LatentState | None]
        Final ``(video_state, audio_state)`` after the denoising loop.
    """
    # Determine device from whichever state is present
    present_state = video_state or audio_state
    if present_state is None:
        raise ValueError("At least one of video_state or audio_state must be provided")
    state_device = present_state.latent.device
    # float64 on CUDA/CPU for ODE numerical stability; MPS has no float64, so degrade to float32.
    hp = highest_precision_float(state_device)

    # Initialize noise generators with different seeds
    if noise_seed_substep is None:
        noise_seed_substep = noise_seed + 10000  # Offset to ensure different seeds
    step_noise_generator = torch.Generator(device=state_device).manual_seed(noise_seed)
    substep_noise_generator = torch.Generator(device=state_device).manual_seed(noise_seed_substep)
    sde_noise_injecting_fn = partial(
        _inject_sde_noise, stepper=stepper, new_noise_fn=new_noise_fn, legacy_mode=legacy_mode
    )
    step_noise_injecting_fn = partial(sde_noise_injecting_fn, step_noise_generator=step_noise_generator, eta=eta)
    # substep eta is always default 0.5 for compatibility with original implementation.
    substep_noise_injecting_fn = partial(sde_noise_injecting_fn, step_noise_generator=substep_noise_generator, eta=0.5)

    if not isinstance(stepper, Res2sDiffusionStep):
        raise ValueError("stepper must be an instance of Res2sDiffusionStep")

    n_full_steps = len(sigmas) - 1
    # inject minimal sigma value to avoid division by zero
    if sigmas[-1] == 0:
        sigmas = torch.cat([sigmas[:-1], torch.tensor([0.0011, 0.0], device=sigmas.device)], dim=0)
    # Compute step sizes in hyperbolic space
    hs = -torch.log(sigmas[1:].to(hp).cpu() / (sigmas[:-1].to(hp).cpu()))

    # Initialize phi cache for reuse across loop iterations
    phi_cache = {}
    c2 = 0.5  # Midpoint for res_2s

    for step_idx in tqdm(range(n_full_steps)):
        sigma = sigmas[step_idx].to(hp)
        sigma_next = sigmas[step_idx + 1].to(hp)

        # Initialize anchor point
        x_anchor_video = video_state.latent.clone().to(hp) if video_state is not None else None
        x_anchor_audio = audio_state.latent.clone().to(hp) if audio_state is not None else None

        # ====================================================================
        # STAGE 1: Evaluate at current point
        # ====================================================================
        video_result, audio_result = denoiser(transformer, video_state, audio_state, sigmas, step_idx)
        denoised_video_1 = video_result.denoised if video_result is not None else None
        denoised_audio_1 = audio_result.denoised if audio_result is not None else None
        if video_state is not None and denoised_video_1 is not None:
            denoised_video_1 = post_process_latent(denoised_video_1, video_state.denoise_mask, video_state.clean_latent)
        if audio_state is not None and denoised_audio_1 is not None:
            denoised_audio_1 = post_process_latent(denoised_audio_1, audio_state.denoise_mask, audio_state.clean_latent)

        h = hs[step_idx].item()

        # Compute RK coefficients (pass phi_cache for caching)
        a21, b1, b2 = get_res2s_coefficients(h, phi_cache, c2)

        # Compute substep sigma, sqrt is a hardcode for c2 = 0.5
        sub_sigma = torch.sqrt(sigma * sigma_next)

        # ====================================================================
        # Compute substep x using RK coefficient a21
        # ====================================================================
        if x_anchor_video is not None and denoised_video_1 is not None:
            eps_1_video = denoised_video_1.to(hp) - x_anchor_video
            x_mid_video = x_anchor_video.to(hp) + h * a21 * eps_1_video
        else:
            eps_1_video = None
            x_mid_video = None

        if x_anchor_audio is not None and denoised_audio_1 is not None:
            eps_1_audio = denoised_audio_1.to(hp) - x_anchor_audio
            x_mid_audio = x_anchor_audio.to(hp) + h * a21 * eps_1_audio
        else:
            eps_1_audio = None
            x_mid_audio = None

        # ====================================================================
        # SDE noise injection at substep
        # ====================================================================
        if x_mid_video is not None and video_state is not None:
            x_mid_video = substep_noise_injecting_fn(
                state=video_state,
                sample=x_anchor_video,
                denoised_sample=x_mid_video,
                sigmas=torch.stack([sigma, sub_sigma]),
                step_idx=0,
            )
        if x_mid_audio is not None and audio_state is not None:
            x_mid_audio = substep_noise_injecting_fn(
                state=audio_state,
                sample=x_anchor_audio,
                denoised_sample=x_mid_audio,
                sigmas=torch.stack([sigma, sub_sigma]),
                step_idx=0,
            )

        # ====================================================================
        # ITERATIVE REFINEMENT (Bong Iteration)
        # ====================================================================
        if bongmath and h < 0.5 and sigma > 0.03:
            for _ in range(bongmath_max_iter):
                if x_mid_video is not None and eps_1_video is not None:
                    x_anchor_video = x_mid_video - h * a21 * eps_1_video
                    eps_1_video = denoised_video_1.to(hp) - x_anchor_video
                if x_mid_audio is not None and eps_1_audio is not None:
                    x_anchor_audio = x_mid_audio - h * a21 * eps_1_audio
                    eps_1_audio = denoised_audio_1.to(hp) - x_anchor_audio

        # ====================================================================
        # STAGE 2: Evaluate at substep point (WITH NOISE)
        # ====================================================================
        mid_video_state = (
            replace(video_state, latent=x_mid_video.to(model_dtype))
            if video_state is not None and x_mid_video is not None
            else None
        )
        mid_audio_state = (
            replace(audio_state, latent=x_mid_audio.to(model_dtype))
            if audio_state is not None and x_mid_audio is not None
            else None
        )

        video_result_2, audio_result_2 = denoiser(
            transformer,
            video_state=mid_video_state,
            audio_state=mid_audio_state,
            sigmas=torch.stack([sub_sigma]).to(sigmas.device),
            step_index=0,
        )
        denoised_video_2 = video_result_2.denoised if video_result_2 is not None else None
        denoised_audio_2 = audio_result_2.denoised if audio_result_2 is not None else None
        if video_state is not None and denoised_video_2 is not None:
            denoised_video_2 = post_process_latent(denoised_video_2, video_state.denoise_mask, video_state.clean_latent)
        if audio_state is not None and denoised_audio_2 is not None:
            denoised_audio_2 = post_process_latent(denoised_audio_2, audio_state.denoise_mask, audio_state.clean_latent)

        # ====================================================================
        # FINAL COMBINATION: Compute x_next using RK coefficients
        # ====================================================================
        if x_anchor_video is not None and eps_1_video is not None and denoised_video_2 is not None:
            eps_2_video = denoised_video_2.to(hp) - x_anchor_video
            x_next_video = x_anchor_video + h * (b1 * eps_1_video + b2 * eps_2_video)
        else:
            x_next_video = None

        if x_anchor_audio is not None and eps_1_audio is not None and denoised_audio_2 is not None:
            eps_2_audio = denoised_audio_2.to(hp) - x_anchor_audio
            x_next_audio = x_anchor_audio + h * (b1 * eps_1_audio + b2 * eps_2_audio)
        else:
            x_next_audio = None

        # ====================================================================
        # SDE NOISE INJECTION AT STEP LEVEL
        # ====================================================================
        if x_next_video is not None and video_state is not None:
            x_next_video = step_noise_injecting_fn(
                state=video_state,
                sample=x_anchor_video,
                denoised_sample=x_next_video,
                sigmas=sigmas,
                step_idx=step_idx,
            )
        if x_next_audio is not None and audio_state is not None:
            x_next_audio = step_noise_injecting_fn(
                state=audio_state,
                sample=x_anchor_audio,
                denoised_sample=x_next_audio,
                sigmas=sigmas,
                step_idx=step_idx,
            )

        # Update states
        if video_state is not None and x_next_video is not None:
            video_state = replace(video_state, latent=x_next_video.to(model_dtype))
        if audio_state is not None and x_next_audio is not None:
            audio_state = replace(audio_state, latent=x_next_audio.to(model_dtype))

    # Final step if we need to fully remove the noise
    if sigmas[-1] == 0:
        video_result_final, audio_result_final = denoiser(transformer, video_state, audio_state, sigmas, n_full_steps)
        denoised_video_1 = video_result_final.denoised if video_result_final is not None else None
        denoised_audio_1 = audio_result_final.denoised if audio_result_final is not None else None
        if video_state is not None and denoised_video_1 is not None:
            denoised_video_1 = post_process_latent(denoised_video_1, video_state.denoise_mask, video_state.clean_latent)
            video_state = replace(video_state, latent=denoised_video_1.to(model_dtype))
        if audio_state is not None and denoised_audio_1 is not None:
            denoised_audio_1 = post_process_latent(denoised_audio_1, audio_state.denoise_mask, audio_state.clean_latent)
            audio_state = replace(audio_state, latent=denoised_audio_1.to(model_dtype))

    return video_state, audio_state


@dataclass(frozen=True)
class _ModalityStep:
    """One modality's resolved inputs for a single ancestral-Euler step.
    Attributes:
        modality: ``"video"`` or ``"audio"``.
        state: The modality's current latent state.
        step_fn: The step callable for this modality, already bound to any
            stepper-specific inputs.
        denoised: Mask-corrected x_0 prediction in float32, or ``None`` when the
            denoiser produced no prediction for this modality (in which case the
            modality is passed through unchanged this step).
    """

    modality: str
    state: LatentState
    step_fn: Callable[..., torch.Tensor]
    denoised: torch.Tensor | None

    @classmethod
    def from_modality_result(
        cls,
        modality: str,
        state: LatentState,
        result: DenoisedLatentResult | None,
        make_step_fn: Callable[[DenoisedLatentResult | None, str], Callable[..., torch.Tensor]],
    ) -> "_ModalityStep":
        """Resolve one modality's step inputs from a denoiser result.
        ``make_step_fn`` is invoked for every present modality, including one
        whose result carries no denoised tensor, so that any validation it
        performs fires in a stable order regardless of the denoiser's output.
        """
        step_fn = make_step_fn(result, modality)
        denoised = None
        if result is not None and result.denoised is not None:
            denoised = post_process_latent(result.denoised.float(), state.denoise_mask, state.clean_latent)
        return cls(modality=modality, state=state, step_fn=step_fn, denoised=denoised)


def _ancestral_euler_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    transformer: X0Model,
    denoiser: Denoiser,
    noise_seed: int,
    new_noise_fn: Callable[[torch.Tensor, torch.Generator], torch.Tensor],
    model_dtype: torch.dtype,
    draw_noise: bool,
    make_step_fn: Callable[[DenoisedLatentResult | None, str], Callable[..., torch.Tensor]],
) -> tuple[LatentState | None, LatentState | None]:
    """
    Shared driver for the ancestral (noise-injecting) Euler denoising loops.
    Steps each present modality in float32, re-applies the conditioning mask
    after noise injection, and short-circuits to the denoised prediction on a
    terminal zero sigma. When both modalities are present, noise is drawn from
    the same seeded generator (video first, audio second) so the random
    sequence is consistent.
    ### Parameters
    draw_noise:
        Whether to draw a noise tensor per modality per step. When ``False``,
        ``noise=None`` is passed to the stepper.
    make_step_fn:
        ``(result, modality) -> step_fn`` factory returning the step callable
        for that modality, with any stepper-specific inputs already bound.
        ``step_fn`` is invoked as
        ``step_fn(sample=, denoised_sample=, sigmas=, step_index=, noise=)``.
        Called once per present modality per step right after the denoiser
        returns (before the terminal-sigma check), so it can also validate the
        denoiser output. ``modality`` is ``"video"`` or ``"audio"``.
    ### Returns
    tuple[LatentState | None, LatentState | None]
        Final ``(video_state, audio_state)`` after the denoising loop.
    """
    if video_state is None and audio_state is None:
        raise ValueError("At least one of video_state or audio_state must be provided")

    present_state = video_state if video_state is not None else audio_state
    generator = torch.Generator(device=present_state.latent.device).manual_seed(noise_seed)
    states: dict[str, LatentState | None] = {"video": video_state, "audio": audio_state}

    for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
        video_result, audio_result = denoiser(transformer, states["video"], states["audio"], sigmas, step_idx)
        pending = [
            _ModalityStep.from_modality_result(modality, state, result, make_step_fn)
            for modality, state, result in (
                ("video", states["video"], video_result),
                ("audio", states["audio"], audio_result),
            )
            if state is not None
        ]

        terminal_step = sigmas[step_idx + 1] == 0
        for step in pending:
            if step.denoised is None:
                continue
            if terminal_step:
                states[step.modality] = replace(step.state, latent=step.denoised.to(model_dtype))
                continue
            noise = new_noise_fn(step.state.latent, generator) if draw_noise else None
            x_next = step.step_fn(
                sample=step.state.latent.float(),
                denoised_sample=step.denoised,
                sigmas=sigmas,
                step_index=step_idx,
                noise=noise,
            )
            if draw_noise:
                x_next = post_process_latent(x_next, step.state.denoise_mask, step.state.clean_latent)
            states[step.modality] = replace(step.state, latent=x_next.to(model_dtype))

        if terminal_step:
            break

    return states["video"], states["audio"]


def euler_ancestral_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    stepper: EulerAncestralDiffusionStep,
    transformer: X0Model,
    denoiser: Denoiser,
    noise_seed: int = -1,
    new_noise_fn: Callable[[torch.Tensor, torch.Generator], torch.Tensor] = _get_plain_noise,
    model_dtype: torch.dtype = torch.bfloat16,
) -> tuple[LatentState | None, LatentState | None]:
    """
    Joint audio-video denoising loop using the ancestral (SDE) Euler sampler.
    Each step takes a deterministic Euler step to an intermediate ``sigma_down``
    and renoises back up to the next sigma; see
    :class:`~ltx_core.components.diffusion_steps.EulerAncestralDiffusionStep`.
    Works with any denoiser (no ``uncond`` prediction required). With
    ``stepper=EulerAncestralDiffusionStep(eta=0.0)`` this reduces to
    :func:`euler_denoising_loop`.
    Either ``video_state`` or ``audio_state`` may be ``None`` for absent
    modalities. When both are present, noise is drawn from the same seeded
    generator (video first, audio second).
    ### Parameters
    sigmas:
        1-D tensor of noise levels defining the sampling schedule.
    video_state:
        Current video :class:`~ltx_core.types.LatentState`, or ``None``.
    audio_state:
        Current audio :class:`~ltx_core.types.LatentState`, or ``None``.
    stepper:
        :class:`~ltx_core.components.diffusion_steps.EulerAncestralDiffusionStep`
        instance carrying ``eta`` and ``s_noise`` parameters.
    transformer:
        The diffusion model passed to the denoiser at each step.
    denoiser:
        Callable implementing :class:`~ltx_pipelines.utils.types.Denoiser`.
    noise_seed:
        Integer seed for the noise generator. Default ``-1``.
    new_noise_fn:
        ``(latent, generator) -> noise`` callable. Defaults to plain
        ``torch.randn`` (no channel-wise normalization). Pass
        :func:`_get_new_noise` for the normalized variant used in res2s.
    model_dtype:
        Dtype for latent state updates. Default ``bfloat16``. Pass
        ``torch.float32`` to keep the sampling trajectory in full precision.
    ### Returns
    tuple[LatentState | None, LatentState | None]
        Final ``(video_state, audio_state)`` after the denoising loop.
    """
    if not isinstance(stepper, EulerAncestralDiffusionStep):
        raise ValueError(f"stepper must be an instance of EulerAncestralDiffusionStep, got {type(stepper).__name__}")

    return _ancestral_euler_denoising_loop(
        sigmas=sigmas,
        video_state=video_state,
        audio_state=audio_state,
        transformer=transformer,
        denoiser=denoiser,
        noise_seed=noise_seed,
        new_noise_fn=new_noise_fn,
        model_dtype=model_dtype,
        # Gated on eta alone: at s_noise=0 the step still applies its variance-preserving
        # rescale, so it must not fall back to the noise-free branch.
        draw_noise=stepper.eta > 0,
        # The ancestral step needs nothing beyond the standard arguments.
        make_step_fn=lambda _result, _modality: stepper.step,
    )


def euler_cfg_pp_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    stepper: EulerCfgPpDiffusionStep,
    transformer: X0Model,
    denoiser: Denoiser,
    noise_seed: int = -1,
    new_noise_fn: Callable[[torch.Tensor, torch.Generator], torch.Tensor] = _get_plain_noise,
    model_dtype: torch.dtype = torch.bfloat16,
) -> tuple[LatentState | None, LatentState | None]:
    """
    Joint audio-video denoising loop using the CFG++ corrected Euler sampler.
    Applies the CFG++ update rule at each step: the ODE derivative is computed
    from the unconditioned denoised prediction rather than the standard velocity,
    and an ancestral DDIM noise injection is applied in the rescaled sigma space.
    Requires a guided denoiser whose :class:`~ltx_pipelines.utils.types.DenoisedLatentResult`
    carries ``uncond`` tensors (i.e. CFG must be enabled).
    Either ``video_state`` or ``audio_state`` may be ``None`` for absent modalities.
    When both are present, noise is drawn from the same seeded generator (video
    first, audio second) to produce a consistent random sequence.
    ### Parameters
    sigmas:
        1-D tensor of noise levels defining the sampling schedule.
    video_state:
        Current video :class:`~ltx_core.types.LatentState`, or ``None``.
    audio_state:
        Current audio :class:`~ltx_core.types.LatentState`, or ``None``.
    stepper:
        :class:`~ltx_core.components.diffusion_steps.EulerCfgPpDiffusionStep`
        instance carrying ``eta`` and ``s_noise`` parameters.
    transformer:
        The diffusion model passed to the denoiser at each step.
    denoiser:
        Callable implementing :class:`~ltx_pipelines.utils.types.Denoiser`.
    noise_seed:
        Integer seed for the noise generator. Default ``-1``.
    new_noise_fn:
        ``(latent, generator) -> noise`` callable. Defaults to plain
        ``torch.randn`` (no channel-wise normalization). Pass
        :func:`_get_new_noise` for the normalized variant used in res2s.
    model_dtype:
        Dtype for latent state updates. Default ``bfloat16``.
    ### Returns
    tuple[LatentState | None, LatentState | None]
        Final ``(video_state, audio_state)`` after the denoising loop.
    """
    if not isinstance(stepper, EulerCfgPpDiffusionStep):
        raise ValueError(f"stepper must be an instance of EulerCfgPpDiffusionStep, got {type(stepper).__name__}")

    def make_step_fn(result: DenoisedLatentResult | None, modality: str) -> Callable[..., torch.Tensor]:
        """Bind the modality's uncond prediction into the CFG++ step."""
        uncond = result.uncond if result is not None else None
        if not isinstance(uncond, torch.Tensor):
            raise ValueError(
                f"euler_cfg_pp_denoising_loop requires {modality} DenoisedLatentResult.uncond to be a tensor. "
                "Use GuidedDenoiser or FactoryGuidedDenoiser with cfg_scale != 1 "
                "or force_uncond_pass=True and a negative_context."
            )
        return partial(stepper.step, uncond_denoised=uncond)

    return _ancestral_euler_denoising_loop(
        sigmas=sigmas,
        video_state=video_state,
        audio_state=audio_state,
        transformer=transformer,
        denoiser=denoiser,
        noise_seed=noise_seed,
        new_noise_fn=new_noise_fn,
        model_dtype=model_dtype,
        draw_noise=stepper.eta > 0 and stepper.s_noise > 0,
        make_step_fn=make_step_fn,
    )
