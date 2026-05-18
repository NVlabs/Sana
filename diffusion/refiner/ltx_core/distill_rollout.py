"""Distillation rollout core: shared-noise teacher/student supervision.

Two modes:
  ``rollout_stepwise`` — per-denoising-step MSE on final predictions.
  ``blockwise``        — per-block hidden state MSE on a single forward pass.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.refiner.vendor.ltx_core.components.diffusion_steps import EulerDiffusionStep
from diffusion.refiner.vendor.ltx_core.components.noisers import GaussianNoiser
from diffusion.refiner.vendor.ltx_core.components.patchifiers import VideoLatentPatchifier
from diffusion.refiner.vendor.ltx_core.tools import VideoLatentTools
from diffusion.refiner.vendor.ltx_core.types import LatentState, VideoLatentShape
from diffusion.refiner.vendor.ltx_core.utils import to_velocity
from diffusion.refiner.vendor.ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES
from diffusion.refiner.vendor.ltx_pipelines.utils.helpers import (
    modality_from_latent_state,
    post_process_latent,
)

from .causal_attention import apply_causal_mask
from .ltx_student_wrapper import LTXStudentWrapper
from .ltx_teacher_wrapper import LTXTeacherWrapper

logger = logging.getLogger(__name__)


def compute_block_loss(
    student_h: torch.Tensor,
    teacher_h: torch.Tensor,
    mode: str = "mse",
) -> torch.Tensor:
    """Compute block-wise alignment loss between student and teacher hidden states.

    Args:
        mode: ``"mse"`` — mean squared error (default, backward-compatible).
              ``"cosine"`` — 1 - cosine_similarity (direction-only, scale-invariant).
    """
    if mode == "cosine":
        cos_sim = F.cosine_similarity(student_h.float(), teacher_h.float(), dim=-1)
        return (1 - cos_sim).mean()
    return F.mse_loss(student_h.float(), teacher_h.float())


# ---------------------------------------------------------------------------
# Block-wise hidden state capture
# ---------------------------------------------------------------------------


@contextmanager
def capture_block_hidden_states(model: nn.Module) -> Iterator[list[torch.Tensor]]:
    """Context manager that hooks every transformer block and captures video hidden states.

    Usage::

        with capture_block_hidden_states(x0_model) as states:
            output = x0_model(video=..., audio=..., perturbations=...)
        # states is now a list of 48 tensors, one per block output
    """
    captured: list[torch.Tensor] = []
    handles: list[torch.utils.hooks.RemovableHook] = []

    velocity_model = model.velocity_model if hasattr(model, "velocity_model") else model
    blocks = velocity_model.transformer_blocks

    def make_hook(idx: int):
        def hook_fn(module, input, output):
            # output is (video_args: TransformerArgs, audio_args: TransformerArgs | None)
            video_args = output[0]
            if video_args is not None:
                captured.append(video_args.x)

        return hook_fn

    for idx, block in enumerate(blocks):
        h = block.register_forward_hook(make_hook(idx))
        handles.append(h)

    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


@dataclass
class StepLoss:
    """Per-step distillation loss."""

    step_index: int
    sigma: float
    mse: torch.Tensor  # scalar, grad-enabled (through student)


@dataclass
class PerceptualLoss:
    """Perceptual temporal loss from partial VAE decoder features."""

    mse: torch.Tensor  # scalar, grad-enabled


@dataclass
class DistillRolloutOutput:
    """Result of one distillation rollout."""

    step_losses: list[StepLoss]
    total_loss: torch.Tensor  # grad-enabled
    final_latent_teacher: torch.Tensor  # [B, C, F, H, W] for logging
    final_latent_student: torch.Tensor  # [B, C, F, H, W] for logging
    block_losses: list[BlockLoss] = None  # optional blockwise losses
    perceptual_loss: PerceptualLoss | None = None


@dataclass
class BlockLoss:
    """Per-block alignment loss."""

    block_index: int
    mse: torch.Tensor  # scalar, grad-enabled


@dataclass
class BlockwiseOutput:
    """Result of one block-wise alignment pass."""

    block_losses: list[BlockLoss]
    total_loss: torch.Tensor  # mean across blocks, grad-enabled
    output_mse: torch.Tensor  # final output MSE (denoised predictions)


def create_shared_noised_state(
    stage2_latent: torch.Tensor,
    *,
    seed: int,
    noise_scale: float | None = None,
    fps: float = 24.0,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[LatentState, VideoLatentTools]:
    """Create a single noised state that both teacher and student will use."""
    B, C, Fr, H, W = stage2_latent.shape
    tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=VideoLatentShape(batch=B, channels=C, frames=Fr, height=H, width=W),
        fps=fps,
    )
    state = tools.create_initial_state(
        device=device,
        dtype=dtype,
        initial_latent=stage2_latent.to(device=device, dtype=dtype),
    )
    scale = float(STAGE_2_DISTILLED_SIGMA_VALUES[0]) if noise_scale is None else float(noise_scale)
    noiser = GaussianNoiser(torch.Generator(device=device).manual_seed(seed))
    return noiser(state, scale), tools


def generate_global_noise(
    stage2_latent: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate noise for the full video once.

    Returns noise in un-patchified latent space ``[B, C, F, H, W]`` so that
    per-window slicing produces consistent noise in overlapping regions.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        *stage2_latent.shape,
        device=device,
        dtype=dtype,
        generator=gen,
    )


def create_window_noised_state(
    stage2_latent_window: torch.Tensor,
    global_noise_window: torch.Tensor,
    *,
    noise_scale: float | None = None,
    fps: float = 24.0,
    device: torch.device,
    dtype: torch.dtype,
    temporal_offset: int = 0,
) -> tuple[LatentState, VideoLatentTools]:
    """Create a noised LatentState for a single window using pre-generated noise.

    Unlike :func:`create_shared_noised_state` which generates noise internally,
    this function accepts a slice of globally-generated noise so that overlap
    regions share identical noise realizations across windows.

    Args:
        stage2_latent_window: ``[B, C, Fw, H, W]`` — window latent.
        global_noise_window:  ``[B, C, Fw, H, W]`` — noise slice (same shape).
        noise_scale: Sigma for the first diffusion step.  ``None`` uses default.
        temporal_offset: Global latent-frame index of this window's first frame.
    """
    B, C, Fw, H, W = stage2_latent_window.shape
    tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=VideoLatentShape(batch=B, channels=C, frames=Fw, height=H, width=W),
        fps=fps,
    )
    state = tools.create_initial_state(
        device=device,
        dtype=dtype,
        initial_latent=stage2_latent_window.to(device=device, dtype=dtype),
        temporal_offset=temporal_offset,
    )
    # Apply noise using the same formula as GaussianNoiser but with pre-generated noise
    scale = float(STAGE_2_DISTILLED_SIGMA_VALUES[0]) if noise_scale is None else float(noise_scale)
    noise_patchified = tools.patchifier.patchify(global_noise_window.to(device=device, dtype=dtype))
    scaled_mask = state.denoise_mask * scale
    noised_latent = noise_patchified * scaled_mask + state.latent * (1 - scaled_mask)
    return replace(state, latent=noised_latent.to(dtype)), tools


def distill_rollout_stepwise(
    *,
    teacher: LTXTeacherWrapper,
    student: LTXStudentWrapper,
    stage2_latent: torch.Tensor,
    v_context: torch.Tensor,
    seed: int = 0,
    fps: float = 24.0,
    num_inference_steps: int | None = None,
    block_loss_weight: float = 0.0,
    output_loss_weight: float = 1.0,
    chunk_sizes: list[int] | None = None,
    block_loss_max_layer: int | None = None,
    partial_decoder: nn.Module | None = None,
    perceptual_loss_weight: float = 0.0,
    loss_space: str = "x0",
    step_sample_weights: list[float] | None = None,
    attention_mode: str = "causal",
    window_radius: int | None = None,
    perceptual_teacher_layers: list[int] | None = None,
    block_loss_mode: str = "mse",
) -> DistillRolloutOutput:
    """Teacher-trajectory distillation with random single-step supervision.

    1. Teacher runs the full rollout (no grad), recording each step's
       (input_state, denoised_output).
    2. One step is randomly sampled.
    3. Student forwards on the SAME input_state as the teacher at that step.
    4. Loss = MSE(student_pred, teacher_pred) in the chosen loss_space.

    Parameters
    ----------
    loss_space : str
        ``"x0"`` — MSE on denoised predictions (default, backward-compatible).
        ``"velocity"`` — MSE on velocity predictions: v = (x_t - x0) / σ.

    This ensures identical inputs for teacher and student at the supervised
    step, and only requires one student forward (saves ~3x memory vs all steps).
    """
    device = teacher.device
    dtype = teacher.dtype

    sigmas = teacher.stage2_sigmas
    if num_inference_steps is not None:
        sigmas = sigmas[: num_inference_steps + 1]
    sigmas = sigmas.to(device=device, dtype=torch.float32)
    stepper = EulerDiffusionStep()
    num_steps = len(sigmas) - 1

    # Shared noised state
    shared_state, video_tools = create_shared_noised_state(
        stage2_latent,
        seed=seed,
        fps=fps,
        device=device,
        dtype=dtype,
    )

    # --- Phase 1: Pick random step + random block ---
    rng = torch.Generator().manual_seed(seed)
    if step_sample_weights is not None and len(step_sample_weights) >= num_steps:
        # Weighted sampling: e.g. [0.5, 0.25, 0.25] biases toward step 0 (high sigma)
        w = torch.tensor(step_sample_weights[:num_steps], dtype=torch.float32)
        selected_step = torch.multinomial(w, 1, generator=rng).item()
    else:
        selected_step = torch.randint(0, num_steps, (1,), generator=rng).item()

    velocity_model = (
        teacher.transformer.velocity_model if hasattr(teacher.transformer, "velocity_model") else teacher.transformer
    )
    total_blocks = len(velocity_model.transformer_blocks)
    block_pool = min(block_loss_max_layer, total_blocks) if block_loss_max_layer is not None else total_blocks
    selected_block = torch.randint(0, block_pool, (1,), generator=rng).item() if block_loss_weight > 0 else -1

    # --- Phase 2: Teacher forward to selected step (no grad) ---
    # Capture: single block (for block loss) + multiple perceptual layers (inline)
    teacher_state = shared_state
    teacher_block_h = None
    teacher_perc_states: dict[int, torch.Tensor] = {}

    with torch.no_grad():
        if selected_block >= 0:
            velocity_model._capture_block_idx = selected_block
        # Set multi-block capture for perceptual teacher layers
        if perceptual_teacher_layers and perceptual_loss_weight > 0:
            velocity_model._capture_block_indices = set(perceptual_teacher_layers)
            velocity_model._captured_block_outputs = {}

        for step_idx in range(selected_step + 1):
            # Reset multi-capture dict each step (we only want the final step's outputs)
            if perceptual_teacher_layers and perceptual_loss_weight > 0:
                velocity_model._captured_block_outputs = {}

            teacher_denoised = teacher.forward_denoise_step(
                video_state=teacher_state,
                v_context=v_context,
                sigmas=sigmas,
                step_index=step_idx,
            )
            if step_idx < selected_step:
                t_denoised_pp = post_process_latent(
                    teacher_denoised,
                    teacher_state.denoise_mask,
                    teacher_state.clean_latent,
                )
                teacher_state = replace(
                    teacher_state,
                    latent=stepper.step(teacher_state.latent, t_denoised_pp, sigmas, step_idx),
                )

        # Grab single block output
        if selected_block >= 0:
            teacher_block_h = getattr(velocity_model, "_captured_block_output", None)
            if teacher_block_h is not None:
                teacher_block_h = teacher_block_h.detach()
            velocity_model._capture_block_idx = None
            velocity_model._captured_block_output = None

        # Grab multi-block perceptual outputs
        if perceptual_teacher_layers and perceptual_loss_weight > 0:
            for li, t in getattr(velocity_model, "_captured_block_outputs", {}).items():
                if t is not None:
                    teacher_perc_states[li] = t.detach()
            velocity_model._capture_block_indices = None
            velocity_model._captured_block_outputs = {}

    teacher_input_state = teacher_state
    teacher_denoised = teacher_denoised.detach().clone()
    sigma = float(sigmas[selected_step].item())
    del teacher_state
    torch.cuda.empty_cache()

    # --- Phase 3: Student forward on SAME input (with causal/sliding-window mask) ---
    # Capture: single block (for block loss) + multiple perceptual layers (inline)
    if selected_block >= 0:
        velocity_model._capture_block_idx = selected_block
    if perceptual_teacher_layers and perceptual_loss_weight > 0:
        velocity_model._capture_block_indices = set(perceptual_teacher_layers)
        velocity_model._captured_block_outputs = {}

    student_denoised = student.forward_denoise_step(
        video_state=teacher_input_state,
        v_context=v_context,
        sigmas=sigmas,
        step_index=selected_step,
        apply_causal=True,
        chunk_sizes=chunk_sizes,
        attention_mode=attention_mode,
        window_radius=window_radius,
    )

    student_block_h = None
    if selected_block >= 0:
        student_block_h = getattr(velocity_model, "_captured_block_output", None)
        velocity_model._capture_block_idx = None
        velocity_model._captured_block_output = None

    # Grab student perceptual layer outputs (with grad)
    student_perc_states: dict[int, torch.Tensor] = {}
    if perceptual_teacher_layers and perceptual_loss_weight > 0:
        student_perc_states = dict(getattr(velocity_model, "_captured_block_outputs", {}))
        velocity_model._capture_block_indices = None
        velocity_model._captured_block_outputs = {}

    # --- Phase 4: Loss ---
    if loss_space == "velocity":
        # Convert denoised (x0) to velocity: v = (x_t - x0) / σ
        x_t = teacher_input_state.latent
        sigma_t = sigmas[selected_step]
        student_v = to_velocity(x_t, sigma_t, student_denoised)
        teacher_v = to_velocity(x_t, sigma_t, teacher_denoised)
        output_mse = F.mse_loss(student_v.float(), teacher_v.float())
    else:
        output_mse = F.mse_loss(student_denoised.float(), teacher_denoised.float())
    total_loss = output_loss_weight * output_mse

    step_losses = [StepLoss(step_index=selected_step, sigma=sigma, mse=output_mse)]
    block_losses = []

    if student_block_h is not None and teacher_block_h is not None:
        block_mse = compute_block_loss(student_block_h, teacher_block_h, mode=block_loss_mode)
        total_loss = total_loss + block_loss_weight * block_mse
        block_losses.append(BlockLoss(block_index=selected_block, mse=block_mse))

    # --- Phase 5: Perceptual loss (optional) ---
    # Two modes:
    #   a) perceptual_teacher_layers: MSE on teacher transformer hidden states (preferred)
    #      Uses inline _capture_block_indices (gradient-checkpoint safe, no hooks)
    #   b) partial_decoder: MSE on VAE decoder features (legacy)
    perceptual_out = None
    if perceptual_loss_weight > 0 and perceptual_teacher_layers and teacher_perc_states:
        # Mode (a): Teacher transformer layer features (captured inline in Phase 2+3)
        perc_losses = []
        for li in perceptual_teacher_layers:
            if li in teacher_perc_states and li in student_perc_states:
                perc_losses.append(F.mse_loss(student_perc_states[li].float(), teacher_perc_states[li].float()))
        del teacher_perc_states, student_perc_states

        if perc_losses:
            perc_mse = torch.stack(perc_losses).mean()
            total_loss = total_loss + perceptual_loss_weight * perc_mse
            perceptual_out = PerceptualLoss(mse=perc_mse)

    elif perceptual_loss_weight > 0 and partial_decoder is not None:
        # Mode (b): Legacy VAE decoder perceptual loss.
        # Teacher path (no grad)
        with torch.no_grad():
            t_denoised_pp = post_process_latent(
                teacher_denoised, teacher_input_state.denoise_mask, teacher_input_state.clean_latent
            )
            t_stepped = replace(
                teacher_input_state,
                latent=stepper.step(teacher_input_state.latent, t_denoised_pp, sigmas, selected_step),
            )
            t_latent = video_tools.unpatchify(video_tools.clear_conditioning(t_stepped)).latent
            t_feat = partial_decoder(t_latent)
            del t_denoised_pp, t_stepped, t_latent

        # Student path (grad flows)
        s_denoised_pp = post_process_latent(
            student_denoised, teacher_input_state.denoise_mask, teacher_input_state.clean_latent
        )
        s_stepped = replace(
            teacher_input_state,
            latent=stepper.step(teacher_input_state.latent, s_denoised_pp, sigmas, selected_step),
        )
        s_latent = video_tools.unpatchify(video_tools.clear_conditioning(s_stepped)).latent
        s_feat = partial_decoder(s_latent)
        del s_denoised_pp, s_stepped, s_latent

        perc_mse = F.mse_loss(s_feat.float(), t_feat.float())
        total_loss = total_loss + perceptual_loss_weight * perc_mse
        perceptual_out = PerceptualLoss(mse=perc_mse)
        del s_feat, t_feat

    del stepper

    with torch.no_grad():
        t_final = video_tools.unpatchify(video_tools.clear_conditioning(teacher_input_state)).latent

    return DistillRolloutOutput(
        step_losses=step_losses,
        total_loss=total_loss,
        final_latent_teacher=t_final,
        final_latent_student=t_final,
        block_losses=block_losses,
        perceptual_loss=perceptual_out,
    )


# ---------------------------------------------------------------------------
# Block-wise alignment
# ---------------------------------------------------------------------------


def distill_blockwise(
    *,
    teacher: LTXTeacherWrapper,
    student: LTXStudentWrapper,
    stage2_latent: torch.Tensor,
    v_context: torch.Tensor,
    seed: int = 0,
    fps: float = 24.0,
    step_index: int = 0,
    block_loss_weight: float = 1.0,
    output_loss_weight: float = 1.0,
    num_sample_blocks: int | None = None,  # None = all blocks, e.g. 8 = random 8 blocks
    chunk_sizes: list[int] | None = None,
    block_loss_max_layer: int | None = None,
) -> BlockwiseOutput:
    """Run one forward pass and compute per-block hidden state alignment loss.

    Same input for teacher (bidirectional) and student (causal LoRA).
    At every transformer block, the student's video hidden state is
    supervised against the teacher's.

    Parameters
    ----------
    teacher / student : wrappers
    stage2_latent : [B, C, F, H, W]
    v_context : encoded prompt embeddings
    seed : shared noise seed
    step_index : which sigma step to run (default 0 = first step)
    block_loss_weight : weight for per-block MSE in total loss
    output_loss_weight : weight for final output MSE in total loss
    """
    device = teacher.device
    dtype = teacher.dtype

    sigmas = teacher.stage2_sigmas.to(device=device, dtype=torch.float32)

    # Shared noised state
    shared_state, _ = create_shared_noised_state(
        stage2_latent,
        seed=seed,
        fps=fps,
        device=device,
        dtype=dtype,
    )

    sigma = sigmas[step_index]

    # --- Determine which blocks to supervise ---
    velocity_model = (
        teacher.transformer.velocity_model if hasattr(teacher.transformer, "velocity_model") else teacher.transformer
    )
    total_blocks = len(velocity_model.transformer_blocks)

    # Limit block supervision to first N layers if configured
    eligible_blocks = min(block_loss_max_layer, total_blocks) if block_loss_max_layer is not None else total_blocks

    if num_sample_blocks is not None and num_sample_blocks < eligible_blocks:
        # Randomly sample block indices from eligible range
        rng = torch.Generator().manual_seed(seed)
        sampled_indices = set(torch.randperm(eligible_blocks, generator=rng)[:num_sample_blocks].tolist())
    else:
        sampled_indices = set(range(eligible_blocks))

    # Temporarily disable gradient checkpointing during hook-based forward passes.
    # Hooks capture intermediate tensors which conflicts with checkpoint recomputation.
    had_grad_ckpt = getattr(velocity_model, "_enable_gradient_checkpointing", False)
    if had_grad_ckpt:
        velocity_model._enable_gradient_checkpointing = False

    # --- Teacher forward: only capture sampled blocks (no grad) ---
    teacher_block_states: dict[int, torch.Tensor] = {}
    teacher_handles: list[torch.utils.hooks.RemovableHook] = []

    def make_teacher_hook(block_idx: int):
        def hook_fn(module, input, output):
            video_args = output[0]
            if video_args is not None:
                teacher_block_states[block_idx] = video_args.x.detach()

        return hook_fn

    with torch.no_grad():
        for idx, block in enumerate(velocity_model.transformer_blocks):
            if sampled_indices is None or idx in sampled_indices:
                teacher_handles.append(block.register_forward_hook(make_teacher_hook(idx)))

        video_mod_teacher = modality_from_latent_state(shared_state, v_context, sigma)
        teacher_denoised, _ = teacher.transformer(
            video=video_mod_teacher,
            audio=None,
            perturbations=None,
        )

        for h in teacher_handles:
            h.remove()

    # --- Student forward: compute per-block loss on-the-fly via hooks ---
    student_state = apply_causal_mask(shared_state, chunk_sizes=chunk_sizes)
    video_mod_student = modality_from_latent_state(student_state, v_context, sigma)

    block_losses: list[BlockLoss] = []
    student_handles: list[torch.utils.hooks.RemovableHook] = []

    def make_student_hook(block_idx: int):
        def hook_fn(module, input, output):
            video_args = output[0]
            if video_args is not None and block_idx in teacher_block_states:
                mse = F.mse_loss(video_args.x.float(), teacher_block_states[block_idx].float())
                block_losses.append(BlockLoss(block_index=block_idx, mse=mse))

        return hook_fn

    for idx, block in enumerate(velocity_model.transformer_blocks):
        if sampled_indices is None or idx in sampled_indices:
            student_handles.append(block.register_forward_hook(make_student_hook(idx)))

    try:
        student_denoised, _ = student.transformer(
            video=video_mod_student,
            audio=None,
            perturbations=None,
        )
    finally:
        for h in student_handles:
            h.remove()
    del teacher_block_states

    # Restore gradient checkpointing
    if had_grad_ckpt:
        velocity_model._enable_gradient_checkpointing = True

    # --- Output MSE ---
    output_mse = F.mse_loss(student_denoised.float(), teacher_denoised.float())

    # --- Total loss ---
    block_mean = torch.stack([bl.mse for bl in block_losses]).mean()
    total_loss = block_loss_weight * block_mean + output_loss_weight * output_mse

    return BlockwiseOutput(
        block_losses=block_losses,
        total_loss=total_loss,
        output_mse=output_mse,
    )


# ---------------------------------------------------------------------------
# Window-consistent distillation with full attention + KV prefix
# ---------------------------------------------------------------------------


def _get_attn1_layers(model: nn.Module) -> list:
    """Get all attn1 (self-attention) layers from transformer blocks."""
    vm = model.velocity_model if hasattr(model, "velocity_model") else model
    return [block.attn1 for block in vm.transformer_blocks]


def _set_kv_cache_capture(attn1_layers: list, enable: bool) -> None:
    for attn in attn1_layers:
        attn._kv_cache_capture = enable


def _get_captured_kv(
    attn1_layers: list,
    boundary_slice: slice,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Extract captured K,V for boundary tokens from all layers (detached)."""
    cache = []
    for attn in attn1_layers:
        k, v = attn._cached_kv  # already detached in attention.py
        cache.append((k[:, boundary_slice, :].clone(), v[:, boundary_slice, :].clone()))
        del attn._cached_kv
    return cache


def _set_kv_cache_prefix(
    attn1_layers: list,
    cache: list[tuple[torch.Tensor, torch.Tensor]] | None,
) -> None:
    for i, attn in enumerate(attn1_layers):
        attn._kv_cache_prefix = cache[i] if cache is not None else None


def _clear_kv_state(attn1_layers: list) -> None:
    for attn in attn1_layers:
        attn._kv_cache_capture = False
        attn._kv_cache_prefix = None
        if hasattr(attn, "_cached_kv"):
            del attn._cached_kv


def _build_windows(
    total_frames: int,
    window_frames: int,
    overlap_frames: int,
) -> list[tuple[int, int]]:
    """Build (start, end) pairs for overlapping windows."""
    stride = window_frames - overlap_frames
    assert stride > 0
    windows = []
    start = 0
    while start < total_frames:
        end = min(start + window_frames, total_frames)
        windows.append((start, end))
        if end == total_frames:
            break
        start += stride
    return windows


@dataclass
class WindowedDistillOutput:
    """Result of one window-consistent distillation step."""

    total_loss: torch.Tensor
    output_mse: torch.Tensor
    block_mse: torch.Tensor | None
    selected_step: int
    target_window: tuple[int, int]
    context_window: tuple[int, int] | None
    _attn1_layers: list | None = None  # internal: for deferred KV cleanup

    def cleanup_kv_state(self) -> None:
        """Clear KV cache state.  Must be called AFTER backward() completes,
        because gradient checkpointing re-runs the forward during backward
        and needs the KV prefix to reproduce correct tensor shapes."""
        if self._attn1_layers is not None:
            _clear_kv_state(self._attn1_layers)
            self._attn1_layers = None


def distill_rollout_windowed(
    *,
    teacher: LTXTeacherWrapper,
    student: LTXStudentWrapper,
    stage2_latent: torch.Tensor,
    v_context: torch.Tensor,
    seed: int = 0,
    fps: float = 24.0,
    num_inference_steps: int | None = None,
    window_latent_frames: int = 5,
    overlap_latent_frames: int = 2,
    output_loss_weight: float = 1.0,
    block_loss_weight: float = 0.0,
    block_loss_max_layer: int | None = None,
    loss_space: str = "x0",
    step_sample_weights: list[float] | None = None,
    p_student_trajectory: float = 0.0,
    use_denoised_kv: bool = False,
    block_loss_mode: str = "mse",
) -> WindowedDistillOutput:
    """Window-consistent distillation: full attention within each window,
    KV cache prefix for cross-window context.

    Training procedure:
      1. Generate global noise for the full video.
      2. Pick a random denoising step and a random (context, target) window pair.
      3. **Teacher trajectory path** (probability ``1 - p_student_trajectory``):
         - Teacher rollout on full video to step k (no grad, bidirectional).
         - Teacher forward on target window at step k → GT.
      4. **Student trajectory path** (probability ``p_student_trajectory``):
         - Student rollout on full video to step k-1 (no grad, causal LoRA).
         - Euler step → student_state_k (student's own imperfect state).
         - Teacher forward on student_state_k target window → GT.
      5. Student: context window forward (no grad) → capture boundary KV.
      6. Student: target window forward (with grad, KV prefix) → prediction.
      7. Loss = MSE(student_pred, teacher_GT).

    Key properties:
      - Window internal attention is full bidirectional (no mask needed).
      - Cross-window context flows only through KV cache (per-layer, detached).
      - Global noise ensures overlap regions are noise-consistent.
      - Global RoPE positions (via temporal_offset) ensure position continuity.
    """
    device = teacher.device
    dtype = teacher.dtype
    B, C, F_lat, H, W = stage2_latent.shape

    sigmas = teacher.stage2_sigmas
    if num_inference_steps is not None:
        sigmas = sigmas[: num_inference_steps + 1]
    sigmas = sigmas.to(device=device, dtype=torch.float32)
    stepper = EulerDiffusionStep()
    num_steps = len(sigmas) - 1

    # --- Build windows ---
    windows = _build_windows(F_lat, window_latent_frames, overlap_latent_frames)
    window_latent_frames - overlap_latent_frames

    # --- Random selections ---
    rng = torch.Generator().manual_seed(seed)
    if step_sample_weights is not None and len(step_sample_weights) >= num_steps:
        w = torch.tensor(step_sample_weights[:num_steps], dtype=torch.float32)
        selected_step = torch.multinomial(w, 1, generator=rng).item()
    else:
        selected_step = torch.randint(0, num_steps, (1,), generator=rng).item()

    # Pick target window (with bias toward windows that need KV cache)
    if len(windows) <= 1:
        target_wi = 0
    else:
        # 20% chance of window 0 (no KV prefix), 80% uniform over rest
        if torch.rand(1, generator=rng).item() < 0.2:
            target_wi = 0
        else:
            target_wi = torch.randint(1, len(windows), (1,), generator=rng).item()

    context_wi = target_wi - 1 if target_wi > 0 else None
    target_ws, target_we = windows[target_wi]

    # Pick random block for block loss
    velocity_model = (
        teacher.transformer.velocity_model if hasattr(teacher.transformer, "velocity_model") else teacher.transformer
    )
    total_blocks = len(velocity_model.transformer_blocks)
    block_pool = min(block_loss_max_layer, total_blocks) if block_loss_max_layer is not None else total_blocks
    selected_block = torch.randint(0, block_pool, (1,), generator=rng).item() if block_loss_weight > 0 else -1

    # --- Global noise (consistent across windows) ---
    global_noise = generate_global_noise(
        stage2_latent,
        seed=seed,
        device=device,
        dtype=dtype,
    )

    # --- Decide trajectory: teacher or student ---
    use_student_trajectory = p_student_trajectory > 0 and torch.rand(1, generator=rng).item() < p_student_trajectory

    # --- Phase 1: Get the input state at the selected denoising step ---
    from .ltx_student_wrapper import LoRALinear

    full_state, full_tools = create_window_noised_state(
        stage2_latent,
        global_noise,
        fps=fps,
        device=device,
        dtype=dtype,
        temporal_offset=0,
    )
    if use_student_trajectory:
        # Student trajectory: student rollout on full video to step k (no grad).
        # Even when selected_step=0 the loop is empty, but we still use the
        # student code path for consistency (LoRA enabled during forward).
        with torch.no_grad():
            LoRALinear.enabled = True
            try:
                rollout_state = full_state
                for si in range(selected_step):
                    sigma = sigmas[si]
                    video_mod = modality_from_latent_state(rollout_state, v_context, sigma)
                    denoised, _ = student.transformer(video=video_mod, audio=None, perturbations=None)
                    denoised_pp = post_process_latent(denoised, rollout_state.denoise_mask, rollout_state.clean_latent)
                    rollout_state = replace(
                        rollout_state,
                        latent=stepper.step(rollout_state.latent, denoised_pp, sigmas, si),
                    )
            finally:
                LoRALinear.enabled = True  # restore (teacher.forward_denoise_step toggles it)
            full_stepped = full_tools.unpatchify(full_tools.clear_conditioning(rollout_state))
            stepped_latent = full_stepped.latent
    else:
        # Teacher trajectory: teacher rollout on full video to step k
        with torch.no_grad():
            rollout_state = full_state
            for si in range(selected_step):
                teacher_denoised = teacher.forward_denoise_step(
                    video_state=rollout_state,
                    v_context=v_context,
                    sigmas=sigmas,
                    step_index=si,
                )
                denoised_pp = post_process_latent(
                    teacher_denoised,
                    rollout_state.denoise_mask,
                    rollout_state.clean_latent,
                )
                rollout_state = replace(
                    rollout_state,
                    latent=stepper.step(rollout_state.latent, denoised_pp, sigmas, si),
                )
            full_stepped = full_tools.unpatchify(full_tools.clear_conditioning(rollout_state))
            stepped_latent = full_stepped.latent
    del full_state, rollout_state, full_stepped
    torch.cuda.empty_cache()

    # Now stepped_latent is [B, C, F_lat, H, W] at the correct noise level for step k.

    # --- Phase 2: Teacher forward on FULL VIDEO (bidirectional) → crop target GT ---
    # Teacher sees ALL frames with full bidirectional attention, giving the highest
    # quality supervision.  We then slice out the target window's tokens as GT.
    # This is critical for temporal consistency: the teacher's output at frame t
    # reflects information from ALL other frames, not just the local window.
    full_teacher_tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=VideoLatentShape(batch=B, channels=C, frames=F_lat, height=H, width=W),
        fps=fps,
    )
    full_teacher_state = full_teacher_tools.create_initial_state(
        device=device,
        dtype=dtype,
        initial_latent=stepped_latent.to(device=device, dtype=dtype),
        temporal_offset=0,
    )

    if selected_block >= 0:
        velocity_model._capture_block_idx = selected_block

    with torch.no_grad():
        teacher_denoised_full = teacher.forward_denoise_step(
            video_state=full_teacher_state,
            v_context=v_context,
            sigmas=sigmas,
            step_index=selected_step,
        )

    # Crop target window tokens from teacher's full-video output.
    # With patch_size=1, each latent frame has H*W tokens, ordered [f0_tokens, f1_tokens, ...].
    tokens_per_frame = H * W
    tgt_token_start = target_ws * tokens_per_frame
    tgt_token_end = target_we * tokens_per_frame
    teacher_denoised = teacher_denoised_full[:, tgt_token_start:tgt_token_end, :].detach().clone()

    teacher_block_h = None
    if selected_block >= 0:
        teacher_block_h_full = getattr(velocity_model, "_captured_block_output", None)
        if teacher_block_h_full is not None:
            teacher_block_h = teacher_block_h_full[:, tgt_token_start:tgt_token_end, :].detach()
        velocity_model._capture_block_idx = None
        velocity_model._captured_block_output = None

    del full_teacher_state, teacher_denoised_full
    torch.cuda.empty_cache()

    # Build target window state for student (needs its own LatentState with correct positions)
    target_latent = stepped_latent[:, :, target_ws:target_we, :, :]
    Fw = target_we - target_ws
    tgt_tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=VideoLatentShape(batch=B, channels=C, frames=Fw, height=H, width=W),
        fps=fps,
    )
    tgt_state = tgt_tools.create_initial_state(
        device=device,
        dtype=dtype,
        initial_latent=target_latent.to(device=device, dtype=dtype),
        temporal_offset=target_ws,
    )

    # --- Phase 3: Student context window → KV cache (no grad) ---
    attn1_layers = _get_attn1_layers(student.transformer)
    kv_cache = None

    if context_wi is not None:
        ctx_ws, ctx_we = windows[context_wi]
        ctx_latent = stepped_latent[:, :, ctx_ws:ctx_we, :, :]
        Fc = ctx_we - ctx_ws
        ctx_tools = VideoLatentTools(
            patchifier=VideoLatentPatchifier(patch_size=1),
            target_shape=VideoLatentShape(batch=B, channels=C, frames=Fc, height=H, width=W),
            fps=fps,
        )
        tokens_per_frame = H * W  # patch_size=1

        with torch.no_grad():
            if use_denoised_kv:
                # Match inference: run full multi-step denoising on context
                # window, then re-forward the final denoised output to
                # capture KV with rich structural information.
                ctx_state = ctx_tools.create_initial_state(
                    device=device,
                    dtype=dtype,
                    initial_latent=ctx_latent.to(device=device, dtype=dtype),
                    temporal_offset=ctx_ws,
                )
                # Full denoising loop (same as inference)
                for si in range(num_steps):
                    sigma = sigmas[si]
                    video_mod = modality_from_latent_state(ctx_state, v_context, sigma)
                    denoised_ctx, _ = student.transformer(video=video_mod, audio=None, perturbations=None)
                    denoised_pp = post_process_latent(denoised_ctx, ctx_state.denoise_mask, ctx_state.clean_latent)
                    ctx_state = replace(
                        ctx_state,
                        latent=stepper.step(ctx_state.latent, denoised_pp, sigmas, si),
                    )
                # Unpatchify final denoised output
                ctx_final = ctx_tools.unpatchify(ctx_tools.clear_conditioning(ctx_state)).latent
                del ctx_state

                # Re-forward on denoised output → capture KV
                ctx_state_clean = ctx_tools.create_initial_state(
                    device=device,
                    dtype=dtype,
                    initial_latent=ctx_final,
                    temporal_offset=ctx_ws,
                )
                _set_kv_cache_capture(attn1_layers, True)
                video_mod = modality_from_latent_state(
                    ctx_state_clean,
                    v_context,
                    sigmas[num_steps - 1],
                )
                _, _ = student.transformer(video=video_mod, audio=None, perturbations=None)
                del ctx_state_clean, ctx_final
            else:
                # Single-pass: forward on noised input → capture KV directly.
                ctx_state = ctx_tools.create_initial_state(
                    device=device,
                    dtype=dtype,
                    initial_latent=ctx_latent.to(device=device, dtype=dtype),
                    temporal_offset=ctx_ws,
                )
                _set_kv_cache_capture(attn1_layers, True)
                sigma = sigmas[selected_step]
                video_mod = modality_from_latent_state(ctx_state, v_context, sigma)
                _, _ = student.transformer(video=video_mod, audio=None, perturbations=None)
                del ctx_state

            # Capture boundary KV (last `overlap` frames)
            boundary_start = (Fc - overlap_latent_frames) * tokens_per_frame
            boundary_slice = slice(boundary_start, Fc * tokens_per_frame)
            kv_cache = _get_captured_kv(attn1_layers, boundary_slice)
            _set_kv_cache_capture(attn1_layers, False)

        torch.cuda.empty_cache()

    # --- Phase 4: Student target window forward (with grad, KV prefix) ---
    if kv_cache is not None:
        _set_kv_cache_prefix(attn1_layers, kv_cache)

    if selected_block >= 0:
        velocity_model._capture_block_idx = selected_block

    try:
        sigma = sigmas[selected_step]
        video_mod = modality_from_latent_state(tgt_state, v_context, sigma)
        student_denoised, _ = student.transformer(video=video_mod, audio=None, perturbations=None)
    finally:
        # NOTE: Do NOT clear _kv_cache_prefix here — gradient checkpointing will
        # recompute the forward during backward, and needs the prefix to reproduce
        # the same K,V shapes.  Only disable capture (no longer needed).
        _set_kv_cache_capture(attn1_layers, False)
        # Always clean up block capture state, even on exception.
        if selected_block >= 0:
            velocity_model._capture_block_idx = None

    student_block_h = None
    if selected_block >= 0:
        student_block_h = getattr(velocity_model, "_captured_block_output", None)
        velocity_model._captured_block_output = None

    # --- Phase 5: Loss ---
    if loss_space == "velocity":
        x_t = tgt_state.latent
        sigma_t = sigmas[selected_step]
        student_v = to_velocity(x_t, sigma_t, student_denoised)
        teacher_v = to_velocity(x_t, sigma_t, teacher_denoised)
        output_mse = F.mse_loss(student_v.float(), teacher_v.float())
    else:
        output_mse = F.mse_loss(student_denoised.float(), teacher_denoised.float())

    total_loss = output_loss_weight * output_mse

    block_mse = None
    if student_block_h is not None and teacher_block_h is not None:
        block_mse = compute_block_loss(student_block_h, teacher_block_h, mode=block_loss_mode)
        total_loss = total_loss + block_loss_weight * block_mse

    del stepper, stepped_latent
    torch.cuda.empty_cache()

    return WindowedDistillOutput(
        total_loss=total_loss,
        output_mse=output_mse,
        block_mse=block_mse,
        selected_step=selected_step,
        target_window=(target_ws, target_we),
        context_window=(windows[context_wi][0], windows[context_wi][1]) if context_wi is not None else None,
        _attn1_layers=attn1_layers,
    )
