"""Teacher-forcing flow matching loss for causal chunk refiner training.

**Two-pass forward** with per-layer KV cache (runs inside FSDP wrapper):

  Pass 1 (no_grad, sigma=0):
    Forward clean context chunks through each transformer block.
    After each block's attn1, capture post-RoPE K,V.

  Pass 2 (with grad, sigma=sigma_t):
    Forward noisy target chunk through each transformer block.
    Inject cached K,V as prefix so target attends to clean context.

The two-pass logic lives in X0Model._forward_teacher_forcing(), triggered
by setting model._tf_context before calling model.forward(). This ensures
FSDP2 root params (patchify_proj, adaln) are properly materialized.

Reference: HunyuanVideo diffusion-forcing (hyvideo1p5_ti2v_pipeline_from_pt_df_s3.py).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from diffusion.refiner.ltx_core.causal_attention import apply_causal_mask
from diffusion.refiner.ltx_core.flow_matching import (
    _augment_refiner_input,
    _make_inference_matched_stage2_endpoint,
)
from diffusion.refiner.ltx_core.ltx_student_wrapper import LTXStudentWrapper
from diffusion.refiner.vendor.ltx_core.components.patchifiers import VideoLatentPatchifier
from diffusion.refiner.vendor.ltx_core.tools import VideoLatentTools
from diffusion.refiner.vendor.ltx_core.types import VideoLatentShape
from diffusion.refiner.vendor.ltx_pipelines.utils.helpers import modality_from_latent_state

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class TeacherForcingFMOutput:
    loss: torch.Tensor
    velocity_mse: torch.Tensor
    x0_mse: torch.Tensor
    sigma_mean: torch.Tensor
    target_chunk_idx: int
    endpoint_abs: torch.Tensor
    target_velocity_abs: torch.Tensor
    pred_velocity_abs: torch.Tensor
    aug_scale_mean: torch.Tensor

    def metrics(self) -> dict[str, torch.Tensor]:
        return {
            "loss_velocity": self.velocity_mse.detach(),
            "loss_x0": self.x0_mse.detach(),
            "sigma_mean": self.sigma_mean.detach(),
            "target_chunk_idx": torch.tensor(
                float(self.target_chunk_idx),
                device=self.loss.device,
            ),
            "endpoint_abs": self.endpoint_abs.detach(),
            "target_velocity_abs": self.target_velocity_abs.detach(),
            "pred_velocity_abs": self.pred_velocity_abs.detach(),
            "aug_scale_mean": self.aug_scale_mean.detach(),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_frame_ranges(chunk_sizes: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    offset = 0
    for cs in chunk_sizes:
        ranges.append((offset, offset + cs))
        offset += cs
    return ranges


def _forward_teacher_forcing(
    student: LTXStudentWrapper,
    context_state: LatentState,
    target_state: LatentState,
    v_context: torch.Tensor,
    context_sigma: torch.Tensor,
    target_sigma: torch.Tensor,
) -> torch.Tensor:
    """Two-pass forward routed through X0Model.forward() for FSDP2 compat.

    Sets ``model._tf_context`` before calling ``model(video=target_mod, ...)``.
    Inside forward(), this triggers ``_forward_teacher_forcing()`` which does:
      - preprocess both context and target (root FSDP params materialized)
      - per-block: context no_grad → capture KV, target grad → inject KV
      - postprocess target → denoised x0
    """
    model = student.transformer  # X0Model (FSDP-wrapped)

    ctx_mod = modality_from_latent_state(context_state, v_context, context_sigma)
    tgt_mod = modality_from_latent_state(target_state, v_context, target_sigma)

    # Set context modality — forward() checks this and branches
    model._tf_context = ctx_mod
    try:
        denoised, _ = model(video=tgt_mod, audio=None, perturbations=None)
    finally:
        model._tf_context = None

    return denoised


# ---------------------------------------------------------------------------
# Core loss
# ---------------------------------------------------------------------------


def compute_teacher_forcing_fm_loss(
    *,
    student: LTXStudentWrapper,
    clean_target_latent: torch.Tensor,
    refiner_start_latent: torch.Tensor,
    v_context: torch.Tensor,
    sigmas: torch.Tensor,
    start_sigma: float,
    fps: float = 16.0,
    chunk_sizes: list[int],
    target_chunk_idx: int | None = None,
    aug_scale_min: float = 0.0,
    aug_scale_max: float = 0.0,
    truncate_future: bool = True,
) -> TeacherForcingFMOutput:
    """Teacher-forcing FM loss: two-pass forward with per-layer KV cache.

    Parameters
    ----------
    clean_target_latent : [B,C,F,H,W]  refiner GT (x0), used as clean context
    refiner_start_latent : [B,C,F,H,W]  sana output (z_sana), used for FM endpoint
    sigmas : [B]  per-sample sigma for the target chunk
    chunk_sizes : e.g. [5,3,3]
    truncate_future : drop frames after target chunk (saves memory)
    """
    if clean_target_latent.shape != refiner_start_latent.shape:
        raise ValueError("Shape mismatch")
    if clean_target_latent.ndim != 5:
        raise ValueError(f"Expected [B,C,F,H,W], got {tuple(clean_target_latent.shape)}")

    batch, channels, total_frames, height, width = clean_target_latent.shape
    device = clean_target_latent.device
    dtype = clean_target_latent.dtype

    if sum(chunk_sizes) != total_frames:
        raise ValueError(f"sum(chunk_sizes)={sum(chunk_sizes)} != {total_frames}")

    # --- 1. Select target chunk ---
    n_chunks = len(chunk_sizes)
    if target_chunk_idx is None:
        target_chunk_idx = random.randint(0, n_chunks - 1)

    chunk_ranges = _chunk_frame_ranges(chunk_sizes)
    target_start, target_end = chunk_ranges[target_chunk_idx]
    height * width

    # --- 2. Augmentation + FM endpoint ---
    augmented_input, endpoint_scales = _augment_refiner_input(
        refiner_start_latent,
        aug_scale_min=aug_scale_min,
        aug_scale_max=aug_scale_max,
    )
    endpoint_latent = _make_inference_matched_stage2_endpoint(
        augmented_input,
        start_sigma=float(start_sigma),
    )

    sigmas_clamped = sigmas.to(device=device, dtype=torch.float32).clamp(
        min=1e-6,
        max=float(start_sigma),
    )

    # --- 3. Build target chunk noisy latent ---
    alpha = sigmas_clamped / float(start_sigma)
    alpha_5d = alpha.view(-1, 1, 1, 1, 1)
    target_clean_5d = clean_target_latent[:, :, target_start:target_end, :, :]
    target_endpoint_5d = endpoint_latent[:, :, target_start:target_end, :, :]
    target_noisy_5d = (1.0 - alpha_5d) * target_clean_5d + alpha_5d * target_endpoint_5d

    # --- 4. Build context (clean x0 for chunks before target) ---
    if target_chunk_idx > 0:
        context_latent = clean_target_latent[:, :, :target_start, :, :]
        context_frames = target_start
        context_chunk_sizes = chunk_sizes[:target_chunk_idx]
    else:
        context_latent = None
        context_frames = 0
        context_chunk_sizes = []

    # --- 5. Patchify target ---
    target_frames = target_end - target_start
    target_tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=VideoLatentShape(
            batch=batch,
            channels=channels,
            frames=target_frames,
            height=height,
            width=width,
        ),
        fps=float(fps),
    )
    target_clean_state = target_tools.create_initial_state(
        device=device,
        dtype=dtype,
        initial_latent=target_clean_5d.to(device=device, dtype=dtype),
        temporal_offset=target_start,
    )
    target_noisy_patchified = target_tools.patchifier.patchify(
        target_noisy_5d.to(device=device, dtype=dtype),
    )
    target_state = replace(target_clean_state, latent=target_noisy_patchified.to(dtype))

    target_clean_patchified = target_clean_state.clean_latent
    target_endpoint_patchified = target_tools.patchifier.patchify(
        target_endpoint_5d.to(device=device, dtype=dtype),
    )

    # --- 6. Patchify context (if any) ---
    if context_latent is not None:
        context_tools = VideoLatentTools(
            patchifier=VideoLatentPatchifier(patch_size=1),
            target_shape=VideoLatentShape(
                batch=batch,
                channels=channels,
                frames=context_frames,
                height=height,
                width=width,
            ),
            fps=float(fps),
        )
        context_state = context_tools.create_initial_state(
            device=device,
            dtype=dtype,
            initial_latent=context_latent.to(device=device, dtype=dtype),
        )
        if len(context_chunk_sizes) > 1:
            context_state = apply_causal_mask(
                context_state,
                chunk_sizes=context_chunk_sizes,
                attention_mode="causal",
            )
    else:
        context_state = None

    # --- 7. Forward ---
    if context_state is not None:
        context_sigma = torch.zeros(batch, device=device, dtype=torch.float32)
        pred_x0 = _forward_teacher_forcing(
            student=student,
            context_state=context_state,
            target_state=target_state,
            v_context=v_context,
            context_sigma=context_sigma,
            target_sigma=sigmas_clamped,
        )
    else:
        # No context (target_chunk_idx == 0): single pass
        pred_x0 = student.forward_denoise(
            video_state=target_state,
            v_context=v_context,
            sigma=sigmas_clamped,
            apply_causal=False,
        )

    # --- 8. Compute loss on target chunk ---
    target_velocity = ((target_endpoint_patchified - target_clean_patchified) / float(start_sigma)).to(torch.float32)

    sigma_view = sigmas_clamped.view(-1, 1, 1).float()
    pred_velocity = (target_noisy_patchified.float() - pred_x0.float()) / sigma_view

    velocity_mse = F.mse_loss(pred_velocity, target_velocity)
    x0_mse = F.mse_loss(pred_x0.float(), target_clean_patchified.float())

    return TeacherForcingFMOutput(
        loss=velocity_mse,
        velocity_mse=velocity_mse,
        x0_mse=x0_mse,
        sigma_mean=sigmas_clamped.mean(),
        target_chunk_idx=target_chunk_idx,
        endpoint_abs=target_endpoint_patchified.float().abs().mean(),
        target_velocity_abs=target_velocity.abs().mean(),
        pred_velocity_abs=pred_velocity.abs().mean(),
        aug_scale_mean=endpoint_scales.float().mean(),
    )
