"""Flow-matching helpers for LTX-2.0 refiner LoRA training.

Semantics (train-inference aligned):

  clean target  x0 : refiner GT VAE latent (z_refiner)
  stage-1 input    : sana output VAE latent (z_sana)
  endpoint      x1 : inference-matched noised stage-2 start
                      x1 = (1 - start_sigma) * z_sana + start_sigma * eps
  sigma_max        : initial stage-2 sigma (0.909375)
  alpha            : sigma / sigma_max
  noisy state   xt : (1 - alpha) * x0 + alpha * x1
  target v         : (x1 - x0) / sigma_max

The model predicts denoised x0. Velocity supervision is obtained by
converting the prediction: v = (xt - x0_hat) / sigma.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from diffusion.refiner.ltx_core.ltx_student_wrapper import LTXStudentWrapper
from diffusion.refiner.vendor.ltx_core.components.patchifiers import VideoLatentPatchifier
from diffusion.refiner.vendor.ltx_core.tools import VideoLatentTools
from diffusion.refiner.vendor.ltx_core.types import VideoLatentShape

# ---------------------------------------------------------------------------
# Timestep samplers
# ---------------------------------------------------------------------------


class TimestepSampler:
    def sample(self, batch_size: int, seq_length: int, device: torch.device) -> torch.Tensor:
        raise NotImplementedError


class UniformTimestepSampler(TimestepSampler):
    def __init__(self, min_value: float = 0.0, max_value: float = 1.0):
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def sample(self, batch_size: int, seq_length: int, device: torch.device) -> torch.Tensor:
        del seq_length
        return torch.rand(batch_size, device=device) * (self.max_value - self.min_value) + self.min_value


class ShiftedLogitNormalTimestepSampler(TimestepSampler):
    """Token-count-aware shifted logit-normal sampler (matches LTX trainer default)."""

    def __init__(self, std: float = 1.0, eps: float = 1e-3, uniform_prob: float = 0.1):
        self.std = float(std)
        self.eps = float(eps)
        self.uniform_prob = float(uniform_prob)
        self.normal_999_percentile = 3.0902 * self.std
        self.normal_005_percentile = -2.5758 * self.std

    @staticmethod
    def _get_shift_for_sequence_length(
        seq_length: int,
        min_tokens: int = 1024,
        max_tokens: int = 4096,
        min_shift: float = 0.95,
        max_shift: float = 2.05,
    ) -> float:
        # Clamp seq_length to [min_tokens, max_tokens] before the linear
        # interpolation — extrapolating to seq_length >> max_tokens pushes
        # mu into the sigmoid saturation region (sigmoid(mu+3) == sigmoid(mu-2.5) == 1.0),
        # which produces 0/0 = NaN in the `stretched` ratio below.
        # 1-minute refiner training has seq_length ~= 106k tokens, far beyond
        # the 4096 the original formula was tuned for.
        seq_length = max(min_tokens, min(int(seq_length), max_tokens))
        slope = (max_shift - min_shift) / (max_tokens - min_tokens)
        intercept = min_shift - slope * min_tokens
        return slope * seq_length + intercept

    def sample(self, batch_size: int, seq_length: int, device: torch.device) -> torch.Tensor:
        mu = self._get_shift_for_sequence_length(int(seq_length))

        normal_samples = torch.randn((batch_size,), device=device) * self.std + mu
        logitnormal_samples = torch.sigmoid(normal_samples)

        percentile_999 = torch.sigmoid(torch.tensor(mu + self.normal_999_percentile, device=device))
        percentile_005 = torch.sigmoid(torch.tensor(mu + self.normal_005_percentile, device=device))

        stretched = (logitnormal_samples - percentile_005) / (percentile_999 - percentile_005)
        stretched = torch.where(stretched >= self.eps, stretched, 2 * self.eps - stretched)
        stretched = torch.clamp(stretched, 0.0, 1.0)

        uniform = (1 - self.eps) * torch.rand((batch_size,), device=device) + self.eps
        prob = torch.rand((batch_size,), device=device)
        return torch.where(prob > self.uniform_prob, stretched, uniform)


def build_timestep_sampler(
    mode: str = "shifted_logit_normal",
    *,
    std: float = 1.0,
    eps: float = 1e-3,
    uniform_prob: float = 0.1,
) -> TimestepSampler:
    if mode == "uniform":
        return UniformTimestepSampler(min_value=eps, max_value=1.0)
    if mode == "shifted_logit_normal":
        return ShiftedLogitNormalTimestepSampler(std=std, eps=eps, uniform_prob=uniform_prob)
    raise ValueError(f"Unsupported timestep sampler mode: {mode}")


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class RefinerFlowMatchingOutput:
    loss: torch.Tensor
    velocity_mse: torch.Tensor
    x0_mse: torch.Tensor
    sigma_mean: torch.Tensor
    endpoint_abs: torch.Tensor
    target_velocity_abs: torch.Tensor
    pred_velocity_abs: torch.Tensor
    aug_scale_mean: torch.Tensor
    raw_start_to_x0_mse: torch.Tensor
    endpoint_to_x0_mse: torch.Tensor

    def metrics(self) -> dict[str, torch.Tensor]:
        return {
            "loss_velocity": self.velocity_mse.detach(),
            "loss_x0": self.x0_mse.detach(),
            "sigma_mean": self.sigma_mean.detach(),
            "endpoint_abs": self.endpoint_abs.detach(),
            "target_velocity_abs": self.target_velocity_abs.detach(),
            "pred_velocity_abs": self.pred_velocity_abs.detach(),
            "aug_scale_mean": self.aug_scale_mean.detach(),
            "raw_start_to_x0_mse": self.raw_start_to_x0_mse.detach(),
            "endpoint_to_x0_mse": self.endpoint_to_x0_mse.detach(),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sigma_view(sigmas: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return sigmas.view(-1, *([1] * (reference.ndim - 1)))


def _augment_refiner_input(
    refiner_input_latent: torch.Tensor,
    *,
    aug_scale_min: float,
    aug_scale_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optional robustness augmentation on the clean stage-1 latent."""
    if aug_scale_max <= 0:
        zeros = torch.zeros(
            (refiner_input_latent.shape[0], 1, 1, 1, 1),
            device=refiner_input_latent.device,
            dtype=refiner_input_latent.dtype,
        )
        return refiner_input_latent, zeros

    if aug_scale_min < 0 or aug_scale_max < aug_scale_min:
        raise ValueError(f"Invalid refiner endpoint augmentation range: min={aug_scale_min}, max={aug_scale_max}")

    scales = torch.empty(
        (refiner_input_latent.shape[0], 1, 1, 1, 1),
        device=refiner_input_latent.device,
        dtype=refiner_input_latent.dtype,
    ).uniform_(float(aug_scale_min), float(aug_scale_max))
    gaussian = torch.randn_like(refiner_input_latent)
    augmented = refiner_input_latent * (1 - scales) + gaussian * scales
    return augmented, scales


def _make_inference_matched_stage2_endpoint(
    refiner_start_latent: torch.Tensor,
    *,
    start_sigma: float,
) -> torch.Tensor:
    """Match the inference initializer exactly:
    x1 = (1 - start_sigma) * z_sana + start_sigma * eps
    """
    noise = torch.randn_like(refiner_start_latent)
    return (1.0 - float(start_sigma)) * refiner_start_latent + float(start_sigma) * noise


# ---------------------------------------------------------------------------
# Core loss function
# ---------------------------------------------------------------------------


def compute_refiner_flow_matching_loss(
    *,
    student: LTXStudentWrapper,
    clean_target_latent: torch.Tensor,
    refiner_start_latent: torch.Tensor,
    v_context: torch.Tensor,
    sigmas: torch.Tensor,
    start_sigma: float,
    fps: float = 24.0,
    attention_mode: str = "bidirectional",
    window_radius: int | None = None,
    chunk_sizes: list[int] | None = None,
    aug_scale_min: float = 0.0,
    aug_scale_max: float = 0.0,
) -> RefinerFlowMatchingOutput:
    """Compute refiner flow-matching loss with inference-matched endpoints.

    Args:
        student: LTXStudentWrapper with trainable LoRA.
        clean_target_latent: [B,C,F,H,W] refiner GT (x0).
        refiner_start_latent: [B,C,F,H,W] sana output (z_sana).
        v_context: Text conditioning for the transformer.
        sigmas: [B] per-sample sigma values in (0, start_sigma].
        start_sigma: Maximum sigma (stage-2 initialization noise level).
        fps: Video FPS for positional encoding.
        attention_mode: "bidirectional" for FM training.
        aug_scale_min/max: Robustness augmentation range (0 = disabled).
    """
    if clean_target_latent.shape != refiner_start_latent.shape:
        raise ValueError(
            "clean_target_latent and refiner_start_latent must have the same shape, "
            f"got {tuple(clean_target_latent.shape)} vs {tuple(refiner_start_latent.shape)}"
        )
    if clean_target_latent.ndim != 5:
        raise ValueError(f"Expected latent tensors as [B,C,F,H,W], got {tuple(clean_target_latent.shape)}")
    if sigmas.ndim != 1 or sigmas.shape[0] != clean_target_latent.shape[0]:
        raise ValueError(
            f"sigmas must be shape [B], got {tuple(sigmas.shape)} for batch {clean_target_latent.shape[0]}"
        )
    if start_sigma <= 0:
        raise ValueError(f"start_sigma must be > 0, got {start_sigma}")

    device = clean_target_latent.device
    dtype = clean_target_latent.dtype
    batch, channels, frames, height, width = clean_target_latent.shape

    # 1) Optional augmentation on the stage-1 latent
    augmented_input, endpoint_scales = _augment_refiner_input(
        refiner_start_latent,
        aug_scale_min=aug_scale_min,
        aug_scale_max=aug_scale_max,
    )

    # 2) Inference-matched endpoint: x1 = (1 - σ_start) * z_sana + σ_start * ε
    endpoint_latent = _make_inference_matched_stage2_endpoint(
        augmented_input,
        start_sigma=float(start_sigma),
    )

    # 3) Patchify latents using vendor tools
    tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=VideoLatentShape(batch=batch, channels=channels, frames=frames, height=height, width=width),
        fps=float(fps),
    )
    clean_state = tools.create_initial_state(
        device=device,
        dtype=dtype,
        initial_latent=clean_target_latent.to(device=device, dtype=dtype),
    )
    clean_patchified = clean_state.clean_latent
    endpoint_patchified = tools.patchifier.patchify(endpoint_latent.to(device=device, dtype=dtype))

    # 4) Build noisy state via FM path interpolation
    sigmas = sigmas.to(device=device, dtype=torch.float32).clamp(min=1e-6, max=float(start_sigma))
    sigma_view = _sigma_view(sigmas, clean_patchified)
    alpha_view = _sigma_view(sigmas / float(start_sigma), clean_patchified)

    # xt = (1 - α) * x0 + α * x1
    noisy_latent = (1.0 - alpha_view) * clean_patchified + alpha_view * endpoint_patchified
    noisy_state = replace(clean_state, latent=noisy_latent.to(dtype))

    # Target velocity: v = (x1 - x0) / σ_start
    target_velocity = ((endpoint_patchified - clean_patchified) / float(start_sigma)).to(torch.float32)

    # 5) Student forward pass
    # bidirectional mode: skip causal mask entirely
    apply_causal = attention_mode != "bidirectional"
    effective_attention_mode = "causal" if attention_mode == "bidirectional" else attention_mode

    pred_x0 = student.forward_denoise(
        video_state=noisy_state,
        v_context=v_context,
        sigma=sigmas,
        apply_causal=apply_causal,
        chunk_sizes=chunk_sizes,
        attention_mode=effective_attention_mode,
        window_radius=window_radius,
    )

    # Ulysses CP: the student's transformer shards the sequence dim internally
    # and returns a local-seq-shard output. We therefore need to compare against
    # the matching local slice of the targets. Local per-rank MSE (.mean()) plus
    # FSDP's mean reduce-scatter across CP ranks recovers the global mean MSE.
    from diffusion.refiner.ltx_core.context_parallel import (
        cp_enabled,
        shard_along_seq,
    )

    if cp_enabled():
        noisy_latent_view = shard_along_seq(noisy_latent, dim=1)
        clean_patchified_view = shard_along_seq(clean_patchified, dim=1)
        endpoint_patchified_view = shard_along_seq(endpoint_patchified, dim=1)
        target_velocity_view = shard_along_seq(target_velocity, dim=1)
        sigma_view_local = shard_along_seq(sigma_view, dim=1) if sigma_view.shape[1] > 1 else sigma_view
    else:
        noisy_latent_view = noisy_latent
        clean_patchified_view = clean_patchified
        endpoint_patchified_view = endpoint_patchified
        target_velocity_view = target_velocity
        sigma_view_local = sigma_view

    # 6) Compute losses (on the local seq shard when CP is on)
    pred_velocity = (noisy_latent_view.float() - pred_x0.float()) / sigma_view_local.float()

    velocity_mse = F.mse_loss(pred_velocity, target_velocity_view)
    x0_mse = F.mse_loss(pred_x0.float(), clean_patchified_view.float())

    # Diagnostic metrics (keep them on the local shard too so they reflect the
    # exact tokens this rank scored against).
    raw_start_patchified = tools.patchifier.patchify(refiner_start_latent.to(device=device, dtype=dtype))
    raw_start_patchified_view = shard_along_seq(raw_start_patchified, dim=1) if cp_enabled() else raw_start_patchified
    raw_start_to_x0_mse = F.mse_loss(raw_start_patchified_view.float(), clean_patchified_view.float())
    endpoint_to_x0_mse = F.mse_loss(endpoint_patchified_view.float(), clean_patchified_view.float())

    return RefinerFlowMatchingOutput(
        loss=velocity_mse,
        velocity_mse=velocity_mse,
        x0_mse=x0_mse,
        sigma_mean=sigmas.mean(),
        endpoint_abs=endpoint_patchified_view.float().abs().mean(),
        target_velocity_abs=target_velocity_view.abs().mean(),
        pred_velocity_abs=pred_velocity.abs().mean(),
        aug_scale_mean=endpoint_scales.float().mean(),
        raw_start_to_x0_mse=raw_start_to_x0_mse,
        endpoint_to_x0_mse=endpoint_to_x0_mse,
    )
