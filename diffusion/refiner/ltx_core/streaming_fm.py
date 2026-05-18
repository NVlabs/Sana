"""Streaming refiner flow-matching loss with sink + sliding history window.

Simplified layout (after dropping the HistoryChannelCompressor): context is
just the clean refiner latent of prior chunks, no ``z_sana`` fusion. This
matches the teacher-forcing / self-forcing paradigm where history is "what
the model has already produced".

Empirically the compressor never learned to use z_sana (Frobenius ratio
sana/refined ~ 0.012 at step 5555), so removing it is free: ~1% net effect
on the forward, 99% reduction in moving parts.

Token layout per training/inference step::

    tokens = [ context_tokens (N_ctx) | current_tokens (N_cur) ]
              └─ sink + W history chunks (z_refiner) ─┘

    sigma_per_token:   0 for context      σ_t for current
    role:              K/V providers      query + loss target

Attention rule (FlexAttention block mask, one compiled kernel per shape)::

    disallowed iff  (q_idx < N_ctx) AND (kv_idx >= N_ctx)

Context queries never see current keys; current queries see everything.

Current chunk uses the truncated-σ refiner recipe (native to w60_10_30)::

    x1 = (1 − σ_start) · z_sana_cur + σ_start · ε        ← z_sana used only here
    α  = σ / σ_start                   (∈ (0, 1])
    xt = (1 − α) · z_refiner_cur + α · x1
    v_target = (x1 − z_refiner_cur) / σ_start
    loss = MSE(v_pred, v_target)       on current tokens only

Training-time augmentation is applied to the context z_refiner chunks to
simulate inference-time drift (the history is model-generated at inference,
not GT). Menu: gaussian re-noise / exposure / temporal blur / channel
dropout / history mixup / token dropout — all curriculum-ramped.
"""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.refiner.ltx_core.ltx_student_wrapper import LTXStudentWrapper
from diffusion.refiner.vendor.ltx_core.components.patchifiers import VideoLatentPatchifier
from diffusion.refiner.vendor.ltx_core.tools import VideoLatentTools
from diffusion.refiner.vendor.ltx_core.types import LatentState, VideoLatentShape

__all__ = [
    "apply_context_augmentation",
    "compute_streaming_fm_loss",
    "StreamingFMOutput",
    "build_streaming_flex_block_mask",
    "install_flex_block_mask_on_video_self_attn",
    "uninstall_flex_block_mask_on_video_self_attn",
    "streaming_flex_mask_scope",
    "sample_chunk_layout",
    "DEFAULT_AUG_FLAGS",
    "run_streaming_forward",
    "run_short_rollout",
    "sample_history_mode",
    "sample_rollout_config",
]


@contextlib.contextmanager
def streaming_flex_mask_scope(
    student: LTXStudentWrapper,
    *,
    n_context_tokens: int,
    n_total_tokens: int,
    device: torch.device,
):
    """Context manager: install the FlexAttention streaming block mask for the
    duration of the ``with`` block, then restore the original attention_fn.

    Spans both forward AND backward — required because gradient checkpointing
    re-runs forward during backward, and any install/uninstall that happens
    only around the forward would leave the recomputation seeing a different
    attention function (→ different number of saved tensors → CheckpointError).
    """
    block_mask = build_streaming_flex_block_mask(
        n_context_tokens=n_context_tokens,
        n_total_tokens=n_total_tokens,
        device=device,
    )
    saved = install_flex_block_mask_on_video_self_attn(student, block_mask)
    try:
        yield
    finally:
        uninstall_flex_block_mask_on_video_self_attn(saved)


# ---------------------------------------------------------------------------
# Augmentation menu (applied to context z_refiner latents only, training-time)
# ---------------------------------------------------------------------------


DEFAULT_AUG_FLAGS: dict[str, bool] = {
    "gaussian_renoise": True,
    "exposure": True,
    "temporal_blur": True,
    "channel_dropout": True,
}


def apply_context_augmentation(
    z_refined: torch.Tensor,
    *,
    training: bool,
    strength: float = 1.0,
    flags: dict[str, bool] | None = None,
) -> torch.Tensor:
    """Frame-aware corruption of a context z_refined latent to simulate drift.

    Simulates what the model will see at inference time: the refiner's own
    past output contains accumulated error. Applying perturbations to the
    clean GT training history closes this train/test distribution gap.
    """
    if not training or strength <= 0.0:
        return z_refined
    if flags is None:
        flags = DEFAULT_AUG_FLAGS
    s = float(max(0.0, min(1.0, strength)))

    x = z_refined
    B, C, T, H, W = x.shape
    device = x.device
    dtype = x.dtype

    if flags.get("gaussian_renoise", False):
        sigma_drift = torch.rand(B, 1, 1, 1, 1, device=device) * (0.1 * s)
        eps = torch.randn_like(x)
        x = (1.0 - sigma_drift).to(dtype) * x + sigma_drift.to(dtype) * eps
    if flags.get("exposure", False):
        scale = 1.0 + (torch.rand(B, C, 1, 1, 1, device=device, dtype=dtype) - 0.5) * (0.3 * s)
        x = x * scale
    if flags.get("temporal_blur", False) and T > 2:
        alpha = torch.rand(B, 1, 1, 1, 1, device=device, dtype=dtype) * (0.2 * s)
        neighbour_avg = 0.5 * (x[:, :, :-2] + x[:, :, 2:])
        blended = (1.0 - alpha) * x[:, :, 1:-1] + alpha * neighbour_avg
        x = torch.cat([x[:, :, :1], blended, x[:, :, -1:]], dim=2)
    if flags.get("channel_dropout", False):
        drop_prob = 0.1 * s
        if drop_prob > 0.0:
            keep = (torch.rand(B, C, 1, 1, 1, device=device) >= drop_prob).to(dtype)
            x = x * keep
    return x


# ---------------------------------------------------------------------------
# FlexAttention block mask for streaming (context KV-only, current full)
# ---------------------------------------------------------------------------


def build_streaming_flex_block_mask(
    *,
    n_context_tokens: int,
    n_total_tokens: int,
    device: torch.device,
):
    """Compile a FlexAttention block mask: context q → context kv only; current q → all.

    disallowed iff  (q_idx < n_context_tokens) AND (kv_idx >= n_context_tokens)
    """
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:
        raise RuntimeError("FlexAttention not available") from exc

    n_ctx = int(n_context_tokens)

    def mask_mod(b, h, q_idx, kv_idx):
        q_is_current = q_idx >= n_ctx
        kv_is_current = kv_idx >= n_ctx
        return q_is_current | (~kv_is_current)  # allowed unless (ctx q AND cur kv)

    compiled = torch.compile(create_block_mask)
    return compiled(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=int(n_total_tokens),
        KV_LEN=int(n_total_tokens),
        device=str(device),
    )


class _FlexBlockMaskAttentionFn:
    _compiled_flex = None

    @classmethod
    def _get_compiled(cls):
        if cls._compiled_flex is None:
            from torch.nn.attention.flex_attention import flex_attention

            cls._compiled_flex = torch.compile(flex_attention, dynamic=False)
        return cls._compiled_flex

    def __init__(self, block_mask):
        self._flex = self._get_compiled()
        self._block_mask = block_mask

    def __call__(self, q, k, v, heads, mask=None):
        del mask
        b, _, qk_dim = q.shape
        dim_head = qk_dim // heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))
        out = self._flex(q, k, v, block_mask=self._block_mask)
        return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


def install_flex_block_mask_on_video_self_attn(
    student: LTXStudentWrapper,
    block_mask,
) -> list[tuple[nn.Module, object]]:
    """Swap ``attn1.attention_function`` on every block to FlexAttention."""
    fn = _FlexBlockMaskAttentionFn(block_mask)
    saved: list[tuple[nn.Module, object]] = []
    for module in student.transformer.modules():
        attn1 = getattr(module, "attn1", None)
        if attn1 is None:
            continue
        if not hasattr(attn1, "attention_function"):
            continue
        saved.append((attn1, attn1.attention_function))
        attn1.attention_function = fn
    return saved


def uninstall_flex_block_mask_on_video_self_attn(
    saved: list[tuple[nn.Module, object]],
) -> None:
    for attn, orig in saved:
        attn.attention_function = orig


# ---------------------------------------------------------------------------
# Core streaming FM loss
# ---------------------------------------------------------------------------


@dataclass
class StreamingFMOutput:
    loss: torch.Tensor
    velocity_mse: torch.Tensor
    x0_mse: torch.Tensor
    sigma_mean: torch.Tensor
    context_tokens: int
    current_tokens: int
    pred_velocity_abs: torch.Tensor
    target_velocity_abs: torch.Tensor

    def metrics(self) -> dict[str, torch.Tensor]:
        dev = self.loss.device
        return {
            "loss_velocity": self.velocity_mse.detach(),
            "loss_x0": self.x0_mse.detach(),
            "sigma_mean": self.sigma_mean.detach(),
            "context_tokens": torch.tensor(float(self.context_tokens), device=dev),
            "current_tokens": torch.tensor(float(self.current_tokens), device=dev),
            "pred_velocity_abs": self.pred_velocity_abs.detach(),
            "target_velocity_abs": self.target_velocity_abs.detach(),
        }


def _build_combined_state(
    context_chunks: list[torch.Tensor],  # each [B, C, T_i, H, W]  z_refiner of history
    context_temporal_offsets: list[int],
    current_noisy_5d: torch.Tensor,  # [B, C, T_cur, H, W]
    current_clean_5d: torch.Tensor,  # [B, C, T_cur, H, W]
    current_temporal_offset: int,
    *,
    fps: float,
    device: torch.device,
    dtype: torch.dtype,
    token_dropout_prob: float = 0.0,
) -> tuple[LatentState, int, int]:
    """Assemble a single LatentState with [context | current] tokens concat'd.

    ``denoise_mask`` is the vendor's binary update mask (1 = denoise this token,
    0 = frozen clean conditioning). ``timesteps_from_mask`` multiplies it by the
    scalar sigma passed to ``forward_denoise``, so per-token AdaLN timesteps
    come out as ``[0 for context, sigma for current]`` — matching intent.

    The scalar ``sigma`` is consumed by ``prompt_adaln`` (cross-attn AdaLN) if
    the base model has one. Keeping ``mask = 0/1`` and passing real sigma
    scalar means that path sees the right value even if we ever swap bases.

    Returns (state, n_context_tokens, n_current_tokens).
    """
    if not context_chunks:
        raise ValueError("streaming FM requires at least one context chunk (sink)")
    patchifier = VideoLatentPatchifier(patch_size=1)
    B = current_noisy_5d.shape[0]
    C = current_noisy_5d.shape[1]
    H = current_noisy_5d.shape[3]
    W = current_noisy_5d.shape[4]

    # Patchify each context chunk with its own temporal_offset
    ctx_latents: list[torch.Tensor] = []
    ctx_positions: list[torch.Tensor] = []
    ctx_clean_latents: list[torch.Tensor] = []
    for chunk, t_off in zip(context_chunks, context_temporal_offsets):
        T_c = chunk.shape[2]
        tools = VideoLatentTools(
            patchifier=patchifier,
            target_shape=VideoLatentShape(batch=B, channels=C, frames=T_c, height=H, width=W),
            fps=float(fps),
        )
        state_c = tools.create_initial_state(
            device=device,
            dtype=dtype,
            initial_latent=chunk.to(device=device, dtype=dtype),
            temporal_offset=int(t_off),
        )
        ctx_latents.append(state_c.latent)
        ctx_positions.append(state_c.positions)
        ctx_clean_latents.append(state_c.clean_latent)

    # Patchify current chunk
    cur_tools = VideoLatentTools(
        patchifier=patchifier,
        target_shape=VideoLatentShape(
            batch=B,
            channels=C,
            frames=current_noisy_5d.shape[2],
            height=H,
            width=W,
        ),
        fps=float(fps),
    )
    cur_clean_state = cur_tools.create_initial_state(
        device=device,
        dtype=dtype,
        initial_latent=current_clean_5d.to(device=device, dtype=dtype),
        temporal_offset=int(current_temporal_offset),
    )
    cur_noisy_patched = cur_tools.patchifier.patchify(
        current_noisy_5d.to(device=device, dtype=dtype),
    )

    # Concat tokens / positions / clean_latent
    latent_combined = torch.cat(ctx_latents + [cur_noisy_patched], dim=1)
    positions_combined = torch.cat(ctx_positions + [cur_clean_state.positions], dim=2)
    clean_combined = torch.cat(ctx_clean_latents + [cur_clean_state.clean_latent], dim=1)

    n_context = sum(c.shape[1] for c in ctx_latents)
    n_current = cur_noisy_patched.shape[1]

    # Optional token dropout on context tokens
    if token_dropout_prob > 0.0:
        keep = (torch.rand(B, n_context, 1, device=device) >= token_dropout_prob).to(latent_combined.dtype)
        keep_full = torch.ones(
            B,
            n_context + n_current,
            1,
            device=device,
            dtype=latent_combined.dtype,
        )
        keep_full[:, :n_context, :] = keep
        latent_combined = latent_combined * keep_full

    # Binary denoise_mask: 0 for context (frozen), 1 for current (update).
    # Per-token AdaLN timestep is computed downstream as ``mask * sigma``.
    denoise_mask = torch.zeros(
        B,
        n_context + n_current,
        1,
        device=device,
        dtype=torch.float32,
    )
    denoise_mask[:, n_context:, 0] = 1.0

    return (
        LatentState(
            latent=latent_combined,
            denoise_mask=denoise_mask,
            positions=positions_combined.to(dtype),
            clean_latent=clean_combined,
            attention_mask=None,
        ),
        n_context,
        n_current,
    )


def _apply_history_mixup(
    context_chunks: list[torch.Tensor],
    *,
    strength: float,
) -> list[torch.Tensor]:
    """Cross-blend adjacent history chunks in latent space (≥ 2 hist chunks only).

    The first element of ``context_chunks`` is the sink and is kept stable;
    only adjacent hist chunks (indices 1+) are mixed.
    """
    if len(context_chunks) < 3 or strength <= 0.0:
        return context_chunks
    new_chunks = list(context_chunks)
    for i in range(1, len(new_chunks) - 1):
        if new_chunks[i].shape != new_chunks[i + 1].shape:
            continue
        B = new_chunks[i].shape[0]
        alpha = torch.rand(B, 1, 1, 1, 1, device=new_chunks[i].device, dtype=new_chunks[i].dtype) * (
            0.1 * float(strength)
        )
        new_chunks[i] = (1.0 - alpha) * new_chunks[i] + alpha * new_chunks[i + 1]
    return new_chunks


def compute_streaming_fm_loss(
    *,
    student: LTXStudentWrapper,
    # context chunks: sink at index 0, history chunks at indices 1..W (z_refiner)
    context_z_ref_list: list[torch.Tensor],
    context_temporal_offsets: list[int],
    # current chunk
    cur_z_sana: torch.Tensor,
    cur_z_ref: torch.Tensor,
    cur_temporal_offset: int,
    # text context
    v_context: torch.Tensor,
    # FM schedule
    sigma: torch.Tensor,
    start_sigma: float = 0.909375,
    fps: float = 16.0,
    # augmentation config
    apply_augmentation: bool = True,
    aug_strength: float = 1.0,
    aug_flags: dict[str, bool] | None = None,
    token_dropout_prob: float = 0.0,
) -> StreamingFMOutput:
    """Streaming FM loss — single forward over [context | current] with block mask.

    Context = z_refiner of sink + history chunks (clean GT at training,
    model's own output at inference). z_sana is ONLY used inside current
    chunk for the truncated-σ FM endpoint construction.
    """

    if cur_z_sana.shape != cur_z_ref.shape:
        raise ValueError("cur_z_sana / cur_z_ref shape mismatch")
    if cur_z_ref.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(cur_z_ref.shape)}")
    if len(context_z_ref_list) != len(context_temporal_offsets):
        raise ValueError("context offsets length mismatch")
    if not context_z_ref_list:
        raise ValueError("streaming FM requires ≥ 1 context chunk (sink)")

    batch, channels, T_cur, height, width = cur_z_ref.shape
    device = cur_z_ref.device
    dtype = cur_z_ref.dtype

    aug_flags = aug_flags if aug_flags is not None else DEFAULT_AUG_FLAGS

    # 1. Apply augmentation to each context chunk (z_refiner only)
    context_chunks: list[torch.Tensor] = []
    for zr in context_z_ref_list:
        zr_a = apply_context_augmentation(
            zr,
            training=apply_augmentation,
            strength=aug_strength,
            flags=aug_flags,
        )
        context_chunks.append(zr_a)

    if apply_augmentation and aug_flags.get("history_mixup", False):
        context_chunks = _apply_history_mixup(context_chunks, strength=aug_strength)

    # 2. Current chunk truncated-σ FM noising
    sigma_clamped = sigma.to(device=device, dtype=torch.float32).clamp(
        min=1e-6,
        max=float(start_sigma),
    )
    eps = torch.randn_like(cur_z_ref)
    x1 = (1.0 - float(start_sigma)) * cur_z_sana + float(start_sigma) * eps
    alpha = (sigma_clamped / float(start_sigma)).view(-1, 1, 1, 1, 1).to(dtype)
    xt = (1.0 - alpha) * cur_z_ref + alpha * x1
    v_target_5d = ((x1 - cur_z_ref) / float(start_sigma)).to(torch.float32)

    # 3. Assemble combined state [context | current] with binary denoise_mask
    state, n_ctx, n_cur = _build_combined_state(
        context_chunks,
        context_temporal_offsets,
        current_noisy_5d=xt,
        current_clean_5d=cur_z_ref,
        current_temporal_offset=cur_temporal_offset,
        fps=fps,
        device=device,
        dtype=dtype,
        token_dropout_prob=(token_dropout_prob if apply_augmentation else 0.0),
    )

    # 4. Single forward. Pass the REAL scalar sigma — per-token AdaLN timesteps
    #    come out as mask*sigma = [0, ..., sigma]; scalar sigma also feeds
    #    prompt_adaln correctly if the base model has one.
    pred_all = student.forward_denoise(
        video_state=state,
        v_context=v_context,
        sigma=sigma_clamped,
        apply_causal=False,
        attention_mode="bidirectional",
    )

    # 5. Slice current tokens; compute velocity loss
    pred_cur = pred_all[:, n_ctx:, :]
    patchifier = VideoLatentPatchifier(patch_size=1)
    xt_patched = patchifier.patchify(xt.to(dtype))
    v_target_patched = patchifier.patchify(v_target_5d.to(dtype)).float()
    clean_patched = patchifier.patchify(cur_z_ref.to(dtype)).float()

    sigma_view = sigma_clamped.view(-1, 1, 1).float()
    pred_v = (xt_patched.float() - pred_cur.float()) / sigma_view

    velocity_mse = F.mse_loss(pred_v, v_target_patched)
    x0_mse = F.mse_loss(pred_cur.float(), clean_patched)

    return StreamingFMOutput(
        loss=velocity_mse,
        velocity_mse=velocity_mse,
        x0_mse=x0_mse,
        sigma_mean=sigma_clamped.mean(),
        context_tokens=n_ctx,
        current_tokens=n_cur,
        pred_velocity_abs=pred_v.abs().mean(),
        target_velocity_abs=v_target_patched.abs().mean(),
    )


# ---------------------------------------------------------------------------
# Inference helper: single chunk forward (reuses same forward path)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def run_streaming_forward(
    *,
    student: LTXStudentWrapper,
    context_z_ref_list: list[torch.Tensor],
    context_temporal_offsets: list[int],
    cur_latent_5d: torch.Tensor,
    cur_clean_5d: torch.Tensor,
    cur_temporal_offset: int,
    v_context: torch.Tensor,
    sigma: torch.Tensor,
    fps: float = 16.0,
) -> torch.Tensor:
    """One forward pass for inference — returns pred_x0 for current chunk tokens.

    Returns ``[B, n_cur, C]``. Caller handles Euler step + unpatchify.
    """
    if not context_z_ref_list:
        raise ValueError("streaming inference requires ≥ 1 context chunk (sink)")

    B, C, T_cur, H, W = cur_latent_5d.shape
    device = cur_latent_5d.device
    dtype = cur_latent_5d.dtype

    context_chunks = [zr.to(device=device, dtype=dtype) for zr in context_z_ref_list]
    sigma_clamped = sigma.to(device=device, dtype=torch.float32).clamp(min=1e-6)

    state, n_ctx, n_cur = _build_combined_state(
        context_chunks,
        context_temporal_offsets,
        current_noisy_5d=cur_latent_5d,
        current_clean_5d=cur_clean_5d,
        current_temporal_offset=cur_temporal_offset,
        fps=fps,
        device=device,
        dtype=dtype,
        token_dropout_prob=0.0,
    )
    with streaming_flex_mask_scope(
        student,
        n_context_tokens=n_ctx,
        n_total_tokens=n_ctx + n_cur,
        device=device,
    ):
        pred_all = student.forward_denoise(
            video_state=state,
            v_context=v_context,
            sigma=sigma_clamped,
            apply_causal=False,
            attention_mode="bidirectional",
        )
    return pred_all[:, n_ctx:, :]


# ---------------------------------------------------------------------------
# Chunk layout sampler
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Self-rollout: generate history on-policy (CFG=1 only, variable steps)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def run_short_rollout(
    student: LTXStudentWrapper,
    z_sana_chunk: torch.Tensor,  # [B, C, T, H, W]
    v_context: torch.Tensor,
    temporal_offset: int,
    *,
    num_steps: int,
    start_sigma: float = 0.909375,
    fps: float = 16.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Run a short self-rollout to produce a synthetic z_refined latent.

    Used during training to mix "rollout history" into the context window,
    matching the distribution the model will actually produce at inference.

    Grad policy: ``torch.inference_mode`` throughout; output is ``.detach()``.
    CFG is NOT used here (single forward per step, matching CFG=1 inference).

    Args:
      z_sana_chunk: sana stage-1 latent for the chunk, ``[B, C, T, H, W]``
      num_steps: Euler steps (from the config's ``step_choices`` pool).
        The LTX2Scheduler produces ``num_steps + 1`` sigma points; we
        truncate to start_sigma so effective update count may be slightly
        smaller for very small num_steps.

    Returns: ``[B, C, T, H, W]`` — a z_refined-equivalent latent produced by
    the model's own rollout, suitable for use as a history context chunk.
    """
    # Lazy imports to avoid top-level circular dependencies
    from diffusion.refiner.vendor.ltx_core.components.diffusion_steps import EulerDiffusionStep
    from diffusion.refiner.vendor.ltx_core.components.schedulers import LTX2Scheduler

    B, C, T, H, W = z_sana_chunk.shape
    device = z_sana_chunk.device
    dtype = z_sana_chunk.dtype

    # Initial noisy latent: x1 = (1 - σ_start) z_sana + σ_start ε (matches inference)
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))
        eps = torch.randn(B, C, T, H, W, generator=gen, device=device, dtype=dtype)
    else:
        eps = torch.randn_like(z_sana_chunk)
    x_t = (1.0 - float(start_sigma)) * z_sana_chunk + float(start_sigma) * eps

    # Build sigma schedule (truncated to start_sigma)
    full = (
        LTX2Scheduler()
        .execute(steps=max(num_steps, 1))
        .to(
            dtype=torch.float32,
            device=device,
        )
    )
    sigmas = None
    for i in range(len(full)):
        if full[i].item() <= float(start_sigma) + 1e-6:
            sigmas = full[i:].clone()
            sigmas[0] = float(start_sigma)
            break
    if sigmas is None or len(sigmas) < 2:
        sigmas = torch.tensor([float(start_sigma), 0.0], device=device, dtype=torch.float32)

    tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=VideoLatentShape(batch=B, channels=C, frames=T, height=H, width=W),
        fps=float(fps),
    )
    state = tools.create_initial_state(
        device=device,
        dtype=dtype,
        initial_latent=x_t,
        temporal_offset=int(temporal_offset),
    )
    stepper = EulerDiffusionStep()

    for step_idx in range(len(sigmas) - 1):
        sigma = sigmas[step_idx]
        pred = student.forward_denoise(
            video_state=state,
            v_context=v_context,
            sigma=sigma,
            apply_causal=False,
            attention_mode="bidirectional",
        )
        new_latent = stepper.step(state.latent, pred, sigmas, step_idx)
        state = replace(state, latent=new_latent)

    final = tools.unpatchify(tools.clear_conditioning(state)).latent
    return final.detach()


# ---------------------------------------------------------------------------
# History-mode curriculum sampler (clean / aug / rollout)
# ---------------------------------------------------------------------------


def sample_history_mode(
    step: int,
    *,
    warmup_steps: int = 500,
    aug_ramp_steps: int = 2000,
    rollout_ramp_steps: int = 2500,
    steady_weights: dict[str, float] | None = None,
    rng: random.Random | None = None,
) -> str:
    """Sample one of ``'clean'`` / ``'aug'`` / ``'rollout'`` via 3-phase curriculum.

    Phase boundaries:
      [0, warmup_steps):                  all clean
      [warmup, warmup+aug_ramp):          clean ramp-down, aug ramp-up (to 0.7)
      [warmup+aug_ramp, +rollout_ramp):   rollout ramp-up, aug & clean settle
      [end, ∞):                            fixed steady_weights

    Defaults give the trajectory:
      step 0      : (1.00, 0.00, 0.00)
      step 1500   : (0.65, 0.35, 0.00)
      step 2500   : (0.30, 0.70, 0.00)
      step 3750   : (0.25, 0.50, 0.25)
      step 5000+  : steady_weights (default 0.20, 0.30, 0.50)
    """
    if steady_weights is None:
        steady_weights = {"clean": 0.2, "aug": 0.3, "rollout": 0.5}
    r = rng if rng is not None else random
    end_aug = warmup_steps + aug_ramp_steps
    end_rollout = end_aug + rollout_ramp_steps

    if step < warmup_steps:
        weights = [1.0, 0.0, 0.0]
    elif step < end_aug:
        t = (step - warmup_steps) / max(aug_ramp_steps, 1)
        weights = [1.0 - t * 0.7, t * 0.7, 0.0]
    elif step < end_rollout:
        t = (step - end_aug) / max(rollout_ramp_steps, 1)
        weights = [0.3 - t * 0.1, 0.7 - t * 0.4, t * 0.5]
    else:
        weights = [
            steady_weights.get("clean", 0.2),
            steady_weights.get("aug", 0.3),
            steady_weights.get("rollout", 0.5),
        ]
    modes = ["clean", "aug", "rollout"]
    return r.choices(modes, weights=weights, k=1)[0]


def sample_rollout_config(
    rng: random.Random | None = None,
    *,
    step_choices: list[int] | None = None,
) -> dict:
    """Sample a rollout config (step count only; CFG is always 1.0)."""
    if step_choices is None:
        step_choices = [1, 3, 5, 10]
    r = rng if rng is not None else random
    return {"num_steps": int(r.choice(step_choices))}


# ---------------------------------------------------------------------------
# Chunk layout sampler
# ---------------------------------------------------------------------------


def sample_chunk_layout(
    total_frames: int,
    *,
    first_chunk: int = 1,
    chunk_size: int = 3,
    window_size: int = 2,
    rng: random.Random | None = None,
) -> tuple[list[tuple[int, int]], list[int], int]:
    """Return ``(chunks, ctx_indices, cur_idx)`` for one video."""
    if total_frames < first_chunk + chunk_size * (window_size + 1):
        raise ValueError(
            f"video has {total_frames} latent frames; need ≥ "
            f"{first_chunk + chunk_size * (window_size + 1)} for W={window_size}"
        )
    chunks: list[tuple[int, int]] = [(0, first_chunk)]
    offset = first_chunk
    while offset + chunk_size <= total_frames:
        chunks.append((offset, offset + chunk_size))
        offset += chunk_size

    n_chunks = len(chunks)
    min_cur_idx = 1 + window_size
    if n_chunks < min_cur_idx + 1:
        raise ValueError(f"only {n_chunks} chunks — need ≥ {min_cur_idx + 1} for W={window_size}")
    r = rng if rng is not None else random
    cur_idx = r.randint(min_cur_idx, n_chunks - 1)
    ctx_indices = [0] + list(range(cur_idx - window_size, cur_idx))
    return chunks, ctx_indices, cur_idx
