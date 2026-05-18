"""Progressive-noise sigma helpers for streaming LTX-2 refiner inference.

Inference-only helpers used by the chunk-causal AR refiner.

Exports:
    - ``build_progressive_sigma_table``
    - ``progressive_prompt_sigma``
"""

from __future__ import annotations

import torch


def _interp_1d(anchors: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Linear interpolation over anchor values at fractional positions."""
    lo = torch.floor(positions).long().clamp(0, anchors.numel() - 1)
    hi = torch.ceil(positions).long().clamp(0, anchors.numel() - 1)
    weight = (positions - lo.to(positions.dtype)).to(anchors.dtype)
    return anchors[lo] * (1.0 - weight) + anchors[hi] * weight


def _build_ltx2_scheduler_resampled_sigmas(
    *,
    num_denoising_steps: int,
    start_sigma: float,
    device: torch.device,
    dtype: torch.dtype,
    source_steps: int = 40,
) -> torch.Tensor:
    """Official LTX2Scheduler shape, adapted to progressive table length.

    The vendor refiner path uses ``LTX2Scheduler`` and then truncates the
    schedule to ``start_sigma``. The truncated effective step count is not
    generally divisible by ``generated_num`` (for the default 40-step schedule
    it is 23), so progressive training/inference needs a fixed-length
    resampling of that same curve. The first point is pinned to ``start_sigma``
    and the terminal zero is included only for interpolation; callers receive
    the first ``num_denoising_steps`` points, matching the table contract.
    """
    from diffusion.refiner.vendor.ltx_core.components.schedulers import LTX2Scheduler

    full = LTX2Scheduler().execute(steps=max(int(source_steps), int(num_denoising_steps)))
    full = full.to(device=device, dtype=torch.float32)
    truncated = None
    for i in range(full.numel()):
        if float(full[i].item()) <= float(start_sigma) + 1.0e-6:
            truncated = full[i:].clone()
            truncated[0] = float(start_sigma)
            break
    if truncated is None:
        raise ValueError(
            f"No LTX2Scheduler sigma <= start_sigma={start_sigma}; schedule={full.detach().cpu().tolist()}"
        )
    if not torch.isclose(truncated[-1], torch.zeros_like(truncated[-1])):
        truncated = torch.cat([truncated, torch.zeros(1, device=device, dtype=torch.float32)])

    positions = torch.linspace(
        0.0,
        float(truncated.numel() - 1),
        int(num_denoising_steps) + 1,
        device=device,
        dtype=torch.float32,
    )[:-1]
    sigmas = _interp_1d(truncated, positions)
    sigmas[0] = float(start_sigma)
    return sigmas.to(dtype=dtype)


def progressive_prompt_sigma(frame_sigmas: torch.Tensor, *, context_num: int) -> torch.Tensor:
    """Prompt-AdalN sigma for progressive mixed-sigma windows.

    Token timesteps are per-frame. Prompt-AdalN has one sigma per batch item,
    so use the mean active target-frame sigma, mirroring the existing ODE
    regression path for mixed-sigma chunks.
    """
    target_sigmas = frame_sigmas[int(context_num) :]
    active = target_sigmas[target_sigmas > 1.0e-6]
    if active.numel() == 0:
        active = frame_sigmas[frame_sigmas > 1.0e-6]
    if active.numel() == 0:
        return frame_sigmas.new_tensor(0.0)
    return active.mean()


def build_ltx_progressive_sigmas(
    *,
    num_denoising_steps: int,
    start_sigma: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    schedule_mode: str = "ltx2_scheduler",
) -> torch.Tensor:
    """Build a descending stage-2 sigma schedule in [start_sigma, 0).

    ``linear`` mirrors WorldCam's linearly sampled schedule with maximum
    ``start_sigma`` instead of 1.0. ``ltx2_scheduler`` resamples the truncated
    official LTX2 curve to the progressive table length. ``ltx_distilled_interpolate``
    linearly interpolates over the LTX stage-2 distilled anchors
    ``[start_sigma, 0.725, 0.421875, 0]``.
    """
    if num_denoising_steps <= 0:
        raise ValueError(f"num_denoising_steps must be > 0, got {num_denoising_steps}")
    if start_sigma <= 0:
        raise ValueError(f"start_sigma must be > 0, got {start_sigma}")

    if schedule_mode == "linear":
        sigmas = torch.linspace(
            float(start_sigma),
            0.0,
            int(num_denoising_steps) + 1,
            device=device,
            dtype=torch.float32,
        )[:-1]
    elif schedule_mode == "ltx_distilled_interpolate":
        anchors = torch.tensor(
            [float(start_sigma), 0.725, 0.421875, 0.0],
            device=device,
            dtype=torch.float32,
        )
        positions = torch.linspace(
            0.0,
            float(anchors.numel() - 1),
            int(num_denoising_steps) + 1,
            device=device,
            dtype=torch.float32,
        )[:-1]
        sigmas = _interp_1d(anchors, positions)
    elif schedule_mode in {"ltx2", "ltx2_scheduler", "ltx2_scheduler_resample"}:
        sigmas = _build_ltx2_scheduler_resampled_sigmas(
            num_denoising_steps=num_denoising_steps,
            start_sigma=float(start_sigma),
            device=device,
            dtype=torch.float32,
        )
    else:
        raise ValueError(
            f"Unsupported progressive schedule_mode={schedule_mode!r}; "
            "expected 'linear', 'ltx2_scheduler', or 'ltx_distilled_interpolate'."
        )

    return sigmas.to(dtype=dtype)


def build_progressive_sigma_table(
    *,
    context_num: int,
    generated_num: int,
    num_denoising_steps: int,
    start_sigma: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    schedule_mode: str = "ltx2_scheduler",
) -> tuple[dict[int, torch.Tensor], int]:
    """WorldCam-style per-frame sigma table adapted to LTX stage-2 sigma range.

    Returns ``(table, stage)`` where ``table[k]`` is a length-``window_frames``
    tensor of per-frame sigmas for sub-step ``k`` (``window_frames =
    context_num + generated_num``). Context tokens are at sigma 0; target
    tokens follow a staggered schedule.
    """
    context_num = int(context_num)
    generated_num = int(generated_num)
    num_denoising_steps = int(num_denoising_steps)
    if context_num <= 0 or generated_num <= 0:
        raise ValueError(f"context_num and generated_num must be positive, got {context_num}, {generated_num}")
    if num_denoising_steps % generated_num != 0:
        raise ValueError(
            "num_denoising_steps must be divisible by generated_num so each "
            f"target frame gets the same number of sub-steps; got {num_denoising_steps} and {generated_num}"
        )

    stage = num_denoising_steps // generated_num
    sigmas = build_ltx_progressive_sigmas(
        num_denoising_steps=num_denoising_steps,
        start_sigma=start_sigma,
        device=device,
        dtype=torch.float32,
        schedule_mode=schedule_mode,
    )
    sigmas_padded = torch.cat([sigmas, torch.zeros(context_num, device=device, dtype=torch.float32)]).flip(0)

    table: dict[int, torch.Tensor] = {}
    total = context_num + generated_num
    for k in range(stage):
        gen_sigmas = sigmas_padded[(context_num + stage - 1 - k) :: stage]
        frame_sigmas = torch.cat([sigmas_padded[:context_num], gen_sigmas])[:total]
        table[k] = frame_sigmas.to(dtype=dtype)
    return table, stage
