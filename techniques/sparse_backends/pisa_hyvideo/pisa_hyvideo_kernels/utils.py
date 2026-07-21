from __future__ import annotations

import math
import torch


def threshold_grid(start: float, end: float, step: float) -> list[float]:
    n = int(round((end - start) / step))
    return [round(start + i * step, 6) for i in range(n + 1)]


def round_int8(x: torch.Tensor) -> torch.Tensor:
    rounded = x + 0.5 * torch.where(x >= 0, 1, -1)
    return rounded.clamp(-128, 127).to(torch.int8)


@torch.no_grad()
def estimate_density(
    q: torch.Tensor,
    k: torch.Tensor,
    thresholds: list[float],
    mode: str,
    block_size: int,
    group_size: int,
    sample_heads: int,
    q_stride_blocks: int,
    q_block_size: int | None = None,
    kv_block_size: int | None = None,
    q_start_offset: int | None = None,
    sink_last_blocks: int = 0,
    sink_last_tokens: int = 0,
) -> list[dict[str, float]]:
    q_block_size = block_size if q_block_size is None else q_block_size
    kv_block_size = block_size if kv_block_size is None else kv_block_size

    bsz, heads, q_len, dim = q.shape
    kv_len = k.shape[-2]
    q_start_offset = kv_len - q_len if q_start_offset is None else q_start_offset
    nt_q = q_len // q_block_size
    nt_kv = kv_len // kv_block_size
    sink_last_blocks = max(int(sink_last_blocks), 0)
    if sink_last_tokens:
        sink_last_blocks = max(sink_last_blocks, math.ceil(int(sink_last_tokens) / kv_block_size))
    usable_q = nt_q * q_block_size
    usable_kv = nt_kv * kv_block_size
    q = q[:, :, :usable_q].float()
    k = k[:, :, :usable_kv].float()
    head_ids = torch.linspace(0, heads - 1, min(sample_heads, heads), device=q.device).round().long().unique()
    q_block_ids = torch.arange(0, nt_q, q_stride_blocks, device=q.device)
    if q_block_ids.numel() and q_block_ids[-1].item() != nt_q - 1:
        q_block_ids = torch.cat([q_block_ids, torch.tensor([nt_q - 1], device=q.device)])

    thresholds_t = torch.tensor([float(t) for t in thresholds], device=q.device, dtype=torch.float32)
    totals_t = torch.zeros((len(thresholds),), device=q.device, dtype=torch.float64)
    slots = 0
    scale = dim**-0.5
    all_kv_block_ids = torch.arange(nt_kv, device=q.device)

    for b in range(bsz):
        for h in head_ids.tolist():
            k_blocks = k[b, h].reshape(nt_kv, kv_block_size, dim)
            kc = k_blocks.mean(dim=1)
            kc_scale = kc.abs().amax(dim=1) / 127.0 + 1e-7
            kc_int8 = round_int8(kc / kc_scale[:, None]).float()
            if mode == "diag":
                kc_mean = kc.mean(dim=0)
                kc_var_diag = torch.clamp((kc * kc).mean(dim=0) - kc_mean * kc_mean, min=0.0)

            for q_idx in q_block_ids.tolist():
                q_start = q_idx * q_block_size
                q_abs_start = q_start_offset + q_start
                q_abs_end = q_abs_start + q_block_size
                kv_start = all_kv_block_ids * kv_block_size
                kv_end = kv_start + kv_block_size
                qb = q[b, h, q_start : q_start + q_block_size]
                local = (kv_start < q_abs_end + kv_block_size) & (kv_end > q_abs_start - kv_block_size)
                sink_last = (sink_last_blocks > 0) & (all_kv_block_ids >= nt_kv - sink_last_blocks)
                force_exact = local | sink_last
                if mode == "original":
                    q_scale = qb.abs().amax() / 127.0 + 1e-7
                    qb_int8 = round_int8(qb / q_scale).float()
                    token_scores = (qb_int8 @ kc_int8.T) * (q_scale * kc_scale[None, :] * scale)
                    col_mean = token_scores.mean(dim=0)
                    col_std = torch.sqrt(
                        torch.clamp(token_scores.square().mean(dim=0) - col_mean.square(), min=0.0) + 1e-6
                    )
                    scores = col_mean[None, :] + thresholds_t[:, None] * col_std[None, :]
                    group_count = math.ceil(nt_kv / group_size)
                    padded = group_count * group_size
                    if padded != nt_kv:
                        pad = padded - nt_kv
                        scores = torch.nn.functional.pad(scores, (0, pad), value=0.0)
                        force_exact_padded = torch.nn.functional.pad(force_exact, (0, pad), value=True)
                        valid = torch.zeros((padded,), device=q.device, dtype=torch.bool)
                        valid[:nt_kv] = True
                    else:
                        force_exact_padded = force_exact
                        valid = torch.ones((nt_kv,), device=q.device, dtype=torch.bool)
                    scores_g = scores.reshape(len(thresholds), group_count, group_size)
                    valid_g = valid.reshape(group_count, group_size)
                    force_exact_g = force_exact_padded.reshape(group_count, group_size)
                    denom = valid_g.sum(dim=1).clamp_min(1).to(torch.float32)
                    group_mean = (scores_g * valid_g[None, :, :]).sum(dim=2) / denom[None, :]
                    group_sq_mean = (scores_g.square() * valid_g[None, :, :]).sum(dim=2) / denom[None, :]
                    group_std = torch.sqrt(torch.clamp(group_sq_mean - group_mean.square(), min=0.0) + 1e-6)
                    cutoff = group_mean[:, :, None] + thresholds_t[:, None, None] * group_std[:, :, None]
                    exact = ((scores_g > cutoff) | force_exact_g[None, :, :]) & valid_g[None, :, :]
                    totals_t += exact.sum(dim=(1, 2)).to(torch.float64)
                elif mode == "diag":
                    qc = qb.mean(dim=0)
                    qc_scores = (kc @ qc) * scale
                    diag_mu = (qc * kc_mean).sum() * scale
                    diag_var = torch.sum((qc * qc) * kc_var_diag) * (scale * scale)
                    diag_std = torch.sqrt(torch.clamp(diag_var, min=0.0) + 1e-6)
                    cutoff = diag_mu + thresholds_t * diag_std
                    exact = (qc_scores[None, :] > cutoff[:, None]) | force_exact[None, :]
                    totals_t += exact.sum(dim=1).to(torch.float64)
                else:
                    raise ValueError(mode)
                slots += nt_kv

    totals = totals_t.detach().cpu().tolist()
    return [{"threshold": float(tau), "density": float(total / slots)} for tau, total in zip(thresholds, totals)]


def calibrate_density(
    q: torch.Tensor,
    k: torch.Tensor,
    mode: str,
    target_density: float,
    block_size: int,
    group_size: int,
    sample_heads: int,
    q_stride_blocks: int,
    coarse_start: float,
    coarse_end: float,
    coarse_step: float,
    fine_radius: float,
    fine_step: float,
    q_block_size: int | None = None,
    kv_block_size: int | None = None,
    q_start_offset: int | None = None,
    sink_last_blocks: int = 0,
    sink_last_tokens: int = 0,
) -> dict[str, object]:
    coarse = estimate_density(
        q,
        k,
        threshold_grid(coarse_start, coarse_end, coarse_step),
        mode,
        block_size,
        group_size,
        sample_heads,
        q_stride_blocks,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        q_start_offset=q_start_offset,
        sink_last_blocks=sink_last_blocks,
        sink_last_tokens=sink_last_tokens,
    )
    coarse_best = min(coarse, key=lambda row: abs(row["density"] - target_density))
    fine_start = max(coarse_start, float(coarse_best["threshold"]) - fine_radius)
    fine_end = min(coarse_end, float(coarse_best["threshold"]) + fine_radius)
    fine = estimate_density(
        q,
        k,
        threshold_grid(fine_start, fine_end, fine_step),
        mode,
        block_size,
        group_size,
        sample_heads,
        q_stride_blocks,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        q_start_offset=q_start_offset,
        sink_last_blocks=sink_last_blocks,
        sink_last_tokens=sink_last_tokens,
    )
    best = min(fine, key=lambda row: abs(row["density"] - target_density))
    return {
        "threshold": best["threshold"],
        "density": best["density"],
        "density_delta": best["density"] - target_density,
        "coarse_best": coarse_best,
        "fine_sweep": fine,
    }
