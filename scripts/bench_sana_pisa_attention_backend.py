#!/usr/bin/env python3
"""Benchmark PISA against dense FlashAttention at the official Sana Video shape."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


DEFAULT_PISA_SOURCE = Path(
    "/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/"
    "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/layers/attention/"
    "backends/piecewise_attn.py"
)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def time_cuda(
    fn: Callable[[], torch.Tensor], *, warmup: int, iterations: int
) -> tuple[dict[str, float | int], torch.Tensor]:
    last = None
    for _ in range(warmup):
        last = fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        last = fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))

    assert last is not None
    return (
        {
            "mean_ms": statistics.fmean(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "p90_ms": percentile(times, 0.90),
            "max_ms": max(times),
            "warmup": warmup,
            "iterations": iterations,
        },
        last,
    )


def benchmark_call(
    fn: Callable[[], torch.Tensor], *, warmup: int, iterations: int
) -> tuple[dict[str, float | int], torch.Tensor]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_bytes = torch.cuda.memory_allocated()
    timing, output = time_cuda(fn, warmup=warmup, iterations=iterations)
    timing["peak_increment_gib"] = (
        torch.cuda.max_memory_allocated() - baseline_bytes
    ) / (1024**3)
    return timing, output


def strict_flash_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> Callable[[], torch.Tensor]:
    def run() -> torch.Tensor:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            )

    return run


def flash_attention_4(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> Callable[[], torch.Tensor]:
    from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
        flash_attn_varlen_func_op,
    )

    q_bnhd = q.transpose(1, 2).contiguous()
    k_bnhd = k.transpose(1, 2).contiguous()
    v_bnhd = v.transpose(1, 2).contiguous()

    def run() -> torch.Tensor:
        return flash_attn_varlen_func_op(
            q=q_bnhd,
            k=k_bnhd,
            v=v_bnhd,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=q_bnhd.shape[1],
            max_seqlen_k=k_bnhd.shape[1],
            softmax_scale=scale,
            causal=False,
            return_softmax_lse=False,
            ver=4,
        ).transpose(1, 2)

    return run


def set_pisa_allocator() -> None:
    import triton
    from sglang.multimodal_gen.runtime.layers.attention.backends.piecewise_attn import (
        _make_tma_allocator,
    )

    triton.set_allocator(_make_tma_allocator())


def prepare_pisa_route(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int,
    density: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from sglang.multimodal_gen.runtime.layers.attention.backends.piecewise_attn import (
        chunk_reduce_qkv,
        taylor_error_block_indices,
    )

    set_pisa_allocator()
    qc, kc, vc, k_var = chunk_reduce_qkv(
        q=q,
        k=k,
        v=v,
        block_size=block_size,
        include_v_centroid=True,
    )
    block_indices = taylor_error_block_indices(
        qc=qc,
        kc=kc,
        k_var=k_var,
        density=density,
        scale=scale,
    )
    return qc, kc, vc, k_var, block_indices


def pisa_route_only(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int,
    density: float,
    scale: float,
) -> Callable[[], torch.Tensor]:
    def run() -> torch.Tensor:
        return prepare_pisa_route(
            q,
            k,
            v,
            block_size=block_size,
            density=density,
            scale=scale,
        )[-1]

    return run


def pisa_forward_only(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor,
    block_indices: torch.Tensor,
    *,
    block_size: int,
    scale: float,
) -> Callable[[], torch.Tensor]:
    from sglang.multimodal_gen.runtime.layers.attention.backends.piecewise_attn import (
        piecewise_attn_fwd,
    )

    def run() -> torch.Tensor:
        set_pisa_allocator()
        output, _ = piecewise_attn_fwd(
            q=q,
            k=k,
            v=v,
            kc=kc,
            vc=vc,
            block_indices=block_indices,
            block_size=block_size,
            scale=scale,
            approx_remainder=True,
        )
        return output

    return run


def pisa_total(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int,
    density: float,
    scale: float,
) -> Callable[[], torch.Tensor]:
    from sglang.multimodal_gen.runtime.layers.attention.backends.piecewise_attn import (
        piecewise_attn_fwd,
    )

    def run() -> torch.Tensor:
        _, kc, vc, _, block_indices = prepare_pisa_route(
            q,
            k,
            v,
            block_size=block_size,
            density=density,
            scale=scale,
        )
        output, _ = piecewise_attn_fwd(
            q=q,
            k=k,
            v=v,
            kc=kc,
            vc=vc,
            block_indices=block_indices,
            block_size=block_size,
            scale=scale,
            approx_remainder=True,
        )
        return output

    return run


def tensor_checks(output: torch.Tensor) -> dict[str, float | bool]:
    sample = output.reshape(-1)[:: max(1, output.numel() // 65536)].float()
    return {
        "all_finite": bool(torch.isfinite(output).all().item()),
        "sample_mean": float(sample.mean().item()),
        "sample_std": float(sample.std().item()),
    }


def source_provenance(path: Path) -> dict[str, str | None]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(path.parents[7]), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"path": str(path), "sha256": digest, "git_commit": commit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=23000)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--densities", default="0.1,0.125,0.25,0.5,0.75")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--pisa-source", type=Path, default=DEFAULT_PISA_SOURCE)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.pisa_source.exists():
        raise FileNotFoundError(args.pisa_source)

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    torch.backends.cuda.enable_cudnn_sdp(False)

    shape = (args.batch, args.heads, args.tokens, args.head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    scale = args.head_dim**-0.5

    dense: dict[str, object] = {}
    dense_errors: dict[str, str] = {}
    dense_reference_name = "sdpa_flash"

    for name, factory in (
        ("sdpa_flash", strict_flash_sdpa),
        ("flash_attention_4", flash_attention_4),
    ):
        print(f"[bench] dense backend={name} shape={shape}", flush=True)
        try:
            timing, output = benchmark_call(
                factory(q, k, v, scale),
                warmup=args.warmup,
                iterations=args.iterations,
            )
            dense[name] = {"timing": timing, "checks": tensor_checks(output)}
            del output
        except Exception as exc:
            dense_errors[name] = repr(exc)
            print(f"[bench] dense backend={name} failed: {exc!r}", flush=True)

    if dense_reference_name not in dense:
        if "flash_attention_4" not in dense:
            raise RuntimeError(f"No dense FlashAttention backend succeeded: {dense_errors}")
        dense_reference_name = "flash_attention_4"
    dense_reference_ms = dense[dense_reference_name]["timing"]["mean_ms"]

    densities = [float(item) for item in args.densities.split(",") if item.strip()]
    pisa_results = []
    for density in densities:
        print(f"[bench] PISA density={density:.6f} shape={shape}", flush=True)
        try:
            _, kc, vc, _, block_indices = prepare_pisa_route(
                q,
                k,
                v,
                block_size=args.block_size,
                density=density,
                scale=scale,
            )
            torch.cuda.synchronize()
            block_count = kc.shape[2]
            exact_blocks = block_indices.shape[-1]
            exact_density = exact_blocks / block_count

            route_timing, route_output = benchmark_call(
                pisa_route_only(
                    q,
                    k,
                    v,
                    block_size=args.block_size,
                    density=density,
                    scale=scale,
                ),
                warmup=args.warmup,
                iterations=args.iterations,
            )
            del route_output

            forward_timing, forward_output = benchmark_call(
                pisa_forward_only(
                    q,
                    k,
                    v,
                    kc,
                    vc,
                    block_indices,
                    block_size=args.block_size,
                    scale=scale,
                ),
                warmup=args.warmup,
                iterations=args.iterations,
            )
            forward_checks = tensor_checks(forward_output)
            del forward_output

            total_timing, total_output = benchmark_call(
                pisa_total(
                    q,
                    k,
                    v,
                    block_size=args.block_size,
                    density=density,
                    scale=scale,
                ),
                warmup=max(2, args.warmup // 2),
                iterations=max(5, args.iterations // 2),
            )
            total_checks = tensor_checks(total_output)
            del total_output

            pisa_results.append(
                {
                    "configured_density": density,
                    "exact_density": exact_density,
                    "actual_sparsity": 1.0 - exact_density,
                    "block_size": args.block_size,
                    "query_blocks": block_indices.shape[-2],
                    "key_blocks": block_count,
                    "exact_blocks_per_query": exact_blocks,
                    "route_only": route_timing,
                    "kernel_with_precomputed_route": forward_timing,
                    "total_route_plus_kernel": total_timing,
                    "kernel_checks": forward_checks,
                    "total_checks": total_checks,
                    "speedup_kernel_only_vs_sdpa_flash": (
                        dense["sdpa_flash"]["timing"]["mean_ms"]
                        / forward_timing["mean_ms"]
                        if "sdpa_flash" in dense
                        else None
                    ),
                    "speedup_total_vs_sdpa_flash": (
                        dense["sdpa_flash"]["timing"]["mean_ms"]
                        / total_timing["mean_ms"]
                        if "sdpa_flash" in dense
                        else None
                    ),
                    "speedup_total_vs_flash_attention_4": (
                        dense["flash_attention_4"]["timing"]["mean_ms"]
                        / total_timing["mean_ms"]
                        if "flash_attention_4" in dense
                        else None
                    ),
                    "speedup_total_vs_selected_dense_reference": (
                        dense_reference_ms / total_timing["mean_ms"]
                    ),
                }
            )
            del kc, vc, block_indices
        except Exception as exc:
            pisa_results.append(
                {"configured_density": density, "error": repr(exc)}
            )
            print(f"[bench] PISA density={density:.6f} failed: {exc!r}", flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    payload = {
        "benchmark_scope": "attention_backend_only_qkv_to_output",
        "official_sana_video_shape": {
            "output_pixels": [193, 736, 1280],
            "vae_stride": [8, 32, 32],
            "latent_grid": [25, 23, 40],
            "tokens": 23000,
            "batch": 2,
            "heads": 10,
            "head_dim": 256,
            "dtype": "bfloat16",
            "attention_mask": None,
            "dropout_p": 0.0,
            "is_causal": False,
            "dense_softmax_calls_per_dit": 8,
        },
        "measured_shape": {
            "batch": args.batch,
            "heads": args.heads,
            "tokens": args.tokens,
            "head_dim": args.head_dim,
            "dtype": "bfloat16",
        },
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "pisa_source": source_provenance(args.pisa_source),
        "settings": vars(args) | {"out": str(args.out), "pisa_source": str(args.pisa_source)},
        "dense_reference_for_primary_speedup": dense_reference_name,
        "dense": dense,
        "dense_errors": dense_errors,
        "pisa": pisa_results,
        "interpretation": {
            "primary_speedup_uses_total_backend_time": True,
            "pisa_total_includes": [
                "qkv block centroid reduction",
                "Taylor-error top-k block routing",
                "exact selected-block attention",
                "approximate remainder attention",
            ],
            "excluded": [
                "qkv projection",
                "qk normalization",
                "RoPE",
                "output gate",
                "output projection",
                "full DiT",
                "full diffusion",
            ],
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    with torch.inference_mode() if torch.cuda.is_available() else nullcontext():
        main()
