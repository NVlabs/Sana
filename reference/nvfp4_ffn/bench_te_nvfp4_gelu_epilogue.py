#!/usr/bin/env python3
# ported from Sol-LTX-Infer scripts/bench_te_nvfp4_gelu_epilogue.py @ 29d0d9e
"""Benchmark TE NVFP4 FFN proj_in with and without GEMM+GELU epilogue.

This targets the LTX-2.x video FFN proj_in shape:
  [M, 4096] x [16384, 4096]^T -> [M, 16384]

Baseline mirrors the current runtime path: Transformer Engine NVFP4 Linear with
bias applied by GEMM, followed by PyTorch GELU(approximate="tanh"). The fused
candidate uses TE's lower-level general_gemm(..., gelu=True) so bias and GELU
are requested from the GEMM epilogue instead of a separate activation kernel.

Reference-only in autovideo: requires CUDA and TransformerEngine NVFP4 support.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def sync() -> None:
    torch.cuda.synchronize()


def make_te_linear(in_features: int, out_features: int, dtype: torch.dtype):
    import transformer_engine.pytorch as te

    layer = te.Linear(
        in_features,
        out_features,
        bias=True,
        params_dtype=dtype,
        device="cuda",
    )
    torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(layer.bias)
    layer.eval()
    return layer


def te_linear_gelu_fused(layer, x: torch.Tensor, fp8_autocast, recipe) -> torch.Tensor:
    from transformer_engine.pytorch.cpp_extensions import general_gemm
    from transformer_engine.pytorch.module.base import _2X_ACC_FPROP, quantize_weight
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager
    from transformer_engine.pytorch.utils import assert_dim_for_fp8_exec, cast_if_needed

    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        inp = layer.prepare_forward(x, allow_non_contiguous=False)
        try:
            if not getattr(layer, "fp8", False):
                raise RuntimeError("TE fp8/nvfp4 state was not enabled")
            weight, bias = layer._get_weight_and_bias_tensors()
            assert_dim_for_fp8_exec(inp, weight)
            (
                input_quantizer,
                weight_quantizer,
                output_quantizer,
                _grad_input_quantizer,
                _grad_weight_quantizer,
                _grad_output_quantizer,
            ) = layer._get_quantizers(
                fp8_output=False,
                fp8_grad=False,
                is_grad_enabled=False,
            )
            input_quantizer.set_usage(rowwise=True, columnwise=False)
            inputmat = input_quantizer(inp)
            weight_quantizer.set_usage(rowwise=True, columnwise=False)
            weightmat, _ = quantize_weight(
                tensor=weight,
                quantizer=weight_quantizer,
                workspace=None,
                update_workspace=True,
                skip_update_flag=None,
                fsdp_group=getattr(layer, "fsdp_group", None),
                workspace_dtype=getattr(layer, "activation_dtype", x.dtype),
                cache=False,
            )
            weightmat.update_usage(rowwise_usage=True)
            bias = cast_if_needed(bias, getattr(layer, "activation_dtype", x.dtype))
            use_split_accumulator = _2X_ACC_FPROP
            fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
            if hasattr(fp8_recipe, "fp8_gemm_fprop"):
                use_split_accumulator = fp8_recipe.fp8_gemm_fprop.use_split_accumulator
            out, *_ = general_gemm(
                weightmat,
                inputmat,
                quantization_params=output_quantizer,
                out_dtype=getattr(layer, "activation_dtype", x.dtype),
                bias=bias,
                gelu=True,
                use_split_accumulator=use_split_accumulator,
            )
            return out
        finally:
            layer.end_forward()


def bench(fn, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        y = fn()
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        y = fn()
    sync()
    total_ms = (time.perf_counter() - start) * 1000.0
    if y.numel() == 0:
        raise RuntimeError("empty output")
    return {"total_ms": total_ms, "avg_ms": total_ms / iters}


def pad_to_te_m(x: torch.Tensor, multiple: int = 16) -> tuple[torch.Tensor, int]:
    original_m = int(x.shape[0])
    pad_rows = (-original_m) % multiple
    if pad_rows:
        return F.pad(x, (0, 0, 0, pad_rows)), original_m
    return x, original_m


def run_case(m: int, dtype: torch.dtype, warmup: int, iters: int) -> dict[str, Any]:
    from transformer_engine.common.recipe import NVFP4BlockScaling
    from transformer_engine.pytorch import fp8_autocast

    recipe = NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        disable_2d_quantization=True,
    )
    layer = make_te_linear(4096, 16384, dtype)
    x = torch.randn((m, 4096), device="cuda", dtype=dtype)

    def baseline():
        x_pad, original_m = pad_to_te_m(x)
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            y = layer(x_pad)
        if int(y.shape[0]) != original_m:
            y = y[:original_m]
        return F.gelu(y, approximate="tanh")

    def fused():
        x_pad, original_m = pad_to_te_m(x)
        y = te_linear_gelu_fused(layer, x_pad, fp8_autocast, recipe)
        if int(y.shape[0]) != original_m:
            y = y[:original_m]
        return y

    y_base = baseline()
    y_fused = fused()
    sync()
    diff = (y_base.float() - y_fused.float()).abs()
    baseline_stats = bench(baseline, warmup, iters)
    fused_stats = bench(fused, warmup, iters)
    return {
        "m": m,
        "shape": [m, 4096, 16384],
        "dtype": str(dtype).replace("torch.", ""),
        "baseline_te_linear_plus_torch_gelu_ms": baseline_stats["avg_ms"],
        "fused_te_general_gemm_gelu_ms": fused_stats["avg_ms"],
        "speedup": baseline_stats["avg_ms"] / fused_stats["avg_ms"],
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[31620, 126480],
        help="Flattened [batch*tokens] dimensions to test.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    results = [run_case(m, dtype, args.warmup, args.iters) for m in args.m]
    payload = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
