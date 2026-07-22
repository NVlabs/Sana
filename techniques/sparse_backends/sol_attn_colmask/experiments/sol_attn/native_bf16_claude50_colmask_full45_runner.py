"""Generic-shape, source-pinned runner for claude50 colmask full45."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch

from experiments.sol_attn.native_bf16_lean6_routeidx_g256_cursor_ballotscatter_packedgmemo_tworowhoist_inlineu32_fusedroute_n128_pair_tmemp_hybrid_massreuse_terminalo_runner import (
    BLOCK_SIZE,
    HEAD_DIM,
    LOGICAL_GROUP_SIZE,
    LSE_GUARD_VALUE,
    OUTPUT_GUARD_VALUE,
    ROUTE_HALVES_PER_GROUP,
    ROUTE_TILE_SIZE,
    TRACE_POISON,
    TRACE_WORDS,
    _to_cute_tensor,
    _validate_prepared,
)


ROOT = Path(__file__).resolve().parents[2]
KERNEL_PATH = ROOT / (
    "kernels/sol_attn_sm100/native_bf16_claude49_g256_colmask_fwd.py"
)
KERNEL_SHA256 = (
    "e4e47b7e5fc2015b41e4462507372651e1f6eaf05ee7ddd54af3cac1301f283b"
)


def _validate_kernel_identity() -> str:
    observed = hashlib.sha256(KERNEL_PATH.read_bytes()).hexdigest()
    if observed != KERNEL_SHA256:
        raise RuntimeError(
            f"claude50 colmask kernel SHA mismatch: {observed} != "
            f"{KERNEL_SHA256}"
        )
    return observed


def make_native_bf16_claude50_colmask_full45_runner(
    T: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor,
    global_threshold: torch.Tensor,
    scale: float,
    *,
    is_causal: bool = False,
    trace_route_masks: bool = False,
    guard_elements: int = 0,
):
    """Compile the colmask candidate without changing any dispatcher."""

    if is_causal:
        raise ValueError("colmask supports only full noncausal attention")
    if not isinstance(trace_route_masks, bool):
        raise ValueError("trace_route_masks must be a bool")
    if (
        isinstance(guard_elements, bool)
        or not isinstance(guard_elements, int)
        or guard_elements < 0
        or guard_elements % 8 != 0
    ):
        raise ValueError(
            "guard_elements must be a nonnegative multiple of eight"
        )
    b, h, num_blocks = _validate_prepared(
        T, q, k, v, kc, vc, global_threshold, scale
    )
    kernel_sha256 = _validate_kernel_identity()
    from kernels.sol_attn_sm100.native_bf16_claude49_g256_colmask_fwd import (
        build_sol_attn_sm100_lean6_routeidx_g256_cursor_ballotscatter_packedgmemo_tworowhoist_inlineu32_fusedroute_n128_pair_tmemp_bf16_fwd,
        build_sol_attn_sm100_lean6_routeidx_g256_cursor_ballotscatter_packedgmemo_tworowhoist_inlineu32_fusedroute_n128_pair_tmemp_bf16_trace_fwd,
    )

    op = (
        build_sol_attn_sm100_lean6_routeidx_g256_cursor_ballotscatter_packedgmemo_tworowhoist_inlineu32_fusedroute_n128_pair_tmemp_bf16_trace_fwd(
            T
        )
        if trace_route_masks
        else build_sol_attn_sm100_lean6_routeidx_g256_cursor_ballotscatter_packedgmemo_tworowhoist_inlineu32_fusedroute_n128_pair_tmemp_bf16_fwd(
            T
        )
    )

    output_storage = None
    lse_storage = None
    if guard_elements:
        output_storage = torch.full(
            (v.numel() + 2 * guard_elements,),
            OUTPUT_GUARD_VALUE,
            device=v.device,
            dtype=v.dtype,
        )
        output = output_storage[
            guard_elements : guard_elements + v.numel()
        ].view_as(v)
        output.fill_(float("nan"))
        lse_numel = b * h * T
        lse_storage = torch.full(
            (lse_numel + 2 * guard_elements,),
            LSE_GUARD_VALUE,
            device=v.device,
            dtype=torch.float32,
        )
        lse = lse_storage[
            guard_elements : guard_elements + lse_numel
        ].view(b, h, T)
        lse.fill_(float("nan"))
    else:
        output = torch.full_like(v, float("nan"))
        lse = torch.full(
            (b, h, T),
            float("nan"),
            device=v.device,
            dtype=torch.float32,
        )

    num_route_tiles = (
        num_blocks + ROUTE_TILE_SIZE - 1
    ) // ROUTE_TILE_SIZE
    num_groups = (
        num_blocks + LOGICAL_GROUP_SIZE - 1
    ) // LOGICAL_GROUP_SIZE
    route_mask_trace = None
    if trace_route_masks:
        route_mask_trace = torch.full(
            (
                b,
                h,
                num_blocks,
                num_groups,
                ROUTE_HALVES_PER_GROUP,
                TRACE_WORDS,
            ),
            TRACE_POISON,
            device=v.device,
            dtype=torch.int32,
        )

    op_args = [
        _to_cute_tensor(q),
        _to_cute_tensor(k),
        _to_cute_tensor(v),
        _to_cute_tensor(output),
        _to_cute_tensor(kc),
        _to_cute_tensor(vc),
        _to_cute_tensor(global_threshold),
        _to_cute_tensor(lse),
        float(scale),
    ]
    if route_mask_trace is not None:
        op_args.append(_to_cute_tensor(route_mask_trace))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    started = time.perf_counter()
    compiled_op = cute.compile(
        op,
        *op_args,
        stream=stream,
        options="--enable-tvm-ffi",
    )
    compile_s = time.perf_counter() - started

    def run():
        compiled_op(*op_args, stream=stream)
        return output

    run.output = output
    run.lse = lse
    run.route_mask_trace = route_mask_trace
    run.route_mask_trace_poison = TRACE_POISON
    run.trace_route_masks = trace_route_masks
    run.num_blocks = num_blocks
    run.num_route_tiles = num_route_tiles
    run.num_groups = num_groups
    run.group_size = LOGICAL_GROUP_SIZE
    run.logical_group_size = LOGICAL_GROUP_SIZE
    run.physical_route_tile_size = ROUTE_TILE_SIZE
    run.route_halves_per_group = ROUTE_HALVES_PER_GROUP
    run.compile_s = compile_s
    run.guard_elements = guard_elements
    run.output_storage = output_storage
    run.lse_storage = lse_storage
    run.kernel_source_sha256 = kernel_sha256
    run.expected_kernel_source_sha256 = KERNEL_SHA256
    run.production_dispatch_registered = False
    run.phase_graph_changed = False
    run.math_changed = False
    run.online_routing_changed = False
    run.shape_or_density_fast_path = False
    return run


__all__ = [
    "KERNEL_SHA256",
    "make_native_bf16_claude50_colmask_full45_runner",
]
