"""Runner for the G512 uniform-outer-cursor codegen diagnostic."""

from __future__ import annotations

import math
import time
import hashlib
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch


BLOCK_SIZE = 64
HEAD_DIM = 128
LOGICAL_GROUP_SIZE = 512
ROUTE_TILE_SIZE = 128
ROUTE_HALVES_PER_GROUP = 4
TRACE_WORDS = 8
TRACE_POISON = -123456789
OUTPUT_GUARD_VALUE = 321.0
LSE_GUARD_VALUE = -654.0
ROOT = Path(__file__).resolve().parents[2]
KERNEL_PATH = ROOT / (
    "kernels/sol_attn_sm100/"
    "native_bf16_lean6_routeidx_g512_cursor_fusedroute_n128_pair_tmemp_hybrid_"
    "massreuse_terminalo_fwd.py"
)
KERNEL_SHA256 = "7723c2ac9ecd2e840182ed2f9d171679536d42be42a7b0a64ca7bdbee68ea47c"


def _validate_kernel_identity() -> str:
    observed = hashlib.sha256(KERNEL_PATH.read_bytes()).hexdigest()
    if observed != KERNEL_SHA256:
        raise RuntimeError(
            f"G512-cursor/N128 codegen diagnostic kernel SHA mismatch: {observed} != "
            f"{KERNEL_SHA256}"
        )
    return observed


def _validate_prepared(
    T: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor,
    global_threshold: torch.Tensor,
    scale: float,
) -> tuple[int, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("lean6 BF16 runner requires CUDA")
    if tuple(torch.cuda.get_device_capability(q.device)) != (10, 0):
        raise RuntimeError("lean6 BF16 runner requires SM100")
    if isinstance(T, bool) or not isinstance(T, int) or T <= 0:
        raise ValueError(f"T must be a positive integer, got {T!r}")
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("Q/K/V must share contiguous [B,H,T,128] shape")
    b, h, t, d = q.shape
    if t != T or d != HEAD_DIM:
        raise ValueError(f"expected T={T}, D=128, got {tuple(q.shape)}")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v, kc, vc)):
        raise TypeError("Q/K/V/KC/VC must all be BF16")
    if any(x.device != q.device for x in (k, v, kc, vc, global_threshold)):
        raise ValueError("all prepared tensors must be on one CUDA device")
    if any(not x.is_contiguous() for x in (q, k, v, kc, vc, global_threshold)):
        raise ValueError("all prepared tensors must be contiguous")
    num_blocks = (T + BLOCK_SIZE - 1) // BLOCK_SIZE
    if tuple(kc.shape) != (b, h, num_blocks, HEAD_DIM):
        raise ValueError(f"invalid KC shape {tuple(kc.shape)}")
    if tuple(vc.shape) != tuple(kc.shape):
        raise ValueError(f"invalid VC shape {tuple(vc.shape)}")
    if global_threshold.dtype != torch.float32:
        raise TypeError("global_threshold must be FP32")
    if tuple(global_threshold.shape) != (b, h, num_blocks):
        raise ValueError(
            f"invalid threshold shape {tuple(global_threshold.shape)}"
        )
    expected_scale = HEAD_DIM**-0.5
    if not math.isclose(float(scale), expected_scale, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"lean6 v1 fixes softmax_scale={expected_scale}, got {scale}"
        )
    return b, h, num_blocks


def _to_cute_tensor(tensor: torch.Tensor):
    return from_dlpack(
        tensor, assumed_align=16, enable_tvm_ffi=True
    ).mark_layout_dynamic(leading_dim=tensor.ndim - 1)


def make_native_bf16_lean6_routeidx_g512_cursor_fusedroute_n128_pair_tmemp_hybrid_massreuse_terminalo_runner(
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
    """Compile one fixed full-grid specialization without changing dispatch."""

    if is_causal:
        raise ValueError("lean6 v1 does not support causal attention")
    if not isinstance(trace_route_masks, bool):
        raise ValueError("trace_route_masks must be a bool")
    if (
        isinstance(guard_elements, bool)
        or not isinstance(guard_elements, int)
        or guard_elements < 0
    ):
        raise ValueError("guard_elements must be a nonnegative integer")
    # Safety-only donor amendment: the BF16 slice is promised 16-byte alignment.
    if guard_elements % 8 != 0:
        raise ValueError("guard_elements must preserve 16-byte alignment")
    b, h, num_blocks = _validate_prepared(
        T, q, k, v, kc, vc, global_threshold, scale
    )
    kernel_sha256 = _validate_kernel_identity()
    from kernels.sol_attn_sm100.native_bf16_lean6_routeidx_g512_cursor_fusedroute_n128_pair_tmemp_hybrid_massreuse_terminalo_fwd import (
        build_sol_attn_sm100_lean6_routeidx_g512_cursor_fusedroute_n128_pair_tmemp_bf16_fwd,
        build_sol_attn_sm100_lean6_routeidx_g512_cursor_fusedroute_n128_pair_tmemp_bf16_trace_fwd,
    )

    op = (
        build_sol_attn_sm100_lean6_routeidx_g512_cursor_fusedroute_n128_pair_tmemp_bf16_trace_fwd(T)
        if trace_route_masks
        else build_sol_attn_sm100_lean6_routeidx_g512_cursor_fusedroute_n128_pair_tmemp_bf16_fwd(T)
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
            (b, h, T), float("nan"), device=v.device, dtype=torch.float32
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
    run.builder_names = (
        "build_sol_attn_sm100_lean6_routeidx_g512_cursor_fusedroute_n128_pair_tmemp_bf16_fwd",
        "build_sol_attn_sm100_lean6_routeidx_g512_cursor_fusedroute_n128_pair_tmemp_bf16_trace_fwd",
    )
    run.kernel_source_sha256 = kernel_sha256
    run.expected_kernel_source_sha256 = KERNEL_SHA256
    run.production_dispatch_registered = False
    run.codegen_parent_kernel_source_sha256 = (
        "70e8eeb87c98941a759f180b0bfdca8d40233d53a80df58fa3883ff6e4f04882"
    )
    run.parent_kernel_source_sha256 = run.codegen_parent_kernel_source_sha256
    run.semantic_parent_kernel_source_sha256 = (
        "7c26c2c184fe968cd3a485f6485b4d810034628ad1564d4de1422e2c6f17d6fb"
    )
    run.codegen_diagnostic_axis = "uniform_outer_loop_induction_only"
    run.phase_graph_changed = False
    run.math_changed = False
    run.shape_or_density_fast_path = False
    run.terminalo_parent_kernel_source_sha256 = (
        "4a7933ebaf288148e0fd63cb69390733736c4fe8d0e3145a8deb82dfffcc5aa2"
    )
    run.exactp_grandparent_kernel_source_sha256 = (
        "65e14fcfbc92396f846174e4570498259dd89f173efe15d6f72c88ad38b90b5f"
    )
    run.grandparent_routep_kernel_source_sha256 = (
        "bc0771cac790ed4c0ccf89356b5890a124c91fee28998e8959452fb994a5cf4f"
    )
    run.grandparent_037_kernel_source_sha256 = (
        "037e6ba686d40e84f0686dc98c46aa5c53e2bdec338aae882cd94a9a0883b221"
    )
    run.route_pair_p_generations_per_group = 0
    run.exact_pair_p_generations_per_pair = 0
    run.exact_pair_p_namedbarrier_threads = 5 * 32
    run.exact_pair_p_empty_edge = "next_pair_score_completion"
    run.pair_o_completion_generations_per_cta = 1
    run.pair_o_empty_generations_per_cta = 0
    run.intermediate_o_completion_edge = "next_pair_score_completion"
    run.physical_tile_end_full_cta_joins = 0
    run.logical_group_end_full_cta_joins = 0
    run.cross_half_exact_carry_capacity = 0
    run.collect_multiple_score_halves_before_execute = False
    run.collect_all_four_index_halves_before_exact = True
    run.pre_exact_full_cta_joins_per_logical_group = 1
    run.pre_exact_full_cta_joins_per_physical_half = 0
    run.route_index_reuse_hb = (
        "warp5_reads_before_final_score_completion; owners_read_before_"
        "final_exact_p_namedbarrier; sole_writer_reuses_after_exact_loop"
    )
    run.cross_group_handoff = (
        "pack_k_pack_v_buffer_free_plus_pair_score_ready"
    )
    run.pre_exact_full_cta_join_retained = True
    run.final_epilogue_full_cta_join_retained = True
    return run


__all__ = [
    "KERNEL_SHA256",
    "make_native_bf16_lean6_routeidx_g512_cursor_fusedroute_n128_pair_tmemp_hybrid_massreuse_terminalo_runner",
]
