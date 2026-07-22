"""SM100 BF16 N256 KV-pack + deferred-softmax-combine mechanism leaf.

This file is deliberately independent from the production SOL Attention dispatcher.
It records one Blackwell-native mechanism:

* four non-contiguous N64 exact blocks are gathered into one logical N256 pack;
* QK is split into two M64xN128 transactions backed by disjoint TMEM score
  regions;
* each half owns an independent online-softmax state ``(m_h, l_h, O_h)``;
* K and V use the same physical SMEM payload in alternating phases;
* the attention mainloop never reduces or combines the two halves; and
* one epilogue combines the two unnormalised outputs with the associative
  online-softmax formula.

The TMEM phase graph is adapted from FlashAttention PR #2224 snapshot
``a8b8251e4705de223ea81e8d96506d9ff51080c8``.  This independent mechanism
leaf passes its fixed non-contiguous exact4 CuTe JIT and B200 correctness gate;
it is not integrated into, or representative of, the production dispatcher.
"""

import argparse
import json
import math
from pathlib import Path
import time
from typing import Optional, Sequence

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import flash_attn.cute.blackwell_helpers as mma_utils
import flash_attn.cute.pipeline as fa_pipeline
import flash_attn.cute.utils as fa_utils
import torch
import triton
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

from kernels.sol_attn_sm100.native_tmem import (
    _add_physical_tmem_base,
    _zero_based_tmem_tensor,
    tcgen05_wait_ld,
    tcgen05_wait_st,
    transpose_m64_p_bf16_lane_halfwords,
)


FA4_PR2224_HEAD = "a8b8251e4705de223ea81e8d96506d9ff51080c8"
MECHANISM_FAMILY = "sm100-bf16-m64-n256-deferred-half-combine-leaf-v1"
VALIDATION_STATUS = "b200-exact4-device-gate-pass"
VALIDATED_CLEAN_SOURCE_SHA256 = (
    "3c57260114b6edb90365213048448a9b2138389608d62159f0aa301a74a463f8"
)

M = 64
N_MEMBER = 64
N_HALF = 128
N_PACK = 256
D = 128
DV = 128
PACK_MEMBERS = 4
HALVES = 2
MEMBERS_PER_HALF = 2
THREADS = 512

QK_QUARTER_INST = (64, 64, 16)
QK_QUARTER_TILE = (64, 64, 64)
QK_HALF_INST = (64, 128, 16)
QK_HALF_TILE = (64, 128, 128)
PV_QUARTER_INST = (64, 64, 16)
PV_QUARTER_TILE = (64, 64, 64)
PV_HALF_INST = (64, 128, 16)
PV_HALF_TILE = (64, 128, 128)
K_QUARTERS_PER_HALF = 4
K_GATHER_STAGES = 8
V_QUARTERS_PER_HALF = 4
V_GATHER_STAGES = 8

# FA4-style 512-column TMEM map.  Scores and outputs never overlap.  BF16 P
# needs 64 FP32-addressed columns and overwrites the consumed upper half of S.
TMEM_COLS = 512
S0_OFFSET = 0
S1_OFFSET = 128
P0_OFFSET = 64
P1_OFFSET = 192
O0_OFFSET = 256
O1_OFFSET = 384
SCORE_COLS_PER_HALF = 128
P_COLS_PER_HALF = 64
O_COLS_PER_HALF = 128

Q_SMEM_BYTES = 16384
KV_PACK_SMEM_BYTES = 65536
HALF_STATS_SMEM_BYTES = 1024
SMEM_PAYLOAD_BYTES = 81920
SMEM_TARGET_BYTES = 86016

SCALE = 1.0 / math.sqrt(D)
SCALE_LOG2 = SCALE * math.log2(math.e)
LOG2E = math.log2(math.e)
LN2 = math.log(2.0)
SEMANTIC_ROW_OFFSET = 0
NONCONTIGUOUS_PACK = (0, 2, 5, 9)

ROLE_MAP = {
    "softmax_half0": (0, 1, 2, 3),
    "softmax_half1": (4, 5, 6, 7),
    "deferred_combine_epilogue": (8, 9, 10, 11),
    "tcgen05_mma": (12,),
    "output_store": (13,),
    "tma_kv_phase": (14,),
    "reserved": (15,),
}

SOURCE_RECEIPT = {
    "fa4_pr2224_head": FA4_PR2224_HEAD,
    "tile": "M64xN256xD128-as-two-independent-M64xN128-halves",
    "tmem_map": "S0[0,128) S1[128,256) O0[256,384) O1[384,512)",
    "p_overlay": "P0=S0+64 P1=S1+64",
    "kv_smem": "one-physical-buffer-K-phase-then-V-phase",
    "physical_smem_authority": "canonical_n128_k_v_layout",
    "k_gather_alias": "four-64x64-quarters-per-physical-N128-half",
    "k_quarter_order": "K0/N0,K0/N1,K1/N0,K1/N1",
    "v_gather_alias": "four-64x64-quarters-per-physical-N128-half",
    "v_quarter_order": "D0/N0,D0/N1,D1/N0,D1/N1",
    "kv_alias_handoff": "QK-complete-K-to-V;PV-complete-V-to-next-K",
    "mainloop_cross_half_reduce": False,
    "mainloop_output_pair_combine": False,
    "epilogue_deferred_combine": True,
    "production_dispatch_changed": False,
    "mechanism_only": True,
    "full_sol_attn": False,
    "b200_validated": True,
    "validated_clean_source_sha256": VALIDATED_CLEAN_SOURCE_SHA256,
    "validated_clean_result": "n256-deferred-g-clean.jsonl",
}


def build_n256_pack_schedule(
    exact_indices: Sequence[int],
) -> tuple[tuple[int, int, int, int], ...]:
    """Group an exact stream into full four-member packs without reordering.

    This first mechanism leaf intentionally admits full packs only.  Generic
    N64/N128 tails belong to the integration milestone, not to this mechanism
    proof.  The restriction is shape-based and has no density-specific route.
    """

    indices = tuple(int(index) for index in exact_indices)
    if len(indices) == 0 or len(indices) % PACK_MEMBERS != 0:
        raise ValueError("N256 mechanism leaf requires non-empty full packs")
    if any(right <= left for left, right in zip(indices, indices[1:])):
        raise ValueError("exact indices must be strictly ascending")
    return tuple(
        tuple(indices[start : start + PACK_MEMBERS])
        for start in range(0, len(indices), PACK_MEMBERS)
    )


def split_pack_halves(
    pack: Sequence[int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Map members 0/1 to half 0 and members 2/3 to half 1."""

    members = tuple(int(index) for index in pack)
    if len(members) != PACK_MEMBERS:
        raise ValueError("one N256 pack has exactly four N64 members")
    return members[:MEMBERS_PER_HALF], members[MEMBERS_PER_HALF:]


def deferred_combine_reference(
    max0: float,
    sum0: float,
    numerator0: Sequence[float],
    max1: float,
    sum1: float,
    numerator1: Sequence[float],
) -> tuple[tuple[float, ...], float]:
    """Pure-Python oracle for the one-time two-half epilogue combine.

    ``numerator_h`` is unnormalised and referenced to ``max_h``.  Keeping this
    representation avoids a local divide followed by a compensating multiply.
    """

    if len(numerator0) != len(numerator1):
        raise ValueError("partial output widths differ")
    combined_max = max(max0, max1)
    alpha0 = 0.0 if max0 == -math.inf else math.exp(max0 - combined_max)
    alpha1 = 0.0 if max1 == -math.inf else math.exp(max1 - combined_max)
    combined_sum = alpha0 * sum0 + alpha1 * sum1
    if combined_sum == 0.0:
        return tuple(0.0 for _ in numerator0), -math.inf
    output = tuple(
        (alpha0 * float(left) + alpha1 * float(right)) / combined_sum
        for left, right in zip(numerator0, numerator1)
    )
    return output, combined_max + math.log(combined_sum)


@cute.jit
def _load_m64_n128_score(
    score_template: cute.Tensor,
    thr_mma_qk: cute.core.ThrMma,
    tmem_base: Int32,
    score_offset: Int32,
    owner_tidx: Int32,
):
    """Load one half's M64xN128 FP32 score tile from its TMEM region."""

    relative_score = _zero_based_tmem_tensor(Float32, score_template.layout)
    load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld16x64bOp(tcgen05.copy.Repetition(64)),
        Float32,
    )
    tiled_load = tcgen05.make_tmem_copy(load_atom, relative_score)
    thread_load = tiled_load.get_slice(owner_tidx)
    source_relative = thread_load.partition_S(relative_score)
    source = _add_physical_tmem_base(
        source_relative, tmem_base + score_offset
    )
    coordinates = thread_load.partition_D(
        thr_mma_qk.partition_C(cute.make_identity_tensor((M, N_HALF)))
    )
    scores = cute.make_fragment(coordinates.shape, Float32)
    cute.copy(tiled_load, source, scores)
    tcgen05_wait_ld()
    cute.arch.fence_view_async_tmem_load()
    return scores, coordinates


@cute.jit
def _store_m64_n128_probability(
    o_template: cute.Tensor,
    probabilities: cute.Tensor,
    tmem_base: Int32,
    p_offset: Int32,
    owner_tidx: Int32,
):
    """Store one half's BF16 P after that half has consumed S."""

    p_layout = cute.composition(
        o_template.layout, cute.make_layout((M, N_HALF // 2))
    )
    relative_p = _zero_based_tmem_tensor(Float32, p_layout)
    store_atom = cute.make_copy_atom(
        tcgen05.copy.St16x64bOp(tcgen05.copy.Repetition(32)),
        Float32,
    )
    tiled_store = tcgen05.make_tmem_copy(store_atom, relative_p)
    thread_store = tiled_store.get_slice(owner_tidx)
    destination = _add_physical_tmem_base(
        thread_store.partition_D(relative_p), tmem_base + p_offset
    )
    packed = cute.make_fragment(
        thread_store.partition_S(
            cute.make_identity_tensor((M, N_HALF // 2))
        ).shape,
        Float32,
    )
    transpose_m64_p_bf16_lane_halfwords(
        probabilities, packed, owner_tidx
    )
    cute.copy(tiled_store, packed, destination)
    tcgen05_wait_st()
    # This fence is the required P_r2t -> tcgen05.PV visibility edge.
    cute.arch.fence_view_async_tmem_store()


@cute.jit
def _rescale_m64_partial_o(
    o_template: cute.Tensor,
    thr_mma_pv: cute.core.ThrMma,
    tmem_base: Int32,
    o_offset: Int32,
    owner_tidx: Int32,
    alpha: Float32,
):
    """Rescale only one half's unnormalised O; never touches its peer."""

    relative_o = _zero_based_tmem_tensor(Float32, o_template.layout)
    correction_width = 16
    relative_fragment = cute.composition(
        relative_o, cute.make_layout((M, correction_width))
    )
    load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld16x64bOp(tcgen05.copy.Repetition(8)), Float32
    )
    store_atom = cute.make_copy_atom(
        tcgen05.copy.St16x64bOp(tcgen05.copy.Repetition(8)), Float32
    )
    thread_load = tcgen05.make_tmem_copy(
        load_atom, relative_fragment
    ).get_slice(owner_tidx)
    thread_store = tcgen05.make_tmem_copy(
        store_atom, relative_fragment
    ).get_slice(owner_tidx)
    source = _add_physical_tmem_base(
        thread_load.partition_S(relative_fragment), tmem_base + o_offset
    )
    destination = _add_physical_tmem_base(
        thread_store.partition_D(relative_fragment), tmem_base + o_offset
    )
    for fragment_idx in cutlass.range_constexpr(DV // correction_width):
        registers = cute.make_fragment(
            thread_load.partition_D(relative_fragment).shape, Float32
        )
        source_i = cute.make_tensor(
            source.iterator + fragment_idx * correction_width, source.layout
        )
        cute.copy(thread_load, source_i, registers)
        tcgen05_wait_ld()
        cute.arch.fence_view_async_tmem_load()
        for i in cutlass.range(cute.size(registers), unroll_full=True):
            registers[i] = Float32(registers[i]) * Float32(alpha)
        destination_i = cute.make_tensor(
            destination.iterator + fragment_idx * correction_width,
            destination.layout,
        )
        cute.copy(thread_store, registers, destination_i)
        tcgen05_wait_st()
    cute.arch.fence_view_async_tmem_store()


@cute.jit
def _load_m64_partial_o(
    o_template: cute.Tensor,
    thr_mma_pv: cute.core.ThrMma,
    tmem_base: Int32,
    o_offset: Int32,
    owner_tidx: Int32,
):
    """Load one M64xD128 partial output with the same ownership map."""

    relative_o = _zero_based_tmem_tensor(Float32, o_template.layout)
    coordinates = thr_mma_pv.partition_C(
        cute.make_identity_tensor((M, DV))
    )
    epilogue_tiler = (
        (
            cute.size(relative_o, mode=[0, 0]),
            cute.size(relative_o, mode=[0, 1]),
        ),
    )
    relative_o = cute.zipped_divide(relative_o, epilogue_tiler)
    coordinates = cute.zipped_divide(coordinates, epilogue_tiler)
    load_atom = cute.make_copy_atom(
        tcgen05.Ld16x256bOp(tcgen05.Repetition.x8), Float32
    )
    tiled_load = tcgen05.make_tmem_copy(
        load_atom, relative_o[None, Int32(0)]
    )
    thread_load = tiled_load.get_slice(owner_tidx)
    source = _add_physical_tmem_base(
        thread_load.partition_S(relative_o), tmem_base + o_offset
    )
    register_coordinates = thread_load.partition_D(coordinates)[
        None, None, Int32(0)
    ]
    registers = cute.make_fragment(register_coordinates.shape, Float32)
    cute.copy(
        tiled_load, source[None, None, Int32(0)], registers
    )
    tcgen05_wait_ld()
    cute.arch.fence_view_async_tmem_load()
    return registers, register_coordinates


@cute.jit
def _online_update_one_half(
    scores: cute.Tensor,
    running_max: Float32,
    running_sum: Float32,
):
    """Update exactly one half-state; no peer state is an argument."""

    local_max = fa_utils.fmax_reduce(scores.load(), arch=100)
    local_max = Float32(local_max) * Float32(SCALE)
    peer_max = cute.arch.shuffle_sync_bfly(local_max, offset=2)
    transaction_max = local_max
    if peer_max > transaction_max:
        transaction_max = peer_max
    new_max = running_max
    if running_max == -Float32.inf or transaction_max > running_max:
        new_max = transaction_max
    alpha = Float32(0.0)
    if running_max != -Float32.inf:
        alpha = cute.math.exp2(
            (running_max - new_max) * Float32(LOG2E), fastmath=True
        )
    probabilities = cute.make_fragment(scores.shape, Float32)
    for i in cutlass.range(cute.size(scores), unroll_full=True):
        probabilities[i] = cute.math.exp2(
            Float32(scores[i]) * Float32(SCALE_LOG2)
            - new_max * Float32(LOG2E),
            fastmath=True,
        )
    transaction_sum = fa_utils.fadd_reduce(
        probabilities.load(), arch=100
    )
    transaction_sum += cute.arch.shuffle_sync_bfly(
        transaction_sum, offset=2
    )
    new_sum = running_sum * alpha + transaction_sum
    return probabilities, new_max, new_sum, alpha


@cute.jit
def _run_independent_half_mainloop(
    half_id: cutlass.Constexpr[int],
    score_offset: Int32,
    p_offset: Int32,
    o_offset: Int32,
    score_template: cute.Tensor,
    o_template: cute.Tensor,
    thr_mma_qk: cute.core.ThrMma,
    thr_mma_pv: cute.core.ThrMma,
    tmem_base: Int32,
    owner_tidx: Int32,
    pack_count: Int32,
    score_pipe,
    p_pipe,
    o_pipe,
    kv_alias_barrier,
    sHalfStats: cute.Tensor,
):
    """Own ``(m_h,l_h,O_h)`` for one half for the whole exact stream."""

    running_max = -Float32.inf
    running_sum = Float32(0.0)
    o_initialized = Int32(0)
    score_consumer = fa_pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, 1
    )
    p_producer = fa_pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, 1
    )
    o_consumer = fa_pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, 1
    )

    for pack_idx in cutlass.range(pack_count, unroll=1):
        score_pipe.consumer_wait(score_consumer)
        # QK completion, not QK issue, transfers the aliased payload from K
        # to V ownership.  Both four-warp half owners join the elected tcgen
        # issuer lane on this alternating named-barrier phase.
        kv_alias_barrier.arrive_and_wait()
        scores, _ = _load_m64_n128_score(
            score_template,
            thr_mma_qk,
            tmem_base,
            score_offset,
            owner_tidx,
        )
        probabilities, next_max, next_sum, alpha = (
            _online_update_one_half(scores, running_max, running_sum)
        )

        # O_h from pack i-1 is complete before this half rescales it.  There
        # is intentionally no wait on, load from, or reduction with peer O.
        if o_initialized != Int32(0):
            # The preceding iteration already waited for this O generation
            # before allowing V -> next K.  Keep the generation acquired until
            # the new alpha is known, then rescale and release it.
            _rescale_m64_partial_o(
                o_template,
                thr_mma_pv,
                tmem_base,
                o_offset,
                owner_tidx,
                alpha,
            )
            o_pipe.consumer_release(o_consumer)
            o_consumer.advance()

        _store_m64_n128_probability(
            o_template,
            probabilities,
            tmem_base,
            p_offset,
            owner_tidx,
        )
        score_pipe.consumer_release(score_consumer)
        score_consumer.advance()
        p_pipe.producer_acquire(p_producer)
        p_pipe.producer_commit(p_producer)
        p_producer.advance()
        running_max = next_max
        running_sum = next_sum
        o_initialized = Int32(1)

        # Hold the shared V payload until both N128 PV operations have truly
        # completed.  Waiting at the end of this iteration (rather than before
        # the next score) avoids a V->K / next-QK dependency cycle.
        o_pipe.consumer_wait(o_consumer)
        kv_alias_barrier.arrive_and_wait()

    # The terminal O generation is complete and was kept acquired above.
    o_pipe.consumer_release(o_consumer)
    o_consumer.advance()
    lane = owner_tidx % Int32(32)
    owner_warp = owner_tidx // Int32(32)
    owner_row = (
        owner_warp * Int32(16)
        + lane // Int32(4)
        + (lane % Int32(2)) * Int32(8)
        + Int32(SEMANTIC_ROW_OFFSET)
    ) & Int32(M - 1)
    if (lane & Int32(2)) == Int32(0):
        sHalfStats[half_id, owner_row, 0] = running_max
        sHalfStats[half_id, owner_row, 1] = running_sum
    cute.arch.fence_view_async_shared()


@cute.jit
def _deferred_combine_epilogue(
    o_template: cute.Tensor,
    thr_mma_pv: cute.core.ThrMma,
    tmem_base: Int32,
    owner_tidx: Int32,
    sHalfStats: cute.Tensor,
    mO: cute.Tensor,
    mLSE: cute.Tensor,
):
    """The only legal cross-half join: one O(M*D) epilogue operation."""

    partial0, coordinates0 = _load_m64_partial_o(
        o_template, thr_mma_pv, tmem_base, Int32(O0_OFFSET), owner_tidx
    )
    partial1, coordinates1 = _load_m64_partial_o(
        o_template, thr_mma_pv, tmem_base, Int32(O1_OFFSET), owner_tidx
    )
    for i in cutlass.range(cute.size(partial0), unroll_full=True):
        row = (
            coordinates0[i][0] + Int32(SEMANTIC_ROW_OFFSET)
        ) & Int32(M - 1)
        column = coordinates0[i][1]
        max0 = Float32(sHalfStats[0, row, 0])
        sum0 = Float32(sHalfStats[0, row, 1])
        max1 = Float32(sHalfStats[1, row, 0])
        sum1 = Float32(sHalfStats[1, row, 1])
        combined_max = max0
        if max1 > combined_max:
            combined_max = max1
        alpha0 = cute.math.exp2(
            (max0 - combined_max) * Float32(LOG2E), fastmath=True
        )
        alpha1 = cute.math.exp2(
            (max1 - combined_max) * Float32(LOG2E), fastmath=True
        )
        combined_sum = alpha0 * sum0 + alpha1 * sum1
        combined_numerator = (
            alpha0 * Float32(partial0[i])
            + alpha1 * Float32(partial1[i])
        )
        mO[row, column] = combined_numerator * cute.arch.rcp_approx(
            combined_sum
        )
        if column == Int32(0):
            mLSE[row] = combined_max + cute.math.log2(
                combined_sum, fastmath=True
            ) * Float32(LN2)


@cute.struct
class SharedStorage:
    q_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    kv_phase_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    score0_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    score1_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    p0_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    p1_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    o0_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    o1_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    half_stats: cute.struct.Align[
        cute.struct.MemRange[Float32, HALVES * M * 2], 128
    ]
    tmem_holding_buf: Int32


@cute.kernel
def n256_deferred_exact_leaf_kernel(
    tiled_qk_half: cute.TiledMma,
    tiled_pv_half: cute.TiledMma,
    tiled_qk_quarter: cute.TiledMma,
    tiled_pv_quarter: cute.TiledMma,
    tma_atom_q: cute.CopyAtom,
    mQ_mkl: cute.Tensor,
    tma_atom_k: cute.CopyAtom,
    mK_quarter_nkl: cute.Tensor,
    tma_atom_v: cute.CopyAtom,
    mV_nkl: cute.Tensor,
    mExactIndices: cute.Tensor,
    mExactCount: cute.Tensor,
    mO: cute.Tensor,
    mLSE: cute.Tensor,
    q_layout: cute.ComposedLayout,
    k_half_layout: cute.ComposedLayout,
    v_half_layout: cute.ComposedLayout,
    k_quarter_layout: cute.ComposedLayout,
    v_quarter_layout: cute.ComposedLayout,
    p_half_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sHalfStats = storage.half_stats.get_tensor(
        cute.make_layout((HALVES, M, 2))
    )
    sQ = smem.allocate_tensor(
        element_type=BFloat16,
        layout=q_layout.outer,
        byte_alignment=128,
        swizzle=q_layout.inner,
    )
    # The physical allocation is the canonical N128 K layout consumed by
    # tcgen05.  The 64x64 K gather quarters, canonical N128 V, and 64x64 V
    # gather quarters are aliases of this allocation.  This direction matters:
    # allocating a gather view and merely proving equal byte counts does not
    # prove that the N128 MMA descriptor observes the intended addresses.
    sKHalf = smem.allocate_tensor(
        element_type=BFloat16,
        layout=k_half_layout.outer,
        byte_alignment=128,
        swizzle=k_half_layout.inner,
    )
    sKQuarter = cute.make_tensor(
        cute.recast_ptr(sKHalf.iterator, k_quarter_layout.inner, BFloat16),
        k_quarter_layout.outer,
    )
    sVHalf = cute.make_tensor(
        cute.recast_ptr(sKHalf.iterator, v_half_layout.inner, BFloat16),
        v_half_layout.outer,
    )
    sVQuarter = cute.make_tensor(
        cute.recast_ptr(sKHalf.iterator, v_quarter_layout.inner, BFloat16),
        v_quarter_layout.outer,
    )
    k_half_bytes = cute.size_in_bytes(BFloat16, k_half_layout)
    v_half_bytes = cute.size_in_bytes(BFloat16, v_half_layout)
    k_quarter_bytes = cute.size_in_bytes(BFloat16, k_quarter_layout)
    v_quarter_bytes = cute.size_in_bytes(BFloat16, v_quarter_layout)
    assert k_half_bytes == KV_PACK_SMEM_BYTES
    assert v_half_bytes == KV_PACK_SMEM_BYTES
    assert k_quarter_bytes == KV_PACK_SMEM_BYTES
    assert v_quarter_bytes == KV_PACK_SMEM_BYTES

    tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    final_join_barrier = pipeline.NamedBarrier(
        barrier_id=2, num_threads=12 * 32
    )
    kv_alias_barrier = pipeline.NamedBarrier(
        barrier_id=3, num_threads=9 * 32
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf, barrier_for_retrieve=tmem_barrier
    )
    tmem.allocate(TMEM_COLS)

    one_thread = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
    half_threads = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, 4 * 32
    )
    q_pipe = fa_pipeline.PipelineTmaUmma.create(
        num_stages=1,
        producer_group=one_thread,
        consumer_group=one_thread,
        tx_count=Q_SMEM_BYTES,
        barrier_storage=storage.q_mbar.data_ptr(),
    )
    kv_phase_pipe = fa_pipeline.PipelineTmaUmma.create(
        num_stages=1,
        producer_group=one_thread,
        consumer_group=one_thread,
        tx_count=KV_PACK_SMEM_BYTES,
        barrier_storage=storage.kv_phase_mbar.data_ptr(),
    )
    score0_pipe = fa_pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=one_thread,
        consumer_group=half_threads,
        barrier_storage=storage.score0_mbar.data_ptr(),
    )
    score1_pipe = fa_pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=one_thread,
        consumer_group=half_threads,
        barrier_storage=storage.score1_mbar.data_ptr(),
    )
    p0_pipe = fa_pipeline.PipelineAsyncUmma.create(
        num_stages=1,
        producer_group=half_threads,
        consumer_group=one_thread,
        barrier_storage=storage.p0_mbar.data_ptr(),
    )
    p1_pipe = fa_pipeline.PipelineAsyncUmma.create(
        num_stages=1,
        producer_group=half_threads,
        consumer_group=one_thread,
        barrier_storage=storage.p1_mbar.data_ptr(),
    )
    o0_pipe = fa_pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=one_thread,
        consumer_group=half_threads,
        barrier_storage=storage.o0_mbar.data_ptr(),
    )
    o1_pipe = fa_pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=one_thread,
        consumer_group=half_threads,
        barrier_storage=storage.o1_mbar.data_ptr(),
    )

    gQ = cute.local_tile(
        mQ_mkl, QK_HALF_TILE, (0, 0, None), proj=(1, None, 1)
    )
    gKQuarter = cute.local_tile(mK_quarter_nkl, (64, 64), (None, None))
    # Generic quarter TMA follows the donor's (D64, N64) global tiling.  Its
    # two coordinates become D-half and exact N64 block respectively.
    gVQuarter = cute.local_tile(mV_nkl, (64, 64), (None, None))
    thr_qk_half = tiled_qk_half.get_slice(0)
    thr_pv_half = tiled_pv_half.get_slice(0)
    tCgQ = thr_qk_half.partition_A(gQ)
    tQsQ, tQgQ = cpasync.tma_partition(
        tma_atom_q,
        0,
        cute.make_layout(1),
        cute.group_modes(sQ, 0, 3),
        cute.group_modes(tCgQ, 0, 3),
    )
    tKsKV, tKgK = cpasync.tma_partition(
        tma_atom_k,
        0,
        cute.make_layout(1),
        cute.group_modes(sKQuarter, 0, 3),
        cute.group_modes(gKQuarter, 0, 2),
    )
    tVsKV, tVgV = cpasync.tma_partition(
        tma_atom_v,
        0,
        cute.make_layout(1),
        cute.group_modes(sVQuarter, 0, 3),
        cute.group_modes(gVQuarter, 0, 2),
    )

    score_shape = tiled_qk_half.partition_shape_C(QK_HALF_TILE[:2])
    score_template = tiled_qk_half.make_fragment_C(score_shape)
    o_shape = tiled_pv_half.partition_shape_C(PV_HALF_TILE[:2])
    o_template = tiled_pv_half.make_fragment_C(o_shape)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(Float32)
    tmem.relinquish_alloc_permit()
    tmem_base = tmem_ptr.toint()
    tScore0 = cute.make_tensor(
        cute.make_ptr(
            Float32,
            tmem_base + Int32(S0_OFFSET),
            cute.AddressSpace.tmem,
            assumed_align=16,
        ),
        score_template.layout,
    )
    tScore1 = cute.make_tensor(
        cute.make_ptr(
            Float32,
            tmem_base + Int32(S1_OFFSET),
            cute.AddressSpace.tmem,
            assumed_align=16,
        ),
        score_template.layout,
    )
    tO0 = cute.make_tensor(
        cute.make_ptr(
            Float32,
            tmem_base + Int32(O0_OFFSET),
            cute.AddressSpace.tmem,
            assumed_align=16,
        ),
        o_template.layout,
    )
    tO1 = cute.make_tensor(
        cute.make_ptr(
            Float32,
            tmem_base + Int32(O1_OFFSET),
            cute.AddressSpace.tmem,
            assumed_align=16,
        ),
        o_template.layout,
    )
    tPStorage0 = cute.make_tensor(tScore0.iterator, p_half_layout.outer)
    tPStorage1 = cute.make_tensor(tScore1.iterator, p_half_layout.outer)
    tP0Base = tiled_pv_half.make_fragment_A(tPStorage0)[
        None, None, None, 0
    ]
    tP1Base = tiled_pv_half.make_fragment_A(tPStorage1)[
        None, None, None, 0
    ]
    p_width_ratio = Float32.width // BFloat16.width
    tP0 = cute.make_tensor(
        tP0Base.iterator
        + tmem_base
        + tmem_base
        + Int32(P0_OFFSET * p_width_ratio),
        tP0Base.layout,
    )
    tP1 = cute.make_tensor(
        tP1Base.iterator
        + tmem_base
        + tmem_base
        + Int32(P1_OFFSET * p_width_ratio),
        tP1Base.layout,
    )
    tCrQ = tiled_qk_half.make_fragment_A(sQ)
    tCrK = tiled_qk_half.make_fragment_B(sKHalf)
    tCrV = tiled_pv_half.make_fragment_B(sVHalf)

    exact_count = Int32(mExactCount[0])
    pack_count = exact_count // Int32(PACK_MEMBERS)

    # TMA role: each ring visit owns the complete 64 KiB buffer.  K is fully
    # consumed and released before V overwrites the same addresses.
    if warp_idx == Int32(14):
        cpasync.prefetch_descriptor(tma_atom_q)
        cpasync.prefetch_descriptor(tma_atom_k)
        cpasync.prefetch_descriptor(tma_atom_v)
        q_producer = fa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        kv_producer = fa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        q_pipe.producer_acquire(q_producer)
        cute.copy(
            tma_atom_q,
            tQgQ[(None, 0)],
            tQsQ[(None, 0)],
            tma_bar_ptr=q_pipe.producer_get_barrier(q_producer),
        )
        q_producer.advance()
        for pack_idx in cutlass.range(pack_count, unroll=1):
            kv_phase_pipe.producer_acquire(kv_producer)
            k_barrier = kv_phase_pipe.producer_get_barrier(kv_producer)
            for half in cutlass.range_constexpr(HALVES):
                block0 = Int32(
                    mExactIndices[
                        pack_idx * Int32(PACK_MEMBERS)
                        + Int32(half * MEMBERS_PER_HALF)
                    ]
                )
                block1 = Int32(
                    mExactIndices[
                        pack_idx * Int32(PACK_MEMBERS)
                        + Int32(half * MEMBERS_PER_HALF + 1)
                    ]
                )
                quarter0 = Int32(half * K_QUARTERS_PER_HALF)
                # Canonical N128 K-major physical order is K0/N0, K0/N1,
                # K1/N0, K1/N1.  Two whole D128xN64 member transactions are
                # member-major and do not preserve this address graph.
                cute.copy(
                    tma_atom_k,
                    tKgK[None, Int32(0), block0],
                    tKsKV[None, quarter0],
                    tma_bar_ptr=k_barrier,
                )
                cute.copy(
                    tma_atom_k,
                    tKgK[None, Int32(0), block1],
                    tKsKV[None, quarter0 + Int32(1)],
                    tma_bar_ptr=k_barrier,
                )
                cute.copy(
                    tma_atom_k,
                    tKgK[None, Int32(1), block0],
                    tKsKV[None, quarter0 + Int32(2)],
                    tma_bar_ptr=k_barrier,
                )
                cute.copy(
                    tma_atom_k,
                    tKgK[None, Int32(1), block1],
                    tKsKV[None, quarter0 + Int32(3)],
                    tma_bar_ptr=k_barrier,
                )
            kv_producer.advance()
            kv_phase_pipe.producer_acquire(kv_producer)
            v_barrier = kv_phase_pipe.producer_get_barrier(kv_producer)
            for half in cutlass.range_constexpr(HALVES):
                block0 = Int32(
                    mExactIndices[
                        pack_idx * Int32(PACK_MEMBERS)
                        + Int32(half * MEMBERS_PER_HALF)
                    ]
                )
                block1 = Int32(
                    mExactIndices[
                        pack_idx * Int32(PACK_MEMBERS)
                        + Int32(half * MEMBERS_PER_HALF + 1)
                    ]
                )
                quarter0 = Int32(half * V_QUARTERS_PER_HALF)
                # Canonical N128 MN-major physical order.  The four quarter
                # transactions are D0/N0, D0/N1, D1/N0, D1/N1.  Two D128xN64
                # copies would instead interleave the middle quarters.
                cute.copy(
                    tma_atom_v,
                    tVgV[None, Int32(0), block0],
                    tVsKV[None, quarter0],
                    tma_bar_ptr=v_barrier,
                )
                cute.copy(
                    tma_atom_v,
                    tVgV[None, Int32(0), block1],
                    tVsKV[None, quarter0 + Int32(1)],
                    tma_bar_ptr=v_barrier,
                )
                cute.copy(
                    tma_atom_v,
                    tVgV[None, Int32(1), block0],
                    tVsKV[None, quarter0 + Int32(2)],
                    tma_bar_ptr=v_barrier,
                )
                cute.copy(
                    tma_atom_v,
                    tVgV[None, Int32(1), block1],
                    tVsKV[None, quarter0 + Int32(3)],
                    tma_bar_ptr=v_barrier,
                )
            kv_producer.advance()

    # BEGIN N256 MAINLOOP -- the two output states remain independent.
    if warp_idx == Int32(12):
        q_consumer = fa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )
        kv_consumer = fa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )
        score0_producer = fa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        score1_producer = fa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        p0_consumer = fa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )
        p1_consumer = fa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )
        o0_producer = fa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        o1_producer = fa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        q_pipe.consumer_wait(q_consumer)
        for pack_idx in cutlass.range(pack_count, unroll=1):
            kv_phase_pipe.consumer_wait(kv_consumer)
            score0_pipe.producer_acquire(score0_producer)
            mma_utils.gemm(
                tiled_qk_half,
                tScore0,
                tCrQ[None, None, None, Int32(0)],
                tCrK[None, None, None, Int32(0)],
                zero_init=True,
            )
            score0_pipe.producer_commit(score0_producer)
            score0_producer.advance()
            score1_pipe.producer_acquire(score1_producer)
            mma_utils.gemm(
                tiled_qk_half,
                tScore1,
                tCrQ[None, None, None, Int32(0)],
                tCrK[None, None, None, Int32(1)],
                zero_init=True,
            )
            score1_pipe.producer_commit(score1_producer)
            score1_producer.advance()
            kv_alias_barrier.arrive_and_wait()
            kv_phase_pipe.consumer_release(kv_consumer)
            kv_consumer.advance()

            kv_phase_pipe.consumer_wait(kv_consumer)
            p0_pipe.consumer_wait(p0_consumer)
            o0_pipe.producer_acquire(o0_producer)
            mma_utils.gemm(
                tiled_pv_half,
                tO0,
                tP0,
                tCrV[None, None, None, Int32(0)],
                zero_init=pack_idx == Int32(0),
            )
            o0_pipe.producer_commit(o0_producer)
            o0_producer.advance()
            p0_pipe.consumer_release(p0_consumer)
            p0_consumer.advance()

            p1_pipe.consumer_wait(p1_consumer)
            o1_pipe.producer_acquire(o1_producer)
            mma_utils.gemm(
                tiled_pv_half,
                tO1,
                tP1,
                tCrV[None, None, None, Int32(1)],
                zero_init=pack_idx == Int32(0),
            )
            o1_pipe.producer_commit(o1_producer)
            o1_producer.advance()
            p1_pipe.consumer_release(p1_consumer)
            p1_consumer.advance()
            kv_alias_barrier.arrive_and_wait()
            kv_phase_pipe.consumer_release(kv_consumer)
            kv_consumer.advance()
        q_pipe.consumer_release(q_consumer)

    if warp_idx >= Int32(0) and warp_idx <= Int32(3):
        _run_independent_half_mainloop(
            0,
            Int32(S0_OFFSET),
            Int32(P0_OFFSET),
            Int32(O0_OFFSET),
            score_template,
            o_template,
            thr_qk_half,
            thr_pv_half,
            tmem_base,
            tidx,
            pack_count,
            score0_pipe,
            p0_pipe,
            o0_pipe,
            kv_alias_barrier,
            sHalfStats,
        )
        final_join_barrier.arrive_and_wait()
    if warp_idx >= Int32(4) and warp_idx <= Int32(7):
        _run_independent_half_mainloop(
            1,
            Int32(S1_OFFSET),
            Int32(P1_OFFSET),
            Int32(O1_OFFSET),
            score_template,
            o_template,
            thr_qk_half,
            thr_pv_half,
            tmem_base,
            tidx - Int32(4 * 32),
            pack_count,
            score1_pipe,
            p1_pipe,
            o1_pipe,
            kv_alias_barrier,
            sHalfStats,
        )
        final_join_barrier.arrive_and_wait()
    # END N256 MAINLOOP

    # BEGIN DEFERRED COMBINE EPILOGUE -- the sole cross-half join.
    if warp_idx >= Int32(8) and warp_idx <= Int32(11):
        final_join_barrier.arrive_and_wait()
        _deferred_combine_epilogue(
            o_template,
            thr_pv_half,
            tmem_base,
            tidx - Int32(8 * 32),
            sHalfStats,
            mO,
            mLSE,
        )
    # END DEFERRED COMBINE EPILOGUE

    cute.arch.barrier()
    tmem.free(tmem_ptr)


@cute.jit
def n256_deferred_exact_leaf_host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    exact_indices: cute.Tensor,
    exact_count: cute.Tensor,
    o: cute.Tensor,
    lse: cute.Tensor,
):
    qk_quarter_op = tcgen05.MmaF16BF16Op(
        BFloat16,
        Float32,
        QK_QUARTER_INST,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    qk_half_op = tcgen05.MmaF16BF16Op(
        BFloat16,
        Float32,
        QK_HALF_INST,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    pv_quarter_op = tcgen05.MmaF16BF16Op(
        BFloat16,
        Float32,
        PV_QUARTER_INST,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.TMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.MN,
    )
    pv_half_op = tcgen05.MmaF16BF16Op(
        BFloat16,
        Float32,
        PV_HALF_INST,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.TMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.MN,
    )
    tiled_qk_quarter = cute.make_tiled_mma(qk_quarter_op)
    tiled_qk_half = cute.make_tiled_mma(qk_half_op)
    tiled_pv_quarter = cute.make_tiled_mma(pv_quarter_op)
    tiled_pv_half = cute.make_tiled_mma(pv_half_op)

    q_layout = sm100_utils.make_smem_layout_a(
        tiled_qk_half, QK_HALF_TILE, BFloat16, 1
    )
    k_half_layout = sm100_utils.make_smem_layout_b(
        tiled_qk_half, QK_HALF_TILE, BFloat16, HALVES
    )
    v_half_layout = sm100_utils.make_smem_layout_b(
        tiled_pv_half, PV_HALF_TILE, BFloat16, HALVES
    )
    k_quarter_layout = sm100_utils.make_smem_layout_b(
        tiled_qk_quarter,
        QK_QUARTER_TILE,
        BFloat16,
        K_GATHER_STAGES,
    )
    v_quarter_layout = sm100_utils.make_smem_layout_b(
        tiled_pv_quarter,
        PV_QUARTER_TILE,
        BFloat16,
        V_GATHER_STAGES,
    )
    p_half_layout = sm100_utils.make_smem_layout_a(
        tiled_pv_half, PV_HALF_TILE, BFloat16, 1
    )

    copy_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    q_tma_atom, q_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        copy_op,
        q,
        cute.select(q_layout, mode=[0, 1, 2]),
        QK_HALF_TILE,
        tiled_qk_half,
    )
    k_dn = cute.make_tensor(k.iterator, cute.select(k.layout, mode=[1, 0]))
    k_quarter_tma_layout = cute.make_composed_layout(
        k_quarter_layout.inner,
        0,
        cute.make_layout((64, 64), stride=(1, 64)),
    )
    k_tma_atom, k_tma_tensor = cpasync.make_tiled_tma_atom(
        copy_op,
        k_dn,
        k_quarter_tma_layout,
        (64, 64),
    )
    v_nk = cute.make_tensor(v.iterator, cute.select(v.layout, mode=[1, 0]))
    v_quarter_tma_layout = cute.make_composed_layout(
        v_quarter_layout.inner,
        0,
        cute.make_layout((64, 64), stride=(1, 64)),
    )
    v_tma_atom, v_tma_tensor = cpasync.make_tiled_tma_atom(
        copy_op,
        v_nk,
        v_quarter_tma_layout,
        (64, 64),
    )
    n256_deferred_exact_leaf_kernel(
        tiled_qk_half,
        tiled_pv_half,
        tiled_qk_quarter,
        tiled_pv_quarter,
        q_tma_atom,
        q_tma_tensor,
        k_tma_atom,
        k_tma_tensor,
        v_tma_atom,
        v_tma_tensor,
        exact_indices,
        exact_count,
        o,
        lse,
        q_layout,
        k_half_layout,
        v_half_layout,
        k_quarter_layout,
        v_quarter_layout,
        p_half_layout,
    ).launch(grid=(1, 1, 1), block=(THREADS, 1, 1))


def contract_record() -> dict[str, object]:
    """Return metadata only; this function does not launch the unvalidated leaf."""

    return {
        "mechanism_family": MECHANISM_FAMILY,
        "validation_status": VALIDATION_STATUS,
        "noncontiguous_pack": NONCONTIGUOUS_PACK,
        "pack_schedule": build_n256_pack_schedule(NONCONTIGUOUS_PACK),
        "roles": ROLE_MAP,
        "tmem_cols": TMEM_COLS,
        "smem_payload_bytes": SMEM_PAYLOAD_BYTES,
        "source_receipt": SOURCE_RECEIPT,
    }


def _wrapped(
    x: torch.Tensor,
    leading_dim: int,
    compact: int,
    dynamic_outer_divisibility: Optional[int] = None,
):
    wrapped = (
        from_dlpack(x, assumed_align=32, enable_tvm_ffi=True)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(
            mode=compact, divisibility=x.shape[compact]
        )
    )
    if dynamic_outer_divisibility is not None:
        wrapped = wrapped.mark_compact_shape_dynamic(
            mode=0, divisibility=dynamic_outer_divisibility
        )
    return wrapped


def _emit(path: Optional[Path], row: dict[str, object]) -> None:
    line = json.dumps(row, sort_keys=True)
    print(line, flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def run_probe(
    output: Optional[Path] = None,
    warmup: int = 10,
    rep: int = 50,
    fail_closed: bool = True,
) -> dict[str, object]:
    """Compile and run the fixed four-block N256 gate on the active CUDA GPU."""

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")
    if not torch.cuda.is_available():
        raise RuntimeError("N256 deferred device gate requires a CUDA GPU")

    exact_index_values = NONCONTIGUOUS_PACK
    exact_blocks = len(exact_index_values)
    if build_n256_pack_schedule(exact_index_values) != (
        NONCONTIGUOUS_PACK,
    ):
        raise AssertionError("fixed N256 pack contract changed")

    torch.manual_seed(25604)
    storage_blocks = max(exact_index_values) + 1
    q = torch.randn((M, D), device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        (storage_blocks * N_MEMBER, D),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        (storage_blocks * N_MEMBER, DV),
        device="cuda",
        dtype=torch.bfloat16,
    )
    exact_indices = torch.tensor(
        exact_index_values, device="cuda", dtype=torch.int32
    )
    exact_count = torch.tensor(
        [exact_blocks], device="cuda", dtype=torch.int32
    )
    exact_indices_before = exact_indices.clone()

    token_indices = (
        exact_indices.long()[:, None] * N_MEMBER
        + torch.arange(N_MEMBER, device="cuda", dtype=torch.long)[None, :]
    ).reshape(-1)
    selected_k = k.index_select(0, token_indices)
    selected_v = v.index_select(0, token_indices)
    scores = q.float() @ selected_k.float().T * SCALE
    reference_lse = torch.logsumexp(scores, dim=-1)
    row_max = scores.amax(dim=-1, keepdim=True)
    p_float = torch.exp(scores - row_max)
    # Match the kernel's BF16 probability operand while retaining an FP32
    # denominator, exactly as in the established N128 device gate.
    reference_o = (
        p_float.to(torch.bfloat16).float() @ selected_v.float()
    ) / p_float.sum(dim=-1, keepdim=True)

    o = torch.full(
        (M, DV), float("nan"), device="cuda", dtype=torch.float32
    )
    lse = torch.full((M,), float("nan"), device="cuda", dtype=torch.float32)
    q_cute = _wrapped(q, 1, 1)
    k_cute = _wrapped(
        k, 1, 1, dynamic_outer_divisibility=N_MEMBER
    )
    v_cute = _wrapped(
        v, 1, 1, dynamic_outer_divisibility=N_MEMBER
    )
    exact_indices_cute = _wrapped(exact_indices, 0, 0)
    exact_count_cute = _wrapped(exact_count, 0, 0)
    o_cute = _wrapped(o, 1, 1)
    lse_cute = _wrapped(lse, 0, 0)

    compile_started = time.perf_counter()
    fn = cute.compile(
        n256_deferred_exact_leaf_host,
        q_cute,
        k_cute,
        v_cute,
        exact_indices_cute,
        exact_count_cute,
        o_cute,
        lse_cute,
        options="--enable-tvm-ffi",
    )
    compile_s = time.perf_counter() - compile_started
    fn(
        q_cute,
        k_cute,
        v_cute,
        exact_indices_cute,
        exact_count_cute,
        o_cute,
        lse_cute,
    )
    torch.cuda.synchronize()
    kernel_ms = float(
        triton.testing.do_bench(
            lambda: fn(
                q_cute,
                k_cute,
                v_cute,
                exact_indices_cute,
                exact_count_cute,
                o_cute,
                lse_cute,
            ),
            warmup=warmup,
            rep=rep,
        )
    )

    exact_indices_immutable = bool(
        torch.equal(exact_indices, exact_indices_before)
    )
    finite = bool(torch.isfinite(o).all() and torch.isfinite(lse).all())
    o_diff = o - reference_o
    lse_diff = lse - reference_lse
    o_max_abs = float(o_diff.abs().max()) if finite else float("inf")
    o_mean_abs = float(o_diff.abs().mean()) if finite else float("inf")
    o_rel_l2 = (
        float(
            torch.linalg.vector_norm(o_diff)
            / torch.linalg.vector_norm(reference_o).clamp_min(1.0e-12)
        )
        if finite
        else float("inf")
    )
    lse_max_abs = float(lse_diff.abs().max()) if finite else float("inf")
    lse_mean_abs = float(lse_diff.abs().mean()) if finite else float("inf")
    lse_rel_l2 = (
        float(
            torch.linalg.vector_norm(lse_diff)
            / torch.linalg.vector_norm(reference_lse).clamp_min(1.0e-12)
        )
        if finite
        else float("inf")
    )
    passes = (
        finite
        and exact_indices_immutable
        and o_max_abs <= 0.08
        and o_mean_abs <= 0.01
        and o_rel_l2 <= 0.01
        and lse_max_abs <= 0.05
        and lse_mean_abs <= 0.005
        and lse_rel_l2 <= 0.005
    )
    row = {
        "event": "n256_deferred_leaf_device_gate",
        "passes": passes,
        "validation_status": (
            "b200-device-gate-pass" if passes else "b200-device-gate-fail"
        ),
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "exact_blocks": exact_blocks,
        "exact_indices": list(exact_index_values),
        "noncontiguous": all(
            right - left > 1
            for left, right in zip(
                exact_index_values, exact_index_values[1:]
            )
        ),
        "n256_transactions": 1,
        "n128_half_qk": 2,
        "n128_half_pv": 2,
        "finite": finite,
        "exact_indices_immutable": exact_indices_immutable,
        "o_max_abs": o_max_abs,
        "o_mean_abs": o_mean_abs,
        "o_rel_l2": o_rel_l2,
        "lse_max_abs": lse_max_abs,
        "lse_mean_abs": lse_mean_abs,
        "lse_rel_l2": lse_rel_l2,
        "compile_s": compile_s,
        "kernel_ms": kernel_ms,
        "warmup": warmup,
        "rep": rep,
        "threads": THREADS,
        "tmem_cols": TMEM_COLS,
        "smem_payload_bytes": SMEM_PAYLOAD_BYTES,
        "mainloop_cross_half_reduce": False,
        "mainloop_output_pair_combine": False,
        "epilogue_deferred_combine": True,
        "mechanism_only": True,
        "full_sol_attn": False,
        "production_dispatch_changed": False,
        "mechanism_family": MECHANISM_FAMILY,
        "source_receipt": SOURCE_RECEIPT,
    }
    _emit(output, row)
    if fail_closed and not passes:
        raise AssertionError(f"N256 deferred device gate failed: {row}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--no-fail-closed", action="store_true")
    args = parser.parse_args()
    if args.contract_only:
        print(json.dumps(contract_record(), sort_keys=True))
        return
    run_probe(
        output=args.output,
        warmup=args.warmup,
        rep=args.rep,
        fail_closed=not args.no_fail_closed,
    )


if __name__ == "__main__":
    main()
