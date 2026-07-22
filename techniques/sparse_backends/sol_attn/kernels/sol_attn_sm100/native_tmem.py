"""Fixed M64/N64 TMEM leaf operations for the native SM100 forward.

Provenance and retained semantics:

* Dao-AILab/flash-attention ``b94e8ec35b6b741ec9099d45559ada36ea9354c8``
  as pinned in ``_fa4/flash_fwd_sm100.py``: correction is an ordered
  TMEM-load, FP32 scale, TMEM-store operation.
* Repository object ``a74f85f:experiments/sol_attn/probe_native_m64_route_g.py``
  (B200 v990): M64 ``Ld16x64b`` score ownership, the lane-xor-2 BF16
  halfword transpose for ``St16x64b``, and the final ``Ld16x256b`` map.
* Repository object
  ``a74f85f:experiments/sol_attn/probe_native_m64_n128_pack2_dual_cohort.py``
  (B200 v994): partition a zero-based TMEM tensor first, then add the
  allocator-returned physical base to each per-thread partition.

These helpers are deliberately narrower than an attention mainloop.  They
fix M=64, N=64, DV=128, one CTA, S=(0, 64), P=(32, 96), and O=128 TMEM
column offsets.  Pipeline acquire/release, generation selection, and CTA
barriers remain the caller's responsibility.  The waits here retire only the
tcgen05 loads or stores issued by the leaf itself.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from cutlass import BFloat16, Float32, Int32
from cutlass._mlir.dialects import llvm


M64 = 64
N64 = 64
DV128 = 128
S_STAGE_OFFSETS = (0, 64)
P_STAGE_OFFSETS = (32, 96)
O_OFFSET = 128


@cute.jit
def tcgen05_wait_ld() -> None:
    """Retire all prior asynchronous tcgen05 register loads for the thread."""

    llvm.inline_asm(
        None,
        [],
        "tcgen05.wait::ld.sync.aligned;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def tcgen05_wait_st() -> None:
    """Retire all prior asynchronous tcgen05 register stores for the thread."""

    llvm.inline_asm(
        None,
        [],
        "tcgen05.wait::st.sync.aligned;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _zero_based_tmem_tensor(element_type, layout):
    """Create a layout-only TMEM tensor whose address is intentionally zero."""

    return cute.make_tensor(
        cute.make_ptr(
            element_type,
            Int32(0),
            cute.AddressSpace.tmem,
            assumed_align=16,
        ),
        layout,
    )


@cute.jit
def _add_physical_tmem_base(relative: cute.Tensor, physical_address: Int32):
    """Rebind a per-thread zero-based partition to its physical TMEM address."""

    return cute.make_tensor(
        cute.make_ptr(
            relative.element_type,
            physical_address + relative.iterator.toint(),
            cute.AddressSpace.tmem,
            assumed_align=16,
        ),
        relative.layout,
    )


@cute.jit
def load_m64_s0_s1_i32(
    score_template: cute.Tensor,
    thr_mma_qk: cute.core.ThrMma,
    physical_tmem_base: Int32,
    tidx: Int32,
):
    """Load both M64xN64 INT32 score stages from physical TMEM.

    ``score_template`` contributes only the native M64 accumulator layout;
    its iterator is never used.  The returned coordinate tensor is the exact
    logical (row, column) owner map for both register fragments.
    """

    assert score_template.element_type == Int32
    assert cute.size(score_template) == M64 * N64

    relative_score = _zero_based_tmem_tensor(
        Int32, score_template.layout
    )
    load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld16x64bOp(tcgen05.copy.Repetition(32)),
        Int32,
    )
    tiled_load = tcgen05.make_tmem_copy(load_atom, relative_score)
    thread_load = tiled_load.get_slice(tidx)
    source_relative = thread_load.partition_S(relative_score)

    source_s0 = _add_physical_tmem_base(
        source_relative,
        physical_tmem_base + Int32(S_STAGE_OFFSETS[0]),
    )
    source_s1 = _add_physical_tmem_base(
        source_relative,
        physical_tmem_base + Int32(S_STAGE_OFFSETS[1]),
    )
    score_identity = thr_mma_qk.partition_C(
        cute.make_identity_tensor((M64, N64))
    )
    score_coordinates = thread_load.partition_D(score_identity)
    score_s0 = cute.make_fragment(score_coordinates.shape, Int32)
    score_s1 = cute.make_fragment(score_coordinates.shape, Int32)

    cute.copy(tiled_load, source_s0, score_s0)
    cute.copy(tiled_load, source_s1, score_s1)
    tcgen05_wait_ld()
    cute.arch.fence_view_async_tmem_load()
    return score_s0, score_s1, score_coordinates


@cute.jit
def load_m64_s_i32(
    score_template: cute.Tensor,
    thr_mma_qk: cute.core.ThrMma,
    physical_tmem_base: Int32,
    tidx: Int32,
    stage: cutlass.Constexpr[int],
):
    """Load one explicitly selected M64xN64 INT32 score stage."""

    assert stage == 0 or stage == 1
    assert score_template.element_type == Int32
    assert cute.size(score_template) == M64 * N64

    relative_score = _zero_based_tmem_tensor(
        Int32, score_template.layout
    )
    load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld16x64bOp(tcgen05.copy.Repetition(32)),
        Int32,
    )
    tiled_load = tcgen05.make_tmem_copy(load_atom, relative_score)
    thread_load = tiled_load.get_slice(tidx)
    source_relative = thread_load.partition_S(relative_score)
    source = _add_physical_tmem_base(
        source_relative,
        physical_tmem_base + Int32(S_STAGE_OFFSETS[stage]),
    )
    score_identity = thr_mma_qk.partition_C(
        cute.make_identity_tensor((M64, N64))
    )
    score_coordinates = thread_load.partition_D(score_identity)
    score = cute.make_fragment(score_coordinates.shape, Int32)

    cute.copy(tiled_load, source, score)
    tcgen05_wait_ld()
    cute.arch.fence_view_async_tmem_load()
    return score, score_coordinates


@cute.jit
def load_m64_s_f32(
    score_template: cute.Tensor,
    thr_mma_qk: cute.core.ThrMma,
    physical_tmem_base: Int32,
    tidx: Int32,
    stage: cutlass.Constexpr[int],
):
    """Load one M64xN64 FP32 BF16-QK score stage from TMEM.

    BF16 tcgen05 QK accumulates directly into FP32 TMEM.  Its accumulator
    ownership is bit-width compatible with the proven INT32 M64 map, but the
    element type must remain FP32 so the specialization never introduces an
    integer-to-float conversion on the score path.
    """

    assert stage == 0 or stage == 1
    assert score_template.element_type == Float32
    assert cute.size(score_template) == M64 * N64

    relative_score = _zero_based_tmem_tensor(
        Float32, score_template.layout
    )
    load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld16x64bOp(tcgen05.copy.Repetition(32)),
        Float32,
    )
    tiled_load = tcgen05.make_tmem_copy(load_atom, relative_score)
    thread_load = tiled_load.get_slice(tidx)
    source_relative = thread_load.partition_S(relative_score)
    source = _add_physical_tmem_base(
        source_relative,
        physical_tmem_base + Int32(S_STAGE_OFFSETS[stage]),
    )
    score_identity = thr_mma_qk.partition_C(
        cute.make_identity_tensor((M64, N64))
    )
    score_coordinates = thread_load.partition_D(score_identity)
    score = cute.make_fragment(score_coordinates.shape, Float32)

    cute.copy(tiled_load, source, score)
    tcgen05_wait_ld()
    cute.arch.fence_view_async_tmem_load()
    return score, score_coordinates


@cute.jit
def transpose_m64_p_bf16_lane_halfwords(
    probabilities: cute.Tensor,
    p_store_registers: cute.Tensor,
    tidx: Int32,
):
    """Form the proven M64 St16x64b BF16 register ordering in place."""

    assert probabilities.element_type == Float32
    assert p_store_registers.element_type == Float32
    p_bf16 = cute.make_tensor(
        cute.recast_ptr(
            p_store_registers.iterator,
            dtype=BFloat16,
        ),
        probabilities.layout,
    )
    assert cute.size(p_bf16) == cute.size(probabilities)

    for i in cutlass.range(
        cute.size(probabilities), unroll_full=True
    ):
        p_bf16[i] = Float32(probabilities[i]).to(BFloat16)

    lane = tidx % Int32(32)
    for i in cutlass.range(
        cute.size(p_store_registers), unroll_full=True
    ):
        low = i * 2
        high = low + 1
        own_low = Float32(p_bf16[low])
        own_high = Float32(p_bf16[high])
        peer_low = cute.arch.shuffle_sync_bfly(own_low, offset=2)
        peer_high = cute.arch.shuffle_sync_bfly(own_high, offset=2)
        if (lane & Int32(2)) == Int32(0):
            p_bf16[high] = peer_low.to(BFloat16)
        else:
            p_bf16[low] = peer_high.to(BFloat16)
    return p_store_registers


@cute.jit
def store_m64_p_bf16(
    o_template: cute.Tensor,
    probabilities: cute.Tensor,
    physical_tmem_base: Int32,
    tidx: Int32,
    stage: cutlass.Constexpr[int],
):
    """Transpose and store one M64xN64 BF16 P tile into P0 or P1."""

    assert stage == 0 or stage == 1
    assert o_template.element_type == Float32
    assert cute.size(o_template) == M64 * DV128

    p_layout = cute.composition(
        o_template.layout,
        cute.make_layout((M64, N64 // 2)),
    )
    relative_p = _zero_based_tmem_tensor(Float32, p_layout)
    store_atom = cute.make_copy_atom(
        tcgen05.copy.St16x64bOp(tcgen05.copy.Repetition(16)),
        Float32,
    )
    tiled_store = tcgen05.make_tmem_copy(store_atom, relative_p)
    thread_store = tiled_store.get_slice(tidx)
    destination_relative = thread_store.partition_D(relative_p)
    destination = _add_physical_tmem_base(
        destination_relative,
        physical_tmem_base + Int32(P_STAGE_OFFSETS[stage]),
    )
    p_store_coordinates = thread_store.partition_S(
        cute.make_identity_tensor((M64, N64 // 2))
    )
    p_store_registers = cute.make_fragment(
        p_store_coordinates.shape, Float32
    )
    transpose_m64_p_bf16_lane_halfwords(
        probabilities,
        p_store_registers,
        tidx,
    )

    cute.copy(tiled_store, p_store_registers, destination)
    tcgen05_wait_st()
    cute.arch.fence_view_async_tmem_store()
    return p_store_registers


@cute.jit
def _make_m64_o_copy_views(
    o_template: cute.Tensor,
    thr_mma_pv: cute.core.ThrMma,
):
    """Return zero-based M64xDV128 TMEM and logical-coordinate views."""

    assert o_template.element_type == Float32
    assert cute.size(o_template) == M64 * DV128
    relative_o = _zero_based_tmem_tensor(Float32, o_template.layout)
    o_coordinates = thr_mma_pv.partition_C(
        cute.make_identity_tensor((M64, DV128))
    )
    epilogue_tiler = (
        (
            cute.size(relative_o, mode=[0, 0]),
            cute.size(relative_o, mode=[0, 1]),
        ),
    )
    return (
        cute.zipped_divide(relative_o, epilogue_tiler),
        cute.zipped_divide(o_coordinates, epilogue_tiler),
    )


@cute.jit
def load_m64_o_fp32_256b(
    o_template: cute.Tensor,
    thr_mma_pv: cute.core.ThrMma,
    physical_tmem_base: Int32,
    tidx: Int32,
):
    """Load the complete M64xDV128 FP32 O tile with the v990 256-bit map."""

    relative_o, o_coordinates = _make_m64_o_copy_views(
        o_template, thr_mma_pv
    )
    load_atom = cute.make_copy_atom(
        tcgen05.Ld16x256bOp(tcgen05.Repetition.x8),
        Float32,
    )
    tiled_load = tcgen05.make_tmem_copy(
        load_atom, relative_o[None, Int32(0)]
    )
    thread_load = tiled_load.get_slice(tidx)
    source_relative = thread_load.partition_S(relative_o)
    source = _add_physical_tmem_base(
        source_relative,
        physical_tmem_base + Int32(O_OFFSET),
    )
    register_coordinates = thread_load.partition_D(o_coordinates)
    register_coordinates = register_coordinates[
        None, None, Int32(0)
    ]
    o_registers = cute.make_fragment(
        register_coordinates.shape, Float32
    )

    cute.copy(
        tiled_load,
        source[None, None, Int32(0)],
        o_registers,
    )
    tcgen05_wait_ld()
    cute.arch.fence_view_async_tmem_load()
    return o_registers, register_coordinates


@cute.jit
def rescale_m64_o_fp32_32b(
    o_template: cute.Tensor,
    thr_mma_pv: cute.core.ThrMma,
    physical_tmem_base: Int32,
    tidx: Int32,
    row_alpha: cute.Tensor,
) -> None:
    """Rescale M64 O with its matched 16-datapath TMEM load/store map.

    The wide M64 epilogue load is ideal for the final one-way global store, but
    its x8 register layout is not the source layout consumed by an x16 wide
    TMEM store.  Correction therefore follows the M64/1CTA FA4 specialization:
    matched ``Ld16x64b/St16x64b`` x8 atoms process eight D16 fragments, while
    logical coordinates select the proper per-row alpha for every register.
    """

    assert row_alpha.element_type == Float32
    assert cute.size(row_alpha) == M64
    assert o_template.element_type == Float32
    assert cute.size(o_template) == M64 * DV128

    relative_o = _zero_based_tmem_tensor(Float32, o_template.layout)
    o_coordinates = thr_mma_pv.partition_C(
        cute.make_identity_tensor((M64, DV128))
    )

    correction_width = 16
    load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld16x64bOp(tcgen05.copy.Repetition(8)),
        Float32,
    )
    store_atom = cute.make_copy_atom(
        tcgen05.copy.St16x64bOp(tcgen05.copy.Repetition(8)),
        Float32,
    )
    relative_fragment = cute.composition(
        relative_o, cute.make_layout((M64, correction_width))
    )
    coordinate_fragment = cute.composition(
        o_coordinates, cute.make_layout((M64, correction_width))
    )
    thread_load = tcgen05.make_tmem_copy(
        load_atom, relative_fragment
    ).get_slice(tidx)
    thread_store = tcgen05.make_tmem_copy(
        store_atom, relative_fragment
    ).get_slice(tidx)
    source_relative = thread_load.partition_S(relative_fragment)
    register_coordinates = thread_load.partition_D(coordinate_fragment)
    destination_relative = thread_store.partition_D(relative_fragment)
    source = _add_physical_tmem_base(
        source_relative,
        physical_tmem_base + Int32(O_OFFSET),
    )
    destination = _add_physical_tmem_base(
        destination_relative,
        physical_tmem_base + Int32(O_OFFSET),
    )

    for column_fragment in cutlass.range_constexpr(
        DV128 // correction_width
    ):
        o_registers = cute.make_fragment(
            register_coordinates.shape, Float32
        )
        source_fragment = cute.make_tensor(
            source.iterator + column_fragment * correction_width,
            source.layout,
        )
        cute.copy(thread_load, source_fragment, o_registers)
        tcgen05_wait_ld()
        cute.arch.fence_view_async_tmem_load()
        for i in cutlass.range(
            cute.size(o_registers), unroll_full=True
        ):
            row = register_coordinates[i][0]
            o_registers[i] = (
                Float32(o_registers[i]) * Float32(row_alpha[row])
            )
        destination_fragment = cute.make_tensor(
            destination.iterator + column_fragment * correction_width,
            destination.layout,
        )
        cute.copy(thread_store, o_registers, destination_fragment)
        tcgen05_wait_st()
    cute.arch.fence_view_async_tmem_store()


@cute.jit
def rescale_m64_o_fp32_32b_scalar(
    o_template: cute.Tensor,
    thr_mma_pv: cute.core.ThrMma,
    physical_tmem_base: Int32,
    tidx: Int32,
    row_alpha: Float32,
) -> None:
    """Rescale the TMEM-O row owned by ``tidx`` with a register alpha.

    The four fused BF16 softmax warps use the same ``tidx`` values and the
    same Ld16x64b/St16x64b ownership map as the former correction warps.  A
    softmax lane therefore already owns the alpha for every O register in its
    slice; no cross-warp shared-memory alpha vector is required.
    """

    assert o_template.element_type == Float32
    assert cute.size(o_template) == M64 * DV128

    relative_o = _zero_based_tmem_tensor(Float32, o_template.layout)
    o_coordinates = thr_mma_pv.partition_C(
        cute.make_identity_tensor((M64, DV128))
    )

    correction_width = 16
    load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld16x64bOp(tcgen05.copy.Repetition(8)),
        Float32,
    )
    store_atom = cute.make_copy_atom(
        tcgen05.copy.St16x64bOp(tcgen05.copy.Repetition(8)),
        Float32,
    )
    relative_fragment = cute.composition(
        relative_o, cute.make_layout((M64, correction_width))
    )
    coordinate_fragment = cute.composition(
        o_coordinates, cute.make_layout((M64, correction_width))
    )
    thread_load = tcgen05.make_tmem_copy(
        load_atom, relative_fragment
    ).get_slice(tidx)
    thread_store = tcgen05.make_tmem_copy(
        store_atom, relative_fragment
    ).get_slice(tidx)
    source_relative = thread_load.partition_S(relative_fragment)
    register_coordinates = thread_load.partition_D(coordinate_fragment)
    destination_relative = thread_store.partition_D(relative_fragment)
    source = _add_physical_tmem_base(
        source_relative,
        physical_tmem_base + Int32(O_OFFSET),
    )
    destination = _add_physical_tmem_base(
        destination_relative,
        physical_tmem_base + Int32(O_OFFSET),
    )

    alpha = Float32(row_alpha)
    for column_fragment in cutlass.range_constexpr(
        DV128 // correction_width
    ):
        o_registers = cute.make_fragment(
            register_coordinates.shape, Float32
        )
        source_fragment = cute.make_tensor(
            source.iterator + column_fragment * correction_width,
            source.layout,
        )
        cute.copy(thread_load, source_fragment, o_registers)
        tcgen05_wait_ld()
        cute.arch.fence_view_async_tmem_load()
        for i in cutlass.range(
            cute.size(o_registers), unroll_full=True
        ):
            o_registers[i] = Float32(o_registers[i]) * alpha
        destination_fragment = cute.make_tensor(
            destination.iterator + column_fragment * correction_width,
            destination.layout,
        )
        cute.copy(thread_store, o_registers, destination_fragment)
        tcgen05_wait_st()
    cute.arch.fence_view_async_tmem_store()


@cute.jit
def store_normalized_m64_o_bhtd(
    o_template: cute.Tensor,
    thr_mma_pv: cute.core.ThrMma,
    physical_tmem_base: Int32,
    tidx: Int32,
    row_sum: cute.Tensor,
    mO_bhtd: cute.Tensor,
    batch_idx: Int32,
    head_idx: Int32,
    query_start: Int32,
    seqlen_q: Int32,
) -> None:
    """Normalize native O and directly store the valid M64 rows to B,H,T,D."""

    assert row_sum.element_type == Float32
    assert cute.size(row_sum) == M64
    assert mO_bhtd.element_type == BFloat16
    assert cute.rank(mO_bhtd) == 4
    o_registers, o_coordinates = load_m64_o_fp32_256b(
        o_template,
        thr_mma_pv,
        physical_tmem_base,
        tidx,
    )
    for i in cutlass.range(
        cute.size(o_registers), unroll_full=True
    ):
        row = o_coordinates[i][0]
        column = o_coordinates[i][1]
        query_idx = query_start + row
        if query_idx < seqlen_q:
            inv_sum = cute.arch.rcp_approx(Float32(row_sum[row]))
            mO_bhtd[batch_idx, head_idx, query_idx, column] = (
                Float32(o_registers[i]) * inv_sum
            ).to(BFloat16)


__all__ = [
    "DV128",
    "M64",
    "N64",
    "O_OFFSET",
    "P_STAGE_OFFSETS",
    "S_STAGE_OFFSETS",
    "load_m64_o_fp32_256b",
    "load_m64_s0_s1_i32",
    "load_m64_s_i32",
    "rescale_m64_o_fp32_32b",
    "rescale_m64_o_fp32_32b_scalar",
    "store_m64_p_bf16",
    "store_normalized_m64_o_bhtd",
    "tcgen05_wait_ld",
    "tcgen05_wait_st",
    "transpose_m64_p_bf16_lane_halfwords",
]
