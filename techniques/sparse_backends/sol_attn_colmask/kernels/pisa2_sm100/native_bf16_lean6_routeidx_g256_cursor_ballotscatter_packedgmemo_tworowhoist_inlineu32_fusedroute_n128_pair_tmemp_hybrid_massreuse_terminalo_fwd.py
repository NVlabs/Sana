"""B9 scheduling/lifecycle parity backport to logical G256.

This isolated candidate starts from the promoted G256 warp-uniform-cursor
kernel (SHA ``62a97eb4``) and adds only the two B9 mechanisms it lacked:

* four warp ballots plus selected-lane rank-addressed scatter replace the
  serial lane-0 exact-index compaction; and
* the terminal epilogue uses the device-proven two-row register ownership to
  hoist row invariants and emit 32 aligned BF16x2 ``st.global.u32`` stores.

Everything else remains the G256 parent graph: two physical N128 halves per
logical group, TMEM256, six warps, two CTA/SM launch contract, one pre-exact
CTA join per G256 group, owner-local FP32 online softmax, mass-reused route
scores, chunked implicit-dual P-to-TMEM, and completion-backed O lifecycle.
There is no shape/density fast path, new barrier, wider physical MMA, routing
algorithm change, or production dispatch registration.

The ballot mask is warp-uniform.  Each exact lane writes one unique ascending
rank computed from preceding-word popcounts plus lower selected lanes.  The
existing full-CTA pre-exact join publishes those disjoint writes to warp 5;
lane 0 remains the sole packet/trace publisher.  The 1 KiB G256 index stream
stays entirely in SMEM.
"""

import math
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import flash_attn.cute.blackwell_helpers as mma_utils
import flash_attn.cute.pipeline as fa_pipeline
import flash_attn.cute.utils as fa_utils
from cutlass import BFloat16, Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned

from experiments.pisa2.probe_native_bf16_n256_deferred_leaf import (
    _load_m64_n128_score as _load_pair_score,
    _online_update_one_half as _online_update_pair,
    _rescale_m64_partial_o as _rescale_pair_o,
)

from kernels.pisa2_sm90._compat import layout_utils
from kernels.pisa2_sm90.selector import (
    pisa2_popc_b32,
    pisa2_route_is_exact,
)
from kernels.pisa2_sm100.native_tmem import (
    _add_physical_tmem_base,
    _zero_based_tmem_tensor,
    load_m64_o_fp32_256b,
    tcgen05_wait_st,
)


M = 64
N_MEMBER = 64
N_PACK_HALF = 128
D = 128
DV = 128
THREADS = 192
PAIR_STAGES = 1
TMEM_COLS = 256
PAIR_SCORE_OFFSET = 0
PAIR_P_OFFSET = 64
O_OFFSET = 128
PACK_QK_INST = (M, N_PACK_HALF, 16)
PACK_QK_TILE = (M, N_PACK_HALF, D)
PACK_PV_INST = (M, DV, 16)
PACK_PV_TILE = (M, DV, N_PACK_HALF)
PACK_QK_QUARTER_INST = (M, N_MEMBER, 16)
PACK_PV_QUARTER_INST = (M, 64, 16)
PACK_QK_GATHER_TILE = (M, N_MEMBER, 64)
PACK_PV_GATHER_TILE = (M, 64, 64)
SCALE = 1.0 / math.sqrt(D)
SCALE_LOG2 = SCALE * math.log2(math.e)
LOG2E = math.log2(math.e)
LN2 = math.log(2.0)
SEMANTIC_ROW_OFFSET = 16
LOGICAL_GROUP_SIZE = 256
ROUTE_TILE_SIZE = 128
ROUTE_HALVES_PER_GROUP = LOGICAL_GROUP_SIZE // ROUTE_TILE_SIZE
ROUTE_MASK_WORDS = 4
# masks[0:4], current-half exact count, append base, cumulative exact count,
# logical-terminal-half flag
PACKET_WORDS = 8
ROUTE_INDEX_CAPACITY = LOGICAL_GROUP_SIZE
PAIR_P_CHUNKS = 4
PAIR_P_CHUNK_PACKED_COLUMNS = (N_PACK_HALF // 2) // PAIR_P_CHUNKS
PAIR_P_PACKED_REGISTERS_PER_THREAD_PER_CHUNK = 8
O_PACKED_STORE_VALUES_PER_WORD = 2
O_PACKED_STORE_ALIGNMENT_BYTES = 4
O_PACKED_STORE_WRITER_THREADS = 4 * 32
O_ROWS_PER_OWNER_THREAD = 2
O_PACKED_WORDS_PER_ROW_PER_THREAD = 16
O_PACKED_COLUMN_STRIDE = 8

ROLE_MAP = {
    "warp0": "two_streamed_n128_route_pv_then_one_g256_exact_stream",
    "warp1": "ballot_masks_selected_lane_prefix_scatter_lane0_packet_publish",
    "warp2_4": "logical_g256_owner_rows_then_combined_exact_stream",
    "warp5": "independent_single_stage_n128_k_v_tma_producer",
}
MECHANISM_FAMILY = (
    "sm100-bf16-g256-b9-parity-ballotscatter-packed-u32-tworow-v1"
)
CANDIDATE_AXIS = (
    "backport_b9_ballot_scatter_and_two_row_packed_u32_only"
)
G256_PARENT_SOURCE_SHA256 = (
    "62a97eb4e08fce517f5c25c1fbaa49c7cbc62942b9cc229c600ff0e6749e12d2"
)
B9_DONOR_SOURCE_SHA256 = (
    "e0c36d40fb640eb4b44dea85ecd8595d0e6a221a7ded3816d64f52608ecf80bb"
)
B7_MAPPING_RECEIPT_SHA256 = (
    "ac19c0b157bb7abbe09403bea24e97efe65c6725102a2a01f305e864433a55fc"
)
CODEGEN_PARENT_SOURCE_SHA256 = (
    "62a97eb4e08fce517f5c25c1fbaa49c7cbc62942b9cc229c600ff0e6749e12d2"
)
AE9_SCALAR_ORACLE_SHA256 = (
    "ae9b1ccefdfdb410fd61906bcd49dba73a61981db2cb0c198cef92207f2a7475"
)
UNCOMPENSATED_V9_SOURCE_SHA256 = (
    "96dc406a55b5b1de34dea68e52afcacd71179ab338d79ed20b1a536c2489d0ad"
)
EXACT_PARENT_SOURCE_SHA256 = (
    "d6f88bff6f95efd7c6feafbd75b72088f793477ffb2c2a301bc7d2954c34ad07"
)
CHUNKP_PARENT_SOURCE_SHA256 = (
    "f44116182aa1cd36b52edda639c6d56cf6b6ad104d8e19aa7799ce0070194074"
)
RAW_G128_PARENT_SOURCE_SHA256 = (
    "7a4fa1f304b73a86651b701394000ef3d511cba346957a798c7a92346ca5f440"
)
ROUTE_PARENT_SOURCE_SHA256 = (
    "295bf9790b3dd86b356a2ce49c3869a2b66c229bc4d49c1de5f8bc2f28691610"
)
TERMINALO_PARENT_SOURCE_SHA256 = (
    "4a7933ebaf288148e0fd63cb69390733736c4fe8d0e3145a8deb82dfffcc5aa2"
)
PARENT_SOURCE_SHA256 = (
    "9d088e199b0e4866bc06a88c953b71490a82209eb9fd776968f936f43c0097ad"
)
EXACTP_GRANDPARENT_SOURCE_SHA256 = (
    "65e14fcfbc92396f846174e4570498259dd89f173efe15d6f72c88ad38b90b5f"
)
ROUTEP_GRANDPARENT_SOURCE_SHA256 = (
    "bc0771cac790ed4c0ccf89356b5890a124c91fee28998e8959452fb994a5cf4f"
)
GRANDPARENT_037_SOURCE_SHA256 = (
    "037e6ba686d40e84f0686dc98c46aa5c53e2bdec338aae882cd94a9a0883b221"
)

G256_CURSOR_BALLOTSCATTER_PACKEDGMEMO_TWOROWHOIST_INLINEU32_FUSEDROUTE_N128_TMEMP_RECEIPT = {
    "immediate_parent_source_sha256": G256_PARENT_SOURCE_SHA256,
    "b9_donor_source_sha256": B9_DONOR_SOURCE_SHA256,
    "codegen_parent_source_sha256": CODEGEN_PARENT_SOURCE_SHA256,
    "diagnostic_axis": "ballot_scatter_plus_two_row_packed_u32_only",
    "logical_schedule_changed": False,
    "phase_graph_changed": False,
    "math_changed": False,
    "shape_or_density_fast_path": False,
    "logical_group_size": LOGICAL_GROUP_SIZE,
    "physical_route_tile_size": ROUTE_TILE_SIZE,
    "route_halves_per_logical_group": ROUTE_HALVES_PER_GROUP,
    "route_transaction": "M64_N128_D128",
    "exact_pack": "append_two_half_masks_then_stream_one_logical_g256_list",
    "max_pairs_per_logical_g256": LOGICAL_GROUP_SIZE // 2,
    "max_pairs_in_one_physical_half": ROUTE_TILE_SIZE // 2,
    "odd_pair": "duplicate_only_at_logical_g256_tail_and_mask_upper64",
    "logical_terminal_isolated_exact1": "duplicate_member_and_mask_upper64_M64_N128",
    "route_mask_words": ROUTE_MASK_WORDS,
    "route_packet_words": PACKET_WORDS,
    "route_index_capacity": ROUTE_INDEX_CAPACITY,
    "carry_capacity": 0,
    "carry_scope": "none_combined_index_stream_pairs_across_half_boundary",
    "collect_both_score_halves_before_execute": False,
    "collect_both_index_halves_before_exact": True,
    "physical_n256_transaction": False,
    "physical_n512_transaction": False,
    "k_v_alias": False,
    "pair_k_stages": PAIR_STAGES,
    "pair_v_stages": PAIR_STAGES,
    "q_bytes": 16 * 1024,
    "pair_k_bytes": 32 * 1024,
    "pair_v_bytes": 32 * 1024,
    "pair_p_smem_bytes": 0,
    "smem_payload_bytes": 80 * 1024,
    "smem_target_bytes": 84 * 1024,
    "smem_control_modeled_bytes": 3992,
    "smem_total_modeled_bytes": 85912,
    "smem_raw_parent_device_bytes": 101632,
    "smem_candidate_device_projected_bytes": 85280,
    "smem_per_sm_bytes": 233472,
    "smem_two_cta_headroom_modeled_bytes": 61648,
    "tmem_columns": TMEM_COLS,
    "tmem_map": {
        "S": PAIR_SCORE_OFFSET,
        "P": PAIR_P_OFFSET,
        "O": O_OFFSET,
    },
    "pair_p_location": "chunked_bf16_tmem_columns_64_127",
    "score_release": "after_all_owner_tmem_loads_before_softmax",
    "o_wait": "deferred_until_next_iteration_rescale_then_final_drain",
    "route_schedule": "stream_half0_route_approx_then_half1_route_approx_then_combined_exact",
    "warp0_schedule": "exact_qk0_prologue_then_pv_i_before_qk_i_plus_1",
    "online_states": 1,
    "route_mass_scratch": "reuse_dead_route_scores_fragment",
    "route_mass_reduction_shape": "unchanged_parent_shape_and_fadd_tree",
    "route_pair_p_generations_per_group": 0,
    "exact_pair_p_generations": 0,
    "route_p_completion": "four_chunk_wait_st_then_tmem_fence_then_packet_barrier",
    "packet_handoff": "unchanged_named_barrier_5x32",
    "exact_p_handoff": "tcgen05_wait_st_then_tmem_fence_then_named_barrier_5x32",
    "exact_p_empty": "next_pair_score_completion_before_next_p_overwrite",
    "pair_o_completion_generations": "one_final_pv_generation_per_cta",
    "pair_o_empty_generations": 0,
    "intermediate_o_completion": "dominated_by_next_pair_score_completion",
    "o_reuse_handoff": "route_packet_or_exact_p_namedbarrier",
    "physical_tile_end_full_cta_joins": 0,
    "logical_group_end_full_cta_joins": 0,
    "pre_exact_full_cta_joins_per_logical_group": 1,
    "cross_half_handoff": "pack_k_pack_v_buffer_free_plus_pair_score_ready_plus_append_cursor",
    "cross_group_handoff": "pack_k_pack_v_buffer_free_plus_pair_score_ready",
    "route_index_happens_before": {
        "disjoint_writers": "each_selected_owner_warp0_lane_writes_one_unique_rank",
        "word_append_order": "preceding_word_popc_plus_selected_lower_lane_popc",
        "half_append_order": "lane0_cumulative_packet_word6_then_rank_addressed_scatter",
        "packet_writer": "owner_warp0_lane0_only_after_structured_loop_reconvergence",
        "indices_publish_to_warp5": "existing_full_cta_preexact_join_orders_all_selected_writer_lanes",
        "warp5_read_before_reuse": "final_exact_K_score_completion_requires_warp5_index_read_and_TMA_production",
        "owner_read_before_reuse": "final_exact_P_namedbarrier_includes_all_four_owner_warps",
        "empty_stream": "no_index_readers",
        "next_group_writer": "same_owner_writer_after_exact_loop",
    },
    "route_packet_reuse_happens_before": (
        "warp0_reads_half_h_packet_before_issuing_half_h_plus_1_QK; "
        "owner_writer_cannot_publish_half_h_plus_1_until_that_QK_completes"
    ),
    "pre_exact_full_cta_join_retained": True,
    "pre_exact_full_cta_join_scope": "one_per_logical_g256_not_one_per_physical_n128",
    "final_epilogue_full_cta_join_retained": True,
    "matched_parent_spill_sass": {
        "stl_pc": ("0xfffc30e03240", "0xfffc30e03280"),
        "ldl_pc": ("0xfffc30e04910", "0xfffc30e04930"),
        "dynamic_stl": 131072,
        "dynamic_ldl": 131072,
    },
    "target_min_blocks_per_mp": 2,
    "owner_peak_modeled_registers": 168,
    "two_cta_register_ceiling": 168,
    "register_budget_requires_device_compile_audit": True,
    "new_cross_warp_handoff": False,
    "new_named_or_cta_barrier": False,
    "terminal_o_transport": (
        "two_row_invariant_hoisted_same_thread_bf16x2_inline_ptx_st_global_u32"
    ),
    "terminal_o_values_per_store": O_PACKED_STORE_VALUES_PER_WORD,
    "terminal_o_store_alignment_bytes": O_PACKED_STORE_ALIGNMENT_BYTES,
    "terminal_o_writer_threads": O_PACKED_STORE_WRITER_THREADS,
    "terminal_o_extra_smem_bytes": 0,
    "terminal_o_tma_transactions": 0,
    "terminal_o_rows_per_owner_thread": O_ROWS_PER_OWNER_THREAD,
    "terminal_o_words_per_row_per_thread": O_PACKED_WORDS_PER_ROW_PER_THREAD,
    "terminal_o_register_strata": ("4*w,4*w+1", "4*w+2,4*w+3"),
    "terminal_o_even_column_formula": "2*(lane_mod_4)+8*w",
    "terminal_o_stats_loads_per_thread": 2,
    "terminal_o_reciprocals_per_thread": 2,
    "terminal_o_global_row_bases_per_thread": 2,
    "terminal_o_coordinate_parity_predicates": 0,
    "terminal_o_lane_exchange": False,
    "mask_generation": "four_vote_ballot_sync_no_shuffle_tree",
    "word_compaction": "all_selected_lanes_single_rank_addressed_scatter",
    "b7_mapping_receipt_sha256": B7_MAPPING_RECEIPT_SHA256,
    "uniform_schedule_all_shapes_and_densities": True,
    "production_dispatch_changed": False,
    "rejected_alias_parent_sha256": ROUTE_PARENT_SOURCE_SHA256,
    "exact_parent_source_sha256": EXACT_PARENT_SOURCE_SHA256,
    "chunkp_parent_source_sha256": CHUNKP_PARENT_SOURCE_SHA256,
    "raw_g128_parent_source_sha256": RAW_G128_PARENT_SOURCE_SHA256,
    "forbidden_phase_edges": (
        "k_to_v_alias_barrier",
        "v_to_k_alias_barrier",
        "arrive_and_wait_w_index",
        "per_group_full_cta_join",
    ),
    "ae9_scalar_oracle_sha256": AE9_SCALAR_ORACLE_SHA256,
    "uncompensated_v9_source_sha256": UNCOMPENSATED_V9_SOURCE_SHA256,
    "parent_candidate_source_sha256": PARENT_SOURCE_SHA256,
    "terminalo_parent_source_sha256": TERMINALO_PARENT_SOURCE_SHA256,
    "exactp_grandparent_source_sha256": EXACTP_GRANDPARENT_SOURCE_SHA256,
    "routep_grandparent_source_sha256": ROUTEP_GRANDPARENT_SOURCE_SHA256,
    "grandparent_037_source_sha256": GRANDPARENT_037_SOURCE_SHA256,
    "pair_p_store": "four_x8_st16x64b_chunked_r2t",
    "pair_p_store_width": 32,
    "pair_p_store_chunks": PAIR_P_CHUNKS,
    "pair_p_packed_registers_per_thread_per_chunk": (
        PAIR_P_PACKED_REGISTERS_PER_THREAD_PER_CHUNK
    ),
    "pair_p_store_coordinate_map": "f441_lane_xor2_prmt_to_st16x64b",
    "p_handoff_precision": "fp32_owner_fragment_until_r2t_then_bf16_tensor_operand",
    "p_to_pv_visibility": "tcgen05_wait_st_then_fence_view_async_tmem_store",
    "p_alias_order": "all_score_loads_then_p_store_then_pv_then_next_qk",
    "validation_status": "b9_parity_static_audit_device_compile_correctness_pending",
    "device_correctness_gate_eligible": True,
}


@dsl_user_op
def _route_fma_rn_f32(
    a: Float32,
    b: Float32,
    c: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _cvt_bf16x2_f32(
    hi: Float32,
    lo: Float32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Round two FP32 values and pack them as ``{lo, hi}`` BF16 bits."""

    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(hi).ir_value(loc=loc, ip=ip),
                Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _store_global_u32_inline(
    ptr: cute.Pointer,
    value: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store one aligned same-row BF16 pair as a single 32-bit word."""

    llvm.inline_asm(
        None,
        [
            ptr.toint().ir_value(),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _prmt_b32(
    a: Int32,
    b: Int32,
    sel: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Select four bytes from packed words ``a`` and ``b``."""

    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(a).ir_value(loc=loc, ip=ip),
                Int32(b).ir_value(loc=loc, ip=ip),
                Int32(sel).ir_value(loc=loc, ip=ip),
            ],
            "prmt.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _store_pair_probability_chunked_tmemp(
    o_template: cute.Tensor,
    probabilities: cute.Tensor,
    tmem_base: Int32,
    p_offset: Int32,
    owner_tidx: Int32,
):
    """Store M64xN128 BF16 P as four live-range-bounded x8 chunks.

    This is the f441 chunked R2T mapping.  Probabilities remain FP32 until
    each x8 fragment is converted, and every chunk waits for its St16x64b
    store before the fragment goes out of scope.
    """

    assert o_template.element_type == Float32
    assert cute.size(o_template) == M * DV
    p_chunk_layout = cute.composition(
        o_template.layout,
        cute.make_layout((M, PAIR_P_CHUNK_PACKED_COLUMNS)),
    )
    relative_chunk = _zero_based_tmem_tensor(Float32, p_chunk_layout)
    store_atom = cute.make_copy_atom(
        tcgen05.copy.St16x64bOp(tcgen05.copy.Repetition(8)),
        Float32,
    )
    tiled_store = tcgen05.make_tmem_copy(store_atom, relative_chunk)
    thread_store = tiled_store.get_slice(owner_tidx)
    destination_relative = thread_store.partition_D(relative_chunk)
    destination = _add_physical_tmem_base(
        destination_relative, tmem_base + p_offset
    )
    p_store_coordinates = thread_store.partition_S(
        cute.make_identity_tensor((M, PAIR_P_CHUNK_PACKED_COLUMNS))
    )
    lane = owner_tidx % Int32(32)

    for chunk_idx in cutlass.range_constexpr(PAIR_P_CHUNKS):
        p_store_registers = cute.make_fragment(
            p_store_coordinates.shape, Float32
        )
        assert (
            cute.size(p_store_registers)
            == PAIR_P_PACKED_REGISTERS_PER_THREAD_PER_CHUNK
        )
        assert (
            cute.size(probabilities)
            == 2 * cute.size(p_store_registers) * PAIR_P_CHUNKS
        )
        p_store_words = cute.make_tensor(
            cute.recast_ptr(p_store_registers.iterator, dtype=Int32),
            p_store_registers.layout,
        )
        probability_base = chunk_idx * (2 * cute.size(p_store_registers))
        for i in cutlass.range(
            cute.size(p_store_registers), unroll_full=True
        ):
            low = probability_base + i * 2
            high = low + 1
            own = _cvt_bf16x2_f32(
                Float32(probabilities[high]),
                Float32(probabilities[low]),
            )
            peer = cute.arch.shuffle_sync_bfly(own, offset=2)
            if (lane & Int32(2)) == Int32(0):
                p_store_words[i] = _prmt_b32(
                    own, peer, Int32(0x5410)
                )
            else:
                p_store_words[i] = _prmt_b32(
                    own, peer, Int32(0x3276)
                )

        destination_chunk = cute.make_tensor(
            destination.iterator
            + chunk_idx * PAIR_P_CHUNK_PACKED_COLUMNS,
            destination.layout,
        )
        cute.copy(tiled_store, p_store_registers, destination_chunk)
        tcgen05_wait_st()

    cute.arch.fence_view_async_tmem_store()


@cute.jit
def _load_pack_k_half(
    tma_atom_pack_k: cute.CopyAtom,
    tPackKgK: cute.Tensor,
    tPackKsK: cute.Tensor,
    block0: Int32,
    block1: Int32,
    quarter0: Int32,
    barrier,
):
    """Gather one canonical N128 K tile as K0/N0,K0/N1,K1/N0,K1/N1."""

    cute.copy(
        tma_atom_pack_k,
        tPackKgK[(None, block0, Int32(0))],
        tPackKsK[(None, quarter0)],
        tma_bar_ptr=barrier,
    )
    cute.copy(
        tma_atom_pack_k,
        tPackKgK[(None, block1, Int32(0))],
        tPackKsK[(None, quarter0 + Int32(1))],
        tma_bar_ptr=barrier,
    )
    cute.copy(
        tma_atom_pack_k,
        tPackKgK[(None, block0, Int32(1))],
        tPackKsK[(None, quarter0 + Int32(2))],
        tma_bar_ptr=barrier,
    )
    cute.copy(
        tma_atom_pack_k,
        tPackKgK[(None, block1, Int32(1))],
        tPackKsK[(None, quarter0 + Int32(3))],
        tma_bar_ptr=barrier,
    )


@cute.jit
def _load_pack_v_half(
    tma_atom_pack_v: cute.CopyAtom,
    tPackVgV: cute.Tensor,
    tPackVsV: cute.Tensor,
    block0: Int32,
    block1: Int32,
    quarter0: Int32,
    barrier,
):
    """Gather one canonical N128 V tile as D0/N0,D0/N1,D1/N0,D1/N1."""

    cute.copy(
        tma_atom_pack_v,
        tPackVgV[(None, Int32(0), block0)],
        tPackVsV[(None, quarter0)],
        tma_bar_ptr=barrier,
    )
    cute.copy(
        tma_atom_pack_v,
        tPackVgV[(None, Int32(0), block1)],
        tPackVsV[(None, quarter0 + Int32(1))],
        tma_bar_ptr=barrier,
    )
    cute.copy(
        tma_atom_pack_v,
        tPackVgV[(None, Int32(1), block0)],
        tPackVsV[(None, quarter0 + Int32(2))],
        tma_bar_ptr=barrier,
    )
    cute.copy(
        tma_atom_pack_v,
        tPackVgV[(None, Int32(1), block1)],
        tPackVsV[(None, quarter0 + Int32(3))],
        tma_bar_ptr=barrier,
    )


@cute.struct
class SharedStorage:
    q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    pack_k_mbar_ptr: cute.struct.MemRange[
        cutlass.Int64, PAIR_STAGES * 2
    ]
    pack_v_mbar_ptr: cute.struct.MemRange[
        cutlass.Int64, PAIR_STAGES * 2
    ]
    pair_score_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    pair_o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    final_stats: cute.struct.Align[
        cute.struct.MemRange[Float32, M * 2], 128
    ]
    route_partial: cute.struct.Align[
        cute.struct.MemRange[Float32, 4 * ROUTE_TILE_SIZE], 16
    ]
    route_packet: cute.struct.Align[
        cute.struct.MemRange[Int32, PACKET_WORDS], 16
    ]
    tmem_holding_buf: Int32
    # Owner-warp 0 lane 0 appends both N128 route masks.  The full-CTA
    # pre-exact join publishes the completed list to warp 5; no HBM indices.
    route_indices: cute.struct.Align[
        cute.struct.MemRange[Int32, ROUTE_INDEX_CAPACITY], 16
    ]


@cute.kernel
def lean6_bf16_routeidx_g256_cursor_ballotscatter_packedgmemo_tworowhoist_inlineu32_fusedroute_n128_pair_tmemp_grid_kernel(
    tiled_pack_qk: cute.TiledMma,
    tiled_pack_pv: cute.TiledMma,
    tma_atom_q: cute.CopyAtom,
    mQ_mkl: cute.Tensor,
    tma_atom_pack_k: cute.CopyAtom,
    mPackK_nkl: cute.Tensor,
    tma_atom_pack_v: cute.CopyAtom,
    mPackV_nkl: cute.Tensor,
    tma_atom_kc: cute.CopyAtom,
    mKC_nkl: cute.Tensor,
    tma_atom_vc: cute.CopyAtom,
    mVC_nkl: cute.Tensor,
    mThreshold_bhn: cute.Tensor,
    mO_bhtd: cute.Tensor,
    mLSE_bht: cute.Tensor,
    mPacketTrace: Optional[cute.Tensor],
    token_count: Int32,
    route_valid_total: Int32,
    num_route_tiles: Int32,
    q_layout: cute.ComposedLayout,
    pack_k_layout: cute.ComposedLayout,
    pack_k_gather_layout: cute.ComposedLayout,
    pack_p_layout: cute.ComposedLayout,
    pack_v_layout: cute.ComposedLayout,
    pack_v_gather_layout: cute.ComposedLayout,
    route_k_layout: cute.ComposedLayout,
    route_v_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    q_block_idx_raw, head_idx_raw, batch_idx_raw = cute.arch.block_idx()
    q_block_idx = Int32(q_block_idx_raw)
    head_idx = Int32(head_idx_raw)
    batch_idx = Int32(batch_idx_raw)

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sFinalStats = storage.final_stats.get_tensor(
        cute.make_layout((M, 2))
    )
    route_partial = storage.route_partial.get_tensor(
        cute.make_layout((4, ROUTE_TILE_SIZE))
    )
    route_packet = storage.route_packet.get_tensor(
        cute.make_layout((PACKET_WORDS,))
    )
    route_indices = storage.route_indices.get_tensor(
        cute.make_layout((ROUTE_INDEX_CAPACITY,))
    )
    sQ = smem.allocate_tensor(
        element_type=BFloat16,
        layout=q_layout.outer,
        byte_alignment=128,
        swizzle=q_layout.inner,
    )
    sPackK = smem.allocate_tensor(
        element_type=BFloat16,
        layout=pack_k_layout.outer,
        byte_alignment=128,
        swizzle=pack_k_layout.inner,
    )
    sPackV = smem.allocate_tensor(
        element_type=BFloat16,
        layout=pack_v_layout.outer,
        byte_alignment=128,
        swizzle=pack_v_layout.inner,
    )
    # One independent physical N128 K stage and one N128 V stage.  Every
    # runtime route/exact transaction stays in this completion domain.
    sPackKGather = cute.make_tensor(
        cute.recast_ptr(
            sPackK.iterator, pack_k_gather_layout.inner, BFloat16
        ),
        pack_k_gather_layout.outer,
    )
    sPackVGather = cute.make_tensor(
        cute.recast_ptr(
            sPackV.iterator, pack_v_gather_layout.inner, BFloat16
        ),
        pack_v_gather_layout.outer,
    )
    # KC/VC and exact K/V have disjoint lifetimes within each runtime group.
    # They reuse the same independent N128 K and V allocations without a
    # cross-operand alias barrier.
    sKC = cute.make_tensor(
        cute.recast_ptr(sPackK.iterator, route_k_layout.inner, BFloat16),
        route_k_layout.outer,
    )
    sVC = cute.make_tensor(
        cute.recast_ptr(sPackV.iterator, route_v_layout.inner, BFloat16),
        route_v_layout.outer,
    )

    tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    score_loaded_barrier = pipeline.NamedBarrier(
        barrier_id=2, num_threads=4 * 32
    )
    final_stats_ready_barrier = pipeline.NamedBarrier(
        barrier_id=3, num_threads=4 * 32
    )
    pack_score_loaded_barrier = pipeline.NamedBarrier(
        barrier_id=4, num_threads=4 * 32
    )
    route_packet_ready_barrier = pipeline.NamedBarrier(
        barrier_id=5, num_threads=5 * 32
    )
    exact_pair_p_ready_barrier = pipeline.NamedBarrier(
        barrier_id=6, num_threads=5 * 32
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_barrier,
    )
    tmem.allocate(TMEM_COLS)

    one_thread = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
    pack_owner_threads = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, 4 * 32
    )
    q_bytes = cute.size_in_bytes(
        BFloat16, cute.select(q_layout, mode=[0, 1, 2])
    )
    route_k_bytes = cute.size_in_bytes(
        BFloat16, cute.select(route_k_layout, mode=[0, 1, 2])
    )
    route_v_bytes = cute.size_in_bytes(
        BFloat16, cute.select(route_v_layout, mode=[0, 1, 2])
    )
    pack_k_bytes = cute.size_in_bytes(
        BFloat16, cute.select(pack_k_layout, mode=[0, 1, 2])
    )
    pack_v_bytes = cute.size_in_bytes(
        BFloat16, cute.select(pack_v_layout, mode=[0, 1, 2])
    )
    assert route_k_bytes == pack_k_bytes
    assert route_v_bytes == pack_v_bytes
    q_pipe = fa_pipeline.PipelineTmaUmma.create(
        num_stages=1,
        producer_group=one_thread,
        consumer_group=one_thread,
        tx_count=q_bytes,
        barrier_storage=storage.q_mbar_ptr.data_ptr(),
    )
    pack_k_pipe = fa_pipeline.PipelineTmaUmma.create(
        num_stages=PAIR_STAGES,
        producer_group=one_thread,
        consumer_group=one_thread,
        tx_count=pack_k_bytes,
        barrier_storage=storage.pack_k_mbar_ptr.data_ptr(),
    )
    pack_v_pipe = fa_pipeline.PipelineTmaUmma.create(
        num_stages=PAIR_STAGES,
        producer_group=one_thread,
        consumer_group=one_thread,
        tx_count=pack_v_bytes,
        barrier_storage=storage.pack_v_mbar_ptr.data_ptr(),
    )
    pair_score_pipe = fa_pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=one_thread,
        consumer_group=pack_owner_threads,
        barrier_storage=storage.pair_score_mbar_ptr.data_ptr(),
    )
    pair_o_pipe = fa_pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=one_thread,
        consumer_group=pack_owner_threads,
        barrier_storage=storage.pair_o_mbar_ptr.data_ptr(),
    )

    mQ_cur = mQ_mkl[None, None, head_idx, batch_idx]
    mPackK_cur = mPackK_nkl[None, None, head_idx, batch_idx]
    mPackV_cur = mPackV_nkl[None, None, head_idx, batch_idx]
    mKC_cur = mKC_nkl[None, None, head_idx, batch_idx]
    mVC_cur = mVC_nkl[None, None, head_idx, batch_idx]
    gQ = cute.local_tile(mQ_cur, (M, D), (None, 0))
    gPackK = cute.local_tile(
        mPackK_cur, (N_MEMBER, 64), (None, None)
    )
    gPackV = cute.local_tile(
        mPackV_cur, (64, N_MEMBER), (None, None)
    )
    gKC = cute.local_tile(mKC_cur, (N_PACK_HALF, D), (None, 0))
    gVC = cute.local_tile(mVC_cur, (DV, N_PACK_HALF), (0, None))
    thr_pack_qk = tiled_pack_qk.get_slice(0)
    thr_pack_pv = tiled_pack_pv.get_slice(0)
    tCgQ = thr_pack_qk.partition_A(gQ)
    tCgKC = thr_pack_qk.partition_B(gKC)
    tCgVC = thr_pack_pv.partition_B(gVC)
    tCrKC = tiled_pack_qk.make_fragment_B(sKC)
    tCrVC = tiled_pack_pv.make_fragment_B(sVC)
    tCrPackQ = tiled_pack_qk.make_fragment_A(sQ)
    tCrPackK = tiled_pack_qk.make_fragment_B(sPackK)
    tCrPackV = tiled_pack_pv.make_fragment_B(sPackV)

    tQsQ, tQgQ = cpasync.tma_partition(
        tma_atom_q,
        0,
        cute.make_layout(1),
        cute.group_modes(sQ, 0, 3),
        cute.group_modes(tCgQ, 0, 3),
    )
    tPackKsK, tPackKgK = cpasync.tma_partition(
        tma_atom_pack_k,
        0,
        cute.make_layout(1),
        cute.group_modes(sPackKGather, 0, 3),
        cute.group_modes(gPackK, 0, 2),
    )
    tPackVsV, tPackVgV = cpasync.tma_partition(
        tma_atom_pack_v,
        0,
        cute.make_layout(1),
        cute.group_modes(sPackVGather, 0, 3),
        cute.group_modes(gPackV, 0, 2),
    )
    tKCsKC, tKCgKC = cpasync.tma_partition(
        tma_atom_kc,
        0,
        cute.make_layout(1),
        cute.group_modes(sKC, 0, 3),
        cute.group_modes(tCgKC, 0, 3),
    )
    tVCsVC, tVCgVC = cpasync.tma_partition(
        tma_atom_vc,
        0,
        cute.make_layout(1),
        cute.group_modes(sVC, 0, 3),
        cute.group_modes(tCgVC, 0, 3),
    )

    pack_score_shape = tiled_pack_qk.partition_shape_C(
        PACK_QK_TILE[:2]
    )
    pack_score_template = tiled_pack_qk.make_fragment_C(pack_score_shape)
    pack_o_shape = tiled_pack_pv.partition_shape_C(PACK_PV_TILE[:2])
    pack_o_template = tiled_pack_pv.make_fragment_C(pack_o_shape)

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(Float32)
    # The 256-column allocation leaves the second half of SM TMEM available to
    # another CTA.  The live allocation remains owned after permit release.
    tmem.relinquish_alloc_permit()
    tmem_base = tmem_ptr.toint()
    pair_tScore = cute.make_tensor(
        cute.make_ptr(
            Float32,
            tmem_base + Int32(PAIR_SCORE_OFFSET),
            cute.AddressSpace.tmem,
            assumed_align=16,
        ),
        pack_score_template.layout,
    )
    pair_tO = cute.make_tensor(
        cute.make_ptr(
            Float32,
            tmem_base + Int32(O_OFFSET),
            cute.AddressSpace.tmem,
            assumed_align=16,
        ),
        pack_o_template.layout,
    )
    # make_fragment_A drops the physical TMEM allocation base and addresses
    # packed BF16 columns in half-column units.  Restore both facts exactly as
    # in f441: 2*tmem_base + 2*PAIR_P_OFFSET names columns 64..127.
    pair_tP_storage = cute.make_tensor(
        pair_tScore.iterator, pack_p_layout.outer
    )
    pair_tP_base = tiled_pack_pv.make_fragment_A(pair_tP_storage)[
        None, None, None, 0
    ]
    pair_tP = cute.make_tensor(
        pair_tP_base.iterator
        + tmem_base
        + tmem_base
        + Int32(PAIR_P_OFFSET * 2),
        pair_tP_base.layout,
    )
    q_producer = fa_pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, 1
    )
    q_consumer = fa_pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, 1
    )
    pack_k_producer = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, PAIR_STAGES
    )
    pack_k_consumer = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, PAIR_STAGES
    )
    pack_v_producer = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, PAIR_STAGES
    )
    pack_v_consumer = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, PAIR_STAGES
    )
    pair_score_producer = fa_pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, 1
    )
    pair_score_consumer = fa_pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, 1
    )
    pair_o_producer = fa_pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, 1
    )
    pair_o_consumer = fa_pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, 1
    )
    route_start_base = Int32(0)
    q_len = token_count - q_block_idx * Int32(M)
    if q_len > Int32(M):
        q_len = Int32(M)
    threshold = Float32(
        mThreshold_bhn[batch_idx, head_idx, q_block_idx]
    )

    if warp_idx == Int32(5):
        cpasync.prefetch_descriptor(tma_atom_q)
        cpasync.prefetch_descriptor(tma_atom_pack_k)
        cpasync.prefetch_descriptor(tma_atom_pack_v)
        cpasync.prefetch_descriptor(tma_atom_kc)
        cpasync.prefetch_descriptor(tma_atom_vc)

        q_pipe.producer_acquire(q_producer)
        q_barrier = q_pipe.producer_get_barrier(q_producer)
        cute.copy(
            tma_atom_q,
            tQgQ[(None, q_block_idx)],
            tQsQ[(None, q_producer.index)],
            tma_bar_ptr=q_barrier,
        )
        q_producer.advance()

    is_owner = warp_idx >= Int32(1) and warp_idx <= Int32(4)
    is_score_consumer = warp_idx <= Int32(4)
    owner_tidx = tidx - Int32(32)

    # One register-resident online state and one TMEM-O initialization bit span
    # every route/exact transaction in every runtime group.
    running_max = -Float32.inf
    running_sum = Float32(0.0)
    owner_o_initialized = Int32(0)
    mma_o_initialized = Int32(0)

    if warp_idx == Int32(0):
        q_pipe.consumer_wait(q_consumer)

    # The outer loop owns one logical G256 exact-index lifetime.  The inner
    # loop consumes each physical score/PV half immediately; it appends only
    # integer indices, never a second score or probability fragment.
    num_logical_groups = (
        num_route_tiles + Int32(ROUTE_HALVES_PER_GROUP - 1)
    ) // Int32(ROUTE_HALVES_PER_GROUP)
    # BEGIN_G256_CURSOR_UNIFORM_INDUCTION
    # arch_make_warp_uniform is a lowering hint, not a value broadcast. Both
    # values are CTA-invariant integer scalars before the hint.
    logical_group_idx = cute.arch.make_warp_uniform(Int32(0))
    remaining_group_tiles = cute.arch.make_warp_uniform(num_route_tiles)
    while logical_group_idx < num_logical_groups:
        is_final_logical_group = (
            logical_group_idx + Int32(1) == num_logical_groups
        )
        group_route_tile_base = logical_group_idx * Int32(
            ROUTE_HALVES_PER_GROUP
        )
        physical_halves_this_group = remaining_group_tiles
        if physical_halves_this_group > Int32(ROUTE_HALVES_PER_GROUP):
            physical_halves_this_group = Int32(ROUTE_HALVES_PER_GROUP)

        for half_idx in cutlass.range(
            physical_halves_this_group, unroll=1
        ):
            route_tile_idx = group_route_tile_base + half_idx
            is_final_route_tile = (
                route_tile_idx + Int32(1) == num_route_tiles
            )
            is_logical_terminal_half = (
                half_idx + Int32(1) == physical_halves_this_group
            )
            route_start = (
                route_start_base
                + route_tile_idx * Int32(ROUTE_TILE_SIZE)
            )
            remaining_route_count = (
                route_valid_total
                - route_tile_idx * Int32(ROUTE_TILE_SIZE)
            )
            valid_route_count = remaining_route_count
            if valid_route_count > Int32(ROUTE_TILE_SIZE):
                valid_route_count = Int32(ROUTE_TILE_SIZE)
            if valid_route_count < Int32(0):
                valid_route_count = Int32(0)

            # One native N128 route transaction shares the independent K/V stages
            # with the exact-pair engine.  Route and exact are separated by
            # a full-CTA phase boundary, so no K<->V alias handoff is required.
            if warp_idx == Int32(5):
                pack_k_pipe.producer_acquire(pack_k_producer)
                route_k_barrier = pack_k_pipe.producer_get_barrier(
                    pack_k_producer
                )
                cute.copy(
                    tma_atom_kc,
                    tKCgKC[(None, route_tile_idx)],
                    tKCsKC[(None, pack_k_producer.index)],
                    tma_bar_ptr=route_k_barrier,
                )
                pack_k_producer.advance()

                pack_v_pipe.producer_acquire(pack_v_producer)
                route_v_barrier = pack_v_pipe.producer_get_barrier(
                    pack_v_producer
                )
                cute.copy(
                    tma_atom_vc,
                    tVCgVC[(None, route_tile_idx)],
                    tVCsVC[(None, pack_v_producer.index)],
                    tma_bar_ptr=route_v_barrier,
                )
                pack_v_producer.advance()

            if warp_idx == Int32(0):
                pack_k_pipe.consumer_wait(pack_k_consumer)
                pair_score_pipe.producer_acquire(pair_score_producer)
                mma_utils.gemm(
                    tiled_pack_qk,
                    pair_tScore,
                    tCrPackQ[None, None, None, q_consumer.index],
                    tCrKC[None, None, None, pack_k_consumer.index],
                    zero_init=True,
                )
                pair_score_pipe.producer_commit(pair_score_producer)
                pair_score_producer.advance()
                pack_k_pipe.consumer_release(pack_k_consumer)
                pack_k_consumer.advance()

            # BEGIN_RUNTIME_GROUP_BODY
        
            # Route generation: four physical owner warps reduce the native N128
            # score tile into one four-word mask.  HBM receives only the diagnostic
            # copy; the compacted exact stream remains resident in SMEM.
            if is_owner:
                pair_score_pipe.consumer_wait(pair_score_consumer)
                score_raw, score_coords = _load_pair_score(
                    pack_score_template,
                    thr_pack_qk,
                    tmem_base,
                    Int32(PAIR_SCORE_OFFSET),
                    owner_tidx,
                )
                pack_score_loaded_barrier.arrive_and_wait()
                pair_score_pipe.consumer_release(pair_score_consumer)
                pair_score_consumer.advance()
                owner_warp = owner_tidx // Int32(32)
                lane = owner_tidx % Int32(32)
                semantic_row = (
                    score_coords[0][0] + Int32(SEMANTIC_ROW_OFFSET)
                ) & Int32(M - 1)
                row_valid = semantic_row < q_len
                lane_col_parity = (lane // Int32(2)) % Int32(2)
                # Column-pair reduction: parity-0 lanes carry column 2*pair and
                # parity-1 lanes carry column 2*pair+1.  The XOR-1/16/8/4
                # butterfly tree never crosses lane column-parity classes
                # ((l^k)//2 keeps (l//2)%2 for k in {1,16,8,4}), so one tree
                # reduces both columns at once; every surviving addition chain
                # sees operand streams identical to the parent's zero-padded
                # passes, and the removed chains only ever accumulated 0.0.
                # Writer lanes 0 and 2 equal the parent's 2*(col%2).
                for pair_idx in cutlass.range_constexpr(ROUTE_TILE_SIZE // 2):
                    my_col = Int32(2 * pair_idx) + lane_col_parity
                    partial = Float32(0.0)
                    if row_valid and my_col < valid_route_count:
                        partial = Float32(score_raw[pair_idx])
                    raw_partial = partial
                    peer_scaled = cute.arch.shuffle_sync_bfly(
                        raw_partial * Float32(SCALE_LOG2), offset=1
                    )
                    partial = _route_fma_rn_f32(
                        raw_partial, Float32(SCALE_LOG2), peer_scaled
                    )
                    partial = partial + cute.arch.shuffle_sync_bfly(
                        partial, offset=16
                    )
                    partial = partial + cute.arch.shuffle_sync_bfly(
                        partial, offset=8
                    )
                    partial = partial + cute.arch.shuffle_sync_bfly(
                        partial, offset=4
                    )
                    if lane == Int32(0):
                        route_partial[owner_warp, 2 * pair_idx] = partial
                    if lane == Int32(2):
                        route_partial[owner_warp, 2 * pair_idx + 1] = partial
        
                cute.arch.fence_view_async_shared()
                score_loaded_barrier.arrive_and_wait()
                if owner_warp == Int32(0):
                    mask0 = Int32(0)
                    mask1 = Int32(0)
                    mask2 = Int32(0)
                    mask3 = Int32(0)

                    # Half 0 starts a fresh G256 stream and half 1 appends to
                    # lane 0's cumulative packet word.  The preceding packet
                    # barrier makes the base warp-uniform before the vote.
                    append_base = Int32(0)
                    if half_idx != Int32(0):
                        append_base = Int32(route_packet[6])

                    # A positive signed shift avoids materializing 1<<31:
                    # lane 0 gets zero and lane 31 gets 0x7fffffff.
                    lane_mask_lt = Int32(0x7FFFFFFF) >> (
                        Int32(31) - lane
                    )
                    preceding_word_count = Int32(0)
                    for word in cutlass.range_constexpr(ROUTE_MASK_WORDS):
                        off = Int32(word * 32) + lane
                        valid = off < valid_route_count
                        exact_pred = False
                        if valid:
                            pair_02 = Float32(route_partial[0, off]) + Float32(
                                route_partial[2, off]
                            )
                            pair_13 = Float32(route_partial[1, off]) + Float32(
                                route_partial[3, off]
                            )
                            col_mean = (pair_02 + pair_13) / Float32(q_len)
                            exact_pred = pisa2_route_is_exact(
                                q_block_idx,
                                route_start + off,
                                col_mean,
                                threshold,
                                valid,
                            )
                        word_mask = Int32(
                            cute.arch.vote_ballot_sync(exact_pred)
                        )
                        lane_rank = (
                            append_base
                            + preceding_word_count
                            + pisa2_popc_b32(word_mask & lane_mask_lt)
                        )
                        if exact_pred:
                            route_indices[lane_rank] = route_start + off
                        if cutlass.const_expr(word == 0):
                            mask0 = word_mask
                        elif cutlass.const_expr(word == 1):
                            mask1 = word_mask
                        elif cutlass.const_expr(word == 2):
                            mask2 = word_mask
                        else:
                            mask3 = word_mask
                        preceding_word_count = (
                            preceding_word_count
                            + pisa2_popc_b32(word_mask)
                        )

                    # Every selected lane has a unique rank; lane 0 remains
                    # the sole packet/trace publisher after reconvergence.
                    exact_count = preceding_word_count
                    if lane == Int32(0):
                        route_rank = append_base + exact_count

                        route_packet[0] = mask0
                        route_packet[1] = mask1
                        route_packet[2] = mask2
                        route_packet[3] = mask3
                        route_packet[4] = exact_count
                        route_packet[5] = append_base
                        route_packet[6] = route_rank
                        terminal_half_word = Int32(0)
                        if is_logical_terminal_half:
                            terminal_half_word = Int32(1)
                        route_packet[7] = terminal_half_word
                        if cutlass.const_expr(mPacketTrace is not None):
                            mPacketTrace[
                                batch_idx,
                                head_idx,
                                q_block_idx,
                                logical_group_idx,
                                half_idx,
                                Int32(0),
                            ] = mask0
                            mPacketTrace[
                                batch_idx,
                                head_idx,
                                q_block_idx,
                                logical_group_idx,
                                half_idx,
                                Int32(1),
                            ] = mask1
                            mPacketTrace[
                                batch_idx,
                                head_idx,
                                q_block_idx,
                                logical_group_idx,
                                half_idx,
                                Int32(2),
                            ] = mask2
                            mPacketTrace[
                                batch_idx,
                                head_idx,
                                q_block_idx,
                                logical_group_idx,
                                half_idx,
                                Int32(3),
                            ] = mask3
                            mPacketTrace[
                                batch_idx,
                                head_idx,
                                q_block_idx,
                                logical_group_idx,
                                half_idx,
                                Int32(4),
                            ] = exact_count
                            mPacketTrace[
                                batch_idx,
                                head_idx,
                                q_block_idx,
                                logical_group_idx,
                                half_idx,
                                Int32(5),
                            ] = append_base
                            mPacketTrace[
                                batch_idx,
                                head_idx,
                                q_block_idx,
                                logical_group_idx,
                                half_idx,
                                Int32(6),
                            ] = route_rank
                            mPacketTrace[
                                batch_idx,
                                head_idx,
                                q_block_idx,
                                logical_group_idx,
                                half_idx,
                                Int32(7),
                            ] = terminal_half_word
                        cute.arch.fence_view_async_shared()
        
                # The selector packet is now immutable.  Reuse the already resident
                # route scores for the non-exact transaction; no offset list or second
                # route-score load is introduced.
                score_loaded_barrier.arrive_and_wait()
                route_exact_count = Int32(route_packet[4])
                has_route_approx = route_exact_count < valid_route_count
                if has_route_approx:
                    packet_mask0 = Int32(route_packet[0])
                    packet_mask1 = Int32(route_packet[1])
                    packet_mask2 = Int32(route_packet[2])
                    packet_mask3 = Int32(route_packet[3])
                    route_scores = cute.make_fragment(score_raw.shape, Float32)
                    for i in cutlass.range(
                        cute.size(score_raw), unroll_full=True
                    ):
                        group_col = score_coords[i][1]
                        word_bits = packet_mask0
                        if group_col >= Int32(32):
                            word_bits = packet_mask1
                        if group_col >= Int32(64):
                            word_bits = packet_mask2
                        if group_col >= Int32(96):
                            word_bits = packet_mask3
                        exact = (
                            word_bits
                            & (Int32(1) << (group_col & Int32(31)))
                        ) != Int32(0)
                        route_score = -Float32.inf
                        if (
                            row_valid
                            and group_col < valid_route_count
                            and not exact
                        ):
                            route_score = Float32(score_raw[i])
                        route_scores[i] = route_score
        
                    local_max = fa_utils.fmax_reduce(
                        route_scores.load(), arch=100
                    )
                    local_max = Float32(local_max) * Float32(SCALE)
                    peer_max = cute.arch.shuffle_sync_bfly(local_max, offset=2)
                    pair_max = local_max
                    if peer_max > pair_max:
                        pair_max = peer_max
        
                    old_max = running_max
                    old_sum = running_sum
                    new_max = old_max
                    if old_max == -Float32.inf or pair_max > old_max:
                        new_max = pair_max
                    row_alpha = Float32(0.0)
                    if old_max != -Float32.inf:
                        row_alpha = cute.math.exp2(
                            (old_max - new_max) * Float32(LOG2E),
                            fastmath=True,
                        )
        
                    route_probabilities = cute.make_fragment(
                        route_scores.shape, Float32
                    )
                    if new_max == -Float32.inf:
                        for i in cutlass.range(
                            cute.size(route_scores), unroll_full=True
                        ):
                            route_probabilities[i] = Float32(0.0)
                    else:
                        for i in cutlass.range(
                            cute.size(route_scores), unroll_full=True
                        ):
                            route_probabilities[i] = cute.math.exp2(
                                Float32(route_scores[i]) * Float32(SCALE_LOG2)
                                - new_max * Float32(LOG2E),
                                fastmath=True,
                            )
                    # ``route_scores`` is dead after the exponentials above.  Use
                    # it as mass scratch so the compiler does not need a second
                    # full N128-shaped fragment while probabilities remain live
                    # for the chunked TMEM-P store below.  Keeping the same shape,
                    # index order, and fadd_reduce call preserves the parent's
                    # floating-point reduction order and every phase edge.
                    for i in cutlass.range(
                        cute.size(route_probabilities), unroll_full=True
                    ):
                        block_idx = route_start + score_coords[i][1]
                        block_length = token_count - block_idx * Int32(N_MEMBER)
                        if block_length > Int32(N_MEMBER):
                            block_length = Int32(N_MEMBER)
                        if block_length < Int32(0):
                            block_length = Int32(0)
                        route_scores[i] = (
                            route_probabilities[i] * Float32(block_length)
                        )
                    current_sum = fa_utils.fadd_reduce(
                        route_scores.load(), arch=100
                    )
                    current_sum += cute.arch.shuffle_sync_bfly(
                        current_sum, offset=2
                    )
                    # KC is a block mean and VC a valid-token sum.  Route mass uses
                    # the true block length while PV still consumes p*VC once.
                    running_sum = old_sum * row_alpha + current_sum
                    running_max = new_max
                    if owner_o_initialized != Int32(0):
                        _rescale_pair_o(
                            pack_o_template,
                            thr_pack_pv,
                            tmem_base,
                            Int32(O_OFFSET),
                            owner_tidx,
                            row_alpha,
                        )
                    _store_pair_probability_chunked_tmemp(
                        pack_o_template,
                        route_probabilities,
                        tmem_base,
                        Int32(PAIR_P_OFFSET),
                        owner_tidx,
                    )
                    owner_o_initialized = Int32(1)
            # Publish the mask/P decision to warp 0.  The route PV is deliberately
            # drained before exact work: this keeps the first semantic candidate
            # free of the rejected K/V alias handoff and makes all-exact,
            # all-approx, odd, and partial-tail paths share one phase boundary.
            if is_score_consumer:
                route_packet_ready_barrier.arrive_and_wait()
            if warp_idx == Int32(0):
                route_exact_count = Int32(route_packet[4])
                route_has_approx = route_exact_count < valid_route_count
                pack_v_pipe.consumer_wait(pack_v_consumer)
                if route_has_approx:
                    mma_utils.gemm(
                        tiled_pack_pv,
                        pair_tO,
                        pair_tP,
                        tCrVC[None, None, None, pack_v_consumer.index],
                        zero_init=mma_o_initialized == Int32(0),
                    )
                    # Half 0 is followed by half-1 route QK.  The terminal
                    # route half is followed by exact QK0 whenever the fused
                    # G256 index stream is nonempty.  Those score completions
                    # prove this PV complete; only a final route-only CTA needs
                    # an explicit O completion here.
                    if (
                        is_final_route_tile
                        and Int32(route_packet[6]) == Int32(0)
                    ):
                        pair_o_pipe.producer_commit(pair_o_producer)
                    mma_o_initialized = Int32(1)
                pack_v_pipe.consumer_release(pack_v_consumer)
                pack_v_consumer.advance()
            if is_owner:
                cumulative_exact_count = Int32(route_packet[6])
                if (
                    is_final_route_tile
                    and cumulative_exact_count == Int32(0)
                ):
                    pair_o_pipe.consumer_wait(pair_o_consumer)

            # route_packet may be reused by the next physical half without a
            # CTA join.  Warp 0 reads this half's packet before it can issue
            # next-half QK; owner-warp 0 cannot overwrite the packet until
            # that QK's pair-score completion has released all owners.
        
        # Both route halves have published their packet/index data and drained
        # approximate PV.  This is the only CTA-wide pre-exact join in the
        # logical G256 group; it publishes the combined list to warp 5.
        cute.arch.barrier()
        # The cumulative count covers half 0 followed by half 1.  Pairing this
        # one ordered stream removes cross-half odd padding without retaining
        # either physical score fragment.
        exact_block_count = Int32(route_packet[6])
        exact_pair_count = (exact_block_count + Int32(1)) // Int32(2)
        pair_count = exact_pair_count
        has_pair_exact = exact_block_count > Int32(0)

        # BEGIN_GENERAL_N128_PAIR
        # Every executable exact count, including a logical-group terminal
        # exact1, stays in the N128 domain.

        # Warp 5 streams one physical N128 K stage and one physical N128 V
        # stage.  A missing odd peer duplicates block0 only for the physical
        # transaction; owners mask all upper-64 scores before softmax.
        if warp_idx == Int32(5) and has_pair_exact:
            for pair_idx in cutlass.range(pair_count, unroll=1):
                ordinal0 = pair_idx * Int32(2)
                block0 = Int32(route_indices[ordinal0])
                block1 = block0
                if ordinal0 + Int32(1) < exact_block_count:
                    block1 = Int32(route_indices[ordinal0 + Int32(1)])

                pack_k_pipe.producer_acquire(pack_k_producer)
                pair_k_barrier = pack_k_pipe.producer_get_barrier(
                    pack_k_producer
                )
                _load_pack_k_half(
                    tma_atom_pack_k,
                    tPackKgK,
                    tPackKsK,
                    block0,
                    block1,
                    pack_k_producer.index * Int32(4),
                    pair_k_barrier,
                )
                pack_k_producer.advance()

                pack_v_pipe.producer_acquire(pack_v_producer)
                pair_v_barrier = pack_v_pipe.producer_get_barrier(
                    pack_v_producer
                )
                _load_pack_v_half(
                    tma_atom_pack_v,
                    tPackVgV,
                    tPackVsV,
                    block0,
                    block1,
                    pack_v_producer.index * Int32(4),
                    pair_v_barrier,
                )
                pack_v_producer.advance()

        if warp_idx == Int32(0) and has_pair_exact:
            # QK0 prologue.  K and score cursors advance exactly once per QK;
            # neither V nor O state is touched until the steady-state PV path.
            pack_k_pipe.consumer_wait(pack_k_consumer)
            pair_score_pipe.producer_acquire(pair_score_producer)
            mma_utils.gemm(
                tiled_pack_qk,
                pair_tScore,
                tCrPackQ[None, None, None, q_consumer.index],
                tCrPackK[None, None, None, pack_k_consumer.index],
                zero_init=True,
            )
            pair_score_pipe.producer_commit(pair_score_producer)
            pair_score_producer.advance()
            # PipelineTmaUmma release is tcgen05-completion-backed.
            pack_k_pipe.consumer_release(pack_k_consumer)
            pack_k_consumer.advance()

            for pair_idx in cutlass.range(pair_count, unroll=1):
                # P aliases the drained upper half of S.  PV must therefore be
                # issued before QK(i+1) overwrites S.  Both instructions are
                # emitted back-to-back by warp 0, retaining the full-G128
                # tcgen05 dependency order without its K/V alias barriers.
                pack_v_pipe.consumer_wait(pack_v_consumer)
                # All four owners have completed their synchronous chunked
                # TMEM stores and the helper's TMEM store fence before this
                # five-warp rendezvous releases the single MMA warp.
                exact_pair_p_ready_barrier.arrive_and_wait()
                mma_utils.gemm(
                    tiled_pack_pv,
                    pair_tO,
                    pair_tP,
                    tCrPackV[None, None, None, pack_v_consumer.index],
                    zero_init=mma_o_initialized == Int32(0),
                )
                # QK(i+1) completion dominates PV(i) completion for every
                # nonterminal transaction on this tcgen05 issuer.  Commit one
                # explicit O-full generation only for the CTA's final PV.
                if (
                    is_final_logical_group
                    and pair_idx + Int32(1) == pair_count
                ):
                    pair_o_pipe.producer_commit(pair_o_producer)
                mma_o_initialized = Int32(1)
                pack_v_pipe.consumer_release(pack_v_consumer)
                pack_v_consumer.advance()

                if pair_idx + Int32(1) < pair_count:
                    pack_k_pipe.consumer_wait(pack_k_consumer)
                    pair_score_pipe.producer_acquire(pair_score_producer)
                    mma_utils.gemm(
                        tiled_pack_qk,
                        pair_tScore,
                        tCrPackQ[None, None, None, q_consumer.index],
                        tCrPackK[
                            None, None, None, pack_k_consumer.index
                        ],
                        zero_init=True,
                    )
                    pair_score_pipe.producer_commit(pair_score_producer)
                    pair_score_producer.advance()
                    pack_k_pipe.consumer_release(pack_k_consumer)
                    pack_k_consumer.advance()

        if is_owner and has_pair_exact:
            for pair_idx in cutlass.range(pair_count, unroll=1):
                pair_score_pipe.consumer_wait(pair_score_consumer)
                # Keep the exact ae9 score-load helper and fragment scope so
                # this diagnostic changes only the P publication ownership.
                pair_scores, pair_coords = _load_pair_score(
                    pack_score_template,
                    thr_pack_qk,
                    tmem_base,
                    Int32(PAIR_SCORE_OFFSET),
                    owner_tidx,
                )
                # Every owner retires the complete score load before the
                # packed P store aliases columns 64..127 of S.
                pack_score_loaded_barrier.arrive_and_wait()
                pair_score_pipe.consumer_release(pair_score_consumer)
                pair_score_consumer.advance()

                ordinal0 = pair_idx * Int32(2)
                block0 = Int32(route_indices[ordinal0])
                has_peer = ordinal0 + Int32(1) < exact_block_count
                block1 = block0
                if has_peer:
                    block1 = Int32(route_indices[ordinal0 + Int32(1)])
                valid0 = token_count - block0 * Int32(N_MEMBER)
                valid1 = Int32(0)
                if has_peer:
                    valid1 = token_count - block1 * Int32(N_MEMBER)
                if valid0 > Int32(N_MEMBER):
                    valid0 = Int32(N_MEMBER)
                if valid1 > Int32(N_MEMBER):
                    valid1 = Int32(N_MEMBER)
                if valid0 < Int32(0):
                    valid0 = Int32(0)
                if valid1 < Int32(0):
                    valid1 = Int32(0)
                semantic_row = (
                    pair_coords[0][0] + Int32(SEMANTIC_ROW_OFFSET)
                ) & Int32(M - 1)
                row_valid = semantic_row < q_len
                for i in cutlass.range(
                    cute.size(pair_scores), unroll_full=True
                ):
                    column = pair_coords[i][1]
                    valid = column < valid0
                    if column >= Int32(N_MEMBER):
                        valid = column - Int32(N_MEMBER) < valid1
                    if not row_valid or not valid:
                        pair_scores[i] = -Float32.inf

                probabilities, next_max, next_sum, row_alpha = (
                    _online_update_pair(
                        pair_scores, running_max, running_sum
                    )
                )
                # For i>0, pair-score completion comes from QK(i), issued
                # after PV(i-1) on the same tcgen05 issuer.  The score wait and
                # load above therefore retire PV(i-1) before this O rescale.
                # Pair0 similarly follows either route QK or route PV->QK0.
                if owner_o_initialized != Int32(0):
                    _rescale_pair_o(
                        pack_o_template,
                        thr_pack_pv,
                        tmem_base,
                        Int32(O_OFFSET),
                        owner_tidx,
                        row_alpha,
                    )
                # The one TMEM P image is free once PV(i-1) completes.  Keep
                # probabilities FP32 until the live-range-bounded chunked R2T.
                _store_pair_probability_chunked_tmemp(
                    pack_o_template,
                    probabilities,
                    tmem_base,
                    Int32(PAIR_P_OFFSET),
                    owner_tidx,
                )
                # The preceding helper performs tcgen05.wait::st for every
                # chunk and a TMEM-store fence.  Publish P to warp 0 with one
                # uniform generation shared by warps 0-4; warp 5 is excluded.
                exact_pair_p_ready_barrier.arrive_and_wait()
                running_max = next_max
                running_sum = next_sum
                owner_o_initialized = Int32(1)

            # There is no successor QK after the CTA's final exact PV.  Keep
            # exactly one completion-backed wait before the epilogue; all
            # earlier groups flow into a successor route QK completion.
            if is_final_logical_group and pair_count > Int32(0):
                pair_o_pipe.consumer_wait(pair_o_consumer)

        # route_indices reuse HB proof for the next logical group:
        # (1) warp 5 reads both indices before producing each pair's K/V, and
        #     final-pair score completion therefore dominates its last read;
        # (2) all owner index reads precede the final exact-P NamedBarrier;
        # (3) owner-warp0/lane0 is the sole next-group writer and reaches it
        #     only after that same exact loop.  For exact_count==0 there are no
        #     readers.  Therefore no group-tail CTA barrier is required.

        # Cross-group progress is carried by the existing K/V buffer-free
        # phases and pair-score ready phase.  There is no CTA-wide group-tail
        # join: the next producer acquire cannot overwrite a live K/V stage,
        # and the next owner score load cannot precede QK completion.
        # END_GENERAL_N128_PAIR
    
        logical_group_idx = cute.arch.make_warp_uniform(
            logical_group_idx + Int32(1)
        )
        remaining_group_tiles = cute.arch.make_warp_uniform(
            remaining_group_tiles - Int32(ROUTE_HALVES_PER_GROUP)
        )
        # END_RUNTIME_GROUP_BODY
    # END_G256_CURSOR_UNIFORM_INDUCTION

    if warp_idx == Int32(0):
        q_pipe.consumer_release(q_consumer)
        q_consumer.advance()

    if is_owner:
        lane = owner_tidx % Int32(32)
        owner_warp = owner_tidx // Int32(32)
        owner_row = (
            owner_warp * Int32(16)
            + lane // Int32(4)
            + (lane % Int32(2)) * Int32(8)
            + Int32(SEMANTIC_ROW_OFFSET)
        ) & Int32(M - 1)
        # Register state remains owner-local for the entire exact stream.  It
        # is published only once here because the final Ld16x256b epilogue
        # remaps rows differently from the Ld16x64b xor-2 score ownership.
        if (lane & Int32(2)) == Int32(0):
            sFinalStats[owner_row, 0] = running_sum
            sFinalStats[owner_row, 1] = running_max
        cute.arch.fence_view_async_shared()
        final_stats_ready_barrier.arrive_and_wait()

        o_regs, o_coords = load_m64_o_fp32_256b(
            pack_o_template,
            thr_pack_pv,
            tmem_base,
            owner_tidx,
        )
        assert cute.size(o_regs) == 64
        assert cute.size(o_coords) == 64

        # B7's device inversion proves that 4*w/4*w+1 belong to one
        # semantic row and 4*w+2/4*w+3 to its row-plus-eight peer.  Hoist
        # validity, final-sum LDS, reciprocal, and row base once per stratum.
        semantic_row0 = (
            owner_warp * Int32(16)
            + lane // Int32(4)
            + Int32(SEMANTIC_ROW_OFFSET)
        ) & Int32(M - 1)
        semantic_row1 = (semantic_row0 + Int32(8)) & Int32(M - 1)
        even_col_base = (lane % Int32(4)) * Int32(2)

        if semantic_row0 < q_len:
            inv_sum0 = cute.arch.rcp_approx(
                Float32(sFinalStats[semantic_row0, 0])
            )
            query_idx0 = q_block_idx * Int32(M) + semantic_row0
            destination_row0 = cute.domain_offset(
                (batch_idx, head_idx, query_idx0, Int32(0)), mO_bhtd
            )
            for word_i in cutlass.range(
                O_PACKED_WORDS_PER_ROW_PER_THREAD, unroll_full=True
            ):
                even_i = word_i * 4
                odd_i = even_i + 1
                even_value = Float32(o_regs[even_i]) * inv_sum0
                odd_value = Float32(o_regs[odd_i]) * inv_sum0
                packed_word = _cvt_bf16x2_f32(
                    Float32(odd_value), Float32(even_value)
                )
                even_col = (
                    even_col_base + word_i * O_PACKED_COLUMN_STRIDE
                )
                _store_global_u32_inline(
                    destination_row0.iterator + even_col, packed_word
                )

        if semantic_row1 < q_len:
            inv_sum1 = cute.arch.rcp_approx(
                Float32(sFinalStats[semantic_row1, 0])
            )
            query_idx1 = q_block_idx * Int32(M) + semantic_row1
            destination_row1 = cute.domain_offset(
                (batch_idx, head_idx, query_idx1, Int32(0)), mO_bhtd
            )
            for word_i in cutlass.range(
                O_PACKED_WORDS_PER_ROW_PER_THREAD, unroll_full=True
            ):
                even_i = word_i * 4 + 2
                odd_i = even_i + 1
                even_value = Float32(o_regs[even_i]) * inv_sum1
                odd_value = Float32(o_regs[odd_i]) * inv_sum1
                packed_word = _cvt_bf16x2_f32(
                    Float32(odd_value), Float32(even_value)
                )
                even_col = (
                    even_col_base + word_i * O_PACKED_COLUMN_STRIDE
                )
                _store_global_u32_inline(
                    destination_row1.iterator + even_col, packed_word
                )

        if (lane & Int32(2)) == Int32(0) and owner_row < q_len:
            query_idx = q_block_idx * Int32(M) + owner_row
            mLSE_bht[batch_idx, head_idx, query_idx] = (
                running_max
                + cute.math.log2(running_sum, fastmath=True) * Float32(LN2)
            )

    cute.arch.barrier()
    tmem.free(tmem_ptr)


@cute.jit
def lean6_bf16_routeidx_g256_cursor_ballotscatter_packedgmemo_tworowhoist_inlineu32_fusedroute_n128_pair_tmemp_grid_host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    kc: cute.Tensor,
    vc: cute.Tensor,
    threshold: cute.Tensor,
    lse: cute.Tensor,
    softmax_scale: Float32,
    packet_trace: Optional[cute.Tensor] = None,
    stream: cuda.CUstream = None,
):
    q, k, v, o, kc, vc = tuple(
        assume_tensor_aligned(t) for t in (q, k, v, o, kc, vc)
    )
    q_mkl, k_nkl, kc_nkl = [
        layout_utils.select(t, [2, 3, 1, 0]) for t in (q, k, kc)
    ]
    v_nkl, vc_nkl = [
        layout_utils.select(t, [3, 2, 1, 0]) for t in (v, vc)
    ]
    token_count = cute.size(q_mkl.shape[0])
    num_blocks = cute.size(kc_nkl.shape[0])
    num_heads = cute.size(q_mkl.shape[2])
    num_batches = cute.size(q_mkl.shape[3])
    num_route_tiles = cute.ceil_div(num_blocks, ROUTE_TILE_SIZE)
    _ = softmax_scale

    pack_qk_op = tcgen05.MmaF16BF16Op(
        BFloat16,
        Float32,
        PACK_QK_INST,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_pack_qk = cute.make_tiled_mma(pack_qk_op)
    pack_pv_op = tcgen05.MmaF16BF16Op(
        BFloat16,
        Float32,
        PACK_PV_INST,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.TMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.MN,
    )
    tiled_pack_pv = cute.make_tiled_mma(pack_pv_op)
    pack_qk_quarter_op = tcgen05.MmaF16BF16Op(
        BFloat16,
        Float32,
        PACK_QK_QUARTER_INST,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_pack_qk_gather = cute.make_tiled_mma(pack_qk_quarter_op)
    pack_pv_quarter_op = tcgen05.MmaF16BF16Op(
        BFloat16,
        Float32,
        PACK_PV_QUARTER_INST,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.TMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.MN,
    )
    tiled_pack_pv_gather = cute.make_tiled_mma(pack_pv_quarter_op)
    q_layout = sm100_utils.make_smem_layout_a(
        tiled_pack_qk, PACK_QK_TILE, BFloat16, 1
    )
    pack_k_layout = sm100_utils.make_smem_layout_b(
        tiled_pack_qk, PACK_QK_TILE, BFloat16, PAIR_STAGES
    )
    pack_v_layout = sm100_utils.make_smem_layout_b(
        tiled_pack_pv, PACK_PV_TILE, BFloat16, PAIR_STAGES
    )
    pack_k_gather_layout = sm100_utils.make_smem_layout_b(
        tiled_pack_qk_gather,
        PACK_QK_GATHER_TILE,
        BFloat16,
        PAIR_STAGES * 4,
    )
    pack_v_gather_layout = sm100_utils.make_smem_layout_b(
        tiled_pack_pv_gather,
        PACK_PV_GATHER_TILE,
        BFloat16,
        PAIR_STAGES * 4,
    )
    pack_p_layout = sm100_utils.make_smem_layout_a(
        tiled_pack_pv, PACK_PV_TILE, BFloat16, 1
    )
    route_k_layout = sm100_utils.make_smem_layout_b(
        tiled_pack_qk, PACK_QK_TILE, BFloat16, PAIR_STAGES
    )
    route_v_layout = sm100_utils.make_smem_layout_b(
        tiled_pack_pv, PACK_PV_TILE, BFloat16, PAIR_STAGES
    )
    copy_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    q_tma_atom, q_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        copy_op,
        q_mkl,
        cute.select(q_layout, mode=[0, 1, 2]),
        PACK_QK_TILE,
        tiled_pack_qk,
    )
    pack_k_tma_layout = cute.make_composed_layout(
        pack_k_gather_layout.inner,
        0,
        cute.make_layout((64, 64), stride=(64, 1)),
    )
    pack_k_tma_atom, pack_k_tma_tensor = cpasync.make_tiled_tma_atom(
        copy_op,
        k_nkl,
        pack_k_tma_layout,
        (64, 64),
    )
    pack_v_tma_layout = cute.make_composed_layout(
        pack_v_gather_layout.inner,
        0,
        cute.make_layout((64, 64), stride=(1, 64)),
    )
    pack_v_tma_atom, pack_v_tma_tensor = cpasync.make_tiled_tma_atom(
        copy_op,
        v_nkl,
        pack_v_tma_layout,
        (64, 64),
    )
    kc_tma_atom, kc_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        copy_op,
        kc_nkl,
        cute.select(route_k_layout, mode=[0, 1, 2]),
        PACK_QK_TILE,
        tiled_pack_qk,
    )
    vc_tma_atom, vc_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        copy_op,
        vc_nkl,
        cute.select(route_v_layout, mode=[0, 1, 2]),
        PACK_PV_TILE,
        tiled_pack_pv,
    )
    lean6_bf16_routeidx_g256_cursor_ballotscatter_packedgmemo_tworowhoist_inlineu32_fusedroute_n128_pair_tmemp_grid_kernel(
        tiled_pack_qk,
        tiled_pack_pv,
        q_tma_atom,
        q_tma_tensor,
        pack_k_tma_atom,
        pack_k_tma_tensor,
        pack_v_tma_atom,
        pack_v_tma_tensor,
        kc_tma_atom,
        kc_tma_tensor,
        vc_tma_atom,
        vc_tma_tensor,
        threshold,
        o,
        lse,
        packet_trace,
        Int32(token_count),
        Int32(num_blocks),
        Int32(num_route_tiles),
        q_layout,
        pack_k_layout,
        pack_k_gather_layout,
        pack_p_layout,
        pack_v_layout,
        pack_v_gather_layout,
        route_k_layout,
        route_v_layout,
    ).launch(
        grid=(num_blocks, num_heads, num_batches),
        block=(THREADS, 1, 1),
        stream=stream,
        min_blocks_per_mp=2,
    )


class Pisa2Sm100Lean6RouteidxG256CursorBallot4FusedrouteN128PairTmempBf16Forward:
    """Logical-G256 fused route with physical N128 transactions."""

    def __init__(
        self,
        T: int,
        *,
        is_causal: bool = False,
        trace_route_masks: bool = False,
    ):
        if isinstance(T, bool) or not isinstance(T, int) or T <= 0:
            raise ValueError(f"T must be a positive integer, got {T!r}")
        if not isinstance(is_causal, bool):
            raise ValueError("is_causal must be a bool")
        if is_causal:
            raise ValueError(
                "lean6 v1 supports only full noncausal attention; diagonal "
                "token-level causal masking has not been validated"
            )
        if not isinstance(trace_route_masks, bool):
            raise ValueError("trace_route_masks must be a bool")
        self.T = T
        self.num_blocks = (T + M - 1) // M
        self.num_route_tiles = (
            self.num_blocks + ROUTE_TILE_SIZE - 1
        ) // ROUTE_TILE_SIZE
        self.num_groups = (
            self.num_blocks + LOGICAL_GROUP_SIZE - 1
        ) // LOGICAL_GROUP_SIZE
        self.is_causal = is_causal
        self.trace_route_masks = trace_route_masks

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        kc: cute.Tensor,
        vc: cute.Tensor,
        global_threshold: cute.Tensor,
        lse: cute.Tensor,
        softmax_scale: Float32,
        route_mask_trace: Optional[cute.Tensor] = None,
        stream: cuda.CUstream = None,
    ):
        # DLPack tensors are deliberately marked layout-dynamic, so their
        # extents are IR values here and cannot be used in Python ``assert``.
        # The explicit runner validates the complete prepared-tensor ABI
        # before creating these CuTe tensors; the kernel still receives the
        # frozen T/NT specialization through its host launch arguments.
        if cutlass.const_expr(self.trace_route_masks):
            assert route_mask_trace is not None
        else:
            assert route_mask_trace is None
        return lean6_bf16_routeidx_g256_cursor_ballotscatter_packedgmemo_tworowhoist_inlineu32_fusedroute_n128_pair_tmemp_grid_host(
            q,
            k,
            v,
            o,
            kc,
            vc,
            global_threshold,
            lse,
            softmax_scale,
            route_mask_trace,
            stream,
        )


def build_pisa2_sm100_lean6_routeidx_g256_cursor_ballotscatter_packedgmemo_tworowhoist_inlineu32_fusedroute_n128_pair_tmemp_bf16_fwd(
    T: int,
    *,
    is_causal: bool = False,
) -> Pisa2Sm100Lean6RouteidxG256CursorBallot4FusedrouteN128PairTmempBf16Forward:
    """Build the ordinary no-trace full-grid sibling."""

    return Pisa2Sm100Lean6RouteidxG256CursorBallot4FusedrouteN128PairTmempBf16Forward(
        T, is_causal=is_causal
    )


def build_pisa2_sm100_lean6_routeidx_g256_cursor_ballotscatter_packedgmemo_tworowhoist_inlineu32_fusedroute_n128_pair_tmemp_bf16_trace_fwd(
    T: int,
    *,
    is_causal: bool = False,
) -> Pisa2Sm100Lean6RouteidxG256CursorBallot4FusedrouteN128PairTmempBf16Forward:
    """Build the route-packet trace specialization for correctness gates."""

    return Pisa2Sm100Lean6RouteidxG256CursorBallot4FusedrouteN128PairTmempBf16Forward(
        T,
        is_causal=is_causal,
        trace_route_masks=True,
    )


__all__ = [
    "AE9_SCALAR_ORACLE_SHA256",
    "CANDIDATE_AXIS",
    "CODEGEN_PARENT_SOURCE_SHA256",
    "G256_CURSOR_BALLOTSCATTER_PACKEDGMEMO_TWOROWHOIST_INLINEU32_FUSEDROUTE_N128_TMEMP_RECEIPT",
    "LOGICAL_GROUP_SIZE",
    "MECHANISM_FAMILY",
    "PARENT_SOURCE_SHA256",
    "Pisa2Sm100Lean6RouteidxG256CursorBallot4FusedrouteN128PairTmempBf16Forward",
    "ROUTE_INDEX_CAPACITY",
    "ROUTE_TILE_SIZE",
    "UNCOMPENSATED_V9_SOURCE_SHA256",
    "build_pisa2_sm100_lean6_routeidx_g256_cursor_ballotscatter_packedgmemo_tworowhoist_inlineu32_fusedroute_n128_pair_tmemp_bf16_fwd",
    "build_pisa2_sm100_lean6_routeidx_g256_cursor_ballotscatter_packedgmemo_tworowhoist_inlineu32_fusedroute_n128_pair_tmemp_bf16_trace_fwd",
]
