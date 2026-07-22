"""Isolated candidate registry for the source-bound SOL Attention full45 harness.

This module changes neither package exports nor production dispatch.  It only
maps an explicit benchmark CLI choice to source files and lazy runner imports.
Keeping the incremental parent in the same record makes each candidate's
correctness diagnostic and source closure auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Callable


DEFAULT_BACKEND = "lean6_routeidx"
G64_N256_SPLITKV_BACKEND = "lean6_routeidx_g64_n256_splitkv"
G128_N256_BACKEND = "lean6_routeidx_g128_n256"
G128_N128_PAIR_QKLA_RAW128_BACKEND = (
    "lean6_routeidx_g128_n128_pair_qkla_raw128"
)
G64_N128_PAIR_BACKEND = "lean6_routeidx_g64_n128_pair"
G64_N128_PAIR_CHUNKP_BACKEND = "lean6_routeidx_g64_n128_pair_chunkp"
G64_N128_PAIR_QKLA_RAW128_BACKEND = (
    "lean6_routeidx_g64_n128_pair_qkla_raw128"
)
G128_N128_MASSREUSE_TERMINALO_BACKEND = (
    "lean6_routeidx_g128_n128_massreuse_terminalo"
)
G128_N128_MASSREUSE_TERMINALO_GROUPJOIN_ELIDE_BACKEND = (
    "lean6_routeidx_g128_n128_massreuse_terminalo_groupjoin_elide"
)
G256_FUSEDROUTE_N128_BACKEND = "lean6_routeidx_g256_fusedroute_n128"
G256_CURSOR_FUSEDROUTE_N128_BACKEND = (
    "lean6_routeidx_g256_cursor_fusedroute_n128"
)
G512_FUSEDROUTE_N128_BACKEND = "lean6_routeidx_g512_fusedroute_n128"
G512_CURSOR_BALLOT4_FUSEDROUTE_N128_BACKEND = (
    "lean6_routeidx_g512_cursor_ballot4_fusedroute_n128"
)
G512_CURSOR_BALLOTSCATTER_FUSEDROUTE_N128_BACKEND = (
    "lean6_routeidx_g512_cursor_ballotscatter_fusedroute_n128"
)


@dataclass(frozen=True)
class Full45CandidateBackend:
    name: str
    kernel_relative: str
    runner_relative: str
    runner_module: str
    runner_factory: str
    parent_kernel_relative: str
    parent_runner_relative: str
    parent_runner_module: str
    parent_runner_factory: str
    route_group_size: int
    route_packet_words: int
    selection_cursor_policy: str
    production_dispatch_registered: bool = False


BACKENDS = {
    DEFAULT_BACKEND: Full45CandidateBackend(
        name=DEFAULT_BACKEND,
        kernel_relative=(
            "kernels/sol_attn_sm100/native_bf16_lean6_routeidx_fwd.py"
        ),
        runner_relative=(
            "experiments/sol_attn/native_bf16_lean6_routeidx_runner.py"
        ),
        runner_module=(
            "experiments.sol_attn.native_bf16_lean6_routeidx_runner"
        ),
        runner_factory="make_native_bf16_lean6_routeidx_runner",
        parent_kernel_relative=(
            "kernels/sol_attn_sm100/native_bf16_lean6_maskhoist_fwd.py"
        ),
        parent_runner_relative=(
            "experiments/sol_attn/native_bf16_lean6_maskhoist_runner.py"
        ),
        parent_runner_module=(
            "experiments.sol_attn.native_bf16_lean6_maskhoist_runner"
        ),
        parent_runner_factory="make_native_bf16_lean6_maskhoist_runner",
        route_group_size=64,
        route_packet_words=3,
        selection_cursor_policy="word0_then_word1_lowbit_clear",
    ),
    G64_N256_SPLITKV_BACKEND: Full45CandidateBackend(
        name=G64_N256_SPLITKV_BACKEND,
        kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g64_n256_splitkv_fwd.py"
        ),
        runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g64_n256_splitkv_runner.py"
        ),
        runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g64_n256_splitkv_runner"
        ),
        runner_factory=(
            "make_native_bf16_lean6_routeidx_g64_n256_splitkv_runner"
        ),
        parent_kernel_relative=(
            "kernels/sol_attn_sm100/native_bf16_lean6_routeidx_fwd.py"
        ),
        parent_runner_relative=(
            "experiments/sol_attn/native_bf16_lean6_routeidx_runner.py"
        ),
        parent_runner_module=(
            "experiments.sol_attn.native_bf16_lean6_routeidx_runner"
        ),
        parent_runner_factory="make_native_bf16_lean6_routeidx_runner",
        route_group_size=64,
        route_packet_words=3,
        selection_cursor_policy="word0_then_word1_lowbit_clear",
    ),
    G128_N256_BACKEND: Full45CandidateBackend(
        name=G128_N256_BACKEND,
        kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g128_n256_fwd.py"
        ),
        runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g128_n256_runner.py"
        ),
        runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g128_n256_runner"
        ),
        runner_factory=(
            "make_native_bf16_lean6_routeidx_g128_n256_runner"
        ),
        parent_kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g128_full_fwd.py"
        ),
        parent_runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g128_full_runner.py"
        ),
        parent_runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g128_full_runner"
        ),
        parent_runner_factory=(
            "make_native_bf16_lean6_routeidx_g128_full_runner"
        ),
        route_group_size=128,
        route_packet_words=5,
        selection_cursor_policy=(
            "word0_then_word1_then_word2_then_word3_lowbit_clear"
        ),
    ),
    G128_N128_PAIR_QKLA_RAW128_BACKEND: Full45CandidateBackend(
        name=G128_N128_PAIR_QKLA_RAW128_BACKEND,
        kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g128_n128_pair_qkla_raw128_fwd.py"
        ),
        runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g128_n128_pair_qkla_raw128_runner.py"
        ),
        runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g128_n128_pair_qkla_raw128_runner"
        ),
        runner_factory=(
            "make_native_bf16_lean6_routeidx_g128_n128_pair_"
            "qkla_raw128_runner"
        ),
        # The graft changes the QK/PV handoff and N128 phase graph while
        # retaining full-G128 routing semantics.  Full-G128 is therefore the
        # closest independently proven diagnostic parent.
        parent_kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g128_full_fwd.py"
        ),
        parent_runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g128_full_runner.py"
        ),
        parent_runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g128_full_runner"
        ),
        parent_runner_factory=(
            "make_native_bf16_lean6_routeidx_g128_full_runner"
        ),
        route_group_size=128,
        route_packet_words=5,
        selection_cursor_policy=(
            "word0_then_word1_then_word2_then_word3_lowbit_clear"
        ),
    ),
    G64_N128_PAIR_BACKEND: Full45CandidateBackend(
        name=G64_N128_PAIR_BACKEND,
        kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g64_n128_pair_fwd.py"
        ),
        runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g64_n128_pair_runner.py"
        ),
        runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g64_n128_pair_runner"
        ),
        runner_factory=(
            "make_native_bf16_lean6_routeidx_g64_n128_pair_runner"
        ),
        parent_kernel_relative=(
            "kernels/sol_attn_sm100/native_bf16_lean6_routeidx_fwd.py"
        ),
        parent_runner_relative=(
            "experiments/sol_attn/native_bf16_lean6_routeidx_runner.py"
        ),
        parent_runner_module=(
            "experiments.sol_attn.native_bf16_lean6_routeidx_runner"
        ),
        parent_runner_factory="make_native_bf16_lean6_routeidx_runner",
        route_group_size=64,
        route_packet_words=3,
        selection_cursor_policy="word0_then_word1_lowbit_clear",
    ),
    G64_N128_PAIR_CHUNKP_BACKEND: Full45CandidateBackend(
        name=G64_N128_PAIR_CHUNKP_BACKEND,
        kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g64_n128_pair_chunkp_fwd.py"
        ),
        runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g64_n128_pair_chunkp_runner.py"
        ),
        runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g64_n128_pair_chunkp_runner"
        ),
        runner_factory=(
            "make_native_bf16_lean6_routeidx_g64_n128_pair_chunkp_runner"
        ),
        # ChunkP changes only the N128 probability R2T live range.  The
        # original N128-pair sibling is therefore the closest diagnostic
        # parent and isolates that single implementation axis.
        parent_kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g64_n128_pair_fwd.py"
        ),
        parent_runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g64_n128_pair_runner.py"
        ),
        parent_runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g64_n128_pair_runner"
        ),
        parent_runner_factory=(
            "make_native_bf16_lean6_routeidx_g64_n128_pair_runner"
        ),
        route_group_size=64,
        route_packet_words=3,
        selection_cursor_policy="word0_then_word1_lowbit_clear",
    ),
    G64_N128_PAIR_QKLA_RAW128_BACKEND: Full45CandidateBackend(
        name=G64_N128_PAIR_QKLA_RAW128_BACKEND,
        kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g64_n128_pair_qkla_raw128_fwd.py"
        ),
        runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g64_n128_pair_qkla_raw128_runner.py"
        ),
        runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g64_n128_pair_qkla_raw128_runner"
        ),
        runner_factory=(
            "make_native_bf16_lean6_routeidx_g64_n128_pair_"
            "qkla_raw128_runner"
        ),
        parent_kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g64_n128_pair_chunkp_fwd.py"
        ),
        parent_runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g64_n128_pair_chunkp_runner.py"
        ),
        parent_runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g64_n128_pair_chunkp_runner"
        ),
        parent_runner_factory=(
            "make_native_bf16_lean6_routeidx_g64_n128_pair_chunkp_runner"
        ),
        route_group_size=64,
        route_packet_words=3,
        selection_cursor_policy="word0_then_word1_lowbit_clear",
    ),
    G128_N128_MASSREUSE_TERMINALO_BACKEND: Full45CandidateBackend(
        name=G128_N128_MASSREUSE_TERMINALO_BACKEND,
        kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g128_n128_pair_tmemp_hybrid_"
            "massreuse_routep_elide_exactp_namedbarrier_terminalo_fwd.py"
        ),
        runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g128_n128_pair_tmemp_hybrid_"
            "massreuse_routep_elide_exactp_namedbarrier_terminalo_runner.py"
        ),
        runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g128_n128_pair_tmemp_hybrid_"
            "massreuse_routep_elide_exactp_namedbarrier_terminalo_runner"
        ),
        runner_factory=(
            "make_native_bf16_lean6_routeidx_g128_n128_pair_tmemp_hybrid_"
            "massreuse_routep_elide_exactp_namedbarrier_terminalo_runner"
        ),
        # Terminal-O changes only the completion edge for intermediate O.
        # Exact-P NamedBarrier is the immediate diagnostic parent and keeps
        # routing, score, softmax, and PV work otherwise identical.
        parent_kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g128_n128_pair_tmemp_hybrid_"
            "massreuse_routep_elide_exactp_namedbarrier_fwd.py"
        ),
        parent_runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g128_n128_pair_tmemp_hybrid_"
            "massreuse_routep_elide_exactp_namedbarrier_runner.py"
        ),
        parent_runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g128_n128_pair_tmemp_hybrid_"
            "massreuse_routep_elide_exactp_namedbarrier_runner"
        ),
        parent_runner_factory=(
            "make_native_bf16_lean6_routeidx_g128_n128_pair_tmemp_hybrid_"
            "massreuse_routep_elide_exactp_namedbarrier_runner"
        ),
        route_group_size=128,
        route_packet_words=5,
        selection_cursor_policy=(
            "word0_then_word1_then_word2_then_word3_lowbit_clear"
        ),
    ),
    G128_N128_MASSREUSE_TERMINALO_GROUPJOIN_ELIDE_BACKEND: Full45CandidateBackend(
        name=G128_N128_MASSREUSE_TERMINALO_GROUPJOIN_ELIDE_BACKEND,
        kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g128_n128_pair_tmemp_hybrid_"
            "massreuse_routep_elide_exactp_namedbarrier_terminalo_"
            "groupjoin_elide_fwd.py"
        ),
        runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g128_n128_pair_tmemp_hybrid_"
            "massreuse_routep_elide_exactp_namedbarrier_terminalo_"
            "groupjoin_elide_runner.py"
        ),
        runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g128_n128_pair_tmemp_hybrid_"
            "massreuse_routep_elide_exactp_namedbarrier_terminalo_"
            "groupjoin_elide_runner"
        ),
        runner_factory=(
            "make_native_bf16_lean6_routeidx_g128_n128_pair_tmemp_hybrid_"
            "massreuse_routep_elide_exactp_namedbarrier_terminalo_"
            "groupjoin_elide_runner"
        ),
        # The candidate removes only two group-tail CTA joins.  Terminal-O is
        # therefore the immediate parent and isolates that phase-graph edit.
        parent_kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g128_n128_pair_tmemp_hybrid_"
            "massreuse_routep_elide_exactp_namedbarrier_terminalo_fwd.py"
        ),
        parent_runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g128_n128_pair_tmemp_hybrid_"
            "massreuse_routep_elide_exactp_namedbarrier_terminalo_runner.py"
        ),
        parent_runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g128_n128_pair_tmemp_hybrid_"
            "massreuse_routep_elide_exactp_namedbarrier_terminalo_runner"
        ),
        parent_runner_factory=(
            "make_native_bf16_lean6_routeidx_g128_n128_pair_tmemp_hybrid_"
            "massreuse_routep_elide_exactp_namedbarrier_terminalo_runner"
        ),
        route_group_size=128,
        route_packet_words=5,
        selection_cursor_policy=(
            "word0_then_word1_then_word2_then_word3_lowbit_clear"
        ),
    ),
    G256_FUSEDROUTE_N128_BACKEND: Full45CandidateBackend(
        name=G256_FUSEDROUTE_N128_BACKEND,
        kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g256_fusedroute_n128_pair_tmemp_"
            "hybrid_massreuse_terminalo_fwd.py"
        ),
        runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g256_fusedroute_n128_pair_tmemp_"
            "hybrid_massreuse_terminalo_runner.py"
        ),
        runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g256_fusedroute_n128_pair_tmemp_"
            "hybrid_massreuse_terminalo_runner"
        ),
        runner_factory=(
            "make_native_bf16_lean6_routeidx_g256_fusedroute_n128_pair_"
            "tmemp_hybrid_massreuse_terminalo_runner"
        ),
        # The immediate parent streams exact pairs after each physical half.
        # This isolates the fused G256 route/control and single pre-exact join.
        parent_kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g256_streaming_n128_pair_tmemp_"
            "hybrid_massreuse_terminalo_fwd.py"
        ),
        parent_runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g256_streaming_n128_pair_tmemp_"
            "hybrid_massreuse_terminalo_runner.py"
        ),
        parent_runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g256_streaming_n128_pair_tmemp_"
            "hybrid_massreuse_terminalo_runner"
        ),
        parent_runner_factory=(
            "make_native_bf16_lean6_routeidx_g256_streaming_n128_pair_"
            "tmemp_hybrid_massreuse_terminalo_runner"
        ),
        # Routing decisions are block-local; the full45 semantic oracle uses
        # its largest supported physical packet G128 while the kernel still
        # fuses two such packets into one logical G256 control group.
        route_group_size=128,
        route_packet_words=5,
        selection_cursor_policy=(
            "half0_word0_to_word3_then_half1_word0_to_word3_lowbit_clear"
        ),
    ),
    G256_CURSOR_FUSEDROUTE_N128_BACKEND: Full45CandidateBackend(
        name=G256_CURSOR_FUSEDROUTE_N128_BACKEND,
        kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g256_cursor_fusedroute_n128_pair_"
            "tmemp_hybrid_massreuse_terminalo_fwd.py"
        ),
        runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g256_cursor_fusedroute_n128_pair_"
            "tmemp_hybrid_massreuse_terminalo_runner.py"
        ),
        runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g256_cursor_fusedroute_n128_pair_"
            "tmemp_hybrid_massreuse_terminalo_runner"
        ),
        runner_factory=(
            "make_native_bf16_lean6_routeidx_g256_cursor_fusedroute_n128_"
            "pair_tmemp_hybrid_massreuse_terminalo_runner"
        ),
        # Only the uniform induction representation changes.  The frozen G256
        # kernel is therefore the exact diagnostic parent; R_sem stays on its
        # physical G128 routing packets.
        parent_kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g256_fusedroute_n128_pair_tmemp_"
            "hybrid_massreuse_terminalo_fwd.py"
        ),
        parent_runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g256_fusedroute_n128_pair_tmemp_"
            "hybrid_massreuse_terminalo_runner.py"
        ),
        parent_runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g256_fusedroute_n128_pair_tmemp_"
            "hybrid_massreuse_terminalo_runner"
        ),
        parent_runner_factory=(
            "make_native_bf16_lean6_routeidx_g256_fusedroute_n128_pair_"
            "tmemp_hybrid_massreuse_terminalo_runner"
        ),
        route_group_size=128,
        route_packet_words=5,
        selection_cursor_policy=(
            "half0_word0_to_word3_then_half1_word0_to_word3_lowbit_clear"
        ),
    ),
    G512_FUSEDROUTE_N128_BACKEND: Full45CandidateBackend(
        name=G512_FUSEDROUTE_N128_BACKEND,
        kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g512_fusedroute_n128_pair_tmemp_"
            "hybrid_massreuse_terminalo_fwd.py"
        ),
        runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g512_fusedroute_n128_pair_tmemp_"
            "hybrid_massreuse_terminalo_runner.py"
        ),
        runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g512_fusedroute_n128_pair_tmemp_"
            "hybrid_massreuse_terminalo_runner"
        ),
        runner_factory=(
            "make_native_bf16_lean6_routeidx_g512_fusedroute_n128_pair_"
            "tmemp_hybrid_massreuse_terminalo_runner"
        ),
        # G512 is an implementation scheduling choice.  Routing decisions are
        # block-local, so correctness and R_sem deliberately stay on physical
        # G128 packets while the candidate retains four such packets until one
        # logical-G512 exact phase.  The passing G256 fused-route kernel is the
        # immediate diagnostic parent.
        parent_kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g256_fusedroute_n128_pair_tmemp_"
            "hybrid_massreuse_terminalo_fwd.py"
        ),
        parent_runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g256_fusedroute_n128_pair_tmemp_"
            "hybrid_massreuse_terminalo_runner.py"
        ),
        parent_runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g256_fusedroute_n128_pair_tmemp_"
            "hybrid_massreuse_terminalo_runner"
        ),
        parent_runner_factory=(
            "make_native_bf16_lean6_routeidx_g256_fusedroute_n128_pair_"
            "tmemp_hybrid_massreuse_terminalo_runner"
        ),
        route_group_size=128,
        route_packet_words=5,
        selection_cursor_policy=(
            "half0_to_half3_each_word0_to_word3_lowbit_clear"
        ),
    ),
    G512_CURSOR_BALLOT4_FUSEDROUTE_N128_BACKEND: Full45CandidateBackend(
        name=G512_CURSOR_BALLOT4_FUSEDROUTE_N128_BACKEND,
        kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g512_cursor_ballot4_fusedroute_n128_"
            "pair_tmemp_hybrid_massreuse_terminalo_fwd.py"
        ),
        runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g512_cursor_ballot4_fusedroute_n128_"
            "pair_tmemp_hybrid_massreuse_terminalo_runner.py"
        ),
        runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g512_cursor_ballot4_fusedroute_n128_"
            "pair_tmemp_hybrid_massreuse_terminalo_runner"
        ),
        runner_factory=(
            "make_native_bf16_lean6_routeidx_g512_cursor_ballot4_fusedroute_"
            "n128_pair_tmemp_hybrid_massreuse_terminalo_runner"
        ),
        parent_kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g512_cursor_fusedroute_n128_pair_"
            "tmemp_hybrid_massreuse_terminalo_fwd.py"
        ),
        parent_runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g512_cursor_fusedroute_n128_pair_"
            "tmemp_hybrid_massreuse_terminalo_runner.py"
        ),
        parent_runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g512_cursor_fusedroute_n128_pair_"
            "tmemp_hybrid_massreuse_terminalo_runner"
        ),
        parent_runner_factory=(
            "make_native_bf16_lean6_routeidx_g512_cursor_fusedroute_n128_"
            "pair_tmemp_hybrid_massreuse_terminalo_runner"
        ),
        route_group_size=128,
        route_packet_words=5,
        selection_cursor_policy=(
            "four_lane_ballot_prefix_rank_then_lowbit_clear_exact_iteration"
        ),
    ),
    G512_CURSOR_BALLOTSCATTER_FUSEDROUTE_N128_BACKEND: Full45CandidateBackend(
        name=G512_CURSOR_BALLOTSCATTER_FUSEDROUTE_N128_BACKEND,
        kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g512_cursor_ballotscatter_"
            "fusedroute_n128_pair_tmemp_hybrid_massreuse_terminalo_fwd.py"
        ),
        runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g512_cursor_ballotscatter_"
            "fusedroute_n128_pair_tmemp_hybrid_massreuse_terminalo_runner.py"
        ),
        runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g512_cursor_ballotscatter_"
            "fusedroute_n128_pair_tmemp_hybrid_massreuse_terminalo_runner"
        ),
        runner_factory=(
            "make_native_bf16_lean6_routeidx_g512_cursor_ballotscatter_"
            "fusedroute_n128_pair_tmemp_hybrid_massreuse_terminalo_runner"
        ),
        parent_kernel_relative=(
            "kernels/sol_attn_sm100/"
            "native_bf16_lean6_routeidx_g512_cursor_fusedroute_n128_pair_"
            "tmemp_hybrid_massreuse_terminalo_fwd.py"
        ),
        parent_runner_relative=(
            "experiments/sol_attn/"
            "native_bf16_lean6_routeidx_g512_cursor_fusedroute_n128_pair_"
            "tmemp_hybrid_massreuse_terminalo_runner.py"
        ),
        parent_runner_module=(
            "experiments.sol_attn."
            "native_bf16_lean6_routeidx_g512_cursor_fusedroute_n128_pair_"
            "tmemp_hybrid_massreuse_terminalo_runner"
        ),
        parent_runner_factory=(
            "make_native_bf16_lean6_routeidx_g512_cursor_fusedroute_n128_"
            "pair_tmemp_hybrid_massreuse_terminalo_runner"
        ),
        route_group_size=128,
        route_packet_words=5,
        selection_cursor_policy=(
            "four_ballots_selected_lane_prefix_rank_direct_scatter"
        ),
    ),
}


def get_backend(name: str) -> Full45CandidateBackend:
    try:
        return BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"unknown full45 candidate backend {name!r}") from exc


def get_backend_for_kernel(kernel_relative: str) -> Full45CandidateBackend:
    """Resolve the unique candidate named by an edge-receipt kernel path."""

    matches = [
        spec
        for spec in BACKENDS.values()
        if spec.kernel_relative == kernel_relative
    ]
    if len(matches) != 1:
        raise ValueError(
            f"unknown or ambiguous full45 kernel path {kernel_relative!r}"
        )
    return matches[0]


def source_paths(
    root: Path,
    backend: str,
    *,
    correctness_harness: Path,
    timing_harness: Path,
) -> dict[str, Path]:
    """Return the existing full45 source-role schema for one backend."""

    spec = get_backend(backend)
    return {
        "kernel": root / spec.kernel_relative,
        "runner": root / spec.runner_relative,
        "backend_registry": Path(__file__).resolve(),
        "harness": correctness_harness,
        "timing_harness": timing_harness,
        "semantic_kernel": (
            root / "kernels/sol_attention_bf16_aligned.py"
        ),
        "parent_kernel": root / spec.parent_kernel_relative,
        "parent_runner": root / spec.parent_runner_relative,
        "prepared_reference": (
            root / "experiments/sol_attn/prepared_reference.py"
        ),
        "semantic_helper": (
            root / "experiments/sol_attn/check_bf16_cutedsl_semantics.py"
        ),
        "legacy_kernel": (
            root / "kernels/sol_attention_bf16_legacy.py"
        ),
        "routing_kernel": root / "kernels/sol_attention.py",
    }


def _factory(module_name: str, attribute: str) -> Callable[..., Any]:
    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    if not callable(value):
        raise TypeError(f"{module_name}.{attribute} is not callable")
    return value


def load_runner_factories(
    backend: str,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Lazily import candidate and incremental-parent factories."""

    spec = get_backend(backend)
    return (
        _factory(spec.runner_module, spec.runner_factory),
        _factory(spec.parent_runner_module, spec.parent_runner_factory),
    )


__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "G128_N128_MASSREUSE_TERMINALO_GROUPJOIN_ELIDE_BACKEND",
    "G128_N128_MASSREUSE_TERMINALO_BACKEND",
    "G128_N128_PAIR_QKLA_RAW128_BACKEND",
    "G128_N256_BACKEND",
    "G256_FUSEDROUTE_N128_BACKEND",
    "G256_CURSOR_FUSEDROUTE_N128_BACKEND",
    "G512_CURSOR_BALLOT4_FUSEDROUTE_N128_BACKEND",
    "G512_FUSEDROUTE_N128_BACKEND",
    "G64_N128_PAIR_BACKEND",
    "G64_N128_PAIR_CHUNKP_BACKEND",
    "G64_N128_PAIR_QKLA_RAW128_BACKEND",
    "G64_N256_SPLITKV_BACKEND",
    "Full45CandidateBackend",
    "get_backend",
    "get_backend_for_kernel",
    "load_runner_factories",
    "source_paths",
]
