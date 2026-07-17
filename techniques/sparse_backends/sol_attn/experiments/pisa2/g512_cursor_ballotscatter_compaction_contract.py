#!/usr/bin/env python3
"""Fail-closed semantic/source/HB contract for G512 ballotscatter compaction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
PARENT_PATH = ROOT / (
    "kernels/pisa2_sm100/"
    "native_bf16_lean6_routeidx_g512_cursor_fusedroute_n128_pair_tmemp_"
    "hybrid_massreuse_terminalo_fwd.py"
)
CANDIDATE_PATH = ROOT / (
    "kernels/pisa2_sm100/"
    "native_bf16_lean6_routeidx_g512_cursor_ballotscatter_fusedroute_n128_pair_"
    "tmemp_hybrid_massreuse_terminalo_fwd.py"
)
RUNNER_PATH = ROOT / (
    "experiments/pisa2/"
    "native_bf16_lean6_routeidx_g512_cursor_ballotscatter_fusedroute_n128_pair_"
    "tmemp_hybrid_massreuse_terminalo_runner.py"
)

PARENT_SHA256 = (
    "d344389a71501eea4da084813a9178967e41c0322dbbbc9553caa0ed07008bc1"
)
CANDIDATE_SHA256 = (
    "75dccb78d282c7741eeb3833ea09af15543438cfe2d207ade9af72d07400feb5"
)
RUNNER_SHA256 = (
    "840ccdee2c8852f580d835357f14e52896052528bf5049ec6f206aa0cbe6aeed"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _between(text: str, begin: str, end: str) -> str:
    start = text.index(begin)
    stop = text.index(end, start) + len(end)
    return text[start:stop]


def _word_indices(raw: int, word: int, route_start: int) -> list[int]:
    result: list[int] = []
    mask = raw & 0xFFFFFFFF
    while mask:
        lowbit = mask & -mask
        bit = lowbit.bit_length() - 1
        result.append(route_start + 32 * word + bit)
        mask &= mask - 1
    return result


def _popc32(raw: int) -> int:
    return bin(raw & 0xFFFFFFFF).count("1")


def _lane_mask_lt(lane: int) -> int:
    """Mirror the candidate's positive signed-Int32 prefix-mask formula."""

    assert 0 <= lane < 32
    return 0x7FFFFFFF >> (31 - lane)


def _parent_epoch(
    masks: Iterable[tuple[int, int, int, int]],
) -> tuple[list[int], list[int]]:
    indices: list[int] = []
    cumulative: list[int] = []
    for half, words in enumerate(masks):
        for word, raw in enumerate(words):
            indices.extend(_word_indices(raw, word, half * 128))
        cumulative.append(len(indices))
    return indices, cumulative


def _candidate_epoch(
    masks: Iterable[tuple[int, int, int, int]],
) -> tuple[list[int], list[int], bool]:
    """Model selected lanes scattering to rank-addressed stream slots."""

    indices: list[int] = []
    cumulative: list[int] = []
    race_free = True
    for half, words in enumerate(masks):
        append_base = 0 if half == 0 else cumulative[-1]
        exact_count = sum(_popc32(raw) for raw in words)
        writes: dict[int, int] = {}
        preceding_word_count = 0
        for word, raw in enumerate(words):
            word_mask = raw & 0xFFFFFFFF
            for lane in range(32):
                if not (word_mask & (1 << lane)):
                    continue
                rank = (
                    append_base
                    + preceding_word_count
                    + _popc32(word_mask & _lane_mask_lt(lane))
                )
                if rank in writes:
                    race_free = False
                writes[rank] = half * 128 + word * 32 + lane
            preceding_word_count += _popc32(word_mask)
        expected_ranks = set(range(append_base, append_base + exact_count))
        race_free = race_free and set(writes) == expected_ranks
        race_free = race_free and len(indices) == append_base
        indices.extend(writes[rank] for rank in sorted(writes))
        cumulative.append(append_base + exact_count)
    return indices, cumulative, race_free


def _mask_cases() -> list[list[tuple[int, int, int, int]]]:
    cases: list[list[tuple[int, int, int, int]]] = []
    patterns = [
        0,
        1,
        2,
        3,
        0x80000000,
        0xFFFFFFFF,
        0xAAAAAAAA,
        0x55555555,
        0x80000001,
    ]
    for half_count in range(1, 5):
        for seed in range(129):
            halves: list[tuple[int, int, int, int]] = []
            state = (seed + 1) * 0x9E3779B1
            for half in range(half_count):
                words: list[int] = []
                for word in range(4):
                    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
                    value = (
                        state
                        if seed >= len(patterns)
                        else patterns[(seed + half + word) % len(patterns)]
                    )
                    words.append(value)
                halves.append(tuple(words))
            cases.append(halves)
    return cases


def contract() -> dict[str, object]:
    parent = PARENT_PATH.read_text()
    candidate = CANDIDATE_PATH.read_text()
    runner = RUNNER_PATH.read_text()

    parent_route = _between(
        parent,
        "if owner_warp == Int32(0):\n                    mask0",
        "# The selector packet is now immutable",
    )
    candidate_route = _between(
        candidate,
        "if owner_warp == Int32(0):\n                    mask0",
        "# The selector packet is now immutable",
    )
    exact_parent = _between(
        parent, "# BEGIN_GENERAL_N128_PAIR", "# END_GENERAL_N128_PAIR"
    )
    exact_candidate = _between(
        candidate, "# BEGIN_GENERAL_N128_PAIR", "# END_GENERAL_N128_PAIR"
    )

    model_cases = _mask_cases()
    modeled = [(_parent_epoch(case), _candidate_epoch(case)) for case in model_cases]
    model_equal = all(parent == candidate[:2] for parent, candidate in modeled)
    model_race_free = all(candidate[2] for _, candidate in modeled)
    synchronization_tokens = (
        "pipeline.NamedBarrier",
        ".arrive_and_wait()",
        "cute.arch.barrier()",
        "fence_view_async_shared()",
    )
    checks = {
        "parent_sha256": _sha(PARENT_PATH) == PARENT_SHA256,
        "candidate_sha256": _sha(CANDIDATE_PATH) == CANDIDATE_SHA256,
        "runner_sha256": _sha(RUNNER_PATH) == RUNNER_SHA256,
        "logical_g512_preserved": "LOGICAL_GROUP_SIZE = 512" in candidate,
        "physical_n128_preserved": "ROUTE_TILE_SIZE = 128" in candidate,
        "tmem256_preserved": "TMEM_COLS = 256" in candidate,
        "six_warps_preserved": "THREADS = 192" in candidate,
        "two_cta_request_preserved": "min_blocks_per_mp=2" in candidate,
        "single_kv_stage_preserved": "PAIR_STAGES = 1" in candidate,
        "no_special_fast_path": (
            '"shape_or_density_fast_path": False' in candidate
            and "run.shape_or_density_fast_path = False" in runner
        ),
        "one_ballot_source_in_constexpr_four_word_loop": (
            candidate_route.count("cute.arch.vote_ballot_sync(exact_pred)") == 1
            and "cutlass.range_constexpr(ROUTE_MASK_WORDS)" in candidate_route
        ),
        "shuffle_tree_removed_from_route_block": (
            parent_route.count("cute.arch.shuffle_sync_down(") == 5
            and candidate_route.count("cute.arch.shuffle_sync_down(") == 0
        ),
        "twenty_unrolled_shuffles_replaced_by_four_ballots": (
            parent_route.count("cute.arch.shuffle_sync_down(") * 4 == 20
            and candidate_route.count("cute.arch.vote_ballot_sync(exact_pred)")
            * 4
            == 4
        ),
        "selected_lane_direct_scatter": (
            "if exact_pred:\n                            route_indices[lane_rank]"
            in candidate_route
            and "preceding_word_count" in candidate_route
            and "pisa2_popc_b32(word_mask & lane_mask_lt)" in candidate_route
            and "if lane < Int32(4):" not in candidate_route
        ),
        "dynamic_lowbit_bfind_loops_removed": (
            "pisa2_bfind_b32(" in parent_route
            and "pisa2_bfind_b32(" not in candidate_route
            and "while compact_mask" not in candidate_route
            and "while m != Int32(0)" not in candidate_route
            and "lowbit =" not in candidate_route
        ),
        "signed_int32_lane31_prefix_safe": (
            "lane_mask_lt = Int32(0x7FFFFFFF) >> ("
            in candidate_route
            and "Int32(31) - lane" in candidate_route
            and _lane_mask_lt(0) == 0
            and _lane_mask_lt(1) == 1
            and _lane_mask_lt(31) == 0x7FFFFFFF
            and all(
                _lane_mask_lt(lane) == ((1 << lane) - 1)
                for lane in range(32)
            )
        ),
        "lane0_packet_publish_only": (
            candidate_route.count("route_packet[0] = mask0") == 1
            and "if lane == Int32(0):\n                        route_rank = append_base + exact_count"
            in candidate_route
        ),
        "no_new_sync_or_handoff": all(
            candidate.count(token) == parent.count(token)
            for token in synchronization_tokens
        ),
        "shared_layout_capacity_unchanged": all(
            candidate.count(token) == parent.count(token)
            for token in (
                "MemRange[Float32, 4 * ROUTE_TILE_SIZE]",
                "MemRange[Int32, PACKET_WORDS]",
                "MemRange[Int32, ROUTE_INDEX_CAPACITY]",
            )
        ),
        "exact_phase_text_unchanged": exact_parent == exact_candidate,
        "math_helpers_unchanged": all(
            parent.count(name) == candidate.count(name)
            for name in (
                "_online_update_pair(",
                "_rescale_pair_o(",
                "_store_pair_probability_chunked_tmemp(",
                "mma_utils.gemm(",
            )
        ),
        "all_516_compaction_models_equal": model_equal and len(model_cases) == 516,
        "all_516_scatter_models_race_free": (
            model_race_free and len(model_cases) == 516
        ),
        "runner_binds_candidate_only": (
            CANDIDATE_SHA256 in runner
            and PARENT_SHA256 in runner
            and "run.phase_graph_changed = False" in runner
            and "run.production_dispatch_registered = False" in runner
        ),
    }
    return {
        "contract": "g512_cursor_ballotscatter_compaction_v1",
        "checks": checks,
        "passes": all(checks.values()),
        "model_case_count": len(model_cases),
        "parent_sha256": _sha(PARENT_PATH),
        "candidate_sha256": _sha(CANDIDATE_PATH),
        "runner_sha256": _sha(RUNNER_PATH),
    }


def main() -> None:
    result = contract()
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
