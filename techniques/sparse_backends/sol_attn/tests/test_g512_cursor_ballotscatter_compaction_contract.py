from __future__ import annotations

from experiments.sol_attn.g512_cursor_ballotscatter_compaction_contract import (
    CANDIDATE_PATH,
    PARENT_PATH,
    contract,
)


def test_contract_passes_all_516_models() -> None:
    result = contract()
    assert result["passes"] is True
    assert result["model_case_count"] == 516
    assert all(result["checks"].values())


def test_only_systemic_owner_warp_axis_is_explicit() -> None:
    candidate = CANDIDATE_PATH.read_text()
    parent = PARENT_PATH.read_text()
    assert candidate.count("cute.arch.vote_ballot_sync(exact_pred)") == 1
    assert candidate.count("route_indices[lane_rank]") == 1
    assert parent.count("route_indices[route_rank]") == 1
    assert "if exact_pred:" in candidate
    assert "if lane < Int32(4):" not in candidate
    assert "sol_attn_bfind_b32(" not in candidate
    assert "lane_mask_lt = Int32(0x7FFFFFFF) >> (" in candidate
    assert "new_cross_warp_handoff\": False" in candidate
    assert '"shape_or_density_fast_path": False' in candidate


def test_selected_lane_scatter_is_ordered_race_free_and_lane31_safe() -> None:
    checks = contract()["checks"]
    for name in (
        "twenty_unrolled_shuffles_replaced_by_four_ballots",
        "selected_lane_direct_scatter",
        "dynamic_lowbit_bfind_loops_removed",
        "signed_int32_lane31_prefix_safe",
        "all_516_compaction_models_equal",
        "all_516_scatter_models_race_free",
        "lane0_packet_publish_only",
    ):
        assert checks[name] is True


def test_physical_and_phase_envelope_is_frozen() -> None:
    checks = contract()["checks"]
    for name in (
        "logical_g512_preserved",
        "physical_n128_preserved",
        "tmem256_preserved",
        "six_warps_preserved",
        "two_cta_request_preserved",
        "single_kv_stage_preserved",
        "no_new_sync_or_handoff",
        "shared_layout_capacity_unchanged",
        "exact_phase_text_unchanged",
        "math_helpers_unchanged",
    ):
        assert checks[name] is True
