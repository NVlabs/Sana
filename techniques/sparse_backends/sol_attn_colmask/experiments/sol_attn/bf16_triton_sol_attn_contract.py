"""Frozen BF16 CuTeDSL-versus-Triton SOL Attention full45 contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "workloads" / "sol_attn_sm100_bf16_vs_triton_sol_attn_full45.json"

TOKENS = (16384, 32768, 65536, 98304, 131072)
BATCH_HEAD_PAIRS = ((1, 32), (2, 24), (4, 12))
DENSITIES = (0.15, 0.10, 0.05)
G_CANDIDATES = (16, 32, 64, 128)
OUTPUT_LIMITS = {"max_abs": 0.08, "mean_abs": 0.01, "rel_l2": 0.01}
LSE_LIMITS = {"max_abs": 0.05, "mean_abs": 0.005, "rel_l2": 0.005}


EXPECTED_CONTRACT = {
    "contract_version": "sol_attn-sm100-bf16-vs-triton-sol_attn-full45-v1",
    "kernel": {
        "layout": "BHTD",
        "dtype_qkv_kc_vc_o": "bfloat16",
        "softmax_state_dtype": "float32",
        "head_dim": 128,
        "block_size": 64,
        "route_group_size": 64,
        "causal": False,
    },
    "reference": {
        "R_sem": {"backend": "triton_bf16_sol_attn", "group_size": 64},
        "R_perf": {
            "backend": "triton_bf16_sol_attn",
            "group_size_candidates": list(G_CANDIDATES),
            "autotune_warmup": 3,
            "autotune_repetitions": 7,
            "selection_rule": (
                "minimum_7_launch_cuda_event_median_after_3_warmups_"
                "smallest_g_tiebreak"
            ),
            "selection_scope": "independent_per_case",
        },
        "secondary": "experiments.sol_attn.prepared_reference",
        "incremental_parent": "native_bf16_lean6_maskhoist_fwd",
    },
    "workload": {
        "sequence_lengths": list(TOKENS),
        "batch_head_pairs": [list(pair) for pair in BATCH_HEAD_PAIRS],
        "target_densities": list(DENSITIES),
        "seed": 0,
        "case_count": 45,
        "case_construction": "cartesian_product_exactly_once",
        "case_key_fields": [
            "sequence_length",
            "batch",
            "heads",
            "target_density",
        ],
        "threshold": {
            "mode": "calibrate_to_target_density",
            "manual_allowed": False,
        },
    },
    "performance": {
        "timer": "CUDA events",
        "warmup": 20,
        "repetitions": 60,
        "jit_included": False,
        "preprocessing_included": False,
        "ratio": "candidate_median_over_selected_R_perf_median",
    },
    "correctness": {
        "route_bitwise_equal_to_R_sem": True,
        "output_limits": dict(OUTPUT_LIMITS),
        "lse_limits": dict(LSE_LIMITS),
        "edge_sequence_lengths": [128, 256, 512, 8192, 16384, 66000],
        "threshold_modes": ["computed", "all_exact", "local_exact", "strict_tie"],
        "immutable_prepared_tensors": True,
        "source_binding_required": True,
    },
    "promotion": {
        "all_45_correct": True,
        "overall_geomean_strictly_less_than": 1.0,
        "all_T_BH_density_subgroup_geomeans_strictly_less_than": 1.0,
        "individual_ratio_at_most": 1.03,
        "marginal_ratio_at_least": 0.98,
        "forward_reverse_relative_tolerance": 0.01,
        "marginal_remeasurement": {
            "orders": ["A-B-B-A", "B-A-A-B"],
            "fresh_process_per_timing_slot": True,
            "trigger_case_ratio_at_least": 0.98,
            "trigger_subgroup_geomean_at_least": 0.98,
            "replace_ratio_with": "geometric_mean_of_forward_and_reverse",
            "required_comparisons": [
                "candidate_vs_R_perf",
                "candidate_vs_incremental_parent",
            ],
        },
    },
    "prohibited": {
        "fast_path_keys": [
            "shape",
            "sequence_length",
            "batch",
            "heads",
            "density",
            "target_density",
            "realized_density",
            "threshold",
            "tau",
            "seed",
            "benchmark_case_id",
            "global_exact_count",
        ],
        "fallback_backends": ["six_warp", "triton", "dense", "pisa0", "two_query"],
    },
}


def _value_error(path: str, detail: str) -> ValueError:
    return ValueError(f"{path}: {detail}")


def _validate_exact(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise _value_error(path, f"expected object, got {type(actual).__name__}")
        expected_keys = set(expected)
        missing = [key for key in expected if key not in actual]
        if missing:
            child = f"{path}.{missing[0]}" if path else missing[0]
            raise _value_error(child, "missing required field")
        extra = [key for key in actual if key not in expected_keys]
        if extra:
            child = f"{path}.{extra[0]}" if path else extra[0]
            raise _value_error(child, "unknown field")
        for key, expected_value in expected.items():
            child = f"{path}.{key}" if path else key
            _validate_exact(actual[key], expected_value, child)
        return

    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise _value_error(path, f"expected list, got {type(actual).__name__}")
        if len(actual) != len(expected):
            raise _value_error(path, f"expected {len(expected)} items, got {len(actual)}")
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
            _validate_exact(actual_value, expected_value, f"{path}[{index}]")
        return

    if type(actual) is not type(expected) or actual != expected:
        raise _value_error(path, f"expected {expected!r}, got {actual!r}")


def validate_contract(contract: Mapping[str, Any]) -> None:
    """Fail closed unless *contract* exactly matches the frozen authority."""

    if not isinstance(contract, Mapping):
        raise _value_error("contract", "expected object")
    if contract.get("contract_version") != EXPECTED_CONTRACT["contract_version"]:
        raise _value_error(
            "contract_version",
            f"expected {EXPECTED_CONTRACT['contract_version']!r}, "
            f"got {contract.get('contract_version')!r}",
        )
    _validate_exact(contract, EXPECTED_CONTRACT, "")


def load_and_validate_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load a JSON contract, validate it, and return an isolated copy."""

    try:
        contract = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"contract: cannot load {path}: {exc}") from exc
    validate_contract(contract)
    return copy.deepcopy(contract)


__all__ = [
    "BATCH_HEAD_PAIRS",
    "CONTRACT_PATH",
    "DENSITIES",
    "G_CANDIDATES",
    "LSE_LIMITS",
    "OUTPUT_LIMITS",
    "TOKENS",
    "load_and_validate_contract",
    "validate_contract",
]
