from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / (
    "kernels/sol_attn_sm100/"
    "native_bf16_lean6_routeidx_g512_cursor_ballotscatter_"
    "fusedroute_n128_pair_tmemp_hybrid_massreuse_terminalo_fwd.py"
)
RUNNER = ROOT / (
    "experiments/sol_attn/"
    "native_bf16_lean6_routeidx_g512_cursor_ballotscatter_"
    "fusedroute_n128_pair_tmemp_hybrid_massreuse_terminalo_runner.py"
)
SUMMARY = ROOT / "evidence/full45/full45-summary.json"
STRICT_GATE = ROOT / "evidence/full45/strict-user-gate.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_evidence_bound_kernel_and_runner_bytes() -> None:
    assert _sha256(KERNEL) == (
        "9930d201104ed6bc035c670283bed08da3ea8114591ceea4c5873ede0caee106"
    )
    assert _sha256(RUNNER) == (
        "e70095b53757cea6934d9e6d87dc1db33da6f337623d426ab7b0c2a510c0c695"
    )


def test_full45_release_gate() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    gate = json.loads(STRICT_GATE.read_text(encoding="utf-8"))
    assert summary["case_count"] == 45
    assert summary["overall"]["geomean"] < 1.0
    assert summary["overall"]["wins"] == 45
    assert summary["worst_case"]["candidate_over_triton_sol_attn"] <= 1.03
    assert gate["passes"] is True
    assert gate["all_45_correct"] is True
    assert gate["all_T_geomeans_lt_1"] is True
    assert gate["all_BH_geomeans_lt_1"] is True
    assert gate["all_density_geomeans_lt_1"] is True
    assert gate["max_ratio_le_1_03"] is True


def test_systemic_schedule_has_no_fast_path() -> None:
    source = KERNEL.read_text(encoding="utf-8")
    assert '"shape_or_density_fast_path": False' in source
    assert "LOGICAL_GROUP_SIZE = 512" in source
    assert "ROUTE_TILE_SIZE = 128" in source
    assert "TMEM_COLS = 256" in source
    assert "THREADS = 192" in source
