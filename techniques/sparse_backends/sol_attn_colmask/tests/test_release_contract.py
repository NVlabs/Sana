from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kernels/pisa2_sm100/native_bf16_claude49_g256_colmask_fwd.py"
RUNNER = ROOT / "experiments/pisa2/native_bf16_claude50_colmask_full45_runner.py"
SUMMARY = ROOT / "evidence/full45/full45-summary.json"
PER_CASE = ROOT / "evidence/full45/per-case-results.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_evidence_bound_kernel_and_runner_bytes() -> None:
    assert _sha256(KERNEL) == (
        "261d1d0e71fc6b907948eb9547adabe9c0c00932db318f8a32932f628b4b1f3e"
    )
    assert _sha256(RUNNER) == (
        "dc6529fd79ac66d3d724b29e2a0c766465ac96f89086d8c4928c532d893201ae"
    )


def test_public_api_signature_is_stable() -> None:
    from sol_attn import make_pisa2_sm100

    parameters = inspect.signature(make_pisa2_sm100).parameters
    assert tuple(parameters) == (
        "T",
        "q",
        "k",
        "v",
        "kc",
        "vc",
        "global_threshold",
        "scale",
        "is_causal",
        "trace_route_masks",
        "guard_elements",
    )
    assert parameters["is_causal"].default is False
    assert parameters["trace_route_masks"].default is False
    assert parameters["guard_elements"].default == 0


def test_full45_release_claims() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["case_count"] == 45
    assert summary["candidate_backend"] == "claude50_g256_colmask"
    assert summary["candidate_kernel_sha256"] == _sha256(KERNEL)
    assert summary["checks"]["all_45_correct"] is True

    pisa2 = summary["references"]["triton_pisa2"]
    assert math.isclose(
        pisa2["overall"]["geomean"], 0.7461636856364493, abs_tol=1e-15
    )
    assert pisa2["wins"] == 45
    assert math.isclose(
        pisa2["worst_case"]["ratio"], 0.7986808060511288, abs_tol=1e-15
    )

    pisa0 = summary["references"]["triton_pisa0"]
    assert math.isclose(
        pisa0["overall"]["geomean"], 1.000324921105, abs_tol=1e-15
    )
    assert pisa0["wins"] == 30
    assert pisa0["worst_case"] == {
        "case_id": "T32768-B4-H12-d0.05",
        "ratio": 1.1695644015437154,
    }


def test_per_case_evidence_is_complete() -> None:
    with PER_CASE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 45
    assert all(row["correctness_pass"] == "True" for row in rows)
    assert sum(row["win_vs_triton_pisa2"] == "True" for row in rows) == 45
    assert sum(row["win_vs_triton_pisa0"] == "True" for row in rows) == 30
