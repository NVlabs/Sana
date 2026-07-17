from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kernels/pisa2_sm100/native_bf16_claude49_g256_colmask_fwd.py"
PROMOTION = ROOT / "evidence/promotion/v1-full-result.json"
SUMMARY = ROOT / "evidence/full45/full45-summary.json"


def test_colmask_mechanism_and_schedule_are_frozen() -> None:
    source = KERNEL.read_text(encoding="utf-8")
    assert 'MECHANISM_FAMILY = "sm100-bf16-g256-packedsel-colmask-v1"' in source
    assert (
        'CANDIDATE_AXIS = "per_column_additive_mask_replaces_select_chains"'
        in source
    )
    assert '"shape_or_density_fast_path": False' in source
    assert "LOGICAL_GROUP_SIZE = 256" in source
    assert "ROUTE_TILE_SIZE = 128" in source
    assert "TMEM_COLS = 256" in source
    assert "THREADS = 192" in source
    assert "add_packed_f32x2" in source


def test_promotion_evidence_binds_resources_and_fixed_point() -> None:
    promotion = json.loads(PROMOTION.read_text(encoding="utf-8"))
    candidate = promotion["static_sass_resource"]["candidate"]
    assert promotion["passes"] is True
    assert candidate["driver"]["registers_per_thread"] == 168
    assert candidate["driver"]["local_bytes_per_thread"] == 0
    assert candidate["resource"]["local"] == [0]
    assert math.isclose(
        promotion["device"]["timing"]["medians_ms"]["candidate"],
        0.7132160067558289,
        abs_tol=1e-15,
    )


def test_verified_numerical_scope_is_bound() -> None:
    promotion = json.loads(PROMOTION.read_text(encoding="utf-8"))
    checks = promotion["device"]["checks"]
    assert checks["route_bitwise"] is True
    assert checks["plain_output_bitwise"] is True
    assert checks["plain_lse_bitwise"] is True
    assert checks["trace_output_bitwise"] is True
    assert checks["trace_lse_bitwise"] is True

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["case_count"] == 45
    assert summary["checks"]["all_45_correct"] is True
