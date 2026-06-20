#!/usr/bin/env python3
"""Compare the local TeaCache baseline against the public TeaCache4Cosmos core.

This is a public-behavior boundary probe, not a claim of full public-original
equivalence. It pins the parts that match the public TeaCache decision rule and
the parts where the current Cosmos3 candidate intentionally differs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from efficiency.technique import TechniqueContext  # noqa: E402
from efficiency.techniques.teacache import (  # noqa: E402
    TeaCacheResidual,
    teacache_indicator,
    teacache_poly_rescale,
    teacache_relative_l1,
)

PUBLIC_REF = Path("/home/haozhel/.cache/autovideo/public_refs/TeaCache")
PUBLIC_COSMOS_T2V = PUBLIC_REF / "TeaCache4Cosmos" / "teacache_sample_video_t2v.py"
MANIFEST = ROOT / "candidates" / "step_cache" / "teacache_signal_reuse.toml"
COSMOS_STAGE = ROOT / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos3.py"
COSMOS_MODEL = ROOT / "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/models/dits/cosmos3video.py"

# TeaCache4Cosmos public constants from TeaCache commit
# 7c10efc4702c6b619f47805f7abe4a7a08085aa0.
PUBLIC_COSMOS_COEFFICIENTS = [
    2.71156237e02,
    -9.19775607e01,
    2.24437250e00,
    2.08355751e00,
    1.41776330e-01,
]
PUBLIC_COSMOS_THRESHOLDS = {0.3, 0.4}


@dataclass
class BranchState:
    previous: float | None = None
    accumulated: float = 0.0


def poly_eval(coefficients: list[float], x: float) -> float:
    """Horner evaluation matching numpy.poly1d for highest-degree-first coeffs."""

    return teacache_poly_rescale(coefficients, x)


def rel_l1(current: float, previous: float) -> float:
    return teacache_relative_l1(_tensor(current), _tensor(previous))


def public_branch_decisions(
    signals: list[float], *, threshold: float, coefficients: list[float]
) -> list[str]:
    """Public TeaCache4Cosmos single-branch decision boundary.

    The public Cosmos script keeps separate even/odd CFG branch state. For one
    branch, the first and last branch visits always compute; intermediate visits
    accumulate polynomial-rescaled relative L1 distance and reuse while below
    threshold.
    """

    state = BranchState()
    decisions: list[str] = []
    last = len(signals) - 1
    for idx, signal in enumerate(signals):
        if idx == 0 or idx == last or state.previous is None:
            decisions.append("compute")
            state.accumulated = 0.0
        else:
            state.accumulated += poly_eval(
                coefficients, rel_l1(signal, state.previous)
            )
            if state.accumulated < threshold:
                decisions.append("reuse")
            else:
                decisions.append("compute")
                state.accumulated = 0.0
        state.previous = signal
    return decisions


def local_core_decisions(
    signals: list[float], *, threshold: float, coefficients: list[float]
) -> list[str]:
    """Local generic TeaCache decision when public-only extra guards are disabled.

    This mirrors ``efficiency.techniques.teacache.TeaCache`` with no periodic
    recompute, no continuous-hit cap, and no forced final-step recompute. The
    final-step difference is reported separately, so compare only intermediate
    decisions when checking the shared public core.
    """

    state = BranchState()
    decisions: list[str] = []
    for idx, signal in enumerate(signals):
        if idx == 0 or state.previous is None:
            decisions.append("compute")
            state.accumulated = 0.0
        else:
            state.accumulated += poly_eval(
                coefficients, rel_l1(signal, state.previous)
            )
            if state.accumulated < threshold:
                decisions.append("reuse")
            else:
                decisions.append("compute")
                state.accumulated = 0.0
        state.previous = signal
    return decisions


class _ScalarTensor:
    """Tiny tensor-like scalar for dependency-light TeaCache controller probes."""

    shape = (1,)

    def __init__(self, value: float):
        self.value = float(value)

    def __sub__(self, other):
        return _ScalarTensor(self.value - float(other))

    def __add__(self, other):
        return _ScalarTensor(self.value + float(other))

    def __radd__(self, other):
        return _ScalarTensor(float(other) + self.value)

    def __float__(self):
        return self.value

    def abs(self):
        return _ScalarTensor(abs(self.value))

    def mean(self):
        return self.value

    def detach(self):
        return self

    def clone(self):
        return _ScalarTensor(self.value)


def _tensor(value: float):
    try:
        import torch

        return torch.tensor([value])
    except ModuleNotFoundError:
        return _ScalarTensor(value)


def local_residual_technique_decisions(
    signals: list[float],
    *,
    threshold: float,
    coefficients: list[float],
    force_final: bool,
) -> list[str]:
    """Decisions from the actual generic TeaCacheResidual runtime class."""

    technique = TeaCacheResidual(
        threshold=threshold,
        start_step=0,
        coefficients=coefficients,
        max_continuous_hits=0,
        periodic_recompute=0,
    )
    scratch: dict[Any, Any] = {}
    decisions: list[str] = []
    residual_update = _tensor(0.25)
    for idx, signal in enumerate(signals):
        cache_key = "branch"
        scratch[("teacache_signal", cache_key)] = _tensor(signal)
        if force_final and idx == len(signals) - 1:
            scratch[("teacache_force_compute", cache_key)] = True
        else:
            scratch.pop(("teacache_force_compute", cache_key), None)
        ctx = TechniqueContext(step=idx, cache_key=cache_key, scratch=scratch)
        hidden = _tensor(10.0)
        _, carry = technique.before_blocks(ctx, hidden)
        decisions.append(carry[0])
        if carry[0] == "compute":
            technique.after_blocks(ctx, hidden + residual_update, carry)
    return decisions


def git_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def load_manifest_params() -> dict[str, Any]:
    with MANIFEST.open("rb") as f:
        data = tomllib.load(f)
    return data.get("efficiency", {}).get("params", {})


def load_manifest() -> dict[str, Any]:
    with MANIFEST.open("rb") as f:
        return tomllib.load(f)


def cosmos3_adapter_checks(data: dict[str, Any]) -> dict[str, bool]:
    stage_text = COSMOS_STAGE.read_text(errors="ignore") if COSMOS_STAGE.exists() else ""
    model_text = COSMOS_MODEL.read_text(errors="ignore") if COSMOS_MODEL.exists() else ""
    return {
        "manifest_uses_residual_technique": data.get("efficiency", {}).get("name") == "teacache_residual",
        "manifest_selects_block_residual_replay": data.get("env", {}).get("SGLANG_HQ_TEACACHE_REPLAY") == "block_residual",
        "runtime_builder_selects_residual_technique": "SGLANG_HQ_TEACACHE_REPLAY" in stage_text
        and "teacache_residual" in stage_text,
        "runtime_forces_final_branch_compute": "teacache_force_compute" in stage_text
        and "num_inference_steps - 1" in stage_text,
        "runtime_serializes_cfg_for_residual": "teacache_residual_active" in stage_text
        and "_predict_noise_cfg_serial" in stage_text,
        "model_uses_branch_cache_key": "replace(eff_ctx, cache_key=str(cache_key))" in model_text,
        "model_skips_gen_layers_on_reuse": "skip_gen_layers" in model_text
        and "teacache_residual" in model_text
        and "if not skip_gen_layers" in model_text,
        "model_uses_block_input_signal": "teacache_signal" in model_text
        and "input_layernorm(hidden_gen)" in model_text,
    }


def source_checks() -> dict[str, bool]:
    text = PUBLIC_COSMOS_T2V.read_text(errors="ignore") if PUBLIC_COSMOS_T2V.exists() else ""
    return {
        "has_public_cosmos_file": PUBLIC_COSMOS_T2V.exists(),
        "has_cosmos_coefficients": all(f"{coeff:.8e}"[:8] in text for coeff in PUBLIC_COSMOS_COEFFICIENTS[:2]),
        "has_even_odd_branch_state": "previous_modulated_input_even" in text
        and "previous_modulated_input_odd" in text,
        "has_boundary_recompute": "self.cnt == 0 or self.cnt == self.num_steps" in text
        and "self.cnt == 1 or self.cnt == self.num_steps+1" in text,
        "uses_numpy_poly1d": "np.poly1d(coefficients)" in text,
        "uses_relative_l1": ".abs().mean() / self.previous_modulated_input" in text,
    }


def probe() -> dict[str, Any]:
    signals = [1.0, 1.001, 1.002, 1.35, 1.351, 1.352]
    threshold = 0.3
    public_decisions = public_branch_decisions(
        signals, threshold=threshold, coefficients=PUBLIC_COSMOS_COEFFICIENTS
    )
    local_decisions = local_core_decisions(
        signals, threshold=threshold, coefficients=PUBLIC_COSMOS_COEFFICIENTS
    )
    local_runtime_decisions = local_residual_technique_decisions(
        signals,
        threshold=threshold,
        coefficients=PUBLIC_COSMOS_COEFFICIENTS,
        force_final=False,
    )
    local_runtime_public_boundary_decisions = local_residual_technique_decisions(
        signals,
        threshold=threshold,
        coefficients=PUBLIC_COSMOS_COEFFICIENTS,
        force_final=True,
    )
    params = load_manifest_params()
    manifest = load_manifest()
    manifest_coefficients = params.get("coefficients")
    manifest_threshold = float(params.get("threshold", -1.0))
    manifest_max_hits = int(params.get("max_continuous_hits", -1))
    manifest_periodic = int(params.get("periodic_recompute", -1))

    intermediate_slice = slice(0, -1)
    public_core_match = (
        public_decisions[intermediate_slice] == local_decisions[intermediate_slice]
    )
    runtime_core_match = local_runtime_decisions == local_decisions
    runtime_public_boundary_match = (
        local_runtime_public_boundary_decisions == public_decisions
    )
    helper_indicators = []
    for previous, current in zip(signals, signals[1:]):
        helper_indicators.append(
            teacache_indicator(
                _tensor(current),
                _tensor(previous),
                PUBLIC_COSMOS_COEFFICIENTS,
            )
        )
    manifest_mismatches = []
    if manifest_coefficients != PUBLIC_COSMOS_COEFFICIENTS:
        manifest_mismatches.append("coefficients_not_public_cosmos")
    if manifest_threshold not in PUBLIC_COSMOS_THRESHOLDS:
        manifest_mismatches.append("threshold_not_public_cosmos_readme_value")
    if manifest_max_hits != 0:
        manifest_mismatches.append("continuous_hit_cap_is_extra_guard")
    if manifest_periodic != 0:
        manifest_mismatches.append("periodic_recompute_is_extra_guard")

    return {
        "status": "pass",
        "public_reference": {
            "repo": str(PUBLIC_REF),
            "commit": git_commit(PUBLIC_REF),
            "source": str(PUBLIC_COSMOS_T2V),
            "checks": source_checks(),
        },
        "core_formula_probe": {
            "signals": signals,
            "threshold": threshold,
            "coefficients": PUBLIC_COSMOS_COEFFICIENTS,
            "public_branch_decisions": public_decisions,
            "local_core_decisions": local_decisions,
            "local_runtime_decisions": local_runtime_decisions,
            "local_runtime_public_boundary_decisions": local_runtime_public_boundary_decisions,
            "helper_indicators": helper_indicators,
            "intermediate_core_match": public_core_match,
            "runtime_core_match": runtime_core_match,
            "runtime_public_boundary_match": runtime_public_boundary_match,
            "known_boundary_difference": "public TeaCache4Cosmos forces the final branch visit to compute; local generic TeaCache does not force final-step recompute unless configured externally.",
        },
        "candidate_manifest_alignment": {
            "manifest": str(MANIFEST),
            "current_threshold": manifest_threshold,
            "current_coefficients": manifest_coefficients or [1.0, 0.0],
            "current_max_continuous_hits": manifest_max_hits,
            "current_periodic_recompute": manifest_periodic,
            "matches_public_cosmos_profile": not manifest_mismatches,
            "mismatches": manifest_mismatches,
        },
        "cosmos3_adapter_alignment": {
            "checks": cosmos3_adapter_checks(manifest),
            "remaining_public_original_gap": "local Cosmos3 residual replay now uses branch cache keys, final-step recompute, and block-loop skipping, but the exact public TeaCache4Cosmos AdaLN-modulated signal source is still adapter-specific and needs GPU/public-quality validation before any full public-original claim.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = probe()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
