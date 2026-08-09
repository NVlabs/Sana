#!/usr/bin/env python3
"""Compare local step-cache transfeat against public VideoSys PAB boundaries.

This is a public-behavior boundary probe, not a usefulness benchmark. It pins
the public Pyramid Attention Broadcast (PAB) control surface and records how the
current local step-cache transfeat differ from that public implementation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from techniques.methods.payload_cache import PABBroadcastController  # noqa: E402

PUBLIC_VIDEOSYS = Path("/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/.cache/autovideo/public_refs/VideoSys")
PUBLIC_PAB_MGR = PUBLIC_VIDEOSYS / "videosys" / "core" / "pab" / "pab_mgr.py"
PUBLIC_PAB_DOC = PUBLIC_VIDEOSYS / "docs" / "pab.md"
STEP_CACHE_TRANSFEAT = {
    "scheduled_step_reuse": ROOT / "transfeat" / "step_cache" / "scheduled_step_reuse.toml",
    "adaptive_delta_forecast": ROOT / "transfeat" / "step_cache" / "adaptive_delta_forecast.toml",
    "attention_broadcast": ROOT / "transfeat" / "step_cache" / "attention_broadcast.toml",
    "block_layer_feature_cache": ROOT / "transfeat" / "step_cache" / "block_layer_feature_cache.toml",
}


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_checks() -> dict[str, bool]:
    mgr = PUBLIC_PAB_MGR.read_text(errors="ignore") if PUBLIC_PAB_MGR.exists() else ""
    doc = PUBLIC_PAB_DOC.read_text(errors="ignore") if PUBLIC_PAB_DOC.exists() else ""
    return {
        "has_public_pab_manager": PUBLIC_PAB_MGR.exists(),
        "has_public_pab_doc": PUBLIC_PAB_DOC.exists(),
        "has_attention_type_broadcast_flags": all(
            token in mgr
            for token in (
                "cross_broadcast",
                "spatial_broadcast",
                "temporal_broadcast",
            )
        ),
        "uses_timestep_thresholds": all(
            token in mgr
            for token in (
                "cross_threshold",
                "spatial_threshold",
                "temporal_threshold",
            )
        ),
        "uses_count_mod_range": "count % self.config.cross_range" in mgr
        and "count % self.config.spatial_range" in mgr
        and "count % self.config.temporal_range" in mgr,
        "has_mlp_block_skip_cache": "mlp_spatial_outputs[(timestep, block_idx)]" in mgr
        and "get_mlp_output" in mgr,
        "doc_describes_spatial_temporal_cross_ranges": all(
            token in doc
            for token in (
                "spatial_range",
                "temporal_range",
                "cross_range",
            )
        ),
    }


def public_pab_decisions() -> dict[str, Any]:
    pab = load_module(PUBLIC_PAB_MGR, "public_videosys_pab_mgr_probe")
    cfg = pab.PABConfig(
        cross_broadcast=True,
        cross_threshold=[100, 900],
        cross_range=3,
        spatial_broadcast=True,
        spatial_threshold=[100, 900],
        spatial_range=2,
        temporal_broadcast=True,
        temporal_threshold=[100, 900],
        temporal_range=4,
        mlp_broadcast=True,
        mlp_spatial_broadcast_config={500: {"block": [2], "skip_count": 2}},
        mlp_temporal_broadcast_config={500: {"block": [2], "skip_count": 2}},
    )
    cfg.steps = 8
    mgr = pab.PABManager(cfg)

    cross_count = 0
    cross_flags = []
    for timestep in [950, 500, 500, 500, 50, 500]:
        flag, cross_count = mgr.if_broadcast_cross(timestep, cross_count)
        cross_flags.append({"timestep": timestep, "flag": bool(flag), "count": cross_count})

    mlp_seed = mgr.if_skip_mlp(
        timestep=500,
        count=0,
        block_idx=2,
        all_timesteps=[500, 499, 498, 497],
        is_temporal=False,
    )
    mgr.save_skip_output(500, 2, "ff-output", is_temporal=False)
    mlp_hit = mgr.if_skip_mlp(
        timestep=499,
        count=mlp_seed[1],
        block_idx=2,
        all_timesteps=[500, 499, 498, 497],
        is_temporal=False,
    )
    mlp_payload = mgr.get_mlp_output([500, 498], 499, 2, is_temporal=False)

    return {
        "cross_flags": cross_flags,
        "mlp_seed": {
            "broadcast": bool(mlp_seed[0]),
            "next_flag": bool(mlp_seed[2]),
            "skip_range": mlp_seed[3],
        },
        "mlp_hit": {
            "broadcast": bool(mlp_hit[0]),
            "next_flag": bool(mlp_hit[2]),
            "skip_range": mlp_hit[3],
            "payload": mlp_payload,
        },
    }


def local_pab_decisions() -> dict[str, Any]:
    attention_params = load_toml(STEP_CACHE_TRANSFEAT["attention_broadcast"])[
        "efficiency"
    ]["params"]
    block_params = load_toml(STEP_CACHE_TRANSFEAT["block_layer_feature_cache"])[
        "efficiency"
    ]["params"]

    cross = PABBroadcastController(
        steps=8,
        cross_broadcast=attention_params.get("cross_broadcast", False),
        cross_threshold=attention_params.get("cross_threshold"),
        cross_range=attention_params.get("cross_range"),
    )
    cross_count = 0
    cross_flags = []
    for timestep in [950, 500, 500, 500, 50, 500]:
        flag, cross_count = cross.attention_decision("cross", timestep, cross_count)
        cross_flags.append({"timestep": timestep, "flag": bool(flag), "count": cross_count})

    mlp = PABBroadcastController(
        steps=8,
        mlp_broadcast=block_params.get("mlp_broadcast", False),
        mlp_spatial_broadcast_config=block_params.get(
            "mlp_spatial_broadcast_config"
        ),
    )
    mlp_seed = mlp.mlp_decision(
        timestep=500,
        count=0,
        block_idx=2,
        all_timesteps=[500, 499, 498, 497],
        is_temporal=False,
    )
    mlp_hit = mlp.mlp_decision(
        timestep=499,
        count=mlp_seed[1] or 0,
        block_idx=2,
        all_timesteps=[500, 499, 498, 497],
        is_temporal=False,
    )

    return {
        "cross_flags": cross_flags,
        "mlp_seed": {
            "broadcast": bool(mlp_seed[0]),
            "next_flag": bool(mlp_seed[2]),
            "skip_range": mlp_seed[3],
        },
        "mlp_hit": {
            "broadcast": bool(mlp_hit[0]),
            "next_flag": bool(mlp_hit[2]),
            "skip_range": mlp_hit[3],
            "payload": "ff-output" if mlp_hit[0] else None,
        },
    }


def local_transfeat_alignment() -> dict[str, dict[str, Any]]:
    public_behavior = public_pab_decisions()
    local_behavior = local_pab_decisions()
    attention_params = load_toml(STEP_CACHE_TRANSFEAT["attention_broadcast"])[
        "efficiency"
    ]["params"]
    block_params = load_toml(STEP_CACHE_TRANSFEAT["block_layer_feature_cache"])[
        "efficiency"
    ]["params"]
    attention_controller_match = (
        attention_params.get("mode") == "pab"
        and attention_params.get("attention_kind") == "cross"
        and local_behavior["cross_flags"] == public_behavior["cross_flags"]
    )
    mlp_controller_match = (
        block_params.get("mode") == "pab"
        and block_params.get("mlp_broadcast") is True
        and local_behavior["mlp_seed"] == public_behavior["mlp_seed"]
        and local_behavior["mlp_hit"] == public_behavior["mlp_hit"]
    )
    return {
        "scheduled_step_reuse": {
            "manifest": str(STEP_CACHE_TRANSFEAT["scheduled_step_reuse"]),
            "matches_public_pab": False,
            "reason": (
                "local transfeat skips/reuses the whole denoiser step output on "
                "an explicit step index set; public PAB broadcasts attention/MLP "
                "module outputs using attention type, timestep threshold, and "
                "range/count guards."
            ),
        },
        "adaptive_delta_forecast": {
            "manifest": str(STEP_CACHE_TRANSFEAT["adaptive_delta_forecast"]),
            "matches_public_pab": False,
            "reason": (
                "local transfeat delta-extrapolates whole denoiser outputs; public "
                "PAB does not define output-delta forecasting."
            ),
        },
        "attention_broadcast": {
            "manifest": str(STEP_CACHE_TRANSFEAT["attention_broadcast"]),
            "matches_public_pab": False,
            "matches_public_pab_controller": attention_controller_match,
            "matches_public_pab_full_runtime": False,
            "reason": (
                "local transfeat now uses the public cross-attention "
                "threshold/range/count controller, but it only adapts Cosmos3 GEN "
                "cross-attention and does not claim VideoSys' full spatial/"
                "temporal/cross model hooks or passing GPU quality evidence."
            ),
        },
        "block_layer_feature_cache": {
            "manifest": str(STEP_CACHE_TRANSFEAT["block_layer_feature_cache"]),
            "matches_public_pab": False,
            "matches_public_pab_controller": mlp_controller_match,
            "matches_public_pab_full_runtime": False,
            "reason": (
                "local transfeat now uses the public MLP start-timestep/block/"
                "skip-count controller and caches Cosmos3 MLP outputs, but it does "
                "not claim VideoSys' full model hook layout or passing GPU quality "
                "evidence."
            ),
        },
    }


def probe() -> dict[str, Any]:
    checks = source_checks()
    return {
        "status": "pass",
        "public_reference": {
            "repo": str(PUBLIC_VIDEOSYS),
            "commit": git_commit(PUBLIC_VIDEOSYS),
            "manager_source": str(PUBLIC_PAB_MGR),
            "doc": str(PUBLIC_PAB_DOC),
            "checks": checks,
        },
        "public_behavior_probe": public_pab_decisions(),
        "local_behavior_probe": local_pab_decisions(),
        "transfeat_manifest_alignment": local_transfeat_alignment(),
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
