#!/usr/bin/env python3
"""Model-agnostic acceleration search harness.

Given a MODEL PROFILE (models/<id>.toml) and the generic SEARCH DIMENSIONS
(loops/<dim>/dimension.toml), report which method families are available and run
lightweight compose diagnostics. This is not the search driver for subagents:
native Codex goals start from search_space/ and inspect/edit inference code
directly.

This skeleton produces a CPU-only launchability/diagnostic view. The eval +
speed-target selection (1.5/2/3x vs baseline) is the GPU stage; see plan_eval()
and the development notes, which are not published.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # py<3.11 (e.g. the sana env, 3.10) ships tomli
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from efficiency import ModelSpec, compose  # noqa: E402
from efficiency.compose import CompositionError  # noqa: E402
from efficiency.registry import build_technique, build_transform  # noqa: E402


def _load_toml(p: Path) -> dict:
    with open(p, "rb") as f:
        return tomllib.load(f)


def load_model_profile(model_id: str) -> dict:
    p = REPO / "models" / f"{model_id}.toml"
    if not p.exists():
        raise SystemExit(f"no model profile: {p} (see models/README.md)")
    return _load_toml(p)


def load_dimensions() -> list[tuple[str, dict]]:
    return [
        (p.parent.name, _load_toml(p))
        for p in sorted((REPO / "loops").glob("*/dimension.toml"))
    ]


def load_tiers() -> dict:
    p = REPO / "evals" / "tiers.toml"
    return _load_toml(p) if p.exists() else {}


def _grid(space: dict) -> list[dict]:
    """Cartesian product of a {param: [values]} search space."""
    if not space:
        return [{}]
    keys = list(space)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(space[k] for k in keys))]


def _build(kind: str, name: str, params: dict):
    return build_transform(name, **params) if kind == "build_transform" else build_technique(name, **params)


def _baseline_tier_counts(method_baselines: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in method_baselines:
        tier = item.get("tier", "unknown")
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def search(model_id: str, verbose: bool = True) -> list[dict]:
    prof = load_model_profile(model_id)

    results = []
    for dim_id, dim in load_dimensions():
        kind = dim.get("kind", "runtime_technique")
        for tech in dim.get("technique", []):
            name = tech["name"]
            cfgs = _grid(tech.get("search_space", {}))
            ok = rej = 0
            reason = ""
            for cfg in cfgs:
                try:
                    item = _build(kind, name, cfg)
                    spec = ModelSpec(
                        name=str(prof.get("spec") or model_id),
                        capabilities=getattr(item, "required_capabilities", frozenset()),
                    )
                    compose([item], spec)
                    ok += 1
                except CompositionError as e:
                    rej += 1
                    reason = str(e).splitlines()[0]
                except Exception as e:  # bad params etc. — surface, don't crash the sweep
                    rej += 1
                    reason = f"{type(e).__name__}: {e}"
            results.append(
                {
                    "dimension": dim_id,
                    "technique": name,
                    "config": len(cfgs),
                    "composable": ok,
                    "rejected": rej,
                    "eligible": True,
                    "compose_ready": ok > 0,
                    "method_baselines": dim.get("method_baseline", []),
                    "reason": reason,
                }
            )

    if verbose:
        print(
            f"# acceleration search — model '{model_id}' "
            f"(target={prof.get('spec', model_id)}, spec_source=technique_required_capabilities)"
        )
        if prof.get("run_script"):
            print(f"#   run_script: {prof['run_script']}")
        if not results:
            print("  (no loops/*/dimension.toml found yet)")
        for r in results:
            mark = "RUN " if r["eligible"] else "skip"
            extra = f"  ({r['rejected']} rejected: {r['reason']})" if r["rejected"] else ""
            print(f"  [{mark}] {r['dimension']}/{r['technique']}: "
                  f"{r['composable']}/{r['config']} compose-diagnostic{extra}")
        elig = [r for r in results if r["eligible"]]
        print(f"# {len(elig)} launchable technique-dimensions, "
              f"{sum(r['composable'] for r in elig)} composable config "
              f"(compose is diagnostic; eval+speed-target selection = GPU stage, stubbed)")
        # each dimension is a fixed-budget frontier loop; tiers are selected after fan-out
        print("# search loop (per dimension):")
        for dim_id, dim in load_dimensions():
            lp = dim.get("loop", {})
            if lp:
                print(f"    {dim_id}: granularity={lp.get('granularity','?')} "
                      f"max_iters={lp.get('max_iters','?')} "
                      f"early_stop={lp.get('early_stop_patience','?')} keep={lp.get('keep','?')}")
            baselines = dim.get("method_baseline", [])
            if baselines:
                counts = _baseline_tier_counts(baselines)
                counts_text = ", ".join(f"{tier}={count}" for tier, count in sorted(counts.items()))
                ids = ", ".join(item.get("id", "unknown") for item in baselines)
                print(f"      method_baselines: {counts_text} [{ids}]")
        tiers = load_tiers()
        if tiers:
            names = [t for t in tiers if t not in ("targets", "provider", "quality_ranking")]
            print(f"# speed-target tiers: {names}  composed targets: {tiers.get('targets', {})}")
    return results


def plan_eval(model_id: str):  # noqa: D401
    """STUB (GPU stage): run each dimension's BOUNDED SEARCH LOOP.

    Per dimension (its [loop]): run the fixed max_iters frontier budget unless a
    real blocker or explicit orchestrator release applies. Structured-negative
    proposals are logged but do not stop the default fixed-budget loop. Each
    iteration picks a hypothesis from search_space/ plus model traces/code,
    records it in a config manifest, renders a run bundle from the model
    profile + cfg, launches via scripts/launch_config.py, collects
    benchmark.json/quality.json, and compares latency + peak_mem + quality vs the
    profile [baseline]. A config is retained when quality improves OR
    speed/memory improves; it is discarded only when neither quality nor
    speed/memory improves. After the budget closes, the main agent selects low/medium/high
    speed-target winners from the retained frontier by joint Gemini+LPIPS quality
    ranking, then integration stacks those winners into final profiles (composed
    targets in tiers.toml [targets]).
    Not run here (no GPU); it is the GPU stage."""
    raise NotImplementedError("eval+tiering is the GPU stage")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="cosmos3", help="model id under models/<id>.toml")
    args = ap.parse_args()
    search(args.model)
