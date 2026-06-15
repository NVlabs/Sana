#!/usr/bin/env python3
"""Model-agnostic acceleration search harness.

Given a MODEL PROFILE (models/<id>.toml) and the generic SEARCH DIMENSIONS
(loops/<dim>/dimension.toml), report which method families are available and run
lightweight compose diagnostics. This is not the search driver for subagents:
native Codex goals start from search_space/ and inspect/edit inference code
directly.

This skeleton produces a CPU-only launchability/diagnostic view. The eval +
risk-tiering (low/mid/high vs baseline) is the GPU stage; see plan_eval() stub
and docs/search-architecture.md.
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

from efficiency import compose, get_model_spec  # noqa: E402
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


def search(model_id: str, verbose: bool = True) -> list[dict]:
    prof = load_model_profile(model_id)
    spec = get_model_spec(prof["spec"])
    if spec is None:
        raise SystemExit(f"no ModelSpec registered for {prof['spec']!r}")
    caps = {c.name.lower() for c in spec.capabilities}

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
                    compose([_build(kind, name, cfg)], spec)
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
                    "candidates": len(cfgs),
                    "composable": ok,
                    "rejected": rej,
                    "eligible": True,
                    "compose_ready": ok > 0,
                    "reason": reason,
                }
            )

    if verbose:
        print(f"# acceleration search — model '{model_id}' (spec={prof['spec']}, caps={sorted(caps)})")
        if prof.get("run_script"):
            print(f"#   run_script: {prof['run_script']}")
        if not results:
            print("  (no loops/*/dimension.toml found yet)")
        for r in results:
            mark = "RUN " if r["eligible"] else "skip"
            extra = f"  ({r['rejected']} rejected: {r['reason']})" if r["rejected"] else ""
            print(f"  [{mark}] {r['dimension']}/{r['technique']}: "
                  f"{r['composable']}/{r['candidates']} compose-diagnostic{extra}")
        elig = [r for r in results if r["eligible"]]
        print(f"# {len(elig)} launchable technique-dimensions, "
              f"{sum(r['composable'] for r in elig)} composable candidates "
              f"(compose is diagnostic; eval+tiering = GPU stage, stubbed)")
        # each dimension is a BOUNDED SEARCH LOOP; tiers define the per-tier quality budgets
        print("# search loop (per dimension):")
        for dim_id, dim in load_dimensions():
            lp = dim.get("loop", {})
            if lp:
                print(f"    {dim_id}: granularity={lp.get('granularity','?')} "
                      f"max_iters={lp.get('max_iters','?')} "
                      f"early_stop={lp.get('early_stop_patience','?')} keep={lp.get('keep','?')}")
        tiers = load_tiers()
        if tiers:
            names = [t for t in tiers if t not in ("targets", "provider")]
            print(f"# risk tiers: {names}  composed targets: {tiers.get('targets', {})}")
    return results


def plan_eval(model_id: str):  # noqa: D401
    """STUB (GPU stage): run each dimension's BOUNDED SEARCH LOOP.

    Per dimension (its [loop]): up to max_iters iterations (early-stop after
    early_stop_patience with no Pareto improvement). Each iteration picks a config
    from search_space/ plus model traces/code, records it in a candidate
    manifest, renders a run bundle from the model profile + cfg, launches via
    scripts/launch_candidate.py, collects
    benchmark.json/quality.json, and compares latency + peak_mem + quality vs the
    profile [baseline]. A candidate is kept only if it beats baseline on latency
    OR peak_mem AND meets a tier's quality budget (evals/tiers.toml); it is binned
    into the loosest tier it satisfies, keeping the best (latency, peak_mem) config
    per tier. The integration stage then stacks per-tier dimension winners into the
    final low/medium/high profiles (composed targets in tiers.toml [targets]).
    Not run here (no GPU). See docs/search-architecture.md."""
    raise NotImplementedError("eval+tiering is the GPU stage; see docs/search-architecture.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="cosmos3", help="model id under models/<id>.toml")
    args = ap.parse_args()
    search(args.model)
