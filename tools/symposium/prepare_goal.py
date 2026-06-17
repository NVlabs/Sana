#!/usr/bin/env python3
"""Create a Symposium/Codex interactive goal bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sanitize(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    out = "".join(ch if ch in allowed else "-" for ch in value.strip())
    return out.strip("-") or "goal"


SANA_PYTHON = "/home/haozhel/lustre/miniconda3/envs/sana/bin/python"
CANONICAL_BASELINE_FRAMES = (
    "/home/haozhel/lustre/auto-video/runs/20260613-175619-baseline/outputs/frames"
)

FANOUT_LOOP_CONTRACT = f"""This is a bounded per-dimension search loop, not a one-candidate target.

Each loop iteration:

1. Observe prior state: read `SEARCH_JOURNAL.md`, current `best_per_tier`,
   rejected failure signatures, and the canonical baseline.
2. Propose the next hypothesis before implementation: what mechanism changes,
   why it should improve over the previous loop, what recorded failure it avoids,
   and what evidence would reject it.
3. Implement exactly one candidate and one manifest. Do not batch unrelated
   mechanisms into one candidate.
4. Preflight with static/unit checks, dry-run rendering, and OFF identity when
   the dimension has an inactive path.
5. Launch through Slurm only after preflight passes.
6. Run the authoritative gate with:
   `{SANA_PYTHON} search/plan_eval.py --assess <run_dir> --baseline-frames {CANONICAL_BASELINE_FRAMES}`
7. Decide:
   - promoted candidate: update `best_per_tier`, record artifacts, then loop;
   - rejected candidate: record a failure signature, then loop with a
     meaningfully different hypothesis;
   - blocker: record the real external dependency and stop;
   - structured negative: stop only after evidence covers the meaningful
     mechanism space.

Stop only at `max_iters`, `early_stop_patience`, a real blocker, structured
negative evidence, or explicit orchestrator release. Collector `quality.json` is
telemetry; promotion authority is OFF identity plus aligned LPIPS plus aligned
pairwise Gemini on canonical baseline frames."""

INTEGRATION_LOOP_CONTRACT = f"""This is the fan-in integration loop. It starts only after
selected fan-out dimensions are terminal, and it is required before the overall
experiment can be called complete.

Each integration iteration:

1. Read every selected dimension's `AGENT-STATUS.json`, `SUMMARY.md`,
   `SEARCH_JOURNAL.md`, candidate manifests, and run artifacts.
2. Reconcile stale status with durable run artifacts, recording which source of
   truth was used for each tier winner.
3. Build one tier plan at a time from eligible per-dimension winners. A winner is
   eligible only for the tier it passed, or a looser tier. Empty tiers must get an
   explicit `no_eligible_profile` blocker.
4. Implement exactly one composed profile in the integration worktree. Preserve
   each component's OFF guard and feature flag, resolve shared-file conflicts,
   and do not quote composed speedup from single-dimension timings.
5. Preflight, launch, collect, and run the authoritative composed gate with:
   `{SANA_PYTHON} search/plan_eval.py --assess <run_dir> --baseline-frames {CANONICAL_BASELINE_FRAMES}`
6. Decide:
   - passed composition: keep it as the tier incumbent, record artifacts, then
     continue searching for a faster compatible composition if budget remains;
   - failed composition: record an interaction failure signature, then loop with
     a repaired merge, reduced subset, or different tier plan;
   - tier blocker: record the real reason no composed profile can be produced;
   - global blocker: record the external dependency and stop.

Stop only when every low/medium/high tier has either a gated composed profile or
an explicit blocker, or when `max_iters`, `early_stop_patience`, or a real
external blocker is reached. A failed composed gate never completes integration
by itself."""


def read_search_space_summary(root: Path, search_space_root: str) -> tuple[str, str]:
    space_path = (root / search_space_root).resolve()
    if not space_path.exists():
        raise SystemExit(
            f"Search-space root is missing: {space_path}. "
            "Import or restore `search_space/` before generating subagent goals."
        )
    rel_space = str(space_path.relative_to(root))
    readme = space_path / "README.md"
    if readme.exists():
        lines = readme.read_text().splitlines()
        summary = "\n".join(lines[:80]).strip()
    else:
        summary = "Search-space README is missing; inspect the method-family files directly."
    return rel_space, summary.replace("```", "~~~")


def implementation_loop_acceptance() -> list[str]:
    return [
        "run the bounded fan-out loop; do not stop after a single candidate success or failure",
        "write a hypothesis before each candidate explaining the expected improvement and the prior failure it avoids",
        "record each rejected candidate in `SEARCH_JOURNAL.md` with a failure signature and next-hypothesis requirement",
        "after a reject, generate a meaningfully different hypothesis instead of repeating the same mechanism with cosmetic parameters",
        "keep promoted candidates in `best_per_tier` and continue searching until max_iters, early_stop, structured negative, real blocker, or orchestrator release",
        "use the authoritative `sana` `search/plan_eval.py --assess` gate with canonical baseline frames for promotion decisions",
        "treat collector `quality.json` Gemini as telemetry, not promotion authority, when it contradicts aligned LPIPS or aligned pairwise Gemini",
    ]


def integration_acceptance() -> list[str]:
    return [
        "read and reconcile every selected dimension's AGENT-STATUS.json, SUMMARY.md, SEARCH_JOURNAL.md, manifests, and run artifacts",
        "build explicit low/medium/high tier plans from eligible per-dimension winners, using `no_eligible_profile` blockers for empty tiers",
        "implement exactly one composed profile per integration iteration, preserving component OFF guards and feature flags",
        "never report composed speedup or quality from single-dimension runs; launch and gate the merged profile itself",
        "run the authoritative `sana` `search/plan_eval.py --assess` gate with canonical baseline frames for every composed profile",
        "if a composed gate fails, record an interaction failure signature and loop with a repaired merge, reduced subset, or different tier plan",
        "write INTEGRATION-STATUS.json, INTEGRATION-JOURNAL.md, composed manifests, run artifacts, per-tier blockers, and a release matrix",
        "finish only when every low/medium/high tier has a gated composed profile or an explicit blocker",
    ]


def dimension_acceptance(dimension: str, role: str) -> list[str]:
    if role == "integration":
        return integration_acceptance()

    if role == "gate":
        return [
            "review the implementation diff and candidate manifest without authoring implementation code",
            "reproduce or inspect the run artifacts required by the candidate contract",
            "write a structured gate verdict JSON covering OFF identity, pixel metrics, LPIPS, Gemini, timing, and tier eligibility",
            "mark the candidate rejected when any required quality artifact is missing, deferred, unavailable, or prose-only",
            "write `SUMMARY.md` with the verdict, evidence paths, and any non-reproducible gaps",
        ]

    common = implementation_loop_acceptance()

    if dimension in {"step_cache", "cache", "caching"}:
        return common + [
            "inspect the cache method families in `search_space/01_cache.md` before proposing implementations",
            "identify at least three caching mechanisms, including TeaCache-style signal reuse, whole-step reuse, block/residual reuse, and attention/KV/output reuse",
            "derive per-layer, per-step, signal, threshold, fallback, and schedule choices from Cosmos3 inference code or traces rather than predefined constants",
            "modify the inference code directly when that is the shortest path to a runnable candidate; do not wait for a predeclared seam",
            "prove OFF identity against the baseline path before reporting speedup",
            "produce at least one runnable candidate manifest or a structured negative result explaining why no candidate is safe",
            "write exact reproduction commands, changed files, run artifacts, and current structured quality-gate status",
        ]
    if dimension == "token_prune":
        return common + [
            "inspect `search_space/02_token_pruning.md` and then inspect Cosmos3 token layout directly in inference code",
            "derive prunable spans, salience signals, compensation policy, layer windows, and step windows from code/traces",
            "prove gather/scatter or masking keeps positional tensors, attention masks, and output restoration aligned",
            "modify the inference code directly when needed; do not require a predeclared prunable-token seam",
            "prove OFF identity against the baseline path before reporting speedup",
            "produce at least one runnable candidate manifest or a structured negative result",
            "write exact reproduction commands, changed files, run artifacts, and current structured quality-gate status",
        ]
    if dimension == "sparse_attention":
        return common + [
            "inspect `search_space/04_sparse_attention.md` before proposing implementations",
            "analyze Cosmos3 self-attention, cross-attention, and joint/GEN attention paths separately",
            "derive per-layer, per-step, per-attention-type routing, dense fallback, block size, density, and sparsity policy from traces or code inspection",
            "modify the inference code directly when needed; do not require a predeclared swappable-attention seam",
            "prove OFF identity and verify dense fallback behavior before reporting speedup",
            "produce at least one runnable candidate manifest or a structured negative result",
            "write exact reproduction commands, changed files, run artifacts, and current structured quality-gate status",
        ]
    if dimension == "nvfp4_ffn":
        return common + [
            "inspect `search_space/03_quantization.md` before proposing implementations",
            "profile or inspect Cosmos3 FFN/linear modules to choose per-module, per-layer, and per-step precision scope and dense guards",
            "record hardware/library prerequisites and fallback policy explicitly",
            "modify the inference/loading code directly when needed; do not require a predeclared precision seam",
            "prove OFF identity against the BF16 baseline path before reporting speedup",
            "produce at least one runnable candidate manifest or a structured negative result",
            "write exact reproduction commands, changed files, run artifacts, and current structured quality-gate status",
        ]
    if dimension == "kwl_fusion":
        return common + [
            "inspect `search_space/05_kernel_fusion.md` before proposing implementations",
            "profile or inspect Cosmos3 hot ops and choose exact or numerically equivalent fusion candidates from evidence",
            "separate lossless operator fusions from approximate techniques and record expected numeric tolerance",
            "modify the inference/build code directly when needed; do not require a predeclared kernel-fusion seam",
            "prove OFF identity and report warm/cold compile state before reporting speedup",
            "produce at least one runnable candidate manifest or a structured negative result",
            "write exact reproduction commands, changed files, run artifacts, and current structured quality-gate status",
        ]
    return common + [
        "inspect the relevant method family in `search_space/` before proposing implementations",
        "derive all model-specific knobs from traces or code inspection rather than predefined constants",
        "modify inference code directly when needed; do not wait for a predeclared interface",
        "keep implementation work in the isolated worktree and declared write scope",
        "produce at least one runnable candidate manifest or a structured negative result",
        "write exact reproduction commands, changed files, run artifacts, and current structured quality-gate status",
    ]


def resolved_write_scope(args: argparse.Namespace) -> list[str]:
    if args.write_scope:
        return args.write_scope
    if args.role == "integration":
        return [
            "Sol-LTX-Infer/",
            "candidates/",
            "integration/",
            "search/",
            "scripts/",
            "evals/",
        ]
    return ["Sol-LTX-Infer/", "candidates/", "loops/", "search/", "scripts/"]


def required_artifacts(role: str) -> list[str]:
    if role == "integration":
        return [
            "`INTEGRATION-STATUS.json`",
            "`INTEGRATION-JOURNAL.md`",
            "composed low/medium/high manifests or explicit per-tier blockers",
            "run bundle artifacts for every launched composed profile",
            "interaction failure signatures for every rejected composition",
            "release matrix separating per-dimension winners from gated composed profiles",
        ]
    return [
        "`AGENT-STATUS.json`",
        "`SEARCH_JOURNAL.md`",
        "candidate manifest or structured negative-result note",
        "run bundle artifacts when a candidate is launched",
        "failure signatures for every rejected candidate",
        "`SUMMARY.md`",
    ]


def render_goal_md(
    args: argparse.Namespace,
    candidate_rel: str,
    search_space_rel: str,
    search_space_summary: str,
) -> str:
    write_scope = resolved_write_scope(args)
    acceptance = "\n".join(f"- {item}" for item in dimension_acceptance(args.dimension, args.role))
    scope = "\n".join(f"- `{item}`" for item in write_scope)
    artifacts = "\n".join(f"- {item}" for item in required_artifacts(args.role))
    contract_heading = "Fan-In Integration Contract" if args.role == "integration" else "Fan-Out Loop Contract"
    contract_text = INTEGRATION_LOOP_CONTRACT if args.role == "integration" else FANOUT_LOOP_CONTRACT
    return f"""# Goal: {args.goal_id}

You are working in an isolated autovideo goal context.

## Role

`{args.role}`

## Objective

{args.objective}

## Search Space Start

Start from the method-family search space, then inspect and modify Cosmos3
inference code directly:

- Search-space root: `{search_space_rel}`
- Relevant dimension: `{args.dimension}`

Search-space summary:

```text
{search_space_summary}
```

Do not use historical recipe archives or fixed grids as startup context.
If `search_space/` is missing or unclear, stop exploration and ask the main
orchestration agent to repair the search-space contract.

## {contract_heading}

Read `docs/fanout-loop-contract.md`. The operational summary for this goal is:

{contract_text}

## Model And Runtime Context

- Target model profile: `models/cosmos3.toml`
- Execution repo: `Sol-LTX-Infer/`
- Primary implementation surface: inspect and modify the model inference path
  under `Sol-LTX-Infer/` directly in this isolated worktree.
- Launcher: `python3 scripts/launch_candidate.py {candidate_rel} --mode dry-run`
- Collector: `python3 scripts/collect_run.py runs/<run-id>`
- Authoritative assess: `{SANA_PYTHON} search/plan_eval.py --assess <run_dir> --baseline-frames {CANONICAL_BASELINE_FRAMES}`
- Quality source of truth: OFF identity, aligned LPIPS, and aligned pairwise
  Gemini from the authoritative gate. `outputs/quality.json` is telemetry and
  not promotion authority when it contradicts aligned gate artifacts.

## Allowed Worktree Scope

{scope}

## Required Artifacts

{artifacts}

## Symposium Step

Use Symposium `interview-harness` first if the objective is still ambiguous.
Produce a final Seed with:

- goal
- constraints
- acceptance criteria
- ontology boundary

## Candidate Contract

- Candidate manifest: `{candidate_rel}`
- Launch dry-run: `python3 scripts/launch_candidate.py {candidate_rel} --mode dry-run`
- Collect: `python3 scripts/collect_run.py runs/<run-id>`
- Assess: `{SANA_PYTHON} search/plan_eval.py --assess <run_dir> --baseline-frames {CANONICAL_BASELINE_FRAMES}`

## Branching Contract

- Root branch: `{args.root_branch}`
- Submodule branch: `{args.submodule_branch}`

## Done When

{acceptance}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--role", choices=("implementation", "gate", "integration"), default="implementation")
    parser.add_argument("--dimension", default="general")
    parser.add_argument("--search-space-root", default="search_space")
    parser.add_argument("--write-scope", action="append", default=[])
    parser.add_argument("--root-branch", default="")
    parser.add_argument("--submodule-branch", default="")
    parser.add_argument("--goals-root", default="goals")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = project_root()
    goal_id = sanitize(args.goal_id)
    args.goal_id = goal_id
    if not args.root_branch:
        args.root_branch = f"codex/{goal_id}"
    if not args.submodule_branch:
        args.submodule_branch = f"codex/{goal_id}-sol"

    candidate = (root / args.candidate).resolve()
    if not candidate.exists():
        raise SystemExit(f"Candidate manifest does not exist: {candidate}")
    candidate_rel = str(candidate.relative_to(root))
    search_space_rel, search_space_summary = read_search_space_summary(root, args.search_space_root)

    goal_dir = (root / args.goals_root / goal_id).resolve()
    if goal_dir.exists() and not args.overwrite:
        raise SystemExit(f"Goal already exists: {goal_dir} (use --overwrite)")
    goal_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(candidate, goal_dir / "candidate.toml")
    (goal_dir / "goal.md").write_text(
        render_goal_md(args, candidate_rel, search_space_rel, search_space_summary)
    )
    context = {
        "goal_id": goal_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": "tools/symposium/prepare_goal.py",
        "target_agent": "codex",
        "mode": "interactive-goal",
        "candidate_manifest": candidate_rel,
        "role": args.role,
        "dimension": args.dimension,
        "search_space_root": search_space_rel,
        "write_scope": resolved_write_scope(args),
        "acceptance_criteria": dimension_acceptance(args.dimension, args.role),
        "loop_contract": {
            "kind": "fan_in_integration_loop" if args.role == "integration" else "bounded_fanout_search_loop",
            "max_iters": 20,
            "early_stop_patience": 5,
            "failed_candidate_action": "record_interaction_failure_and_loop"
            if args.role == "integration"
            else "reject_log_and_loop",
            "successful_candidate_action": "keep_composed_tier_incumbent_and_loop"
            if args.role == "integration"
            else "keep_best_per_tier_and_loop",
            "authoritative_python": SANA_PYTHON,
            "canonical_baseline_frames": CANONICAL_BASELINE_FRAMES,
            "promotion_authority": [
                "off_identity",
                "aligned_lpips",
                "aligned_pairwise_gemini",
                "speed_or_memory_improvement",
            ],
            "collector_quality_json": "telemetry_not_promotion_authority_when_contradicted",
            "global_done_requires_integration": True,
        },
        "root_branch": args.root_branch,
        "submodule_branch": args.submodule_branch,
        "objective": args.objective,
        "symposium": {
            "skill": "interview-harness",
            "vendor": "tools/symposium/vendor/Symposium",
        },
    }
    (goal_dir / "context.json").write_text(json.dumps(context, indent=2, sort_keys=True) + "\n")
    print(goal_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
