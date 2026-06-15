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


def dimension_acceptance(dimension: str, role: str) -> list[str]:
    if role == "gate":
        return [
            "review the implementation diff and candidate manifest without authoring implementation code",
            "reproduce or inspect the run artifacts required by the candidate contract",
            "write a structured gate verdict JSON covering OFF identity, pixel metrics, LPIPS, Gemini, timing, and tier eligibility",
            "mark the candidate rejected when any required quality artifact is missing, deferred, unavailable, or prose-only",
            "write `SUMMARY.md` with the verdict, evidence paths, and any non-reproducible gaps",
        ]

    if dimension in {"step_cache", "cache", "caching"}:
        return [
            "inspect the cache method families in `search_space/01_cache.md` before proposing implementations",
            "identify at least three caching mechanisms, including TeaCache-style signal reuse, whole-step reuse, block/residual reuse, and attention/KV/output reuse",
            "derive per-layer, per-step, signal, threshold, fallback, and schedule choices from Cosmos3 inference code or traces rather than predefined constants",
            "modify the inference code directly when that is the shortest path to a runnable candidate; do not wait for a predeclared seam",
            "prove OFF identity against the baseline path before reporting speedup",
            "produce at least one runnable candidate manifest or a structured negative result explaining why no candidate is safe",
            "write exact reproduction commands, changed files, run artifacts, and current structured quality-gate status",
        ]
    if dimension == "token_prune":
        return [
            "inspect `search_space/02_token_pruning.md` and then inspect Cosmos3 token layout directly in inference code",
            "derive prunable spans, salience signals, compensation policy, layer windows, and step windows from code/traces",
            "prove gather/scatter or masking keeps positional tensors, attention masks, and output restoration aligned",
            "modify the inference code directly when needed; do not require a predeclared prunable-token seam",
            "prove OFF identity against the baseline path before reporting speedup",
            "produce at least one runnable candidate manifest or a structured negative result",
            "write exact reproduction commands, changed files, run artifacts, and current structured quality-gate status",
        ]
    if dimension == "sparse_attention":
        return [
            "inspect `search_space/04_sparse_attention.md` before proposing implementations",
            "analyze Cosmos3 self-attention, cross-attention, and joint/GEN attention paths separately",
            "derive per-layer, per-step, per-attention-type routing, dense fallback, block size, density, and sparsity policy from traces or code inspection",
            "modify the inference code directly when needed; do not require a predeclared swappable-attention seam",
            "prove OFF identity and verify dense fallback behavior before reporting speedup",
            "produce at least one runnable candidate manifest or a structured negative result",
            "write exact reproduction commands, changed files, run artifacts, and current structured quality-gate status",
        ]
    if dimension == "nvfp4_ffn":
        return [
            "inspect `search_space/03_quantization.md` before proposing implementations",
            "profile or inspect Cosmos3 FFN/linear modules to choose per-module, per-layer, and per-step precision scope and dense guards",
            "record hardware/library prerequisites and fallback policy explicitly",
            "modify the inference/loading code directly when needed; do not require a predeclared precision seam",
            "prove OFF identity against the BF16 baseline path before reporting speedup",
            "produce at least one runnable candidate manifest or a structured negative result",
            "write exact reproduction commands, changed files, run artifacts, and current structured quality-gate status",
        ]
    if dimension == "kwl_fusion":
        return [
            "inspect `search_space/05_kernel_fusion.md` before proposing implementations",
            "profile or inspect Cosmos3 hot ops and choose exact or numerically equivalent fusion candidates from evidence",
            "separate lossless operator fusions from approximate techniques and record expected numeric tolerance",
            "modify the inference/build code directly when needed; do not require a predeclared kernel-fusion seam",
            "prove OFF identity and report warm/cold compile state before reporting speedup",
            "produce at least one runnable candidate manifest or a structured negative result",
            "write exact reproduction commands, changed files, run artifacts, and current structured quality-gate status",
        ]
    return [
        "inspect the relevant method family in `search_space/` before proposing implementations",
        "derive all model-specific knobs from traces or code inspection rather than predefined constants",
        "modify inference code directly when needed; do not wait for a predeclared interface",
        "keep implementation work in the isolated worktree and declared write scope",
        "produce at least one runnable candidate manifest or a structured negative result",
        "write exact reproduction commands, changed files, run artifacts, and current structured quality-gate status",
    ]


def resolved_write_scope(args: argparse.Namespace) -> list[str]:
    return args.write_scope or ["Sol-LTX-Infer/", "candidates/", "loops/", "search/", "scripts/"]


def render_goal_md(
    args: argparse.Namespace,
    candidate_rel: str,
    search_space_rel: str,
    search_space_summary: str,
) -> str:
    write_scope = resolved_write_scope(args)
    acceptance = "\n".join(f"- {item}" for item in dimension_acceptance(args.dimension, args.role))
    scope = "\n".join(f"- `{item}`" for item in write_scope)
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

## Model And Runtime Context

- Target model profile: `models/cosmos3.toml`
- Execution repo: `Sol-LTX-Infer/`
- Primary implementation surface: inspect and modify the model inference path
  under `Sol-LTX-Infer/` directly in this isolated worktree.
- Launcher: `python3 scripts/launch_candidate.py {candidate_rel} --mode dry-run`
- Collector: `python3 scripts/collect_run.py runs/<run-id>`
- Quality source of truth: `outputs/quality.json` and gate verdict JSON, never prose-only logs.

## Allowed Worktree Scope

{scope}

## Required Artifacts

- `AGENT-STATUS.json`
- `SEARCH_JOURNAL.md`
- candidate manifest or structured negative-result note
- run bundle artifacts when a candidate is launched
- `SUMMARY.md`

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
    parser.add_argument("--role", choices=("implementation", "gate"), default="implementation")
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
