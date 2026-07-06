#!/usr/bin/env python3
"""Workflow-local reviewer node."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from nodes.managed_codex import run_managed_codex
from workflow_types import NodeContext, NodeResult


REVIEWER_STATUS = "REVIEWER-STATUS.json"
RESUME_FILE = "STOP_HOOK_RESUME.md"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def rel_to(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def reviewer_goal_dir(ctx: NodeContext) -> Path:
    return Path(str(ctx.state.get("reviewer_goal_dir") or ctx.goal_dir.parent / f"{ctx.goal_dir.name}-reviewer"))


def status_result(ctx: NodeContext, status: dict[str, Any]) -> NodeResult | None:
    if status.get("status") == "accepted" and status.get("decision") == "accept":
        return NodeResult(
            "accepted",
            updates={
                "reviewer_status": status,
                "reviewer_reason": status.get("reason") or "reviewer_accepted",
            },
            artifacts=[REVIEWER_STATUS],
            message=status.get("reason") or "reviewer_accepted",
        )
    if status.get("status") == "discarded" and status.get("decision") == "discard":
        return NodeResult(
            "discarded",
            updates={
                "reviewer_status": status,
                "reviewer_reason": status.get("reason") or "reviewer_discarded",
            },
            artifacts=[REVIEWER_STATUS],
            message=status.get("reason") or "reviewer_discarded",
        )
    if status.get("status") == "needs_executor_resume":
        followups = status.get("required_followups") or []
        return NodeResult(
            "needs_executor_resume",
            updates={
                "reviewer_status": status,
                "reviewer_reason": status.get("reason") or "reviewer_requested_executor_resume",
                "reviewer_followups": followups,
            },
            artifacts=[REVIEWER_STATUS],
            message=status.get("reason") or "reviewer_requested_executor_resume",
        )
    return None


def create_reviewer_goal(ctx: NodeContext, goal_dir: Path) -> None:
    if (goal_dir / "goal.md").exists() and (goal_dir / "context.json").exists():
        return
    if ctx.dry_run:
        return
    context = read_json(ctx.goal_dir / "context.json")
    review_id = goal_dir.name
    target_id = ctx.goal_dir.name
    candidate = context.get("candidate_manifest") or "candidates/hunyuan_diffusers_baseline.toml"
    objective = (
        f"Review executor goal {target_id}: inspect implementation diff, run artifacts, "
        "single-DiT/module evaluation evidence, infra reliability, numerical drift semantics, "
        "and remaining kernel/module optimization space. Only this reviewer may request exit "
        "or discard. Full diffusion/Gemini validation is a terminal workflow gate after the "
        "reviewer requests exit; otherwise request executor resume."
    )
    cmd = [
        sys.executable,
        "tools/symposium/prepare_goal.py",
        "--goal-id",
        review_id,
        "--candidate",
        str(candidate),
        "--objective",
        objective,
        "--dimension",
        str(context.get("dimension") or "kwl_fusion"),
        "--role",
        "gate",
        "--model-id",
        str(context.get("model_id") or "hunyuan_diffusers"),
        "--run-id",
        str(context.get("run_id") or ctx.state.get("experiment_uid") or ""),
        "--root-branch",
        str(context.get("root_branch") or ""),
        "--submodule-branch",
        str(context.get("submodule_branch") or ""),
        "--goals-root",
        rel_to(ctx.worktree, ctx.goal_dir.parent),
        "--overwrite",
    ]
    proc = subprocess.run(cmd, cwd=ctx.worktree, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "prepare reviewer goal failed")

    extra = f"""

## KR Reviewer Workflow Contract

You are the only role allowed to request workflow exit or discard this executor
workflow. The executor may not make final discard decisions. Inspect:

- target executor goal: `{target_id}`
- executor status: `AGENT-STATUS.json`
- executor journal and summary: `SEARCH_JOURNAL.md`, `SUMMARY.md`
- implementation diff and candidate manifests
- single-DiT/module-level artifacts such as `gate_assess.json`, `dit_eval.json`,
  and `microbench.json`

### Loop And Terminal Evaluation

Inside the normal loop, do not require full denoising/full diffusion evidence.
The workflow runner only requires smooth single-DiT/module-level evidence before
calling you.

If you write `status=accepted` or `status=discarded`, that means "attempt
workflow exit." The runner will then call the terminal full diffusion assessment
and require Gemini visual quality to pass before the workflow actually exits.
If terminal full evaluation is missing, blocked, or Gemini says visual quality is
not acceptable, the runner will send the case back to you; then normally write
`status=needs_executor_resume` with concrete repair/retry follow-ups.

### Discard Standard

Discard is allowed only when all conditions hold at the DiT/module evaluation
level:

- there is smooth single-DiT/module-level evidence with durable artifacts;
- the method has no meaningful speed, memory, or correctness/quality proxy
  improvement at that level;
- the negative result is not caused by Slurm cancellation, no-output hang,
  missing stdout/stderr, quota/filesystem trouble, missing logs, or another
  out-of-method condition;
- you judge that this method has no credible remaining operator/module-level
  optimization space.

Do not discard for microbench numerical drift alone. If the algorithm is
mathematically correct and has no semantic error, retain the method or request
more single-DiT/module evidence. If terminal validation has already failed,
request executor repair or final-run retry through `needs_executor_resume`. If
there is a semantic/algorithm error, request executor rewrite; do not discard
the method.

Do not discard for infra failures. Own the method through retry: request a
rerun with heartbeat/logging, a different node, or a smaller diagnostic launch.

Write `{REVIEWER_STATUS}` at the repository root before exiting:

```json
{{
  "schema_version": 1,
  "reviewer_goal_id": "{review_id}",
  "target_goal_id": "{target_id}",
  "status": "accepted",
  "decision": "accept",
  "reason": "single-DiT evidence is ready for terminal full diffusion/Gemini validation",
  "required_followups": [],
  "evidence": ["runs/<candidate-id>_microbench/gate_assess.json"]
}}
```

For final discard, use:

```json
{{
  "schema_version": 1,
  "reviewer_goal_id": "{review_id}",
  "target_goal_id": "{target_id}",
  "status": "discarded",
  "decision": "discard",
  "reason": "single-DiT evidence shows no improvement and reviewer finds no remaining operator-level refinement space",
  "required_followups": [],
  "evidence": ["runs/<candidate-id>_microbench/gate_assess.json"],
  "discard_checks": {{
    "smooth_dit_eval": true,
    "no_speed_memory_quality_gain": true,
    "not_infra_or_collection_failure": true,
    "no_remaining_operator_refinement": true
  }}
}}
```

In every other case, write `"status": "needs_executor_resume"` and put concrete
follow-ups in `required_followups`. Do not modify implementation code as
reviewer.
"""
    goal_md = goal_dir / "goal.md"
    goal_md.write_text(goal_md.read_text() + extra)
    reviewer_context = read_json(goal_dir / "context.json")
    reviewer_context.update(
        {
            "review_target_goal_id": target_id,
            "review_target_goal_dir": rel_to(ctx.worktree, ctx.goal_dir),
            "reviewer_status_file": REVIEWER_STATUS,
        }
    )
    write_json(goal_dir / "context.json", reviewer_context)


def build_prompt(goal_dir: Path) -> str:
    prompt = (goal_dir / "goal.md").read_text() if (goal_dir / "goal.md").exists() else ""
    resume = goal_dir / RESUME_FILE
    if resume.exists():
        prompt += "\n\n## Workflow Resume\n\n" + resume.read_text()
        resume.replace(goal_dir / "STOP_HOOK_RESUME.last.md")
    return prompt


def run_codex_reviewer(ctx: NodeContext, goal_dir: Path) -> tuple[int, str, str, Path]:
    goal_dir.mkdir(parents=True, exist_ok=True)
    out = goal_dir / "agent_last.md"
    if ctx.dry_run:
        out.write_text("[dry-run] reviewer node skipped\n")
        return 0, "", "", out
    prompt = build_prompt(goal_dir)
    result = run_managed_codex(
        ctx,
        source_goal_dir=goal_dir,
        node_name="reviewer",
        role="gate",
        prompt=prompt,
        required_artifacts=[REVIEWER_STATUS],
        output_path=out,
    )
    return result.returncode, result.stdout_tail[-2000:], result.stderr_tail[-2000:], out


def run(ctx: NodeContext) -> NodeResult:
    status_path = ctx.worktree / REVIEWER_STATUS
    existing = status_result(ctx, read_json(status_path))
    if existing:
        return existing

    goal_dir = reviewer_goal_dir(ctx)
    try:
        create_reviewer_goal(ctx, goal_dir)
    except RuntimeError as exc:
        return NodeResult("failed", updates={"reviewer_error": str(exc)}, message=str(exc))

    code, stdout_tail, stderr_tail, out = run_codex_reviewer(ctx, goal_dir)
    status = read_json(status_path)
    result = status_result(ctx, status)
    if result:
        result.updates.update(
            {
                "reviewer_exit_code": code,
                "reviewer_output": rel_to(ctx.worktree, out),
                "reviewer_stdout_tail": stdout_tail,
                "reviewer_stderr_tail": stderr_tail,
            }
        )
        result.artifacts.append(rel_to(ctx.worktree, out))
        return result

    return NodeResult(
        "invalid_status",
        updates={
            "reviewer_exit_code": code,
            "reviewer_output": rel_to(ctx.worktree, out),
            "reviewer_stdout_tail": stdout_tail,
            "reviewer_stderr_tail": stderr_tail,
            "reviewer_status": status,
            "reviewer_reason": "missing_or_invalid_reviewer_status",
        },
        artifacts=[rel_to(ctx.worktree, out)] if out.exists() else [],
        message="missing_or_invalid_reviewer_status",
    )
