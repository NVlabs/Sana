#!/usr/bin/env python3
"""Create a Symposium/Codex interactive goal bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover - used by py3.10 envs
    import tomli as tomllib


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
DEFAULT_MAX_ITERS = 40
DEFAULT_EARLY_STOP_PATIENCE = 0
DEFAULT_MODEL_ID = "cosmos3"
DIMENSION_SEARCH_DOCS = {
    "step_cache": "01_cache.md",
    "cache": "01_cache.md",
    "caching": "01_cache.md",
    "token_prune": "02_token_pruning.md",
    "nvfp4_ffn": "03_quantization.md",
    "sparse_attention": "04_sparse_attention.md",
    "kwl_fusion": "05_kernel_fusion.md",
}
SEARCH_SPACE_GOAL_OMIT_PREFIXES = (
    "- Original source:",
    "- Imported source:",
    "- Branch:",
    "- Commit:",
    "- Source path:",
    "See `SOURCE.json`",
)
HISTORICAL_RECORD_IGNORE_PATHS = (
    "`ORCHESTRATOR-LOG.md`",
    "`RELEASE.md`",
    "`RELEASE-fanout.md`",
    "`evals/verdicts/*.json`",
    "`candidates/cosmos3_*.toml`",
    "`output/launch_orchestrator.sh`",
    "`output/orchestrator-prompt.txt`",
    "`output/orchestrator.log`",
    "`output/wtest_*.txt`",
    "`.symposium/archive/`",
    "`.symposium/scratch/codex-goal-sessions/`",
    "`.symposium/scratch/e2e-workflow-goals/`",
    "`.symposium/scratch/test-goals/`",
    "`.symposium/scratch/test-search-space-import/`",
    "`output/fanout/`",
    "`output/fanout_loop_*/`",
    "`output/fanout_runs/*` except the active run id",
    "`runs/*step-cache*` / `runs/*stepcache*`",
    "`runs/*tokenprune*`",
    "`runs/*teacache*`",
    "`runs/*kwl*`",
    "`runs/*nvfp4*`",
    "`runs/*sparse*`",
)
HISTORICAL_RECORD_GLOBS = (
    "ORCHESTRATOR-LOG.md",
    "RELEASE.md",
    "RELEASE-fanout.md",
    "candidates/cosmos3_*.toml",
    "evals/verdicts/*.json",
    "output/launch_orchestrator.sh",
    "output/orchestrator-prompt.txt",
    "output/orchestrator.log",
    "output/wtest_*.txt",
    ".symposium/archive",
    ".symposium/scratch/codex-goal-sessions",
    ".symposium/scratch/e2e-workflow-goals",
    ".symposium/scratch/test-goals",
    ".symposium/scratch/test-search-space-import",
    "output/fanout",
    "output/fanout_loop_*",
    "output/fanout_runs/*",
    "runs/*step-cache*",
    "runs/*stepcache*",
    "runs/*tokenprune*",
    "runs/*teacache*",
    "runs/*kwl*",
    "runs/*nvfp4*",
    "runs/*sparse*",
)
HISTORICAL_RECORD_POLICY = (
    "clean_start_current_experiment_only: do not read stale optimization "
    "reports, verdicts, worktrees, session-state files, or candidate run "
    "directories unless the main orchestrator explicitly passes them as "
    "current-experiment inputs."
)
RUN_ID_ENV_VARS = (
    "SYMPOSIUM_CURRENT_RUN_ID",
    "AUTO_VIDEO_RUN_ID",
    "RUN_ID",
)

FANOUT_LOOP_CONTRACT = f"""This is a bounded per-dimension search loop, not a one-candidate target.

Each loop iteration:

1. Observe current-experiment state only: read this goal's `SEARCH_JOURNAL.md`,
   retained frontier candidates, discarded/rejected signatures, and the
   canonical baseline.
2. Propose the next hypothesis before implementation: what mechanism changes,
   why it should improve over the previous loop, what recorded failure it avoids,
   and what evidence would reject it.
3. Implement exactly one candidate and one manifest. Do not batch unrelated
   mechanisms into one candidate.
4. Preflight with static/unit checks, dry-run rendering, and OFF identity when
   the dimension has an inactive path.
5. Launch through Slurm only after preflight passes.
6. Run the authoritative gate with:
   `{SANA_PYTHON} search/plan_eval.py --assess <run_dir> --baseline-frames {CANONICAL_BASELINE_FRAMES} --out <run_dir>/assess_verdict.json`
7. Decide:
   - quality improved or speed improved: retain the candidate in
     `frontier_candidates`, record the improvement axis, then loop;
   - quality did not improve and speed did not improve or regressed: discard the
     candidate with the reason, then loop;
   - hard-invalid candidate such as missing artifacts, broken OFF identity, or
     runtime failure: reject it with a failure signature, then loop;
   - blocker: record the real external dependency and stop;
   - structured negative: record it as a proposal/failure signature and continue
     the fixed-budget loop unless the main orchestrator explicitly releases the
     dimension.

The default fan-out mode is fixed-budget frontier search: run the
`max_iters={DEFAULT_MAX_ITERS}` budget unless there is a real external blocker or
explicit orchestrator release. Do not stop because a candidate failed an old
per-tier quality threshold, and do not let a dimension agent unilaterally
terminate the budget with `structured_negative`.
`early_stop_patience={DEFAULT_EARLY_STOP_PATIENCE}`
means patience early stop is disabled unless the main orchestrator explicitly
enables a non-default mode.

The loop does not use hard LPIPS/Gemini quality thresholds for lossy generative
dimensions. LPIPS and aligned pairwise Gemini are recorded together as quality
evidence. When budget fires, write `status=terminal_pending_review` rather than
treating the dimension as globally complete. The main orchestrator selects
quality-best winners for the 1.5x, 2.0x, and 3.0x speed targets from retained
frontier candidates, using Gemini artifact severity/status and LPIPS together,
or reopens the dimension with a new direction, requests validation, drops it, or
marks a blocker. Numeric checks, tolerance declarations, OFF identity,
silent-fallback detection, and precision-support proof are diagnostics unless a
candidate contract explicitly declares a reliable hard gate. Collector
`quality.json` is telemetry; the authoritative gate provides quality and speed
evidence for frontier retention and final speed-target selection."""

INTEGRATION_LOOP_CONTRACT = f"""This is the fan-in integration loop. It starts only after
selected fan-out dimensions are terminal, and it is required before the overall
experiment can be called complete.

Each integration iteration:

1. Read every selected dimension's current-experiment `AGENT-STATUS.json`,
   `SUMMARY.md`, `SEARCH_JOURNAL.md`, candidate manifests, and run artifacts.
2. Reconcile stale status with durable run artifacts, recording which source of
   truth was used for each tier winner.
3. Build one delivery-target plan at a time from retained per-dimension winners.
   Targets are low=1.5x, medium=2.0x, and high=3.0x. Within each speed target,
   prefer the best combined quality evidence: aligned pairwise Gemini severity
   and status first, aligned LPIPS second, then higher speed as a tie-breaker.
   Empty targets must get an explicit `no_eligible_profile` blocker.
4. Implement exactly one composed profile in the integration worktree. Preserve
   each component's OFF guard and feature flag, resolve shared-file conflicts,
   and do not quote composed speedup from single-dimension timings.
5. Preflight, launch, collect, and run the authoritative composed gate with:
   `{SANA_PYTHON} search/plan_eval.py --assess <run_dir> --baseline-frames {CANONICAL_BASELINE_FRAMES} --out <run_dir>/assess_verdict.json`
6. Decide:
   - passed composition: keep it as the tier incumbent, record artifacts, then
     continue searching for a faster compatible composition if budget remains;
   - failed composition: record an interaction failure signature, then loop with
     a repaired merge, reduced subset, or different tier plan;
   - tier blocker: record the real reason no composed profile can be produced;
   - global blocker: record the external dependency and stop.

Stop only when every 1.5x/2.0x/3.0x delivery target has either a gated composed
profile or an explicit blocker, or when `max_iters` or a real external blocker
is reached. A failed composed gate never completes integration by itself."""


def loop_contract_values(args: argparse.Namespace) -> dict:
    values = {
        "max_iters": args.max_iters,
        "early_stop_patience": args.early_stop_patience,
        "loop_mode": "fixed_budget_frontier",
        "frontier_keep_rule": "keep_if_quality_improves_or_speed_or_memory_improves",
        "frontier_discard_rule": "discard_if_no_quality_improvement_and_no_speed_or_memory_improvement",
        "tier_selection": "after_budget_select_1p5x_2x_3x_speed_targets_by_best_gemini_and_lpips_quality",
        "failed_candidate_action": "record_interaction_failure_and_loop"
        if args.role == "integration"
        else "discard_or_reject_log_and_loop",
        "successful_candidate_action": "keep_composed_tier_incumbent_and_loop"
        if args.role == "integration"
        else "retain_frontier_candidate_and_loop",
    }
    if args.role != "integration" and args.dimension == "kwl_fusion":
        values.update(
            {
                "frontier_keep_rule": "keep_if_quality_improves_or_latency_or_peak_memory_improves_with_kwl_semantic_boundary",
                "frontier_discard_rule": "discard_if_no_quality_or_numeric_improvement_and_no_speed_or_memory_improvement",
            }
        )
    return values


def candidate_retention_rule(args: argparse.Namespace) -> str:
    if args.role != "integration" and args.dimension == "kwl_fusion":
        return (
            "retain_kwl_candidate_if_quality_improves_or_latency_or_peak_memory_"
            "improves_with_off_identity_and_semantic_boundary"
        )
    return "retain_if_quality_improves_or_speed_or_memory_improves_discard_if_neither_improves"


def quality_source_of_truth(args: argparse.Namespace) -> list[str]:
    source = [
        "off_identity",
        "aligned_lpips",
        "aligned_pairwise_gemini",
        "speed_or_memory_improvement",
    ]
    if args.role != "integration" and args.dimension == "kwl_fusion":
        source.insert(1, "module_level_tensor_diff_when_available")
        source.insert(2, "declared_numeric_tolerance")
    if args.role != "integration" and args.dimension == "nvfp4_ffn":
        source.insert(1, "numeric_or_precision_checks_when_reliable")
    return source


def dimension_loop_note(dimension: str, role: str) -> str:
    if role == "integration":
        return ""
    if dimension != "kwl_fusion":
        return ""
    return """## KWL Quality-Gated Frontier

For `kwl_fusion`, apply the KWL-specific retention rule from
`loops/kwl_fusion/acceptance.md`: run the full fixed-budget frontier loop,
retain candidates that improve latency, peak memory, aligned quality, or
reliable numeric stability, then let final low/medium/high selection pick the
best retained profiles by speed target and quality ranking. ON bit-exactness is
not required; record the declared tolerance class and aligned quality evidence.
Reject candidates that change scheduler, step count, token set, attention
semantics, cache/prune semantics, quantization policy, prompt/guidance, LoRA
state, resolution, frame count, or output shape.
"""


def historical_record_policy_md() -> str:
    ignored = "\n".join(f"- {item}" for item in HISTORICAL_RECORD_IGNORE_PATHS)
    return f"""## Historical Record Policy

This goal is a clean-start current-experiment loop. Do not use previous
optimization reports, verdicts, old worktrees, archived tmux captures, stale
session-state files, or old candidate run directories as priors. Use only:

- the canonical baseline frames at `{CANONICAL_BASELINE_FRAMES}`;
- `search_space/`, `loops/<dimension>/`, model/runtime code, and this goal's
  own `SEARCH_JOURNAL.md` / `AGENT-STATUS.json`;
- current-experiment sibling dimension artifacts only when this is an
  integration goal and the main orchestrator explicitly selected them.

Ignore these stale-record locations unless the main orchestrator explicitly
passes a path as part of the current experiment:

{ignored}

The Codex goal launcher enforces this at startup: by default it removes stale
optimization records outside the active `run_id` and then refuses to start if
any are still visible. Set `SYMPOSIUM_PRESERVE_HISTORY_RECORDS=1` only when an
operator intentionally wants to inspect stale records outside the goal loop.
"""


def active_run_root(root: Path, run_id: str = "") -> Path | None:
    if not run_id:
        return None
    resolved_root = root.resolve()
    parts = resolved_root.parts
    marker = ("output", "fanout_runs", run_id)
    for idx in range(len(parts) - len(marker) + 1):
        if tuple(parts[idx : idx + len(marker)]) == marker:
            return Path(*parts[: idx + len(marker)])
    return (root / "output" / "fanout_runs" / run_id).resolve()


def infer_run_id(root: Path, provided: str = "") -> str:
    if provided:
        return provided
    for name in RUN_ID_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    parts = root.resolve().parts
    for idx in range(len(parts) - 2):
        if parts[idx : idx + 2] == ("output", "fanout_runs"):
            return parts[idx + 2]
    return ""


def find_stale_optimization_records(root: Path, run_id: str = "") -> list[str]:
    """Return stale optimization records visible from the checkout root."""

    active_run = active_run_root(root, run_id)
    records: list[str] = []
    seen: set[str] = set()
    for pattern in HISTORICAL_RECORD_GLOBS:
        for path in sorted(root.glob(pattern)):
            resolved = path.resolve()
            if active_run and (resolved == active_run or active_run in resolved.parents):
                continue
            rel = path.relative_to(root).as_posix()
            if rel not in seen:
                seen.add(rel)
                records.append(rel)
    return records


def check_stale_records(root: Path, run_id: str = "") -> int:
    records = find_stale_optimization_records(root, run_id)
    if not records:
        suffix = f" for active run {run_id}" if run_id else ""
        print(f"no stale optimization records visible{suffix}")
        return 0
    print("stale optimization records visible:", file=sys.stderr)
    for record in records:
        print(f"  {record}", file=sys.stderr)
    return 5


def remove_stale_optimization_records(root: Path, run_id: str = "") -> list[str]:
    """Delete stale optimization records and return the relative paths removed."""

    removed: list[str] = []
    for rel in find_stale_optimization_records(root, run_id):
        path = root / rel
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        else:
            continue
        removed.append(rel)
    return removed


def clean_stale_records(root: Path, run_id: str = "") -> int:
    records = remove_stale_optimization_records(root, run_id)
    if not records:
        suffix = f" for active run {run_id}" if run_id else ""
        print(f"no stale optimization records to remove{suffix}")
        return 0
    print("removed stale optimization records:")
    for record in records:
        print(f"  {record}")
    return 0


def summarize_search_space(path: Path) -> str:
    summary_lines: list[str] = []
    previous_blank = False
    for line in path.read_text().splitlines()[:120]:
        if line.startswith(SEARCH_SPACE_GOAL_OMIT_PREFIXES):
            continue
        is_blank = line.strip() == ""
        if is_blank and previous_blank:
            continue
        summary_lines.append(line)
        previous_blank = is_blank
    return "\n".join(summary_lines).strip().replace("```", "~~~")


def read_search_space_summary(root: Path, search_space_root: str, dimension: str) -> tuple[str, str, str]:
    space_path = (root / search_space_root).resolve()
    if not space_path.exists():
        raise SystemExit(
            f"Search-space root is missing: {space_path}. "
            "Import or restore `search_space/` before generating subagent goals."
        )
    rel_space = str(space_path.relative_to(root))
    search_doc = DIMENSION_SEARCH_DOCS.get(dimension)
    if search_doc:
        summary_path = space_path / search_doc
    else:
        summary_path = space_path / "README.md"
    if summary_path.exists():
        summary = summarize_search_space(summary_path)
        summary_rel = str(summary_path.relative_to(root))
    else:
        summary = "Search-space README is missing; inspect the method-family files directly."
        summary_rel = rel_space
    return rel_space, summary_rel, summary


def read_dimension_metadata(root: Path, dimension: str) -> dict:
    dim_file = root / "loops" / dimension / "dimension.toml"
    if not dim_file.exists():
        return {}
    with dim_file.open("rb") as handle:
        return tomllib.load(handle)


def method_baseline_catalog_md(method_baselines: list[dict]) -> str:
    if not method_baselines:
        return """## Method Baseline Catalog

No method-baseline catalog is declared for this dimension. Treat the
search-space document as authoritative, but explicitly record whether each
candidate is wired, candidate-wired, runtime-patched, or probe-only.
"""
    lines = [
        "## Method Baseline Catalog",
        "",
        "Use this catalog to avoid overfitting the search to the first wired helper.",
        "`tier=wired` means a candidate can start from existing code; `candidate_wired`",
        "means a helper/env exists but target-runtime consumption still needs proof;",
        "`runtime_patch` means the candidate must patch the live inference path;",
        "`upper_bound_probe` is diagnostic and must not become a delivery winner unless",
        "it later gains full quality evidence and safe fallback behavior.",
        "",
    ]
    for item in method_baselines:
        lines.extend(
            [
                f"- `{item.get('id', 'unknown')}` [{item.get('tier', 'unknown')}/{item.get('status', 'unknown')}]",
                f"  family: `{item.get('family', 'unknown')}`",
                f"  description: {item.get('description', '').strip()}",
                f"  entrypoint: `{item.get('entrypoint', 'unspecified')}`",
                f"  required work: {item.get('required_work', '').strip()}",
            ]
        )
    return "\n".join(lines) + "\n"


def implementation_loop_acceptance() -> list[str]:
    return [
        "run the fixed-budget fan-out loop; do not stop after a single candidate success or failure",
        "write a hypothesis before each candidate explaining the expected improvement and the prior failure it avoids",
        "record each candidate in `SEARCH_JOURNAL.md` with quality evidence, speed evidence, retention decision, and next-hypothesis requirement",
        "retain a candidate when quality improves or speed/memory improves, even if it is not yet selected for a 1.5x/2.0x/3.0x speed target",
        "discard a candidate only when neither quality nor speed/memory improves; reject hard-invalid candidates with a failure signature",
        "after a discard or reject, generate a meaningfully different hypothesis instead of repeating the same mechanism with cosmetic parameters",
        "continue searching until max_iters, real blocker, or explicit orchestrator release; a dimension-agent structured_negative proposal is logged but does not stop the default fixed-budget loop",
        "track `no_improve_count` as telemetry; it must not stop the default fixed-budget frontier loop",
        "record every candidate verdict with `tools/symposium/loop_control.py record-candidate`, then run `decide-next` and `validate-status` before continuing",
        "on max_iters, write `status=terminal_pending_review` and recommend select_tiers_for_integration, restart_with_new_direction, validate, drop, or mark_blocked for main-agent review",
        "use the authoritative `sana` `search/plan_eval.py --assess` gate with canonical baseline frames for retention evidence and final speed-target selection",
        "do not apply hard LPIPS/Gemini thresholds during lossy generative search; record aligned pairwise Gemini and LPIPS together, then rank quality by Gemini severity/status plus LPIPS after the budget closes",
        "treat collector `quality.json` Gemini as telemetry, not the only quality authority, when it contradicts aligned LPIPS or aligned pairwise Gemini",
    ]


def integration_acceptance() -> list[str]:
    return [
        "read and reconcile every selected dimension's AGENT-STATUS.json, SUMMARY.md, SEARCH_JOURNAL.md, manifests, and run artifacts",
        "build explicit 1.5x/2.0x/3.0x delivery-target plans from eligible per-dimension winners, using `no_eligible_profile` blockers for empty targets",
        "within each speed target, choose the best quality profile using aligned pairwise Gemini severity/status and aligned LPIPS together, then higher speed as tie-breaker",
        "implement exactly one composed profile per integration iteration, preserving component OFF guards and feature flags",
        "never report composed speedup or quality from single-dimension runs; launch and gate the merged profile itself",
        "run the authoritative `sana` `search/plan_eval.py --assess` gate with canonical baseline frames for every composed profile",
        "if a composed gate fails, record an interaction failure signature and loop with a repaired merge, reduced subset, or different tier plan",
        "record composed delivery candidates with `--purpose delivery`; record upper-bound or unsafe high probes with `--purpose blocker_probe` or `--purpose unsafe_probe` so they cannot become tier incumbents",
        "write INTEGRATION-STATUS.json, INTEGRATION-JOURNAL.md, composed manifests, run artifacts, per-tier blockers, and a release matrix",
        "run `python3 tools/fanout_audit.py --run <fanout_run_id_or_path>` before declaring the workflow complete",
        "finish only when every 1.5x/2.0x/3.0x target has a gated composed profile or an explicit blocker",
    ]


def dimension_acceptance(dimension: str, role: str) -> list[str]:
    if role == "integration":
        return integration_acceptance()

    if role == "gate":
        return [
            "review the implementation diff and candidate manifest without authoring implementation code",
            "reproduce or inspect the run artifacts required by the candidate contract",
            "write a structured gate verdict JSON covering OFF identity, pixel metrics, LPIPS, Gemini, timing, speed-target bucket, and quality-ranking evidence",
            "mark the candidate rejected when any required quality artifact is missing, deferred, unavailable, or prose-only",
            "write `SUMMARY.md` with the verdict, evidence paths, and any non-reproducible gaps",
        ]

    common = implementation_loop_acceptance()

    if dimension in {"step_cache", "cache", "caching"}:
        return common + [
            "inspect the cache method families in `search_space/01_cache.md` before proposing implementations",
            "identify at least five caching mechanisms, including TeaCache-style signal reuse, EasyCache-style runtime-adaptive transform reuse, PAB-style attention broadcast, block/residual/FFN reuse, and token-wise, CFG-aware, content-adaptive, or predictive/delta caching when applicable",
            "derive per-layer, per-step, signal, threshold, fallback, and schedule choices from target-model inference code or traces rather than predefined constants",
            "modify the inference code directly when that is the shortest path to a runnable candidate; do not wait for a predeclared seam",
            "prove OFF identity against the baseline path before reporting speedup",
            "produce at least one runnable candidate manifest or a structured-negative proposal for orchestrator review; do not use the proposal to stop the default fixed-budget loop",
            "write exact reproduction commands, changed files, run artifacts, and current structured speed/quality evidence status",
        ]
    if dimension == "token_prune":
        return common + [
            "inspect `search_space/02_token_pruning.md` and then inspect target-model token layout directly in inference code",
            "identify at least five token-reduction mechanisms, including pruning, merging, masking, region-aware or dynamics-aware selection, and mediator-token, cluster-aware, context-token, token-wise caching, or dynamic token-density policies when applicable",
            "derive prunable spans, salience signals, compensation policy, layer windows, and step windows from code/traces",
            "prove gather/scatter or masking keeps positional tensors, attention masks, and output restoration aligned",
            "modify the inference code directly when needed; do not require a predeclared prunable-token seam",
            "prove OFF identity against the baseline path before reporting speedup",
            "produce at least one runnable candidate manifest or a structured-negative proposal for orchestrator review; do not use the proposal to stop the default fixed-budget loop",
            "write exact reproduction commands, changed files, run artifacts, and current structured speed/quality evidence status",
        ]
    if dimension == "sparse_attention":
        return common + [
            "inspect `search_space/04_sparse_attention.md` before proposing implementations",
            "run and record attention preflight before GPU search: attention call timing, token/frame/tile layout, sequence length, dominant attention path, available sparse kernels/backends, dense fallback, OFF identity, and env/config consumption proof",
            "analyze target-model self-attention, cross-attention, and joint/GEN attention paths separately",
            "identify at least five training-free sparse-attention families, including piecewise/PISA, Sparse-VideoGen-style spatial/temporal head routing, SVG2-style semantic permutation, AdaSpa-style online precise search and mask reuse, SpargeAttn-style proxy masks, LVSA-style rotating anchors, SVOO-style QK co-clustering, HASTE-style head-wise budgets, or MInference-style dynamic patterns when applicable",
            "derive per-layer, per-step, per-head, per-attention-type routing, dense fallback, block/window/anchor policy, mask refresh, density, kernel/backend path, and sparsity policy from traces or code inspection",
            "measure mask-search, permutation, gather/scatter, and fallback overhead separately from sparse attention kernel time",
            "modify the inference code directly when needed; do not require a predeclared swappable-attention seam",
            "prove OFF identity and verify dense fallback behavior before reporting speedup",
            "produce at least one runnable candidate manifest or a structured-negative proposal for orchestrator review; do not use the proposal to stop the default fixed-budget loop",
            "write exact reproduction commands, changed files, run artifacts, and current structured speed/quality evidence status",
        ]
    if dimension == "nvfp4_ffn":
        return common + [
            "inspect `search_space/03_quantization.md` before proposing implementations",
            "run and record NVFP4 preflight before GPU search: GPU architecture, TransformerEngine import/version, NVFP4BlockScaling availability, FP4 GEMM backend availability, minimal TE/loader smoke, OFF identity, and env-consumption proof",
            "profile or inspect target-model hot linear modules, including FFN/MLP, attention projections, and output projections, to choose module scope, layer guards, step guards, TE recipe flags, fused epilogue path, backend, padding policy, and fallback policy",
            "separate already-wired runtime env axes from metadata-only axes that require candidate-side loader wiring",
            "record hardware/library prerequisites, warm/cold compile state, backend selection, and fallback policy explicitly",
            "modify the inference/loading code directly when needed; do not require a predeclared precision seam",
            "prove OFF identity against the BF16 baseline path before reporting speedup",
            "record reliable numeric precision checks, silent-fallback detection, and BF16 fallback integrity; only treat a numeric gate as hard when the candidate contract explicitly declares it",
            "produce at least one runnable candidate manifest or a structured-negative proposal for orchestrator review; do not use the proposal to stop the default fixed-budget loop",
            "write exact reproduction commands, changed files, run artifacts, and current structured speed/quality evidence status",
        ]
    if dimension == "kwl_fusion":
        return common + [
            "inspect `search_space/05_kernel_fusion.md` before proposing implementations",
            "run and record KWL preflight before GPU search: hot-path evidence, launch count, memory traffic, tensor shapes, dtype, backend availability, compile/graph state, OFF identity, fallback behavior, and semantic boundary proof",
            "identify at least six KWL method families, including exact-preferred and quality-gated approximate variants across GEMM epilogues, norm/modulation/residual fusion, attention-adjacent dense fusion, compile or CUDA graph capture, layout/copy elimination, launch batching, stream overlap, decode/postprocess fusion, or backend selection when applicable",
            "profile or inspect target-model hot ops and choose exact, numerically tolerant, or quality-gated approximate kernel/backend candidates from evidence",
            "separate KWL-safe kernel/backend approximations from algorithm changes; route cache, prune, sparse-attention, scheduler, or quantization-policy changes to other dimensions",
            "retain speed or memory candidates when OFF identity passes and latency or peak memory improves; ON bit-exactness is not required",
            "retain aligned quality or reliable numeric-stability candidates when those signals improve, even without a speedup",
            "record expected numeric tolerance as bit-exact, dtype-rounding-only, reduction-order drift, FMA/epilogue drift, fast-math drift, or approximate-kernel drift",
            "record cold compile, warm compile, autotune, graph replay, and cache-reuse timing modes separately",
            "reject semantic changes to scheduler, step count, token set, attention semantics, cache/prune semantics, quantization policy, prompt/guidance, LoRA state, resolution, frame count, or output shape",
            "modify the inference/build code directly when needed; do not require a predeclared kernel-fusion seam",
            "produce at least one runnable candidate manifest or a structured-negative proposal for orchestrator review; do not use the proposal to stop the default fixed-budget loop",
            "write exact reproduction commands, changed files, run artifacts, and current structured speed/quality evidence status",
        ]
    return common + [
        "inspect the relevant method family in `search_space/` before proposing implementations",
        "derive all model-specific knobs from traces or code inspection rather than predefined constants",
        "modify inference code directly when needed; do not wait for a predeclared interface",
        "keep implementation work in the isolated worktree and declared write scope",
        "produce at least one runnable candidate manifest or a structured-negative proposal for orchestrator review; do not use the proposal to stop the default fixed-budget loop",
        "write exact reproduction commands, changed files, run artifacts, and current structured speed/quality evidence status",
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
            "composed 1.5x/2.0x/3.0x delivery manifests or explicit per-target blockers",
            "run bundle artifacts for every launched composed profile",
            "interaction failure signatures for every rejected composition",
            "release matrix separating per-dimension winners from gated composed profiles",
        ]
    return [
        "`AGENT-STATUS.json` maintained by `tools/symposium/loop_control.py`",
        "`SEARCH_JOURNAL.md` updated once per recorded candidate",
        "candidate manifest or structured-negative proposal note",
        "run bundle artifacts when a candidate is launched",
        "failure signatures for every rejected candidate",
        "`SUMMARY.md`",
    ]


def render_goal_md(
    args: argparse.Namespace,
    candidate_rel: str,
    search_space_rel: str,
    search_doc_rel: str,
    search_space_summary: str,
    method_baselines: list[dict],
) -> str:
    write_scope = resolved_write_scope(args)
    acceptance = "\n".join(f"- {item}" for item in dimension_acceptance(args.dimension, args.role))
    scope = "\n".join(f"- `{item}`" for item in write_scope)
    artifacts = "\n".join(f"- {item}" for item in required_artifacts(args.role))
    contract_heading = "Fan-In Integration Contract" if args.role == "integration" else "Fan-Out Loop Contract"
    contract_text = INTEGRATION_LOOP_CONTRACT if args.role == "integration" else FANOUT_LOOP_CONTRACT
    loop_values = loop_contract_values(args)
    kwl_loop_note = dimension_loop_note(args.dimension, args.role)
    method_baseline_catalog = method_baseline_catalog_md(method_baselines)
    return f"""# Goal: {args.goal_id}

You are working in an isolated autovideo goal context.

## Role

`{args.role}`

## Objective

{args.objective}

## Search Space Start

Start from the method-family search space, then inspect and modify the target-model
inference code directly:

- Search-space root: `{search_space_rel}`
- Relevant dimension: `{args.dimension}`
- Relevant search doc: `{search_doc_rel}`

Relevant search-space summary:

```text
{search_space_summary}
```

Do not use historical recipe archives or fixed grids as startup context.
If `search_space/` is missing or unclear, stop exploration and ask the main
orchestration agent to repair the search-space contract.

{method_baseline_catalog}

{historical_record_policy_md()}

## {contract_heading}

Read `docs/fanout-loop-contract.md`. The operational summary for this goal is:

{contract_text}

{kwl_loop_note}

## Loop Control

- `max_iters`: {loop_values["max_iters"]}
- `early_stop_patience`: {loop_values["early_stop_patience"]}
- loop mode: `{loop_values["loop_mode"]}`
- frontier keep rule: `{loop_values["frontier_keep_rule"]}`
- frontier discard rule: `{loop_values["frontier_discard_rule"]}`
- final speed-target selection: `{loop_values["tier_selection"]}`
- speed targets: `low=1.5x`, `medium=2.0x`, `high=3.0x`
- quality ranking: aligned pairwise Gemini severity/status + aligned LPIPS,
  then higher speed as tie-breaker. LPIPS and Gemini are both considered; LPIPS
  alone is not the selector.
- failed candidate action: `{loop_values["failed_candidate_action"]}`
- successful candidate action: `{loop_values["successful_candidate_action"]}`
- terminal handoff: write `terminal_pending_review` with frontier candidates,
  discarded/rejected candidates, failure signatures, remaining hypotheses, and
  an `agent_recommendation` for the main orchestrator.

Runtime controller commands:

```bash
python3 tools/symposium/loop_control.py init --dimension {args.dimension} --goal-id {args.goal_id} --max-iters {loop_values["max_iters"]} --early-stop-patience {loop_values["early_stop_patience"]} --loop-mode fixed_budget_frontier
python3 tools/symposium/loop_control.py record-candidate --candidate-id <id> --decision <quality_improved|speed_improved|quality_and_speed_improved|discarded_regression|rejected|blocked|structured_negative> --reason "<short reason>" [--purpose frontier|delivery|evidence|blocker_probe|unsafe_probe|control] [--improvement-axis quality|speed|both|none] [--tier low|medium|high] [--run-dir <run_dir>] [--evidence <run_dir>/assess_verdict.json]
python3 tools/symposium/loop_control.py add-evidence --candidate-id <id> --evidence <run_dir>/assess_verdict.json --reason "backfilled authoritative gate artifact"
python3 tools/symposium/loop_control.py decide-next
python3 tools/symposium/loop_control.py validate-status
python3 tools/symposium/loop_control.py status-summary
```

Call `record-candidate` after every authoritative gate. For run-backed
candidates, evidence must include a durable authoritative gate artifact:
`assess_verdict.json`, `verdict.json`, `gate_assess.json`, or
`reject_note.json`. Collector-only telemetry such as `outputs/quality.json`
cannot by itself retain or reject a candidate. Use `add-evidence` only to
backfill a current-experiment record after the durable gate artifact exists. If
`decide-next` returns `terminal_pending_review` or `blocked`, stop candidate
search and hand the status to the main orchestrator. Watchers must treat
`complete`, `terminal_pending_review`, and `blocked` as terminal states by using
`status-summary` or JSON parsing; do not grep only for `status=complete`.

## Model And Runtime Context

- Target model profile: `models/{args.model_id}.toml`
- Execution repo: `Sol-LTX-Infer/`
- Primary implementation surface: inspect and modify the model inference path
  under `Sol-LTX-Infer/` directly in this isolated worktree.
- Launcher: `python3 scripts/launch_candidate.py {candidate_rel} --mode dry-run`
- Collector: `python3 scripts/collect_run.py runs/<run-id>`
- Authoritative assess: `{SANA_PYTHON} search/plan_eval.py --assess <run_dir> --baseline-frames {CANONICAL_BASELINE_FRAMES} --out <run_dir>/assess_verdict.json`
- Quality source of truth: OFF identity, aligned LPIPS, and aligned pairwise
  Gemini from the authoritative gate. `outputs/quality.json` is telemetry and
  not the quality source of truth when it contradicts aligned gate artifacts.
  For generative dimensions, LPIPS/Gemini are ranking evidence rather than hard
  absolute thresholds; numeric checks are diagnostics unless a candidate
  contract explicitly declares a reliable hard gate.

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
- Assess: `{SANA_PYTHON} search/plan_eval.py --assess <run_dir> --baseline-frames {CANONICAL_BASELINE_FRAMES} --out <run_dir>/assess_verdict.json`

## Branching Contract

- Root branch: `{args.root_branch}`
- Submodule branch: `{args.submodule_branch}`

## Done When

{acceptance}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-stale-records", action="store_true")
    parser.add_argument("--clean-stale-records", action="store_true")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--goal-id")
    parser.add_argument("--candidate")
    parser.add_argument("--objective")
    parser.add_argument("--role", choices=("implementation", "gate", "integration"), default="implementation")
    parser.add_argument("--dimension", default="general")
    parser.add_argument("--search-space-root", default="search_space")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--write-scope", action="append", default=[])
    parser.add_argument("--root-branch", default="")
    parser.add_argument("--submodule-branch", default="")
    parser.add_argument("--goals-root", default="goals")
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = project_root()
    args.run_id = infer_run_id(root, args.run_id)
    if args.clean_stale_records:
        return clean_stale_records(root, args.run_id)
    if args.check_stale_records:
        return check_stale_records(root, args.run_id)
    missing = [
        name
        for name in ("goal_id", "candidate", "objective")
        if not getattr(args, name)
    ]
    if missing:
        parser.error(
            "the following arguments are required unless --check-stale-records "
            "or --clean-stale-records is used: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
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
    search_space_rel, search_doc_rel, search_space_summary = read_search_space_summary(
        root,
        args.search_space_root,
        args.dimension,
    )
    dimension_metadata = read_dimension_metadata(root, args.dimension)
    method_baselines = dimension_metadata.get("method_baseline", [])
    model_profile = root / "models" / f"{args.model_id}.toml"
    if not model_profile.exists():
        raise SystemExit(f"Model profile does not exist: {model_profile}")
    if args.max_iters < 1:
        raise SystemExit("--max-iters must be >= 1")
    if args.early_stop_patience < 0:
        raise SystemExit("--early-stop-patience must be >= 0")

    goal_dir = (root / args.goals_root / goal_id).resolve()
    if goal_dir.exists() and not args.overwrite:
        raise SystemExit(f"Goal already exists: {goal_dir} (use --overwrite)")
    goal_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(candidate, goal_dir / "candidate.toml")
    (goal_dir / "goal.md").write_text(
        render_goal_md(
            args,
            candidate_rel,
            search_space_rel,
            search_doc_rel,
            search_space_summary,
            method_baselines,
        )
    )
    context = {
        "goal_id": goal_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": "tools/symposium/prepare_goal.py",
        "target_agent": "codex",
        "mode": "interactive-goal",
        "candidate_manifest": candidate_rel,
        "run_id": args.run_id,
        "role": args.role,
        "dimension": args.dimension,
        "search_space_root": search_space_rel,
        "search_space_doc": search_doc_rel,
        "method_baselines": method_baselines,
        "model_id": args.model_id,
        "model_profile": str(model_profile.relative_to(root)),
        "write_scope": resolved_write_scope(args),
        "acceptance_criteria": dimension_acceptance(args.dimension, args.role),
        "history_policy": {
            "mode": "clean_start_current_experiment_only",
            "policy": HISTORICAL_RECORD_POLICY,
            "startup_enforcement": "clean_stale_records_outside_active_run_id_then_check",
            "ignore_paths": list(HISTORICAL_RECORD_IGNORE_PATHS),
            "allowed_prior_state": [
                "this_goal_SEARCH_JOURNAL",
                "this_goal_AGENT_STATUS",
                "canonical_baseline_frames",
                "current_experiment_selected_sibling_artifacts_for_integration_only",
            ],
        },
        "loop_contract": {
            "kind": "fan_in_integration_loop" if args.role == "integration" else "bounded_fanout_search_loop",
            **loop_contract_values(args),
            "no_improve_counter": "telemetry_increment_on_discard_or_reject_reset_on_quality_or_speed_improvement",
            "early_stop_exit_status": "terminal_pending_review",
            "main_agent_review_actions": [
                "select_tiers_for_integration",
                "accept_frontier_for_integration",
                "restart_with_new_direction",
                "request_validation",
                "mark_blocked",
                "drop_dimension",
            ],
            "authoritative_python": SANA_PYTHON,
            "canonical_baseline_frames": CANONICAL_BASELINE_FRAMES,
            "quality_source_of_truth": quality_source_of_truth(args),
            "candidate_retention": candidate_retention_rule(args),
            "collector_quality_json": "telemetry_not_promotion_authority_when_contradicted",
            "speed_targets": {"low": 1.5, "medium": 2.0, "high": 3.0},
            "quality_ranking": [
                "aligned_pairwise_gemini_max_artifact_severity",
                "aligned_pairwise_gemini_overall",
                "aligned_lpips_max",
                "higher_speedup_tie_breaker",
            ],
            "hard_quality_thresholds": "disabled_by_default_numeric_gates_require_explicit_candidate_contract",
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
