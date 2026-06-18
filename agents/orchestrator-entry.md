# Orchestrator Entry — drive end-to-end acceleration (read this first)

**You are the MAIN orchestration agent.** Given a served model, drive the full
acceleration search and deliver **three speed-target configs** (low / medium / high).
You **orchestrate** — you do not implement techniques yourself. You spawn one agent
per search dimension, gate their results, integrate the winners, and deliver.

This is the single entry point. Everything else hangs off it.

Launch this as a normal main-agent Codex session. Do **not** run `/goal follow`
on this file. Native goal mode is reserved for implementation and gate
subagents that the main agent creates under `goals/<goal-id>/`.

---

## 0. Read these first (the system you drive)
- `docs/search-architecture.md` — the three planes + the search & eval pipeline.
- `docs/fanout-loop-contract.md` — the per-dimension loop state machine, gate
  authority, failure signatures, and stopping rules.
- `models/README.md` — the target model profile and run environment.
- `docs/codex-goal-mode.md` and `docs/multi-agent-orchestration.md` — how to
  spawn, correct, gate, and release per-dimension native goal sessions.
- `efficiency/README.md` — the generic acceleration engine.

## 1. Mental model (3 planes, 3 levels)
- **GENERIC** (never model-specific): `efficiency/` (engine), `loops/<dim>/`
  (search dimensions), `search/` (harness), `evals/tiers.toml` (1.5/2/3x targets),
  `search_space/` (the method-family search space).
- **MODEL/RUNTIME SURFACE**: `models/<id>.toml` plus the live inference code under
  `Sol-LTX-Infer/`.
- A dimension is **launchable** when the main agent decides it is worth exploring.
  Do not block launch on predeclared seams. Subagents may write directly in the
  inference path; seams/compose are post-hoc diagnostics and merge aids.

## 2. End-to-end procedure
```
0. ONBOARD/FREEZE  ensure models/<id>.toml + efficiency/models/<id>_spec.py
                   (model-onboarding.md). Run the official-config baseline and
                   record it in the profile [baseline]. The profile [env] must
                   carry the working runtime env (PYTHON_BIN, HF_HOME/HF_HUB_CACHE,
                   COSMOS3_CACHE, HF_HUB_OFFLINE) or the cluster run fails.
1. SCAN            python search/search.py --model <id>
                   -> method families, loop budgets, compose diagnostics, and the 3 targets.
                   Use this as an observation pass, not as a launch gate.
2. FAN OUT         one native Codex goal session per selected dimension:
                   create a fresh RUN_ID for the experiment and keep all
                   worktrees under `output/fanout_runs/$RUN_ID/`.
                   Do not reuse `output/fanout/`, `output/fanout_loop_*`, old
                   verdicts, old release reports, or archived session captures.
                   own worktree+branch+isolated CODEX_HOME; goal.md includes
                   objective, search-space-start, artifacts, acceptance criteria,
                   and the fan-out loop contract.
                   Start through `tools/symposium/codex_goal_session.py start`,
                   which opens interactive Codex and sends `/goal follow <goal.md>`.
                   Each agent starts from `search_space/` and
                   `loops/<dim>/exploration.md`, then runs a bounded loop:
                   observe current-experiment results -> propose a better hypothesis ->
                   implement exactly one candidate -> preflight -> launch ->
                   authoritative gate -> retain/discard/reject and loop. It can directly
                   modify `Sol-LTX-Infer/`; do not wait for an exposed seam/interface:
                       -> plan_eval.render_candidate(profile, technique, cfg)
                       -> scripts/launch_candidate.py --mode sbatch --confirm-submit
                       -> sana search/plan_eval.py --assess (benchmark + aligned quality)
                       -> retain frontier if quality OR speed improves
                       -> final tier selection after budget
                     budget: fixed max_iters=40, early_stop_patience=0.
                   Candidate failure does not finish the dimension; it must
                   produce a failure signature and loop unless max_iters,
                   a real blocker, or explicit orchestrator release applies.
                   A structured_negative proposal is logged, not a dimension-level stop.
                   Require every subagent to initialize and update
                   `AGENT-STATUS.json` through `tools/symposium/loop_control.py`.
3. GATE (you)      never auto-merge. Per candidate enforce the authoritative quality:
                   off_identity -> aligned LPIPS -> aligned pairwise NVIDIA-Gemini.
                   LPIPS and Gemini are both used for quality ranking within
                   speed targets; LPIPS alone is not the selector. Collector
                   `quality.json` Gemini is telemetry, not the quality source of truth,
                   when it disagrees with the aligned gate. Read each branch diff, run the
                   gate, merge sequentially (by SHA), reconcile shared-file conflicts.
4. INTEGRATE       mandatory fan-in stage after all selected dimensions are
                   terminal and the main agent has selected low/medium/high
                   winners from retained frontiers. Start a native integration
                   goal in a clean worktree. It stacks eligible dimension winners into composed
                   low/medium/high profiles, resolves code/config interactions,
                   then re-gates each merged profile on GPU. A failed composed
                   gate records an interaction failure and loops with a repaired
                   merge, reduced subset, or different tier plan. Empty tiers get
                   explicit blockers. Target composed speedups 1.5x / 2.0x /
                   3.0x (evals/tiers.toml [targets]).
5. DELIVER         the final matrix: per tier -> config (feature flags) + speedup +
                   peak-mem + quality verdict + rollback. Write the release report.
```

## 3. Spawn recipe (per dimension) — the fan-out
For each selected dimension, prepare a goal bundle and start a managed native
Codex goal session:
```
RUN_ID=${RUN_ID:-$(date -u +fanout_%Y%m%dT%H%M%SZ)}
FANOUT_ROOT=$PWD/output/fanout_runs/$RUN_ID
WT=$FANOUT_ROOT/<dim>
BR=codex/${RUN_ID}-<dim>
git worktree add -b $BR $WT <BASE>
CH=$WT/.codex-home; mkdir -p $CH; cp ~/.codex/{auth.json,config.toml} $CH/
(cd $WT && python3 tools/symposium/prepare_goal.py \
    --clean-stale-records \
    --run-id $RUN_ID)
(cd $WT && python3 tools/symposium/prepare_goal.py \
    --goal-id <dim> \
    --candidate candidates/baseline.toml \
    --dimension <dim> \
    --role implementation \
    --run-id $RUN_ID \
    --write-scope Sol-LTX-Infer/ \
    --root-branch $BR \
    --submodule-branch ${BR}-sol \
    --objective "Explore and implement the <dim> dimension from search_space/ by directly inspecting and modifying the target-model inference code.")
CODEX_HOME=$CH python3 tools/symposium/codex_goal_session.py start \
  --worktree $WT --name ${RUN_ID}-<dim> goals/<dim>
```
Use `status`, `capture`, and `send` to track and correct live subagents. When a
candidate is ready, spawn a separate gate goal (`--role gate`) or run the main
authoritative gate yourself; then review, gate, and merge by SHA. Cap N to what
you will review; CPU-only dimension prep runs on the park node, GPU candidate
runs go through Slurm. If a candidate fails, require the subagent to record the
failure signature and propose a meaningfully different next hypothesis. Use
`release` after the agent is closed or rejected; for repeated cap violations or
duplicate job launches, release the session instead of relying only on queued
steering text.

Before reusing a checkout for a new workflow, clean both durable stale records
and live tmux resources from the previous run. The filesystem cleaner does not
own tmux panes, so check them explicitly:

```bash
python3 tools/symposium/prepare_goal.py --clean-stale-records --run-id "$RUN_ID"
tmux ls | rg "$RUN_ID" || true
# Prefer managed release when the goal state exists:
python3 tools/symposium/codex_goal_session.py list
python3 tools/symposium/codex_goal_session.py release goals/<dim> \
  --worktree "$FANOUT_ROOT/<dim>" \
  --name ${RUN_ID}-<dim> \
  --note "stale run cleanup"
# If the state file was already removed, kill the exact old run sessions by name:
tmux kill-session -t ${RUN_ID}-<dim>
```

Do not leave old `workflow_$RUN_ID` or `$RUN_ID-<dim>`/`${RUN_ID}_<dim>` tmux
sessions alive after deleting their `output/fanout_runs/$RUN_ID` directory; they
are stale runtime state and can keep launching jobs or confuse status review.

## 3.1 Spawn recipe — the fan-in integration loop
After every selected fan-out dimension is terminal and no dimension has active
Slurm/collector/gate work, start exactly one integration goal. Do not mark the
experiment complete before this stage produces composed artifacts or explicit
per-tier blockers.

Use the runtime controller first; it reviews dimension status, refuses to start
while any dimension is still running or invalid, avoids duplicate integration
sessions, and starts the integration goal when the fan-out review reaches
`tier_selection_pending` or `integration_pending`:

```
RUN_ID=${RUN_ID:?reuse the same fanout run id}
FANOUT_ROOT=$PWD/output/fanout_runs/$RUN_ID
python3 tools/symposium/loop_control.py ensure-integration \
  --fanout-root "$FANOUT_ROOT" \
  --run-id "$RUN_ID" \
  --base <BASE>
```

The manual recipe below is a fallback/debug equivalent of the controller path.

```
RUN_ID=${RUN_ID:?reuse the same fanout run id}
FANOUT_ROOT=$PWD/output/fanout_runs/$RUN_ID
WT=$FANOUT_ROOT/integration
BR=codex/${RUN_ID}-integration
git worktree add -b $BR $WT <BASE>
CH=$WT/.codex-home; mkdir -p $CH; cp ~/.codex/{auth.json,config.toml} $CH/
(cd $WT && python3 tools/symposium/prepare_goal.py \
    --clean-stale-records \
    --run-id $RUN_ID)
(cd $WT && python3 tools/symposium/prepare_goal.py \
    --goal-id integration \
    --candidate candidates/baseline.toml \
    --dimension integration \
    --role integration \
    --run-id $RUN_ID \
    --root-branch $BR \
    --submodule-branch ${BR}-sol \
    --objective "Integrate fan-out winners into gated composed low, medium, and high delivery profiles for the 1.5x, 2.0x, and 3.0x speed targets. Read each dimension status and durable run artifacts, build target plans, merge one composed profile per iteration, launch GPU generation, run the authoritative aligned gate, rank quality with Gemini and LPIPS together, and loop on failures until every target has a composed artifact or explicit blocker.")
CODEX_HOME=$CH python3 tools/symposium/codex_goal_session.py start \
  --worktree $WT --name ${RUN_ID}-integration goals/integration
```

Integration acceptance:

- low/medium/high each has a gated composed manifest and video, or an explicit
  blocker such as `no_eligible_profile`;
- every composed profile has its own benchmark, collect artifacts, aligned LPIPS,
  and aligned pairwise Gemini verdict against the canonical baseline;
- each speed target chooses the best quality profile using Gemini severity/status
  and LPIPS together, then speed as tie-breaker;
- `INTEGRATION-STATUS.json` and `INTEGRATION-JOURNAL.md` separate per-dimension
  winners from true composed delivery profiles;
- failed merges or quality regressions are recorded as interaction failures and
  looped, not treated as completion.

## 4. Cluster / GPU
- Runs: `scripts/launch_candidate.py <candidate> --mode sbatch --confirm-submit`
  (HSG `batch`, 4 GPU/node). The profile `[env]` supplies the cache/python.
- Collect: `scripts/collect_run.py <run_dir>` -> `benchmark.json` + frames.
- Assess + speed-target bucket: `/home/haozhel/lustre/miniconda3/envs/sana/bin/python search/plan_eval.py --assess <run_dir> --baseline-frames <canonical-frames>`.
- Quality authority: OFF identity + aligned LPIPS + aligned pairwise Gemini.
  `quality.json` collector Gemini can be logged, but it cannot override the
  aligned gate.

## 5. Guardrails (hard)
- Dimensions stay model-agnostic; model specifics live ONLY in `models/` + the spec.
- During fan-out, quality is not a hard per-tier retention gate. Keep a candidate
  if quality improves or speed/memory improves; discard it if quality does not
  improve and speed/memory does not improve or regresses. Low/medium/high are
  1.5x/2.0x/3.0x speed targets; post-budget selection ranks quality with both
  aligned pairwise Gemini and LPIPS.
- OFF==baseline (byte/numeric) on guarded paths; WARMUP before quoting timings;
  never claim speedup from cold compile.
- Bounded loops (fixed max_iters); log any truncation; review before merge.
- A failed candidate gate means reject/log/loop, not finish. A successful
  candidate means retain it in `frontier_candidates` when quality or speed
  improves and keep searching for a better point until a stop condition applies.
- Every fan-out dimension defaults to max_iters=40 and early_stop_patience=0.
  Discarded/rejected candidates increment `no_improve_count` as telemetry;
  retained quality or speed improvements reset it. When a dimension hits budget,
  treat it as `terminal_pending_review` and decide whether to select tier
  winners from the frontier, reopen with a new direction, validate, integrate,
  drop, or mark a blocker.
- A dimension-agent `structured_negative` decision is a proposal/failure
  signature, not a fixed-budget loop stop.
- Runtime state is not prose-only: require `tools/symposium/loop_control.py`
  `init`, `record-candidate`, `decide-next`, and `validate-status` on every
  dimension loop. For main-agent fan-in review, use
  `python3 tools/symposium/loop_control.py review-dimensions --glob "$FANOUT_ROOT/*/AGENT-STATUS.json"`.
- The next candidate after a reject must address the recorded failure signature;
  do not allow cosmetic parameter sweeps of an already-disproven mechanism.
- Fan-out terminal state is not global completion. The run completes only after
  the integration loop produces composed low/medium/high artifacts or explicit
  per-tier blockers.
- Kill duplicate collectors/jobs for the same run; never let two processes race
  to write the same quality artifacts.

## 6. File map / quick commands
| What | Where |
| --- | --- |
| search harness | `search/search.py` (scan), `search/plan_eval.py` (render/assess/tier) |
| model adapter | `models/<id>.toml`, `efficiency/models/<id>_spec.py` |
| dimensions | `loops/<dim>/{dimension.toml, exploration.md, acceptance.md}` |
| search space | `search_space/` |
| engine | `efficiency/` (`selftest.py` = compose/seam test) |
| speed targets | `evals/tiers.toml`; visual rubric `evals/rubrics/gemini_visual_artifact_gate.md` |
| docs | `docs/{search-architecture,model-onboarding}.md` |
```bash
python search/search.py --model <id>                  # method families + diagnostics + speed targets
/home/haozhel/lustre/miniconda3/envs/sana/bin/python search/plan_eval.py --assess <run_dir> --baseline-frames <canonical-frames>   # benchmark+aligned quality+speed bucket
python scripts/launch_candidate.py <candidate> --mode sbatch --confirm-submit
~/lustre/miniconda3/envs/sana/bin/python efficiency/selftest.py        # 23/23
```

## 7. Current target — Cosmos3 (first model)
- Baseline verified (~130s, `models/cosmos3.toml [baseline]`).
- `search --model cosmos3`: **6 launchable method families** — `step_cache`, `teacache`,
  `token_prune`, `nvfp4_ffn`, `kwl_fusion`, `sparse_attention`.
- Subagents should inspect and patch the Cosmos3 inference path directly. Any
  adapter/seam cleanup is a main-agent integration concern after a candidate is
  proven useful.
- GPU eval harness: **VERIFIED end-to-end** (job 3294303, `runs/20260613-175619-baseline`):
  launch (sbatch) -> GPU run (127.83s) -> `collect_run.py` (`benchmark.json`, denoise 119.18s)
  -> Gemini pairwise judge (`overall=pass`, no artifacts) -> `tier_of` -> speed bucket, via
  `/home/haozhel/lustre/miniconda3/envs/sana/bin/python search/plan_eval.py --assess <run_dir> --baseline-frames <canonical-frames>`. NOTE: that candidate
  *is* the baseline config (no technique wired into the Cosmos3 denoise loop yet) -> 1.02x is
  run-variance and below the 1.5x low target. Real speedups need a technique wired into the runtime.

Supersedes the pre-search `agents/launch-agent.md` for the model-agnostic search era.
