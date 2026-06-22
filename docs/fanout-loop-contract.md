# Fan-Out Search Loop Contract

This contract exists because a native goal agent must not treat a dimension as a
single target checklist. Each dimension is a bounded search loop that proposes,
implements, gates, learns from, and improves candidates until a real stopping
condition is reached.

## Core Rule

A single candidate gate failure never completes a dimension. The agent must log
the failure cause, update the search state, and generate the next evidence-backed
hypothesis unless one of these stop conditions is true:

- `max_iters` reached.
- A real blocker prevents meaningful progress and is recorded with the external
  dependency needed to unblock it.
- The main orchestrator explicitly releases the dimension after reviewing the
  retained frontier and remaining hypotheses.

In default fixed-budget mode, a dimension agent may record a
`structured_negative` proposal, but it does not stop the loop by itself. The
runtime controller stores it as a failure signature and the loop continues until
budget, a real blocker, or explicit orchestrator release.

A successful candidate also does not complete the dimension by itself. During
fan-out, a candidate is retained when **quality improves** or **speed/memory
improves**. It is discarded only when neither quality nor speed/memory improves.
The loop continues until the fixed budget, a real blocker, or
main-orchestrator release.

Default fan-out budget is `max_iters = 40` and
`early_stop_patience = 0`, which means patience early stop is disabled in the
default fixed-budget frontier mode. `no_improve_count` remains telemetry for the
main agent, but it does not stop the default loop.

Low/medium/high are selected **after** the dimension budget closes, but they are
delivery speed targets, not hard LPIPS quality thresholds:

- low: best-quality profile at or above 1.5x.
- medium: best-quality profile at or above 2.0x.
- high: best-quality profile at or above 3.0x.

Within each speed target, quality ranking considers aligned pairwise Gemini and
LPIPS together: Gemini artifact severity/status first, aligned LPIPS second, then
higher speed as a tie-breaker. LPIPS alone is not the selector, and lossy
generative dimensions do not use absolute LPIPS/Gemini thresholds as hard gates
during loop retention or final target selection.

KWL/kernel optimization follows the same fixed-budget frontier rule as the
lossy dimensions: run the configured budget, retain candidates that improve
latency, peak memory, aligned quality, or reliable numeric stability, then pick
the low/medium/high winners after the budget closes. KWL does not require ON
bit-exactness, but it still has a strict semantic boundary. Semantic changes to
scheduler, step count, tokens, attention density, cache/prune behavior,
quantization policy, prompt state, LoRA state, frame count, resolution, or output
shape are rejects rather than KWL candidates.

Low-precision numeric dimensions may record reliable numeric checks such as OFF
identity for disabled paths, BF16 fallback integrity, precision-support proof,
and silent-fallback detection. They still record LPIPS and aligned pairwise
Gemini for cross-profile quality ranking, and they should only promote a hard
numeric gate when the candidate contract explicitly declares it.

The whole experiment is not complete when the fan-out dimensions become idle or
terminal. After fan-out closes, the main orchestrator must run the fan-in
integration stage, or record a real integration blocker. Empty queues, closed
tmux panes, and per-dimension summaries are only preconditions for integration.

## Per-Dimension State Machine

Each implementation goal follows this loop:

0. Initialize runtime loop control.
   Before proposing candidates, create the machine-readable state:

   ```bash
   python3 tools/symposium/loop_control.py init \
     --dimension <dim> \
     --goal-id <goal-id> \
     --max-iters 40 \
     --early-stop-patience 0 \
     --loop-mode fixed_budget_frontier
   ```

1. Observe state.
   Read `search_space/`, `loops/<dim>/`, this goal's current-experiment
   `SEARCH_JOURNAL.md`, current frontier candidates, discarded/rejected
   signatures, and the canonical baseline. Do not read stale `output/fanout/`,
   `output/fanout_loop_*`, release reports, archived captures, or verdict JSON
   from earlier experiments unless the main orchestrator explicitly passes them
   as current-experiment evidence.

2. Propose next hypothesis.
   Write a short hypothesis before implementation: what mechanism changes, why it
   should improve over the previous loop, what failure it is expected to avoid,
   and which prior failure signature it is not repeating.

3. Implement one candidate.
   Patch only the isolated worktree, keep OFF behavior guarded, and write a
   single candidate manifest/run plan. Do not batch unrelated mechanisms into one
   candidate.

4. Preflight locally.
   Run relevant static/unit checks, dry-run candidate rendering, and an OFF
   identity check whenever the dimension has an inactive path.

5. Launch and collect.
   Submit the GPU run only after preflight passes. Use a unique run directory and
   job name. Do not launch a replacement for a cancelled job unless the
   orchestrator explicitly says the cancellation was accidental.

6. Authoritative gate.
   Quality assessment must use the configured `sana` Python and canonical
   baseline frames:

   ```bash
   /lustre/fsw/portfolios/nvr/users/yitongl/miniconda3/envs/hunyuanvideo15/bin/python search/plan_eval.py \
     --assess <run_dir> \
     --baseline-frames /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/agent_deploy/Sol-LTX-Infer/runs/20260613-175619-baseline/outputs/frames \
     --out <run_dir>/assess_verdict.json
   ```

   The authoritative gate provides:
   `OFF identity` + aligned LPIPS over the canonical frame pairs + aligned
   pairwise Gemini + speed/memory evidence. Collector `quality.json` is useful
   telemetry, but its video-sampled Gemini result is not the sole quality
   authority when it disagrees with the aligned gate. For lossy generative
   dimensions, LPIPS and Gemini are joint quality-ranking evidence rather than
   absolute hard thresholds.

7. Decide and loop.
   If quality improves or speed/memory improves, retain the candidate in
   `frontier_candidates`, record the improvement axis, and continue to step 1. If
   neither quality nor speed/memory improves, record `discarded_regression`,
   increment telemetry, and continue to step 1. If
   the candidate is hard-invalid, write a `rejected` failure signature and
   continue to step 1 with a meaningfully different hypothesis. If blocked, stop
   and write `SUMMARY.md`. If the agent believes the mechanism space is
   structured-negative, record `structured_negative` with the evidence and a
   remaining-hypothesis note; the fixed-budget loop still continues unless the
   runtime controller returns `terminal_pending_review` or `blocked`.

   Every gate verdict must be recorded through the runtime controller:

   ```bash
   python3 tools/symposium/loop_control.py record-candidate \
     --candidate-id <id> \
     --decision <quality_improved|speed_improved|quality_and_speed_improved|discarded_regression|rejected|blocked|structured_negative> \
     --reason "<short reason>" \
     --purpose <frontier|delivery|evidence|blocker_probe|unsafe_probe|control> \
     --improvement-axis <quality|speed|both|none> \
     --run-dir <run_dir> \
     --evidence <run_dir>/assess_verdict.json
   python3 tools/symposium/loop_control.py decide-next
   python3 tools/symposium/loop_control.py validate-status
   python3 tools/symposium/loop_control.py status-summary
   ```

   For run-backed candidates, `record-candidate` requires a durable
   authoritative gate artifact in evidence: `assess_verdict.json`,
   `verdict.json`, `gate_assess.json`, or `reject_note.json`. Collector-only
   files such as `outputs/quality.json` are supplementary telemetry and cannot
   by themselves retain or reject a candidate. If an earlier current-experiment
   record is missing a durable gate artifact, write or regenerate the artifact
   first, then backfill the status through:

   ```bash
   python3 tools/symposium/loop_control.py add-evidence \
     --candidate-id <id> \
     --evidence <run_dir>/assess_verdict.json \
     --reason "backfilled authoritative gate artifact"
   python3 tools/symposium/loop_control.py validate-status
   ```

   The agent may not continue candidate search when `decide-next` returns
   `terminal_pending_review` or `blocked`.

   `purpose` keeps delivery records distinct from diagnostic evidence:

   - `frontier`: normal fan-out candidate retained for later tier selection.
   - `delivery`: composed integration candidate eligible for `best_per_tier`.
   - `evidence`: measured supporting evidence that must not become a tier winner.
   - `blocker_probe`: bounded probe used only to prove a target blocker.
   - `unsafe_probe`: intentionally unsafe/aggressive probe, never a delivery winner.
   - `control`: baseline, OFF identity, warmup, profile, or other non-scored run.

   Watchers and monitors must use `status-summary` or JSON parsing rather than
   grepping for one terminal string. Valid terminal states are `complete`,
   `terminal_pending_review`, and `blocked`.

When `max_iters` fires, the dimension enters
`status=terminal_pending_review`, not global completion. The main orchestrator
reviews the exit report and decides whether to accept the current frontier,
restart the dimension with a new direction, request validation, drop the
dimension, mark a blocker, or move it into fan-in integration.

## Fan-In Integration State Machine

After every selected dimension is terminal, the main orchestrator starts one
integration goal in a clean worktree. The integration goal is also a loop, not a
one-shot merge.

1. Read standings.
   Load every dimension's `AGENT-STATUS.json`, `SUMMARY.md`, candidate manifests,
   run artifacts, and rejected failure signatures. Prefer durable run artifacts
   over stale status fields when the two disagree, and record the reconciliation.

2. Build delivery-target plans.
   For each speed target (1.5x, 2.0x, 3.0x), choose eligible retained
   per-dimension winners and composed candidates by best joint quality:
   aligned pairwise Gemini severity/status first, aligned LPIPS second, then
   higher speed. If a target has no eligible winners, write an explicit
   `no_eligible_profile` blocker for that target instead of silently omitting it.

3. Implement one composed profile.
   Merge the chosen winners into one runtime/config profile for exactly one
   speed target.
   Resolve shared-file conflicts by preserving each winner's OFF guard and
   feature flag. Do not report composed speedup from single-dimension runs.

4. Preflight and launch.
   Render the composed manifest/profile, run local checks, prove guarded OFF
   behavior, then submit the GPU generation for the composed profile.

5. Authoritative composed gate.
   Re-run benchmark, collection, aligned LPIPS, and aligned pairwise Gemini
   against the canonical baseline. A composed profile inherits no quality verdict
   from its components; it must produce its own speed and joint quality evidence.

6. Decide and loop.
   If the composed profile improves a target bucket, keep it as that target's
   incumbent and continue searching for a faster or higher-quality compatible
   composition until the integration stop rules apply. If it fails, record an
   interaction failure signature and return to step 2 with a repaired merge, a
   reduced subset, or a different target plan.

   Integration records should mark gated target profiles with
   `--purpose delivery`. Non-delivery rows such as high-target upper-bound
   probes should use `--purpose blocker_probe` or `--purpose unsafe_probe`; they
   may support a blocker but must not update `best_per_tier`.

7. Final audit.
   Before reporting workflow completion, run:

   ```bash
   python3 tools/fanout_audit.py --run <fanout_run_id_or_path>
   ```

   The audit checks terminal integration status, release artifacts, pending
   release-matrix rows, durable evidence files, manifest `source_runs`, live
   Slurm jobs, live local assessment/launch processes, and run metadata
   lifecycle history.

Integration stops only when every 1.5x/2.0x/3.0x target has either a gated
composed profile or an explicit blocker, or when `max_iters` or a real external
blocker is reached. A failed composed gate never completes integration by itself.

## Failure Signature

Every rejected candidate writes a compact entry in `SEARCH_JOURNAL.md`:

- `candidate_id`, manifest, run directory, commit SHA.
- Gate status: OFF identity, speed/memory, LPIPS, pairwise Gemini, speed target.
- Root cause class: `quality_cliff`, `perf_regression`, `off_identity_break`,
  `runtime_error`, `judge_inconclusive`, `implementation_noop`, or
  `orchestrator_cancelled`.
- Evidence path: exact artifact JSON/log paths.
- Next hypothesis requirement: what must change before a similar candidate can be
  retried.

The next candidate may not repeat the same failure signature with only cosmetic
parameter changes. It must change the mechanism, fix the diagnosed bug, or move
to a different region of the search space.

## Structured Negative

In default fixed-budget frontier mode, `no_improve_count` is telemetry rather
than an automatic stop. A dimension-agent `structured_negative` is a proposal,
not a terminal state. It must be logged with:

- mechanisms tested;
- retained frontier candidates and discarded/rejected candidates;
- best speed/memory point, best quality point, and why neither justifies more
  variants;
- common root cause across rejected candidates;
- why the remaining untested variants are redundant or lower value.

The runtime controller records that proposal as a failure signature and adds a
remaining hypothesis. The loop continues to `max_iters` unless the main
orchestrator explicitly releases the dimension or a real blocker appears.

Frontier improvement means one of:

- a speed target is reached or a target bucket gets a better joint-quality
  candidate;
- a target bucket gets a meaningful latency or peak-memory improvement;
- quality improves at comparable speed;
- a previous failure mode becomes a gated pass;
- a new mechanism is proven useful enough to test during integration.

## Main-Orchestrator Duties

The main orchestration agent owns cross-agent safety:

- Preflight the gate stack before fan-out: `sana` env, LPIPS import, canonical
  baseline frame count, and baseline-vs-baseline identity.
- Treat `quality.json` collector Gemini as non-authoritative when aligned LPIPS
  or aligned pairwise gate contradict it.
- Kill duplicate collectors or duplicate Slurm jobs for the same run before they
  race on artifacts.
- When a closed/rejected dimension keeps submitting jobs, release the session
  rather than relying on queued steering text.
- Do not mark a dimension dead merely because its tmux session is idle; verify
  no process, Slurm job, or queued input remains.
- Once selected dimensions are terminal, choose 1.5x/2.0x/3.0x target winners
  from retained frontier candidates using Gemini+LPIPS quality ranking, then call
  `python3 tools/symposium/loop_control.py ensure-integration --fanout-root <run-root> --run-id <run-id> --base <base>`.
  The runtime trigger starts the integration goal when the fan-out review is
  ready, no-ops when integration is already running/complete, and refuses to
  start while dimensions are still running or invalid. Track integration with
  the same job/process/queue discipline as the fan-out agents.
- Do not mark the experiment complete until integration has produced
  low/medium/high composed artifacts or explicit per-tier blockers.
- Never press or submit a stray commit/finalization prompt unless the user asked
  for that action.

## Required Outputs

At close, each dimension produces:

- `SEARCH_JOURNAL.md` with one entry per candidate and all failure signatures.
- `AGENT-STATUS.json` with `status`, `iters_used`, `frontier_candidates`,
  `discarded_candidates`, `rejected_candidates`, `no_improve_count`,
  `terminal_reason` or `blocker`, `remaining_hypotheses`,
  `agent_recommendation`, and `next_commands`.
- Candidate manifests and run artifacts for every launched candidate.
- `SUMMARY.md` explaining winners, rejects, remaining hypotheses, and whether the
  dimension is promotable, blocked, or structured-negative.

At global close, integration produces:

- `INTEGRATION-STATUS.json` with per-tier state, chosen components, composed
  incumbents, rejected compositions, blockers, and next commands.
- `INTEGRATION-JOURNAL.md` with one entry per composed profile attempt and every
  interaction failure signature.
- Final 1.5x/2.0x/3.0x composed manifests or explicit per-target blockers.
- Run artifacts, benchmark, aligned quality artifacts, and videos for every
  launched composed profile.
- A release matrix that clearly separates per-dimension winners from gated
  composed delivery profiles.
