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
- `early_stop_patience` reached with no Pareto improvement or no new diagnostic
  information.
- A real blocker prevents meaningful progress and is recorded with the external
  dependency needed to unblock it.
- The dimension has enough evidence for a structured negative: the mechanism was
  tested across its meaningful knobs, failures share a root cause, and remaining
  variants are redundant.

A successful candidate also does not complete the dimension by itself. It is kept
as the current best per tier, then the loop continues looking for a better
candidate until the same stopping rules apply or the main orchestrator releases
the session.

The whole experiment is not complete when the fan-out dimensions become idle or
terminal. After fan-out closes, the main orchestrator must run the fan-in
integration stage, or record a real integration blocker. Empty queues, closed
tmux panes, and per-dimension summaries are only preconditions for integration.

## Per-Dimension State Machine

Each implementation goal follows this loop:

1. Observe state.
   Read `search_space/`, `loops/<dim>/`, prior `SEARCH_JOURNAL.md`, current
   best-per-tier candidates, failed signatures, and the canonical baseline.

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
   /home/haozhel/lustre/miniconda3/envs/sana/bin/python search/plan_eval.py \
     --assess <run_dir> \
     --baseline-frames /home/haozhel/lustre/auto-video/runs/20260613-175619-baseline/outputs/frames
   ```

   Promotion authority is:
   `OFF identity` + aligned LPIPS over the canonical frame pairs + aligned
   pairwise Gemini + speed/memory improvement. Collector `quality.json` is useful
   telemetry, but its video-sampled Gemini result is not promotion authority when
   it disagrees with the aligned gate.

7. Decide and loop.
   If promoted, update `best_per_tier` and continue to step 1. If rejected, write
   a failure signature and continue to step 1 with a meaningfully different
   hypothesis. If blocked or structured-negative, stop and write `SUMMARY.md`.

## Fan-In Integration State Machine

After every selected dimension is terminal, the main orchestrator starts one
integration goal in a clean worktree. The integration goal is also a loop, not a
one-shot merge.

1. Read standings.
   Load every dimension's `AGENT-STATUS.json`, `SUMMARY.md`, candidate manifests,
   run artifacts, and rejected failure signatures. Prefer durable run artifacts
   over stale status fields when the two disagree, and record the reconciliation.

2. Build tier plans.
   For each risk tier, choose eligible per-dimension winners whose own gate
   passed that tier or a stricter tier. If a tier has no eligible winners, write
   an explicit `no_eligible_profile` blocker for that tier instead of silently
   omitting it.

3. Implement one composed profile.
   Merge the chosen winners into one runtime/config profile for exactly one tier.
   Resolve shared-file conflicts by preserving each winner's OFF guard and
   feature flag. Do not report composed speedup from single-dimension runs.

4. Preflight and launch.
   Render the composed manifest/profile, run local checks, prove guarded OFF
   behavior, then submit the GPU generation for the composed profile.

5. Authoritative composed gate.
   Re-run benchmark, collection, aligned LPIPS, and aligned pairwise Gemini
   against the canonical baseline. A composed profile inherits no quality verdict
   from its components; it must pass its own gate.

6. Decide and loop.
   If the composed profile passes, keep it as the tier incumbent and continue
   searching for a faster compatible composition until the integration stop
   rules apply. If it fails, record an interaction failure signature and return
   to step 2 with a repaired merge, a reduced subset, or a different tier plan.

Integration stops only when every tier has either a gated composed profile or an
explicit blocker, or when `max_iters`, `early_stop_patience`, or a real external
blocker is reached. A failed composed gate never completes integration by itself.

## Failure Signature

Every rejected candidate writes a compact entry in `SEARCH_JOURNAL.md`:

- `candidate_id`, manifest, run directory, commit SHA.
- Gate status: OFF identity, speed/memory, LPIPS, pairwise Gemini, tier.
- Root cause class: `quality_cliff`, `perf_regression`, `off_identity_break`,
  `runtime_error`, `judge_inconclusive`, `implementation_noop`, or
  `orchestrator_cancelled`.
- Evidence path: exact artifact JSON/log paths.
- Next hypothesis requirement: what must change before a similar candidate can be
  retried.

The next candidate may not repeat the same failure signature with only cosmetic
parameter changes. It must change the mechanism, fix the diagnosed bug, or move
to a different region of the search space.

## Early Stop And Structured Negative

`early_stop_patience` counts iterations with no Pareto improvement and no new
diagnostic information. A dimension may stop early as structured negative only
when the summary explains:

- mechanisms tested;
- best speed/memory point and why it failed quality, or best clean point and why
  it failed speed;
- common root cause across rejected candidates;
- why the remaining untested variants are redundant or lower value.

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
- Once all selected dimensions are terminal, launch the integration goal and
  track it with the same job/process/queue discipline as the fan-out agents.
- Do not mark the experiment complete until integration has produced
  low/medium/high composed artifacts or explicit per-tier blockers.
- Never press or submit a stray commit/finalization prompt unless the user asked
  for that action.

## Required Outputs

At close, each dimension produces:

- `SEARCH_JOURNAL.md` with one entry per candidate and all failure signatures.
- `AGENT-STATUS.json` with `status`, `iters_used`, `best_per_tier`,
  `rejected_candidates`, `early_stop_reason` or `blocker`, and `next_commands`.
- Candidate manifests and run artifacts for every launched candidate.
- `SUMMARY.md` explaining winners, rejects, remaining hypotheses, and whether the
  dimension is promotable, blocked, or structured-negative.

At global close, integration produces:

- `INTEGRATION-STATUS.json` with per-tier state, chosen components, composed
  incumbents, rejected compositions, blockers, and next commands.
- `INTEGRATION-JOURNAL.md` with one entry per composed profile attempt and every
  interaction failure signature.
- Final low/medium/high composed manifests or explicit per-tier blockers.
- Run artifacts, benchmark, aligned quality artifacts, and videos for every
  launched composed profile.
- A release matrix that clearly separates per-dimension winners from gated
  composed delivery profiles.
