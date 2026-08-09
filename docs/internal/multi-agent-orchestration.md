# Multi-Agent Orchestration

This document sketches the next control-plane layer after M1.5: multiple agents
launching independent jobs with isolated source trees.

## Core Problem

Acceleration transfeat will eventually need code edits inside `Sol-LTX-Infer`.
If two agents share one submodule checkout, they can conflict on:

- git branches
- uncommitted patches
- generated compile caches
- Slurm output paths
- model/runtime environment variables

The orchestration layer should therefore give each agent an isolated execution
workspace.

## Recommended Shape

Run the main agent as a normal Codex orchestration session, not as a native goal.
The main agent observes the whole system, decides which dimensions to wake,
sends mid-run corrections, spawns independent gate agents, merges approved work,
and releases resources. Native `/goal follow` is only for implementation and
gate subagents.

Use one root repo worktree per agent goal, each with its own initialized
`Sol-LTX-Infer` submodule checkout. The current repo remains the coordinator;
goal worktrees are disposable execution sandboxes.

```text
autovideo/
  output/
    fanout_runs/
      fanout_YYYYMMDDTHHMMSSZ/
        token-prune/
          autovideo/
            Sol-LTX-Infer/
            runs/
        step-cache/
          autovideo/
            Sol-LTX-Infer/
            runs/
```

Each goal gets:

- `goal_id`
- root git branch, for example `codex/fanout_YYYYMMDDTHHMMSSZ-token-prune`
- submodule branch, for example `codex/fanout_YYYYMMDDTHHMMSSZ-token-prune-sol`
- independent `runs/<goal-id>/...`
- independent Slurm job names
- pinned `base_commit` and recorded resulting commits

## Lifecycle

```text
create goal
  -> create root worktree
  -> remove stale optimization records from that worktree
  -> initialize submodule
  -> create submodule branch
  -> write goal.md/context.json
  -> enter interactive Codex
  -> send /goal follow <goal.md>
  -> run dimension loop:
       observe -> hypothesize -> implement one transfeat
       -> preflight -> launch -> collect -> authoritative gate
       -> retain/discard/reject and loop
  -> spawn independent gate goal or main-gate promising transfeat
  -> stop only at max_iters, real blocker, or explicit release
  -> terminal_pending_review for main-agent tier-selection/restart/validate/integrate decision
  -> summarize and close
  -> release session resources
```

Root and submodule branches are separate because the root repo tracks orchestration
files and the submodule repo tracks implementation code.

## Agent Claim File

Before launch, create:

```text
runs/<goal-id>/agent.json
```

with:

```json
{
  "goal_id": "token-prune",
  "agent": "codex",
  "root_branch": "codex/fanout_YYYYMMDDTHHMMSSZ-token-prune",
  "submodule_branch": "codex/fanout_YYYYMMDDTHHMMSSZ-token-prune-sol",
  "transfeat": "transfeat/token_prune_feat_norm_075.toml",
  "status": "claimed"
}
```

This gives humans and other agents a cheap lock/coordination point without
requiring a central service.

## Transfeat Ownership

Each transfeat manifest should add an optional ownership block once we start
parallel goals:

```toml
[agent]
goal_id = "token-prune"
owner = "codex"
root_branch = "codex/fanout_YYYYMMDDTHHMMSSZ-token-prune"
submodule_branch = "codex/fanout_YYYYMMDDTHHMMSSZ-token-prune-sol"
write_scope = [
  "Sol-LTX-Infer/",
]
```

Agents should only write inside their declared scope, but that scope should
normally expose the full inference repo so exploration is not blocked by missing
interfaces.

## Slurm Isolation

Every job should use:

- unique run directory
- unique `OUT_DIR`
- unique Slurm job name
- shared model cache read-only where possible
- per-goal compile cache if compiler behavior is being measured

For apples-to-apples perf numbers, explicitly record whether compile caches were
cold or warm.

## Loop And Gate Discipline

Each native implementation goal follows `docs/fanout-loop-contract.md`. A single
transfeat failure is not a completed dimension: the agent records the failure
signature and proposes a different next hypothesis. A single success is also not
completion: it is retained in the frontier when quality or speed improves, and
the loop continues until a stop condition or orchestrator release.

Default fan-out loop budget is fixed max_iters=40 with early_stop_patience=0,
which disables patience early stop. Discarded/rejected transfeat increment
`no_improve_count` as telemetry; retained quality or speed improvements reset it.
Budget exits are `terminal_pending_review` handoffs to the main agent, which
decides whether to select low/medium/high winners from the frontier, reopen the
dimension with a new direction, validate, integrate, drop, or mark blocked.
A structured-negative decision from a dimension agent is logged as a
proposal/failure signature and does not stop the default fixed-budget loop by
itself.

Promotion decisions use the authoritative aligned gate, not prose and not
collector-only video-sampled Gemini:

- OFF identity when applicable;
- aligned LPIPS against canonical baseline frames;
- aligned pairwise Gemini;
- latency or peak-memory improvement.

Final low/medium/high winners are 1.5x/2.0x/3.0x speed targets. Within each
target, the selector ranks quality using aligned pairwise Gemini severity/status
and aligned LPIPS together, then speed as a tie-breaker. LPIPS alone is not the
selector.

The main agent should kill duplicate collectors/jobs for the same run and release
closed sessions that keep launching redundant jobs.

Stale cleanup includes tmux. `prepare_goal.py --clean-stale-records` removes old
reports, verdicts, worktrees, and run directories, but it cannot clean live tmux
sessions. Before starting a fresh workflow in a reused checkout, run
`tmux ls | rg "$RUN_ID"` for the old run id and release/kill exact matches. Use
`python3 tools/symposium/codex_goal_session.py release ... --worktree <WT> --name
<session>` when state files exist; use `tmux kill-session -t
<exact-session-name>` only for leftover sessions whose state files were already
removed. Old tmux sessions are runtime state, not harmless logs.

Fan-out terminal state is not global completion. After selected dimensions close,
the main agent must choose 1.5x/2.0x/3.0x target winners from retained frontiers,
then call the runtime integration trigger:

```bash
python3 tools/symposium/loop_control.py ensure-integration \
  --fanout-root output/fanout_runs/$RUN_ID \
  --run-id $RUN_ID \
  --base <BASE>
```

The trigger refuses to start while any dimension is still running or invalid,
no-ops when integration is already running/complete, and starts one fan-in goal
when the review reaches `tier_selection_pending` or `integration_pending`.
That goal stacks eligible winners, launches composed GPU runs, and re-gates each
merged profile.
The experiment is complete only when every 1.5x/2.0x/3.0x target has a composed
artifact or an explicit integration blocker.

## Goal Mode Bridge

Goal mode must not be a non-interactive fire-and-forget shell command. The bridge
should:

1. create or select an isolated goal worktree
2. write a goal prompt file
3. open an interactive Codex session in that worktree
4. send `/goal follow <goal.md>` inside the interactive session
5. expose `status`, `capture`, `send`, `attach`, `stop`, and `release` controls
6. keep all run artifacts in that goal worktree

The adapter lives at `tools/symposium/start_codex_goal.sh`; the tmux-backed
manager is `tools/symposium/codex_goal_session.py`. The main agent can run the
manager from the coordinator checkout and pass `--worktree <agent-worktree>` so
session state remains centrally visible while Codex edits and runs inside the
agent's isolated worktree.

See `docs/codex-goal-mode.md` for the bridge contract.
