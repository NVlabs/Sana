# Multi-Agent Orchestration

This document sketches the next control-plane layer after M1.5: multiple agents
launching independent jobs with isolated source trees.

## Core Problem

Acceleration candidates will eventually need code edits inside `Sol-LTX-Infer`.
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
  worktrees/
    goals/
      20260612-token-prune/
        autovideo/
          Sol-LTX-Infer/
          runs/
      20260612-step-cache/
        autovideo/
          Sol-LTX-Infer/
          runs/
```

Each goal gets:

- `goal_id`
- root git branch, for example `codex/token-prune`
- submodule branch, for example `codex/token-prune-sol`
- independent `runs/<goal-id>/...`
- independent Slurm job names
- pinned `base_commit` and recorded resulting commits

## Lifecycle

```text
create goal
  -> create root worktree
  -> initialize submodule
  -> create submodule branch
  -> write goal.md/context.json
  -> enter interactive Codex
  -> send /goal follow <goal.md>
  -> run dimension loop:
       observe -> hypothesize -> implement one candidate
       -> preflight -> launch -> collect -> authoritative gate
       -> keep/reject and loop
  -> spawn independent gate goal or main-gate promising candidates
  -> stop only at max_iters, early_stop, real blocker, structured negative, or release
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
  "root_branch": "codex/token-prune",
  "submodule_branch": "codex/token-prune-sol",
  "candidate": "candidates/token_prune_feat_norm_075.toml",
  "status": "claimed"
}
```

This gives humans and other agents a cheap lock/coordination point without
requiring a central service.

## Candidate Ownership

Each candidate manifest should add an optional ownership block once we start
parallel goals:

```toml
[agent]
goal_id = "token-prune"
owner = "codex"
root_branch = "codex/token-prune"
submodule_branch = "codex/token-prune-sol"
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
candidate failure is not a completed dimension: the agent records the failure
signature and proposes a different next hypothesis. A single success is also not
completion: it updates best_per_tier and the loop continues until a stop
condition or orchestrator release.

Promotion decisions use the authoritative aligned gate, not prose and not
collector-only video-sampled Gemini:

- OFF identity when applicable;
- aligned LPIPS against canonical baseline frames;
- aligned pairwise Gemini;
- latency or peak-memory improvement.

The main agent should kill duplicate collectors/jobs for the same run and release
closed sessions that keep launching redundant jobs.

Fan-out terminal state is not global completion. After the selected dimensions
close, the main agent must start one fan-in integration goal that stacks eligible
per-tier winners, launches composed GPU runs, and re-gates each merged profile.
The experiment is complete only when every low/medium/high tier has a composed
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
