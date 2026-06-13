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
  -> enter interactive agent mode
  -> launch candidate
  -> collect run
  -> summarize and close
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
  "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/models/dits/cosmos3video.py",
  "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/efficiency/models/cosmos3_spec.py",
]
```

Agents should only write inside their declared scope.

## Slurm Isolation

Every job should use:

- unique run directory
- unique `OUT_DIR`
- unique Slurm job name
- shared model cache read-only where possible
- per-goal compile cache if compiler behavior is being measured

For apples-to-apples perf numbers, explicitly record whether compile caches were
cold or warm.

## Goal Mode Bridge

Claude-to-Codex goal mode should not be a non-interactive fire-and-forget shell
command if the target mode requires interactive approval. The bridge should:

1. create or select an isolated goal worktree
2. write a goal prompt file
3. open an interactive Codex session in that worktree
4. pass the goal prompt as the initial instruction
5. keep all run artifacts in that goal worktree

The current adapter lives at `tools/symposium/start_codex_goal.sh`. It validates
the goal bundle and refuses to launch without a TTY and a Codex command.

See `docs/codex-goal-mode.md` for the bridge contract.
