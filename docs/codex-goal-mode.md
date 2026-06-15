# Codex Goal Mode Bridge

This bridge lets the main orchestration agent hand off a bounded implementation
or gate task into Codex interactive mode.

The main orchestration agent itself is **not** a native goal. It runs as a
normal Codex session, reads `agents/orchestrator-entry.md`, observes the system,
spawns subagents, sends corrections, gates results, merges, and releases
resources. Only implementation and gate subagents follow `goals/<goal-id>/goal.md`.

## Constraint

Goal mode requires an interactive Codex session. Treat it as a human/agent
handoff, not as a background shell command. A non-interactive launcher may
prepare the workspace and prompt, but the actual goal execution must enter the
interactive mode adapter.

Symposium is vendored at `tools/symposium/vendor/Symposium` and adapted through
`tools/symposium/`. The project can start Claude Code goal sessions through
`tools/symposium/start_claude_goal.sh`; true Codex goal mode still requires a
real interactive Codex launcher configured as `CODEX_GOAL_COMMAND`.

## Handoff Contract

The bridge creates a goal bundle:

```text
goals/<goal-id>/
  goal.md
  context.json
  candidate.toml
```

`goal.md` is followed from inside Codex interactive mode with:

```text
/goal follow goals/<goal-id>/goal.md
```

It must include its own acceptance criteria and search-space-start section.

`context.json` records:

```json
{
  "goal_id": "token-prune",
  "created_by": "claude",
  "target_agent": "codex",
  "mode": "interactive-goal",
  "root_branch": "codex/token-prune",
  "submodule_branch": "codex/token-prune-sol",
  "candidate_manifest": "candidates/token_prune_feat_norm_075.toml",
  "write_scope": [],
  "launch_mode": "dry-run"
}
```

## Interactive Adapter Shape

The adapter script is:

```bash
tools/symposium/start_codex_goal.sh goals/<goal-id>
```

Responsibilities:

1. validate `context.json`
2. require an interactive TTY
3. source `.symposium/goal-mode.env`
4. require `codex` or `CODEX_GOAL_COMMAND`
5. open Codex interactive mode without passing the goal body as a normal prompt
6. send `/goal follow <goal.md>` into the interactive session

The adapter must not silently fall back to non-interactive execution.

## Commands

Install Symposium skills for Codex:

```bash
python3 tools/symposium/install_project_skills.py --target codex
```

Probe readiness:

```bash
python3 tools/symposium/probe_goal_mode.py --json
```

Configure the local launcher:

```bash
cp .symposium/goal-mode.env.example .symposium/goal-mode.env
# edit CODEX_GOAL_COMMAND to point at the interactive Codex CLI
```

Prepare a goal:

```bash
python3 tools/symposium/prepare_goal.py \
  --goal-id sparse-attention \
  --candidate candidates/baseline.toml \
  --objective "Use Symposium to refine sparse attention into a bounded Codex goal."
```

Start interactive mode when a Codex launcher exists:

```bash
tools/symposium/start_codex_goal.sh goals/sparse-attention
```

For detached managed sessions, use:

```bash
python3 tools/symposium/codex_goal_session.py start --worktree output/fanout/sparse-attention goals/sparse-attention
python3 tools/symposium/codex_goal_session.py capture --worktree output/fanout/sparse-attention goals/sparse-attention
python3 tools/symposium/codex_goal_session.py send --worktree output/fanout/sparse-attention goals/sparse-attention --text "Please pause and summarize status." --enter
python3 tools/symposium/codex_goal_session.py release --worktree output/fanout/sparse-attention goals/sparse-attention --note "done"
```

Start an interactive Claude goal session:

```bash
tools/symposium/start_claude_goal.sh goals/sparse-attention
```

## Goal Prompt Template

```markdown
# Goal: <goal-id>

You are working in an isolated autovideo worktree.

## Objective

<one bounded implementation or experiment objective>

## Search Space Start

- Search-space root: `search_space`
- Dimension brief: `loops/<dim>/exploration.md`
- Implementation surface: inspect and modify `Sol-LTX-Infer/` directly in the
  isolated worktree.

## Source Ownership

- Root branch: `<root_branch>`
- Submodule branch: `<submodule_branch>`
- Write scope:
  - `<path>`

## Candidate Contract

- Candidate manifest: `<candidate_manifest>`
- Launch with: `python3 scripts/launch_candidate.py <candidate> --mode dry-run`
- Collect with: `python3 scripts/collect_run.py runs/<run-id>`

## Done When

- implementation or manifest change is complete
- dry-run bundle can be generated
- relevant tests or static checks pass
- final notes explain artifacts and remaining blockers
```

## Remaining Follow-Up

After the actual local/NRT Codex interactive command is available, compare:

- how it creates interactive sessions
- how it passes goal prompts
- whether it uses tmux, pty, thread APIs, or a custom command
- how it records session IDs and logs
- how it handles cancellation and resume

Then update `.symposium/goal-mode.env` with the exact `CODEX_GOAL_COMMAND`.
