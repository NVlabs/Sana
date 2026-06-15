# Symposium Tool Adapter

This directory vendors and adapts [Q00/Symposium](https://github.com/Q00/Symposium)
for `autovideo`. The upstream code is copied into this repo as regular source
files, not as a git submodule. See `VENDOR.json` for provenance.

Symposium is a Socratic skill pack for turning vague requests into precise,
testable Seeds. In this repo it is used before Codex implementation work:

```text
vague acceleration idea
  -> Symposium interview-harness
  -> final Seed / acceptance criteria
  -> Codex interactive goal mode
  -> candidate launch + collection
```

## Layout

| Path | Purpose |
| --- | --- |
| `VENDOR.json` | Upstream source URL and commit provenance. |
| `vendor/Symposium/` | Vendored upstream Symposium source files. |
| `install_project_skills.py` | Install Symposium skills into this project root. |
| `probe_goal_mode.py` | Check whether Symposium skills and interactive Codex goal-mode prerequisites are present. |
| `prepare_goal.py` | Create a goal bundle with `goal.md`, `context.json`, and candidate manifest. |
| `codex_goal_session.py` | Manage detached interactive Codex goal sessions through tmux. |
| `start_claude_goal.sh` | Start an interactive Claude session with the goal prompt. |
| `start_codex_goal.sh` | Guarded interactive adapter. It refuses to run without a TTY and Codex command. |
| `../../.symposium/goal-mode.env.example` | Example machine-local launcher configuration. |

## Install Symposium Skills Locally

Install skills for Codex in this project:

```bash
python3 tools/symposium/install_project_skills.py --target codex
```

Install for Claude:

```bash
python3 tools/symposium/install_project_skills.py --target claude
```

The copied skill files are generated local state and are ignored by git. The
tracked source of truth remains `vendor/Symposium/skills`.

## Launcher Configuration

Project-local launcher settings live in `.symposium/goal-mode.env`. That file is
ignored because it contains machine-specific paths. The tracked template is
`.symposium/goal-mode.env.example`.

On this machine, the configured Claude launcher is:

```bash
CLAUDE_GOAL_COMMAND="$HOME/.local/bin/claude"
```

Set `CODEX_GOAL_COMMAND` in `.symposium/goal-mode.env` when a real interactive
Codex launcher is available:

```bash
export CODEX_GOAL_COMMAND="/absolute/path/to/codex -C /absolute/path/to/auto-video --no-alt-screen"
```

The command must start interactive Codex. Goal text is not passed as a normal
CLI prompt; managed sessions send `/goal follow <goal.md>` after the pane starts.

## Probe

```bash
python3 tools/symposium/probe_goal_mode.py
```

The probe checks:

- the Symposium submodule
- project-local Codex/Claude skill install
- whether an interactive Codex command is available
- whether an interactive Claude command is available
- whether the current shell has a TTY

## Prepare A Goal

```bash
python3 tools/symposium/prepare_goal.py \
  --goal-id sparse-attention \
  --candidate candidates/baseline.toml \
  --dimension sparse_attention \
  --role implementation \
  --objective "Explore sparse attention from search_space/ by directly inspecting and modifying Cosmos3 inference code."
```

This writes:

```text
goals/<goal-id>/
  goal.md
  context.json
  candidate.toml
```

Each generated `goal.md` includes its own search-space-start section, required
artifacts, write scope, and acceptance criteria. Subagents should not need to
infer acceptance criteria from external orchestration docs.

## Start Codex Goal Mode

```bash
tools/symposium/start_codex_goal.sh goals/<goal-id>
```

This script is intentionally guarded. It only starts an interactive session when:

- stdin/stdout are attached to a TTY
- a Codex command is available through `PATH`, or `CODEX_GOAL_COMMAND` is set

Direct use starts interactive Codex and prints the native command to run:
`/goal follow goals/<goal-id>/goal.md`. For unattended fanout, prefer the
managed tmux session below, which sends that slash command automatically.

## Managed Codex Goal Sessions

Use the tmux-backed manager when an agent or human needs to monitor and keep
interacting with a Codex goal without owning the terminal forever.

Start a detached goal session:

```bash
python3 tools/symposium/codex_goal_session.py start goals/<goal-id>
```

Start a detached goal session in an isolated worktree while keeping the session
registry in the coordinator checkout:

```bash
python3 tools/symposium/codex_goal_session.py start \
  --worktree output/fanout/<goal-id> \
  --name <goal-id> \
  goals/<goal-id>
```

Check whether it is alive:

```bash
python3 tools/symposium/codex_goal_session.py status goals/<goal-id>
```

Capture the current screen:

```bash
python3 tools/symposium/codex_goal_session.py capture goals/<goal-id> --lines 80
```

Send follow-up text:

```bash
python3 tools/symposium/codex_goal_session.py send goals/<goal-id> \
  --text "Please pause after summarizing status." --enter
```

Attach interactively:

```bash
python3 tools/symposium/codex_goal_session.py attach goals/<goal-id>
```

Stop the session:

```bash
python3 tools/symposium/codex_goal_session.py stop goals/<goal-id>
```

Release resources and mark the session state released:

```bash
python3 tools/symposium/codex_goal_session.py release goals/<goal-id> \
  --note "gate complete"
```

Session metadata is written under `.symposium/scratch/codex-goal-sessions/`.

## Start Claude Goal Mode

```bash
tools/symposium/start_claude_goal.sh goals/<goal-id>
```

This starts Claude Code interactively with the generated `goal.md` prompt. It is
useful for Symposium interview/refinement and for testing the interactive
handoff shape, but it is not a substitute for true Codex goal mode.
