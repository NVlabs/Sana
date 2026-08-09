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
  -> transfeat launch + collection
```

## Layout

| Path | Purpose |
| --- | --- |
| `VENDOR.json` | Upstream source URL and commit provenance. |
| `vendor/Symposium/` | Vendored upstream Symposium source files. |
| `install_project_skills.py` | Install Symposium skills into this project root. |
| `probe_goal_mode.py` | Check whether Symposium skills and interactive Codex goal-mode prerequisites are present. |
| `prepare_goal.py` | Create a goal bundle with `goal.md`, `context.json`, and transfeat manifest. |
| `codex_goal_session.py` | Manage detached Codex autorun sessions and their exact tmux panes. |
| `start_claude_goal.sh` | Start an interactive Claude session with the goal prompt. |
| `start_codex_goal.sh` | Resolve `codex_auto_run.py` and launch a guarded Codex TUI goal session. |
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

Configure the autorun launcher, model, and sandbox in
`.symposium/goal-mode.env`:

```bash
export CODEX_AUTORUN="$HOME/codex_auto_run.py"
export CODEX_AUTORUN_MODEL="gpt-5.6-sol"
export CODEX_AUTORUN_SANDBOX="workspace-write"
```

`start_codex_goal.sh` also checks `$HOME/codex_auto_run.py` and
`$HOME/code/codex_exec/codex_auto_run.py` when `CODEX_AUTORUN` is unset. It
starts the managed TUI without an argv-sized prompt, waits for the Codex UI,
then delivers the goal through a file-backed tmux buffer. It explicitly selects
`gpt-5.6-sol` and uses `workspace-write` with `on-request` approvals. Do not set
`--bypass`.

## Probe

```bash
python3 tools/symposium/probe_goal_mode.py
```

The probe checks:

- the Symposium submodule
- project-local Codex/Claude skill install
- whether `codex_auto_run.py` is available with workspace-write support
- whether an interactive Codex command is available
- whether an interactive Claude command is available
- whether the current shell has a TTY

## Prepare A Goal

```bash
python3 tools/symposium/prepare_goal.py \
  --goal-id sparse-attention \
  --transfeat transfeat/wan22_ti2v_5b/baseline.toml \
  --dimension sparse_attention \
  --role implementation \
  --run-id ${RUN_ID:-} \
  --objective "Explore sparse attention from search_space/ by directly inspecting and modifying the target-model inference code."
```

This writes:

```text
goals/<goal-id>/
  goal.md
  context.json
  transfeat.toml
```

Each generated `goal.md` includes its own search-space-start section, fan-out
loop contract, required artifacts, write scope, and acceptance criteria.
Subagents should not need to infer acceptance criteria from external
orchestration docs. In particular, a failed transfeat gate means
discard-or-reject/log/loop; a successful transfeat means retain it in the
frontier when quality or speed improves and continue until max_iters, a real
blocker, or explicit orchestrator release. A structured-negative decision is
logged as a proposal/failure signature and does not stop the default
fixed-budget loop. The default fan-out budget is fixed max_iters=40 with
early_stop_patience=0, and budget exits should be written as
terminal_pending_review for main-agent 1.5x/2.0x/3.0x target selection and
review. Target selection ranks quality with aligned pairwise Gemini and LPIPS
together; LPIPS alone is not the selector.

Per-dimension goals embed only the relevant method-family document, for example
`step_cache` gets `search_space/01_cache.md` rather than the whole search-space
index.

Runtime loop accounting is machine-checked:

```bash
python3 tools/symposium/loop_control.py init --dimension <dim> --goal-id <goal-id> --max-iters 40 --early-stop-patience 0 --loop-mode fixed_budget_frontier
python3 tools/symposium/loop_control.py record-transfeat --transfeat-id <id> --decision rejected --reason "<reason>"
python3 tools/symposium/loop_control.py decide-next
python3 tools/symposium/loop_control.py validate-status
```

## Start Codex Goal Mode

```bash
tools/symposium/start_codex_goal.sh goals/<goal-id>
```

This script delegates to `codex_auto_run.py`. Direct use attaches when a TTY is
available; non-interactive use automatically detaches. The launcher rejects
full-access sandbox values and always passes the selected model explicitly.

## Managed Codex Goal Sessions

Use the manager when an agent or human needs to monitor and keep interacting
with a Codex goal without owning the terminal forever. The manager does not
create an outer tmux session; it records the exact pane created by
`codex_auto_run.py`.

Start a detached goal session:

```bash
python3 tools/symposium/codex_goal_session.py start goals/<goal-id>
```

Start a detached goal session in an isolated worktree while keeping the session
registry in the coordinator checkout:

```bash
RUN_ID=${RUN_ID:-$(date -u +fanout_%Y%m%dT%H%M%SZ)}
WT=output/fanout_runs/$RUN_ID/<goal-id>
# after creating the isolated worktree:
(cd $WT && python3 tools/symposium/prepare_goal.py --clean-stale-records --run-id $RUN_ID)
python3 tools/symposium/codex_goal_session.py start \
  --worktree $WT \
  --name ${RUN_ID}-<goal-id> \
  goals/<goal-id>
```

Use a fresh `RUN_ID` for each experiment. New worktrees should run
`prepare_goal.py --clean-stale-records --run-id $RUN_ID` before starting Codex.
Do not reuse `output/fanout/`, `output/fanout_loop_*`, old `evals/verdicts/*.json`,
release reports, or
archived session captures as startup context for a new goal.

Clean live tmux state separately when a run is abandoned or before reusing a
checkout. The stale-record cleaner intentionally manages files, not running
terminal sessions:

```bash
tmux ls | rg "$RUN_ID" || true
python3 tools/symposium/codex_goal_session.py list
python3 tools/symposium/codex_goal_session.py release goals/<goal-id> \
  --worktree "$WT" \
  --name ${RUN_ID}-<goal-id> \
  --note "stale run cleanup"
```

If the session registry was already deleted, kill only exact old run sessions:

```bash
tmux kill-session -t ${RUN_ID}-<goal-id>
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

### Autorun Lifecycle

Managed goals are persistent Codex TUI sessions. The autorun watcher handles
recognized command/edit/network approval overlays while Codex remains in
`workspace-write`; it does not approve full-access screens or enable bypass.
Use `status`, `capture`, and `send` to supervise progress and provide follow-up
instructions. Use `stop` or `release` only for the exact recorded session.

Autorun does not invoke `stop_hook.py after-agent`, because the interactive TUI
does not exit after every response. Executors must update durable
`AGENT-STATUS.json`, `SEARCH_JOURNAL.md`, and evaluation artifacts before
requesting handoff. The orchestrator inspects those artifacts and starts the
reviewer goal. The reviewer writes `REVIEWER-STATUS.json` at the worktree root:

```json
{
  "schema_version": 1,
  "target_goal_id": "kwl-fusion",
  "status": "accepted",
  "decision": "accept",
  "reason": "smooth full evaluation exists and no credible local optimization remains",
  "required_followups": [],
  "evidence": ["runs/<run-id>/assess_verdict.json"]
}
```

If the reviewer writes `"status": "needs_executor_resume"`, send the required
follow-up to the still-live executor session or resume the persisted Codex
conversation. A workflow is accepted only when the reviewer writes
`status="accepted"` and `decision="accept"`. The launcher consumes a pre-existing
`STOP_HOOK_RESUME.md` once for migration compatibility, but it does not create a
process-exit resume loop.

## Indexed Experiment Isolation

Use indexed experiments when running many agents in parallel or when a new run
must not see prior agent edits. Each experiment id gets a fresh git worktree
from a shared baseline commit plus private `goals/`, `runs/`, `state/`, and
compile/cache directories.

Create a clean Hunyuan KWL experiment from the current committed baseline:

```bash
python3 tools/symposium/experiment.py create \
  --experiment-id 15001 \
  --base-ref <baseline-commit-or-branch> \
  --dimension kwl_fusion \
  --model-id hunyuan_diffusers
```

This writes:

```text
output/experiments/15001/
  experiment.json
  worktree/
    goals/kwl-fusion/
    runs/
    state/
    caches/tmp/
    caches/triton/
    caches/torch_extensions/
```

Start it:

```bash
python3 tools/symposium/experiment.py start --experiment-id 15001
```

Check or stop it:

```bash
python3 tools/symposium/experiment.py status --experiment-id 15001
python3 tools/symposium/experiment.py stop --experiment-id 15001
```

Isolation rules:

- `create` refuses to reuse an existing experiment id.
- the worktree starts from `base_sha`, not from dirty coordinator files;
- `start` refuses to run if local `AGENT-STATUS.json`, `SEARCH_JOURNAL.md`,
  `SUMMARY.md`, or `runs/` already exists, unless `--resume` is passed;
- `TMPDIR`, `TRITON_CACHE_DIR`, and `TORCH_EXTENSIONS_DIR` point inside the
  experiment worktree;
- tmux session names include the experiment id, for example
  `exp-15001-kwl-fusion`.

For a truly common starting point across many ids, commit the control-plane
changes first and pass that same commit as `--base-ref` to every `create`
command. Uncommitted coordinator edits are recorded in `experiment.json` but are
not copied into the experiment worktree.

## Start Claude Goal Mode

```bash
tools/symposium/start_claude_goal.sh goals/<goal-id>
```

This starts Claude Code interactively with the generated `goal.md` prompt. It is
useful for Symposium interview/refinement and for testing the interactive
handoff shape, but it is not a substitute for true Codex goal mode.
