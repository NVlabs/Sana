#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: tools/symposium/start_codex_goal.sh goals/<goal-id>" >&2
  exit 2
fi

GOAL_DIR="$1"
if [[ ! -d "$GOAL_DIR" ]]; then
  echo "Goal directory does not exist: $GOAL_DIR" >&2
  exit 2
fi
if [[ ! -f "$GOAL_DIR/goal.md" || ! -f "$GOAL_DIR/context.json" ]]; then
  echo "Goal directory must contain goal.md and context.json: $GOAL_DIR" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export AUTO_VIDEO_HISTORY_POLICY="clean_start_current_experiment_only"
export AUTO_VIDEO_GOAL_DIR="$GOAL_DIR"
CURRENT_RUN_ID="${SYMPOSIUM_CURRENT_RUN_ID:-${AUTO_VIDEO_RUN_ID:-${RUN_ID:-}}}"

if [[ -z "$CURRENT_RUN_ID" && -f "$GOAL_DIR/context.json" ]]; then
  CURRENT_RUN_ID="$(python3 - "$GOAL_DIR/context.json" <<'PY' || true
import json
import sys

try:
    value = json.loads(open(sys.argv[1], encoding="utf-8").read()).get("run_id", "")
except Exception:
    value = ""
print(value or "")
PY
)"
fi
if [[ -z "$CURRENT_RUN_ID" && "$ROOT" =~ /output/fanout_runs/([^/]+)(/|$) ]]; then
  CURRENT_RUN_ID="${BASH_REMATCH[1]}"
fi
if [[ -n "$CURRENT_RUN_ID" ]]; then
  export SYMPOSIUM_CURRENT_RUN_ID="$CURRENT_RUN_ID"
fi

if [[ "${SYMPOSIUM_PRESERVE_HISTORY_RECORDS:-0}" != "1" || "${SYMPOSIUM_CLEAN_HISTORY_RECORDS:-0}" == "1" ]]; then
  python3 tools/symposium/prepare_goal.py --clean-stale-records --run-id "$CURRENT_RUN_ID"
fi

if [[ "${SYMPOSIUM_ALLOW_HISTORY_RECORDS:-0}" != "1" ]]; then
  if ! python3 tools/symposium/prepare_goal.py --check-stale-records --run-id "$CURRENT_RUN_ID"; then
    echo "Refusing to start goal because stale optimization records are visible in this checkout." >&2
    echo "Move/delete them, start from a clean run-id worktree, or set SYMPOSIUM_ALLOW_HISTORY_RECORDS=1 explicitly." >&2
    exit 5
  fi
fi

ENV_FILE="${SYMPOSIUM_GOAL_ENV:-$ROOT/.symposium/goal-mode.env}"
if [[ "${SYMPOSIUM_SKIP_GOAL_ENV:-0}" != "1" && -f "$ENV_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$ENV_FILE"
fi

# Detached tmux servers can retain an environment from before shell-level API
# credentials were configured. Recover the exported key from the user's login
# shell without printing or persisting its value in experiment files.
if [[ -z "${NVIDIA_API_KEY:-}" ]]; then
  LOGIN_NVIDIA_API_KEY="$(bash -lc 'printf "%s" "${NVIDIA_API_KEY:-}"' 2>/dev/null || true)"
  if [[ -n "$LOGIN_NVIDIA_API_KEY" ]]; then
    export NVIDIA_API_KEY="$LOGIN_NVIDIA_API_KEY"
  fi
  unset LOGIN_NVIDIA_API_KEY
fi
export PATH="$HOME/.local/bin:$HOME/bin:$HOME/.codex/bin:$PATH"
if [[ -z "${TERM:-}" || "$TERM" == "dumb" ]]; then
  export TERM=xterm-256color
fi

resolve_codex_autorun() {
  local transfeat
  if [[ -n "${CODEX_AUTORUN:-}" ]]; then
    if [[ -f "$CODEX_AUTORUN" && -x "$CODEX_AUTORUN" ]]; then
      printf '%s\n' "$CODEX_AUTORUN"
      return 0
    fi
    echo "CODEX_AUTORUN is not an executable file: $CODEX_AUTORUN" >&2
    return 1
  fi
  for transfeat in "$HOME/codex_auto_run.py" "$HOME/code/codex_exec/codex_auto_run.py"; do
    if [[ -f "$transfeat" && -x "$transfeat" ]]; then
      printf '%s\n' "$transfeat"
      return 0
    fi
  done
  return 1
}

AUTORUN_BIN="$(resolve_codex_autorun)" || {
  echo "No codex_auto_run.py launcher found. Set CODEX_AUTORUN." >&2
  exit 4
}
AUTORUN_MODEL="${CODEX_AUTORUN_MODEL:-gpt-5.6-sol}"
AUTORUN_SANDBOX="${CODEX_AUTORUN_SANDBOX:-workspace-write}"
case "$AUTORUN_SANDBOX" in
  read-only|workspace-write) ;;
  *)
    echo "CODEX_AUTORUN_SANDBOX must be read-only or workspace-write; got: $AUTORUN_SANDBOX" >&2
    exit 4
    ;;
esac

CODEX_BIN="${CODEX_AUTORUN_CODEX_BINARY:-}"
if [[ -z "$CODEX_BIN" && -n "${CODEX_GOAL_COMMAND:-}" ]]; then
  read -r -a _CODEX_TOKENS <<< "$CODEX_GOAL_COMMAND"
  CODEX_BIN="${_CODEX_TOKENS[0]}"
fi

PROMPT_FILE="$GOAL_DIR/goal.md"
TEMP_PROMPT=""
cleanup_prompt() {
  if [[ -n "$TEMP_PROMPT" && -f "$TEMP_PROMPT" ]]; then
    rm -f "$TEMP_PROMPT"
  fi
}
trap cleanup_prompt EXIT

RESUME_FILE="$GOAL_DIR/STOP_HOOK_RESUME.md"
if [[ -f "$RESUME_FILE" ]]; then
  TEMP_PROMPT="$(mktemp "${TMPDIR:-/tmp}/symposium-codex-prompt.XXXXXX")"
  chmod 600 "$TEMP_PROMPT"
  {
    cat "$GOAL_DIR/goal.md"
    printf '\n\n## Workflow Resume\n\n'
    cat "$RESUME_FILE"
  } > "$TEMP_PROMPT"
  mv "$RESUME_FILE" "$GOAL_DIR/STOP_HOOK_RESUME.last.md"
  PROMPT_FILE="$TEMP_PROMPT"
fi

AUTORUN_ARGS=(
  "$AUTORUN_BIN"
  -C "$ROOT"
)
# Orchestrator/executor agents (workflow_lite) must spawn nested tmux sessions
# and reach Slurm; the workspace-write sandbox blocks the tmux socket under /tmp
# (Landlock). Opt into Codex's no-sandbox mode when explicitly requested by a
# trusted, operator-launched autonomous run.
if [[ "${SYMPOSIUM_AUTORUN_BYPASS:-0}" == "1" ]]; then
  AUTORUN_ARGS+=(--bypass)
else
  AUTORUN_ARGS+=(--sandbox "$AUTORUN_SANDBOX")
fi
if [[ "${SYMPOSIUM_AUTORUN_DETACH:-0}" == "1" || ! -t 0 || ! -t 1 ]]; then
  AUTORUN_ARGS+=(--detach)
fi
if [[ -n "${SYMPOSIUM_AUTORUN_SESSION_PREFIX:-}" ]]; then
  AUTORUN_ARGS+=(--session-prefix "$SYMPOSIUM_AUTORUN_SESSION_PREFIX")
fi
if [[ -n "${CODEX_AUTORUN_RUNTIME_DIR:-}" ]]; then
  AUTORUN_ARGS+=(--runtime-dir "$CODEX_AUTORUN_RUNTIME_DIR")
fi
if [[ "${CODEX_AUTORUN_AUTO_TRUST_DIRECTORY:-0}" == "1" ]]; then
  AUTORUN_ARGS+=(--auto-trust-directory)
fi
if [[ -n "$CODEX_BIN" ]]; then
  AUTORUN_ARGS+=(--codex-binary "$CODEX_BIN")
fi
AUTORUN_ARGS+=(-- --model "$AUTORUN_MODEL" --config check_for_update=false)

echo "Starting Codex autorun goal session for $GOAL_DIR"
echo "Goal file: $GOAL_DIR/goal.md"
echo "History policy: $AUTO_VIDEO_HISTORY_POLICY"
echo "autorun: $AUTORUN_BIN"
if [[ "${SYMPOSIUM_AUTORUN_BYPASS:-0}" == "1" ]]; then
  echo "model: $AUTORUN_MODEL | sandbox: BYPASS (no-sandbox, no approvals)"
else
  echo "model: $AUTORUN_MODEL | sandbox: $AUTORUN_SANDBOX | approvals: on-request"
fi
echo

if ! LAUNCH_OUTPUT="$("${AUTORUN_ARGS[@]}" 2>&1)"; then
  printf '%s\n' "$LAUNCH_OUTPUT" >&2
  exit 1
fi

AUTORUN_SESSION="$(
  printf '%s\n' "$LAUNCH_OUTPUT" \
    | sed -n 's/^Codex running in tmux session: //p' \
    | tail -n 1
)"
if [[ -z "$AUTORUN_SESSION" ]]; then
  printf '%s\n' "$LAUNCH_OUTPUT" >&2
  echo "Could not determine the tmux session created by codex_auto_run.py." >&2
  exit 1
fi

# codex_auto_run.py currently expands prompt-file contents into the tmux
# process argv. Large workflow prompts exceed tmux's command-size limit, so
# launch the managed TUI first and deliver the prompt through a file-backed
# tmux buffer instead.
PANE_READY=0
for _ in $(seq 1 120); do
  PANE_COMMAND="$(tmux display-message -p -t "$AUTORUN_SESSION" '#{pane_current_command}' 2>/dev/null || true)"
  if [[ "$PANE_COMMAND" == "codex" ]]; then
    PANE_CAPTURE="$(tmux capture-pane -p -J -t "$AUTORUN_SESSION" -S -30 2>/dev/null || true)"
    if [[ "$PANE_CAPTURE" == *"OpenAI Codex"* ]]; then
      PANE_READY=1
      break
    fi
  fi
  sleep 0.25
done
if [[ "$PANE_READY" != "1" ]]; then
  tmux kill-session -t "$AUTORUN_SESSION" 2>/dev/null || true
  printf '%s\n' "$LAUNCH_OUTPUT" >&2
  echo "Codex TUI did not become ready for prompt delivery." >&2
  exit 1
fi

BUFFER_NAME="symposium-${AUTORUN_SESSION//[^A-Za-z0-9_-]/-}-$$"
tmux load-buffer -b "$BUFFER_NAME" "$PROMPT_FILE"
tmux paste-buffer -p -d -b "$BUFFER_NAME" -t "$AUTORUN_SESSION"
tmux send-keys -t "$AUTORUN_SESSION" Enter

printf '%s\n' "$LAUNCH_OUTPUT"
