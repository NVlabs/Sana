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
export PATH="$HOME/.local/bin:$HOME/bin:$HOME/.codex/bin:$PATH"
if [[ -z "${TERM:-}" || "$TERM" == "dumb" ]]; then
  export TERM=xterm-256color
fi

# Exec mode: codex runs NON-INTERACTIVELY with the goal as its prompt.
# Rationale: the interactive TUI goal-mode does not render/accept input headless
# in a detached tmux pane (codex 0.133.0); `codex exec` does. No TTY is required,
# and stdin is /dev/null so exec never blocks reading stdin. The goal text is
# passed as the prompt instead of an interactive `/goal follow` slash command.

# Resolve the codex binary (first token of CODEX_GOAL_COMMAND, else `codex`).
if [[ -n "${CODEX_GOAL_COMMAND:-}" ]]; then
  read -r -a _CODEX_TOKENS <<< "$CODEX_GOAL_COMMAND"
  CODEX_BIN="${_CODEX_TOKENS[0]}"
elif command -v codex >/dev/null 2>&1; then
  CODEX_BIN="codex"
else
  echo "No codex command found. Set CODEX_GOAL_COMMAND." >&2
  exit 4
fi

# Default: no sandbox (the worktree is the containment) so codex does not require
# bubblewrap. Override via CODEX_EXEC_FLAGS when a usable sandbox/approval is set.
CODEX_EXEC_FLAGS="${CODEX_EXEC_FLAGS:---dangerously-bypass-approvals-and-sandbox}"

echo "Starting Codex EXEC goal session for $GOAL_DIR (non-interactive)"
echo "Goal file: $GOAL_DIR/goal.md"
echo "History policy: $AUTO_VIDEO_HISTORY_POLICY"
echo "codex bin: $CODEX_BIN | flags: $CODEX_EXEC_FLAGS"
echo

# shellcheck disable=SC2086  # CODEX_EXEC_FLAGS is intentionally word-split
exec "$CODEX_BIN" exec -C "$ROOT" $CODEX_EXEC_FLAGS \
  -o "$GOAL_DIR/agent_last.md" \
  "$(cat "$GOAL_DIR/goal.md")" </dev/null
