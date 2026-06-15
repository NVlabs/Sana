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

ENV_FILE="${SYMPOSIUM_GOAL_ENV:-$ROOT/.symposium/goal-mode.env}"
if [[ "${SYMPOSIUM_SKIP_GOAL_ENV:-0}" != "1" && -f "$ENV_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$ENV_FILE"
fi
export PATH="$HOME/.local/bin:$HOME/bin:$HOME/.codex/bin:$PATH"
if [[ -z "${TERM:-}" || "$TERM" == "dumb" ]]; then
  export TERM=xterm-256color
fi

if [[ ! -t 0 || ! -t 1 ]]; then
  echo "Codex goal mode requires an interactive TTY; refusing non-interactive launch." >&2
  exit 4
fi

if [[ -n "${CODEX_GOAL_COMMAND:-}" ]]; then
  read -r -a CODEX_CMD <<< "$CODEX_GOAL_COMMAND"
elif command -v codex >/dev/null 2>&1; then
  CODEX_CMD=(codex)
else
  echo "No codex command found. Set CODEX_GOAL_COMMAND to the interactive Codex launcher." >&2
  exit 4
fi

echo "Starting interactive Codex goal session for $GOAL_DIR"
echo "Goal file: $GOAL_DIR/goal.md"
echo "Run inside Codex: /goal follow $GOAL_DIR/goal.md"
echo

exec "${CODEX_CMD[@]}"
