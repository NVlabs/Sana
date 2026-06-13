#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: tools/symposium/start_claude_goal.sh goals/<goal-id>" >&2
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
  echo "Claude goal mode requires an interactive TTY; refusing non-interactive launch." >&2
  exit 4
fi

if [[ -n "${CLAUDE_GOAL_COMMAND:-}" ]]; then
  read -r -a CLAUDE_CMD <<< "$CLAUDE_GOAL_COMMAND"
elif command -v claude >/dev/null 2>&1; then
  CLAUDE_CMD=(claude)
else
  echo "No claude command found. Set CLAUDE_GOAL_COMMAND in .symposium/goal-mode.env." >&2
  exit 4
fi

GOAL_PROMPT="$(cat "$GOAL_DIR/goal.md")"

echo "Starting interactive Claude goal session for $GOAL_DIR"
echo "Goal prompt: $GOAL_DIR/goal.md"
echo

exec "${CLAUDE_CMD[@]}" "$GOAL_PROMPT"
