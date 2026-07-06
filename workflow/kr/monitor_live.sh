#!/usr/bin/env bash
set -u

watch_mode=0
interval=60

if [[ "${1:-}" == "--watch" ]]; then
  watch_mode=1
  interval="${2:-60}"
  shift 2
fi

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 [--watch seconds] <experiment-dir>..." >&2
  exit 2
fi

emit_experiment() {
  local exp_dir="$1"
  local worktree="$exp_dir/worktree"
  echo "-- $exp_dir --"

  if [[ ! -d "$worktree" ]]; then
    echo "missing worktree: $worktree"
    return
  fi

  if [[ -f "$worktree/state/workflow-kr-events.jsonl" ]]; then
    tail -n 6 "$worktree/state/workflow-kr-events.jsonl"
  else
    echo "missing workflow event log"
  fi

  stat -c "workflow_state bytes=%s mtime=%y" "$worktree/state/workflow-kr-state.json" 2>/dev/null || true
  stat -c "agent_status bytes=%s mtime=%y" "$worktree/AGENT-STATUS.json" 2>/dev/null || true
  stat -c "agent_last bytes=%s mtime=%y" "$worktree/goals/kwl-retention/agent_last.md" 2>/dev/null || true

  git -C "$worktree" status --short 2>/dev/null | head -40 || true
}

emit_once() {
  printf "\n=== %s ===\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  tmux ls 2>/dev/null | grep -E 'kr-hunyuan|kr-monitor' || true
  pgrep -af 'workflow/kr/workflow.py run' || true
  squeue -u "${USER:-yitongl}" || true

  for exp_dir in "$@"; do
    emit_experiment "$exp_dir"
  done
}

while true; do
  emit_once "$@"
  if [[ "$watch_mode" -ne 1 ]]; then
    break
  fi
  sleep "$interval"
done
