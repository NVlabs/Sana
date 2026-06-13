#!/usr/bin/env bash
set -euo pipefail

TARGET=""
SCOPE="project"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="${2:-}"
      shift 2
      ;;
    --scope)
      SCOPE="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$TARGET" != "codex" && "$TARGET" != "claude" ]]; then
  echo "Usage: ./scripts/install.sh --target codex|claude [--scope project|global]" >&2
  exit 2
fi

if [[ "$SCOPE" != "project" && "$SCOPE" != "global" ]]; then
  echo "--scope must be project or global" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$SCOPE" == "global" ]]; then
  if [[ "$TARGET" == "codex" ]]; then
    DEST="$HOME/.codex/skills"
  else
    DEST="$HOME/.claude/skills"
  fi
else
  if [[ "$TARGET" == "codex" ]]; then
    DEST="$ROOT/.codex/skills"
  else
    DEST="$ROOT/.claude/skills"
  fi
fi

mkdir -p "$DEST"
for skill in "$ROOT"/skills/*; do
  [[ -d "$skill" ]] || continue
  name="$(basename "$skill")"
  rm -rf "$DEST/$name"
  cp -R "$skill" "$DEST/$name"
done

echo "Installed Symposium skills to $DEST"
