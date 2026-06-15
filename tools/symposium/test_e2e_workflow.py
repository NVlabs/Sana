#!/usr/bin/env python3
"""CPU-only E2E smoke test for the main-agent -> native-subagent workflow."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
SANA_PY = Path.home() / "lustre/miniconda3/envs/sana/bin/python"
SEARCH_PY = str(SANA_PY) if SANA_PY.exists() else PY
DIMENSIONS = {
    "step_cache": "01_cache.md",
    "token_prune": "02_token_pruning.md",
    "nvfp4_ffn": "03_quantization.md",
    "sparse_attention": "04_sparse_attention.md",
    "kwl_fusion": "05_kernel_fusion.md",
}


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise AssertionError(f"{' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc


def main() -> int:
    search_space = ROOT / "search_space"
    check("search_space exists", search_space.is_dir())
    for file_name in DIMENSIONS.values():
        check(f"search_space/{file_name} exists", (search_space / file_name).exists())

    reference_files = list((ROOT / "reference").glob("**/*")) if (ROOT / "reference").exists() else []
    check("legacy reference files removed", not any(path.is_file() for path in reference_files))

    scratch = ROOT / ".symposium/scratch/e2e-workflow-goals"
    if scratch.exists():
        shutil.rmtree(scratch)

    for dim in DIMENSIONS:
        dim_dir = ROOT / "loops" / dim
        check(f"{dim} has exploration", (dim_dir / "exploration.md").exists())
        check(f"{dim} has acceptance", (dim_dir / "acceptance.md").exists())
        check(f"{dim} has no references.md", not (dim_dir / "references.md").exists())
        text = (dim_dir / "dimension.toml").read_text()
        check(f"{dim} has no fixed search grid", "[technique.search_space]" not in text)
        check(f"{dim} has no legacy seeds", "[[seeds]]" not in text)

        goal_id = f"e2e-{dim}"
        run(
            [
                PY,
                "tools/symposium/prepare_goal.py",
                "--goal-id",
                goal_id,
                "--candidate",
                "candidates/baseline.toml",
                "--dimension",
                dim,
                "--role",
                "implementation",
                "--objective",
                f"Explore {dim} by reading search_space and directly modifying inference code.",
                "--goals-root",
                str(scratch.relative_to(ROOT)),
            ]
        )
        goal_dir = scratch / goal_id
        goal = (goal_dir / "goal.md").read_text()
        context = json.loads((goal_dir / "context.json").read_text())
        check(f"{dim} goal has search-space section", "## Search Space Start" in goal)
        check(f"{dim} goal exposes inference repo", "Sol-LTX-Infer/" in goal)
        check(f"{dim} goal says direct modify", "modify" in goal and "inference code" in goal)
        check(f"{dim} goal has acceptance criteria", bool(context["acceptance_criteria"]))
        check(f"{dim} context search_space_root", context["search_space_root"] == "search_space")

    session_help = run([PY, "tools/symposium/codex_goal_session.py", "start", "--help"]).stdout
    check("session manager supports native goal start", "goal_dir" in session_help)
    check("session manager supports isolated worktree", "--worktree" in session_help)

    search_out = run([SEARCH_PY, "search/search.py", "--model", "cosmos3"]).stdout
    check("search reports launchable families", "launchable technique-dimensions" in search_out)
    check("search reports compose diagnostic", "compose-diagnostic" in search_out)

    print("e2e workflow smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
