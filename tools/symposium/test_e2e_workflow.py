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
    for search_doc in search_space.glob("*.md"):
        text = search_doc.read_text()
        check(f"{search_doc.name} is model-agnostic", "Cosmos3" not in text and "cosmos3" not in text)

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
        check(f"{dim} loop max_iters is 40", "max_iters = 40" in text)
        check(f"{dim} loop early_stop_patience is 0", "early_stop_patience = 0" in text)
        check(f"{dim} loop keeps frontier", 'keep = "frontier_config"' in text)

        goal_id = f"e2e-{dim}"
        run(
            [
                PY,
                "tools/symposium/prepare_goal.py",
                "--goal-id",
                goal_id,
                "--config",
                "config/wan22_ti2v_5b/baseline.toml",
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
        check(f"{dim} goal has method baseline catalog", "## Method Baseline Catalog" in goal)
        check(f"{dim} goal has history policy", "## Historical Record Policy" in goal)
        check(f"{dim} goal exposes inference repo", "Sol-LTX-Infer/" in goal)
        check(f"{dim} goal says direct modify", "modify" in goal and "inference code" in goal)
        check(f"{dim} goal uses target-model wording", "target-model" in goal and "inference code" in goal)
        check(f"{dim} goal does not hard-code Cosmos3 in generic wording", "Cosmos3 inference code" not in goal)
        check(f"{dim} goal has acceptance criteria", bool(context["acceptance_criteria"]))
        check(f"{dim} context search_space_root", context["search_space_root"] == "search_space")
        check(f"{dim} context search_space_doc", context["search_space_doc"] == f"search_space/{DIMENSIONS[dim]}")
        check(f"{dim} context method baselines", bool(context["method_baselines"]))
        check(f"{dim} has at least one wired/config baseline", any(item["tier"] in {"wired", "config_wired"} for item in context["method_baselines"]))
        check(f"{dim} context history policy", context["history_policy"]["mode"] == "clean_start_current_experiment_only")
        check(f"{dim} goal uses relevant search doc", f"Relevant search doc: `search_space/{DIMENSIONS[dim]}`" in goal)
        check(f"{dim} context max_iters", context["loop_contract"]["max_iters"] == 40)
        check(
            f"{dim} context review handoff",
            context["loop_contract"]["early_stop_exit_status"] == "terminal_pending_review",
        )

    session_help = run([PY, "tools/symposium/codex_goal_session.py", "start", "--help"]).stdout
    check("session manager supports native goal start", "goal_dir" in session_help)
    check("session manager supports isolated worktree", "--worktree" in session_help)

    search_out = run([SEARCH_PY, "search/search.py", "--model", "cosmos3"]).stdout
    check("search reports launchable families", "launchable technique-dimensions" in search_out)
    check("search reports compose diagnostic", "compose-diagnostic" in search_out)
    check("search reports method baselines", "method_baselines:" in search_out)

    print("e2e workflow smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
