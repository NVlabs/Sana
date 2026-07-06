from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_create(tmp_path: Path, experiment_uid: str = "hunyuan-kernel_aw-9999") -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/create_model_experiment.py",
            "--model",
            "hunyuan_diffusers",
            "--workflow-uid",
            "kernel_aw",
            "--experiment-uid",
            experiment_uid,
            "--experiments-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(proc.stdout)


def test_create_model_experiment_copies_baseline_closure_only(tmp_path: Path) -> None:
    metadata = run_create(tmp_path)
    worktree = Path(metadata["worktree"])

    assert (worktree / "runtime/hunyuan_diffusers_baseline/gpu_infer.py").exists()
    assert (worktree / "runtime/hunyuan_diffusers_baseline/scripts/run_hunyuan_diffusers_gpu.sh").exists()
    assert not (worktree / "runtime/hunyuan_diffusers_baseline/step_cache_runtime.py").exists()
    assert (worktree / "candidates/hunyuan_diffusers_baseline.toml").exists()
    assert (worktree / "models/hunyuan_diffusers.toml").exists()
    assert (worktree / "models/hunyuan_diffusers/model.toml").exists()
    assert (worktree / "scripts/launch_candidate.py").exists()
    assert (worktree / "scripts/collect_run.py").exists()
    assert (worktree / "search/plan_eval.py").exists()
    assert (worktree / "tools/vision/nvidia_gemini_judge.py").exists()
    assert (worktree / "efficiency/candidate_manifest.py").exists()
    assert (worktree / "goals/kernel_aw/goal.md").exists()
    assert (worktree / "goals/kernel_aw/context.json").exists()

    assert not (worktree / "efficiency/techniques").exists()
    assert not (worktree / "efficiency/transforms").exists()
    assert not (worktree / "candidates/kwl_fusion").exists()
    assert not (worktree / "candidates/step_cache").exists()
    assert not (worktree / "search_space").exists()
    assert not (worktree / "workflow").exists()
    assert not any("__pycache__" in path.parts for path in worktree.rglob("*"))

    manifest = json.loads((worktree / "state/baseline_copy_manifest.json").read_text())
    assert manifest["copy_mode"] == "allowlist_minimal_runnable_closure"
    assert "candidates/hunyuan_diffusers_baseline.toml" in manifest["copied_paths"]
    assert all("__pycache__" not in rel for rel in manifest["copied_paths"])
    context = json.loads((worktree / "goals/kernel_aw/context.json").read_text())
    assert context["model_uid"] == "hunyuan_diffusers"
    assert context["aspect"] == "kernel"

    dry_run = subprocess.run(
        [
            sys.executable,
            "scripts/launch_candidate.py",
            "candidates/hunyuan_diffusers_baseline.toml",
            "--mode",
            "dry-run",
        ],
        cwd=worktree,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert "hunyuan_diffusers_baseline" in dry_run.stdout


def test_create_model_experiment_rejects_existing_experiment(tmp_path: Path) -> None:
    run_create(tmp_path, "hunyuan-kernel_aw-9998")
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/create_model_experiment.py",
            "--model",
            "hunyuan_diffusers",
            "--workflow-uid",
            "kernel_aw",
            "--experiment-uid",
            "hunyuan-kernel_aw-9998",
            "--experiments-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode != 0
    assert "refusing to overwrite" in proc.stderr.lower()


def test_create_model_experiment_rejects_legacy_workflow_uid_by_default(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/create_model_experiment.py",
            "--model",
            "hunyuan_diffusers",
            "--workflow-uid",
            "kr",
            "--experiment-uid",
            "hunyuan-kr-9997",
            "--experiments-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode != 0
    assert "<aspect>_<two_letter_code>" in proc.stderr


def test_sana_experiment_applies_hot_benchmark_overlay(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/create_model_experiment.py",
            "--model",
            "sana_video",
            "--workflow-uid",
            "attention_pa",
            "--experiment-uid",
            "sana-attention_pa-9996",
            "--experiments-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    metadata = json.loads(proc.stdout)
    worktree = Path(metadata["worktree"])
    overlay = ROOT / "models/sana_video/overlays/inference_video_scripts/inference_sana_video.py"
    installed = worktree / "external/sana_standalone/inference_video_scripts/inference_sana_video.py"

    assert metadata["baseline"]["overlay_copy_count"] == 1
    assert metadata["baseline_overlays"][0]["dest"] == str(installed.relative_to(worktree))
    assert installed.read_bytes() == overlay.read_bytes()
