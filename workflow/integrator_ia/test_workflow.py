#!/usr/bin/env python3
"""Contract tests for the workflow-local Integrator IA graph."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


WORKFLOW_DIR = Path(__file__).resolve().parent
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

from nodes.codex_visual_reviewer.node import decode_blind_verdict, selected_recipe_runs  # noqa: E402
from nodes.final_gate.node import run as run_final_gate  # noqa: E402
from nodes.integration_gate.node import run as run_integration_gate  # noqa: E402
from nodes.source_gate.node import run as run_source_gate  # noqa: E402
from workflow import transition  # noqa: E402
from workflow_types import NodeContext, NodeResult  # noqa: E402


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context(root: Path, worktree: Path, config: dict[str, Any] | None = None) -> NodeContext:
    return NodeContext(
        root=root,
        workflow_dir=WORKFLOW_DIR,
        worktree=worktree,
        goal_dir=worktree / "goals" / "integrator_ia",
        state_path=worktree / "state" / "workflow-integrator_ia-state.json",
        event_log=worktree / "state" / "workflow-integrator_ia-events.jsonl",
        state={"experiment_uid": "sana-integrator_ia-0001"},
        config=config or {},
        env={},
    )


def create_donors(root: Path) -> tuple[Path, Path, Path]:
    kernel = root / "donors" / "sana-kernel_aw-0005" / "worktree"
    pisa = root / "donors" / "sana-attention_pa-0002" / "worktree"
    cache = root / "donors" / "sana-cache_ca-0005" / "worktree"

    (kernel / "candidates").mkdir(parents=True)
    (kernel / "candidates" / "kernel.toml").write_text(
        "id = \"kernel-v1\"\n[verification]\nimplementation_files = [\"external/kernel.py\"]\n"
    )
    write_json(kernel / "runs" / "kernel" / "gate.json", {"speedup": 2.0})
    write_json(
        kernel / "AGENT-STATUS.json",
        {
            "experiment_uid": "sana-kernel_aw-0005",
            "status": "running",
            "canonical_on_manifest": {
                "manifest_path": "candidates/kernel.toml",
                "candidate_ids": ["kernel-a", "kernel-b"],
                "latest_integrated_gate": "runs/kernel/gate.json",
                "latest_source_current_on_speedup": 2.0,
            },
        },
    )

    (pisa / "candidates").mkdir(parents=True)
    (pisa / "candidates" / "pisa.toml").write_text(
        "id = \"pisa-v1\"\n[patch]\ntouch_points = [\"external/pisa.py\"]\n"
    )
    write_json(pisa / "runs" / "pisa" / "assess_verdict.json", {"codex_visual_overall": "pass"})
    write_json(
        pisa / "PISA-RECIPES.json",
        {
            "recipes": {
                "visually_indistinguishable": {
                    "status": "measured",
                    "candidate_id": "pisa-v1",
                    "run_dir": "runs/pisa",
                    "assess_verdict": "runs/pisa/assess_verdict.json",
                    "speedup": 1.1,
                    "codex_visual_overall": "pass",
                    "max_artifact_severity": "none",
                    "launch": {"candidate_manifest": "candidates/pisa.toml"},
                }
            }
        },
    )

    (cache / "candidates").mkdir(parents=True)
    (cache / "candidates" / "cache-v1.toml").write_text("id = \"cache-v1\"\n")
    write_json(cache / "runs" / "cache" / "assess_verdict.json", {"codex_visual_overall": "pass"})
    write_json(
        cache / "AGENT-STATUS.json",
        {
            "experiment_uid": "sana-cache_ca-0005",
            "status": "running",
            "evidence": ["external/cache.py"],
            "candidates": [
                {
                    "candidate_id": "cache-v1",
                    "family": "taylorseer",
                    "run_dir": "runs/cache",
                    "speedup": 1.2,
                    "codex_visual_overall": "pass",
                    "max_artifact_severity": "none",
                }
            ],
        },
    )

    for donor, component in ((kernel, "kernel"), (pisa, "pisa"), (cache, "cache")):
        source = donor / "external" / f"{component}.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"COMPONENT = \"{component}\"\n")
    return kernel, pisa, cache


def pin_sources(root: Path, worktree: Path) -> tuple[NodeContext, dict[str, Any]]:
    kernel, pisa, cache = create_donors(root)
    ctx = context(
        root,
        worktree,
        {
            "kernel_delivery": str(kernel),
            "kernel_manifest": "",
            "pisa_delivery": str(pisa),
            "pisa_recipe": "visually_indistinguishable",
            "cache_delivery": str(cache),
            "cache_candidate": "cache-v1",
        },
    )
    result = run_source_gate(ctx)
    assert result.outcome == "ready", result.message
    inventory = json.loads((worktree / "state" / "integration-source-inventory.json").read_text())
    assert inventory["status"] == "ready"
    assert inventory["sources"]["kernel"]["selection"]["candidate_ids"] == ["kernel-a", "kernel-b"]
    assert inventory["sources"]["pisa"]["selection"]["candidate_id"] == "pisa-v1"
    assert inventory["sources"]["cache"]["selection"]["candidate_id"] == "cache-v1"
    for component in ("kernel", "pisa", "cache"):
        implementations = inventory["sources"][component]["implementation_files"]
        assert len(implementations) == 1
        assert (worktree / implementations[0]["snapshot_path"]).is_file()
    return ctx, inventory


def inventory_artifact(source: dict[str, Any], role: str) -> dict[str, Any]:
    return next(item for item in source["artifacts"] if item["role"] == role)


TIMING_SCOPE = "warm_single_sample_text_encoder_through_vae_decode"


def timing_contract() -> dict[str, Any]:
    return {
        "scope": TIMING_SCOPE,
        "warmup_completed": True,
        "cuda_synchronized": True,
        "aggregation": "median",
        "sample_count": 5,
        "included_stages": ["text_encoder_compute", "dit_denoise", "vae_decode"],
        "excluded_stages": [
            "process_startup",
            "model_load",
            "text_encoder_load",
            "vae_load",
            "compile",
            "warmup",
            "video_encode",
            "video_write",
        ],
    }


def materialize_valid_integration(ctx: NodeContext, inventory: dict[str, Any]) -> Path:
    lock_files = []
    for component in ("kernel", "pisa", "cache"):
        implementation = inventory["sources"][component]["implementation_files"][0]
        source = ctx.worktree / implementation["snapshot_path"]
        destination = ctx.worktree / "integrated" / f"{component}.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        lock_files.append(
            {
                "component": component,
                "source": str(source),
                "source_sha256": sha256(source),
                "destination": str(destination.relative_to(ctx.worktree)),
                "destination_sha256": sha256(destination),
                "merge_mode": "semantic_port",
            }
        )

    inventory_path = ctx.worktree / "state" / "integration-source-inventory.json"
    sources = inventory["sources"]
    write_json(
        ctx.worktree / "INTEGRATION-SOURCES.lock.json",
        {
            "schema_version": 1,
            "workflow_uid": "integrator_ia",
            "inventory_path": "state/integration-source-inventory.json",
            "inventory_sha256": sha256(inventory_path),
            "sources": {
                "kernel": {
                    "candidate_ids": sources["kernel"]["selection"]["candidate_ids"],
                    "manifest_sha256": inventory_artifact(sources["kernel"], "kernel_manifest")["sha256"],
                },
                "pisa": {
                    "candidate_id": "pisa-v1",
                    "recipe": "visually_indistinguishable",
                    "manifest_sha256": inventory_artifact(sources["pisa"], "pisa_manifest")["sha256"],
                },
                "cache": {
                    "candidate_id": "cache-v1",
                    "manifest_sha256": inventory_artifact(sources["cache"], "cache_manifest")["sha256"],
                },
            },
            "files": lock_files,
        },
    )

    config = {
        "sample_nums": 5,
        "image_size": 720,
        "num_frames": 193,
        "fps": 24,
        "steps": 50,
        "cfg_scale": 8,
        "flow_shift": 12,
        "motion_score": 20,
    }
    components = {
        "conservative": {
            "kernel": {
                "enabled": True,
                "candidate_ids": ["kernel-a", "kernel-b"],
                "dispatches": 100,
                "fallbacks": 0,
            },
            "pisa": {
                "enabled": False,
                "candidate_id": "pisa-v1",
                "dispatches": 0,
                "fallbacks": 0,
            },
            "cache": {"enabled": False, "candidate_id": "cache-v1", "calls": 0, "hits": 0},
        },
        "balanced": {
            "kernel": {
                "enabled": True,
                "candidate_ids": ["kernel-a", "kernel-b"],
                "dispatches": 100,
                "fallbacks": 0,
            },
            "pisa": {
                "enabled": False,
                "candidate_id": "pisa-v1",
                "dispatches": 0,
                "fallbacks": 0,
            },
            "cache": {"enabled": True, "candidate_id": "cache-v1", "calls": 250, "hits": 80},
        },
        "aggressive": {
            "kernel": {
                "enabled": True,
                "candidate_ids": ["kernel-a", "kernel-b"],
                "dispatches": 100,
                "fallbacks": 0,
            },
            "pisa": {
                "enabled": True,
                "candidate_id": "pisa-v1",
                "dispatches": 100,
                "fallbacks": 0,
            },
            "cache": {"enabled": True, "candidate_id": "cache-v1", "calls": 250, "hits": 80},
        },
    }
    totals = {"conservative": 80.0, "balanced": 70.0, "aggressive": 60.0}
    recipes: dict[str, Any] = {}
    for tier, total in totals.items():
        run_dir = ctx.worktree / "runs" / f"integrated-{tier}"
        outputs = run_dir / "outputs"
        outputs.mkdir(parents=True)
        write_json(
            outputs / "benchmark.json",
            {
                "timing": timing_contract(),
                "aggregate": {
                    "text_encoder_s": 5.0,
                    "dit_denoise_s": total - 15.0,
                    "vae_decode_s": 9.0,
                    "sample_total_s": total,
                },
                "config": config,
            },
        )
        write_json(outputs / "run_config.json", config)
        stats_components = json.loads(json.dumps(components[tier]))
        stats_components["pisa"]["exact_phase_calls"] = 40 if tier == "aggressive" else 0
        stats_components["pisa"]["approximate_remainder_phase_calls"] = 60 if tier == "aggressive" else 0
        write_json(outputs / "integration_stats.json", {"components": stats_components})
        for index in range(5):
            (outputs / f"prompt_{index:03d}.mp4").write_bytes(b"video")
        performance = {
            "baseline_sample_total_s": 120.0,
            "candidate_sample_total_s": total,
            "speedup": 120.0 / total,
        }
        recipes[tier] = {
            "candidate_id": f"integrated-{tier}-v1",
            "run_dir": str(run_dir.relative_to(ctx.worktree)),
            "components": components[tier],
            "settings": {"profile": tier},
            "performance": performance,
        }

    baseline_outputs = ctx.worktree / "runs" / "integrated-baseline" / "outputs"
    baseline_outputs.mkdir(parents=True)
    write_json(
        baseline_outputs / "benchmark.json",
        {
            "timing": timing_contract(),
            "aggregate": {
                "text_encoder_s": 5.0,
                "dit_denoise_s": 105.0,
                "vae_decode_s": 9.0,
                "sample_total_s": 120.0,
            },
            "config": config,
        },
    )
    for index in range(5):
        (baseline_outputs / f"prompt_{index:03d}.mp4").write_bytes(b"video")

    condition_values = (
        ("baseline", "000", 120.0),
        ("kernel_only", "100", 90.0),
        ("pisa_only", "010", 110.0),
        ("cache_only", "001", 105.0),
        ("kernel_pisa", "110", 85.0),
        ("kernel_cache", "101", 70.0),
        ("pisa_cache", "011", 95.0),
        ("full_stack", "111", 60.0),
    )
    conditions = {
        name: {
            "status": "measured",
            "bits": bits,
            "sample_total_s": total,
            "speedup": 120.0 / total,
        }
        for name, bits, total in condition_values
    }
    conditions["baseline"]["run_dir"] = "runs/integrated-baseline"
    matrix_recipes = {
        tier: {
            "candidate_id": recipe["candidate_id"],
            "run_dir": recipe["run_dir"],
            "performance": recipe["performance"],
        }
        for tier, recipe in recipes.items()
    }
    write_json(
        ctx.worktree / "COMPOSITION-MATRIX.json",
        {
            "schema_version": 2,
            "all_off_identity": True,
            "timing_contract": timing_contract(),
            "conditions": conditions,
            "recipes": matrix_recipes,
        },
    )
    write_json(
        ctx.worktree / "INTEGRATION-STATUS.json",
        {
            "schema_version": 2,
            "workflow_uid": "integrator_ia",
            "experiment_uid": "sana-integrator_ia-0001",
            "status": "ready_for_visual",
            "source_lock": "INTEGRATION-SOURCES.lock.json",
            "composition_matrix": "COMPOSITION-MATRIX.json",
            "recipes": recipes,
            "owned_jobs": [],
            "terminal_reason": "",
        },
    )
    return ctx.worktree / recipes["conservative"]["run_dir"]


def test_source_and_integration_contracts() -> None:
    with TemporaryDirectory() as raw:
        root = Path(raw)
        worktree = root / "integration"
        worktree.mkdir()
        ctx, inventory = pin_sources(root, worktree)
        kernel_implementation = inventory["sources"]["kernel"]["implementation_files"][0]
        snapshot = worktree / kernel_implementation["snapshot_path"]
        pinned_hash = sha256(snapshot)
        Path(kernel_implementation["path"]).write_text("COMPONENT = \"moving-donor\"\n")
        assert sha256(snapshot) == pinned_hash
        run_dir = materialize_valid_integration(ctx, inventory)

        gate = run_integration_gate(ctx)
        assert gate.outcome == "ready", gate.message
        assert gate.updates["integration_run"] == "runs/integrated-conservative"

        selected, reason = selected_recipe_runs(ctx)
        assert not reason
        assert selected["conservative"] == run_dir.resolve()
        assert selected["aggressive"] == (worktree / "runs" / "integrated-aggressive").resolve()

        snapshot.write_text("tampered = true\n")
        snapshot_rejected = run_integration_gate(ctx)
        assert snapshot_rejected.outcome == "needs_retry"
        assert "source_snapshot_hash_mismatch:kernel:3" in snapshot_rejected.updates["integration_issues"]

        stats_path = worktree / "runs" / "integrated-aggressive" / "outputs" / "integration_stats.json"
        stats = json.loads(stats_path.read_text())
        stats["components"]["pisa"]["fallbacks"] = 1
        write_json(stats_path, stats)
        rejected = run_integration_gate(ctx)
        assert rejected.outcome == "needs_retry"
        assert (
            "integration_stats_counter_mismatch:aggressive:pisa:fallbacks"
            in rejected.updates["integration_issues"]
        )


def test_final_gate_requires_reconciliation_then_accepts_delivery() -> None:
    with TemporaryDirectory() as raw:
        root = Path(raw)
        worktree = root / "integration"
        worktree.mkdir()
        ctx, inventory = pin_sources(root, worktree)
        materialize_valid_integration(ctx, inventory)
        status_path = worktree / "INTEGRATION-STATUS.json"
        status = json.loads(status_path.read_text())
        quality_evidence = {
            "conservative": {"severity": "low", "medium_prompts": 0, "eligible": ["conservative", "balanced", "aggressive"]},
            "balanced": {"severity": "medium", "medium_prompts": 1, "eligible": ["balanced", "aggressive"]},
            "aggressive": {"severity": "medium", "medium_prompts": 3, "eligible": ["aggressive"]},
        }
        assessments: dict[str, dict[str, Any]] = {}
        for tier, evidence in quality_evidence.items():
            run_dir = worktree / status["recipes"][tier]["run_dir"]
            verdict = run_dir / "codex_visual_verdict.json"
            write_json(verdict, {"overall": "pass", "quality_tier": tier})
            performance = status["recipes"][tier]["performance"]
            assess = {
                "visual_provider": "codex",
                "quality_tier": tier,
                "timing_scope": TIMING_SCOPE,
                "codex_visual_overall": "pass",
                "eligible_tiers": evidence["eligible"],
                "max_artifact_severity": evidence["severity"],
                "medium_affected_prompt_count": evidence["medium_prompts"],
                "baseline_sample_total_s": performance["baseline_sample_total_s"],
                "candidate_sample_total_s": performance["candidate_sample_total_s"],
                "speedup": performance["speedup"],
                "lpips_max": 0.04,
                "quality_blockers": [],
                "codex_visual_verdict": str(verdict.relative_to(worktree)),
            }
            write_json(run_dir / "assess_verdict.json", assess)
            assessments[tier] = assess
        first = run_final_gate(ctx)
        assert first.outcome == "needs_finalize", first.message

        status["status"] = "complete"
        write_json(status_path, status)
        manifest = worktree / "candidates" / "integrated-recipes.toml"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("id = \"integrated-recipes-v1\"\n")
        source_lock = worktree / "INTEGRATION-SOURCES.lock.json"
        implementations = [
            worktree / "integrated" / "kernel.py",
            worktree / "integrated" / "pisa.py",
            worktree / "integrated" / "cache.py",
        ]
        write_json(
            worktree / "INTEGRATION-DELIVERY.json",
            {
                "schema_version": 2,
                "workflow_uid": "integrator_ia",
                "experiment_uid": "sana-integrator_ia-0001",
                "source_lock": {"path": "INTEGRATION-SOURCES.lock.json", "sha256": sha256(source_lock)},
                "integrated_manifest": {
                    "path": "candidates/integrated-recipes.toml",
                    "sha256": sha256(manifest),
                },
                "implementation_files": [
                    {"path": str(path.relative_to(worktree)), "sha256": sha256(path)}
                    for path in implementations
                ],
                "timing_contract": {"scope": TIMING_SCOPE},
                "recipes": {
                    tier: {
                        "candidate_id": recipe["candidate_id"],
                        "activation_env": {"SANA_INTEGRATED_RECIPE": tier},
                        "components": recipe["components"],
                        "settings": recipe["settings"],
                        "run": {
                            "run_dir": recipe["run_dir"],
                            "benchmark": recipe["run_dir"] + "/outputs/benchmark.json",
                            "integration_stats": recipe["run_dir"] + "/outputs/integration_stats.json",
                        },
                        "performance": recipe["performance"],
                        "quality": {
                            "tier": tier,
                            "codex_visual_overall": "pass",
                            "assess_verdict": recipe["run_dir"] + "/assess_verdict.json",
                            "max_artifact_severity": assessments[tier]["max_artifact_severity"],
                            "lpips_max": assessments[tier]["lpips_max"],
                        },
                    }
                    for tier, recipe in status["recipes"].items()
                },
            },
        )
        final = run_final_gate(ctx)
        assert final.outcome == "smooth", final.message

        delivery_path = worktree / "INTEGRATION-DELIVERY.json"
        delivery = json.loads(delivery_path.read_text())
        delivery["recipes"]["balanced"]["performance"]["speedup"] = 9.9
        write_json(delivery_path, delivery)
        rejected = run_final_gate(ctx)
        assert rejected.outcome == "needs_retry"
        assert "delivery_performance_mismatch:balanced:speedup" in rejected.updates["final_issues"]


def test_tiered_visual_policy_relaxes_medium_without_accepting_severe_loss() -> None:
    isolated_medium = {
        "degraded_side": "left",
        "max_severity": "medium",
        "differences": [],
        "per_prompt": [
            {"prompt_index": 0, "degraded_side": "left", "max_severity": "medium"},
            {"prompt_index": 1, "degraded_side": "neither", "max_severity": "none"},
        ],
    }
    assert decode_blind_verdict(isolated_medium, "left", "conservative")["overall"] == "fail"
    assert decode_blind_verdict(isolated_medium, "left", "balanced")["overall"] == "pass"

    broad_medium = {
        **isolated_medium,
        "per_prompt": [
            {"prompt_index": 0, "degraded_side": "left", "max_severity": "medium"},
            {"prompt_index": 1, "degraded_side": "left", "max_severity": "medium"},
        ],
    }
    assert decode_blind_verdict(broad_medium, "left", "balanced")["overall"] == "fail"
    assert decode_blind_verdict(broad_medium, "left", "aggressive")["overall"] == "pass"

    severe = {**broad_medium, "max_severity": "high"}
    assert decode_blind_verdict(severe, "left", "aggressive")["overall"] == "fail"


def test_graph_transitions_are_explicit_and_repairable() -> None:
    state: dict[str, Any] = {}
    assert transition(state, "check_sources", NodeResult("ready")) == "executor"
    assert transition(state, "executor", NodeResult("exited")) == "check_integration"
    assert transition(state, "check_integration", NodeResult("ready")) == "visual_review"
    assert transition(state, "visual_review", NodeResult("reviewed")) == "check_final"
    assert transition(state, "check_final", NodeResult("needs_retry", message="quality")) == "write_resume"
    assert state["resume_target"] == "executor"

    complete: dict[str, Any] = {}
    assert transition(complete, "check_final", NodeResult("smooth")) == "done"
    assert complete["status"] == "done"

    blocked: dict[str, Any] = {}
    assert transition(blocked, "check_sources", NodeResult("source_blocked")) == "blocked"
    assert blocked["status"] == "blocked"


if __name__ == "__main__":
    test_source_and_integration_contracts()
    test_final_gate_requires_reconciliation_then_accepts_delivery()
    test_tiered_visual_policy_relaxes_medium_without_accepting_severe_loss()
    test_graph_transitions_are_explicit_and_repairable()
    print("integrator_ia workflow tests passed")
