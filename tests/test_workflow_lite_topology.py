#!/usr/bin/env python3
"""Self-contained registration tests for the workflow-lite topology executor."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_last_json(text: str) -> dict:
    start = text.find("{")
    assert start >= 0, text
    return json.loads(text[start:])


def test_registry_and_owned_scope() -> None:
    with (ROOT / "workflow_lite/techniques.toml").open("rb") as handle:
        registry = tomllib.load(handle)

    assert registry["default_order"] == ["kernel", "cache", "pisa"]
    topology = registry["techniques"]["topology"]
    assert topology == {
        "workflow_uid": "topology_ta",
        "scope": "workflow/topology_ta/nodes/codex_executor/topology_scope.md",
        "correctness": "lossless",
    }

    scope = ROOT / topology["scope"]
    assert scope.is_file()
    text = scope.read_text()
    for required in (
        "context/sequence parallelism",
        "tensor parallelism",
        "expert parallel",
        "FSDP",
        "process-group",
        "no_silent_fallback",
        'component = "topology"',
    ):
        assert required in text
    assert "Do not switch FA2/cuDNN" in text
    assert (ROOT / "workflow/topology_ta/nodes/codex_executor/interface.toml").is_file()

    kernel_scope = (
        ROOT / "workflow/kernel_aw/nodes/codex_executor/kernel_scope.md"
    ).read_text()
    assert "Cross-rank partitioning and scheduling are owned by the `topology` executor" in kernel_scope


def test_orchestrator_defaults_and_validates_techniques() -> None:
    command = [
        sys.executable,
        "workflow_lite/run_orchestrated_experiment.py",
        "--model",
        "lingbot_video",
        "--dry-run",
    ]
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    payload = parse_last_json(proc.stdout)
    assert payload["techs"] == ["kernel", "cache", "pisa", "topology"]
    assert payload["tech_selection"] == "model_profile"
    assert payload["technique_specs"]["topology"]["workflow_uid"] == "topology_ta"

    bernini = subprocess.run(
        [
            sys.executable,
            "workflow_lite/run_orchestrated_experiment.py",
            "--model",
            "bernini",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    bernini_payload = parse_last_json(bernini.stdout)
    assert bernini_payload["techs"] == ["kernel", "cache", "pisa"]
    assert bernini_payload["tech_selection"] == "registry_default"

    for raw, expected in (("kernel,unknown", "unknown --techs"), ("topology,topology", "duplicates")):
        invalid = subprocess.run(
            command[:-1] + ["--techs", raw, "--dry-run"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert invalid.returncode != 0
        assert expected in invalid.stderr


def test_override_baseline_freezes_selected_run_metrics() -> None:
    orchestrator = load_module(
        "workflow_lite_orchestrator_topology_test",
        "workflow_lite/run_orchestrated_experiment.py",
    )
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        run_dir = tmp / "fresh-baseline-run"
        outputs = run_dir / "outputs"
        outputs.mkdir(parents=True)
        (outputs / "frames").mkdir()
        (outputs / "benchmark.json").write_text(
            json.dumps(
                {
                    "total_s": 123.0,
                    "denoise_s": 99.0,
                    "timing_scope": "fresh_override_scope",
                    "max_device_memory_used_mib": 4567.0,
                    "config": {"world_size": 4},
                }
            )
        )
        frozen = orchestrator.freeze_baseline(
            "lingbot_video", tmp / "BASELINE.json", str(run_dir)
        )
        assert frozen["total_s"] == 123.0
        assert frozen["denoise_s"] == 99.0
        assert frozen["timing_scope"] == "fresh_override_scope"
        assert frozen["peak_memory_mib"] == 4567.0
        assert frozen["world_size"] == 4
        assert frozen["source"] == "override_run_dir"


def test_topology_verification_contract() -> None:
    verifier = load_module("verify_delivery_topology_test", "workflow_lite/bin/verify_delivery.py")
    assert {"topology", "topology_ta"}.issubset(verifier.LOSSLESS_TECHS)
    assert {"topology", "topology_ta"} == verifier.TOPOLOGY_TECHS
    assert verifier.expected_world_size("lingbot_video", {}) == 4

    with tempfile.TemporaryDirectory() as raw:
        run_dir = Path(raw)
        run_id = run_dir.name
        config_id = "topology_config"
        point = {
            "config_id": config_id,
            "performance": {
                "frontier_axis": "latency",
                "baseline_total_s": 100.0,
                "config_total_s": 80.0,
                "speedup": 1.25,
            },
        }
        outputs = run_dir / "outputs"
        outputs.mkdir()
        topology_evidence = {
            "world_size": 4,
            "active_ranks": [0, 1, 2, 3],
            "all_ranks_participated": True,
            "no_silent_fallback": True,
            "process_groups": [{"kind": "cp", "ranks": [0, 1, 2, 3]}],
            "rank_map": [{"rank": rank, "cp": rank} for rank in range(4)],
            "placement": {"transformer": "fsdp(cp_world)"},
            "collectives": [{"kind": "all_to_all", "calls": 96}],
        }
        evidence = {
            "config_id": config_id,
            "run_id": run_id,
            "baseline_steps": 48,
            "config_steps": 48,
            "baseline_dit_calls": 48,
            "config_dit_calls": 48,
            "method_argument": "The rank partition and collectives compute the same global function.",
            "topology": topology_evidence,
        }
        source = run_dir / "runtime/model.py"
        source.parent.mkdir(parents=True)
        source.write_text("TOPOLOGY_IMPL = True\n")
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        (outputs / "equivalence.json").write_text(json.dumps(evidence))
        (outputs / "topology_preflight.json").write_text(
            json.dumps(
                {
                    "config_id": config_id,
                    "run_id": run_id,
                    "status": "pass",
                    "world_size": 4,
                    "checks": [{"name": "coverage", "passed": True}],
                }
            )
        )
        (outputs / "topology_manifest.json").write_text(
            json.dumps(
                {
                    "config_id": config_id,
                    "run_id": run_id,
                    "world_size": 4,
                    **{
                        key: topology_evidence[key]
                        for key in ("process_groups", "rank_map", "placement", "collectives")
                    },
                    "source_hashes": {"runtime/model.py": source_digest},
                }
            )
        )
        (outputs / "topology_trace.json").write_text(
            json.dumps(
                {
                    "config_id": config_id,
                    "run_id": run_id,
                    "world_size": 4,
                    "active_ranks": [0, 1, 2, 3],
                    "collectives": [{"kind": "all_to_all", "calls": 96}],
                    "fallbacks": {"baseline": 0, "single_rank": 0},
                    "per_rank": [
                        {
                            "rank": rank,
                            "participated": True,
                            "total_s": 80.0 + rank / 100,
                            "peak_memory_mib": 1000,
                        }
                        for rank in range(4)
                    ],
                }
            )
        )
        (outputs / "benchmark.json").write_text(
            json.dumps(
                {
                    "total_s": 80.0,
                    "timing_scope": "frozen_request_scope",
                    "max_device_memory_used_mib": 1000,
                }
            )
        )

        correctness_issues, correctness = verifier.check_correctness(point, run_dir)
        topology_issues, topology = verifier.check_topology_evidence(
            point, run_dir, expected_world=4
        )
        assert correctness_issues == []
        assert correctness["method_argument_present"] is True
        assert topology_issues == []
        assert topology["world_size"] == 4
        assert topology["collective_count"] == 1
        performance_issues, performance = verifier.check_performance_evidence(
            point,
            run_dir,
            {
                "total_s": 100.0,
                "timing_scope": "frozen_request_scope",
                "peak_memory_mib": 1200,
            },
            require_complete=True,
            require_improvement=True,
        )
        assert performance_issues == []
        assert performance["speedup"] == 1.25
        assert performance["latency_improved"] is True

        missing_axis = json.loads(json.dumps(point))
        del missing_axis["performance"]["frontier_axis"]
        missing_axis_issues, _ = verifier.check_performance_evidence(
            missing_axis,
            run_dir,
            {"total_s": 100.0, "timing_scope": "frozen_request_scope"},
            require_complete=True,
            require_improvement=True,
        )
        assert "topology_frontier_axis_missing_or_invalid" in missing_axis_issues

        benchmark_path = outputs / "benchmark.json"
        benchmark = json.loads(benchmark_path.read_text())
        benchmark["timing_scope"] = "wrong_scope"
        benchmark_path.write_text(json.dumps(benchmark))
        timing_issues, _ = verifier.check_performance_evidence(
            point,
            run_dir,
            {"total_s": 100.0, "timing_scope": "frozen_request_scope"},
            require_complete=True,
            require_improvement=True,
        )
        assert "config_timing_scope_mismatch" in timing_issues
        benchmark["timing_scope"] = "frozen_request_scope"
        benchmark_path.write_text(json.dumps(benchmark))

        memory_point = json.loads(json.dumps(point))
        memory_point["performance"] = {
            "frontier_axis": "peak_memory",
            "baseline_total_s": 100.0,
            "config_total_s": 105.0,
            "speedup": 100.0 / 105.0,
        }
        benchmark["total_s"] = 105.0
        benchmark["max_device_memory_used_mib"] = 900.0
        benchmark_path.write_text(json.dumps(benchmark))
        trace_path = outputs / "topology_trace.json"
        trace = json.loads(trace_path.read_text())
        for rank_record in trace["per_rank"]:
            rank_record["peak_memory_mib"] = 900.0
        trace_path.write_text(json.dumps(trace))
        memory_issues, memory_performance = verifier.check_performance_evidence(
            memory_point,
            run_dir,
            {
                "total_s": 100.0,
                "timing_scope": "frozen_request_scope",
                "peak_memory_mib": 1200.0,
            },
            require_complete=True,
            require_improvement=True,
        )
        assert memory_issues == []
        assert memory_performance["memory_improved"] is True
        for rank_record in trace["per_rank"]:
            rank_record["peak_memory_mib"] = 1.0
        trace_path.write_text(json.dumps(trace))
        fake_memory_issues, _ = verifier.check_performance_evidence(
            memory_point,
            run_dir,
            {
                "total_s": 100.0,
                "timing_scope": "frozen_request_scope",
                "peak_memory_mib": 1200.0,
            },
            require_complete=True,
            require_improvement=True,
        )
        assert "topology_trace_benchmark_peak_memory_mismatch" in fake_memory_issues
        benchmark["total_s"] = 80.0
        benchmark["max_device_memory_used_mib"] = 1000.0
        benchmark_path.write_text(json.dumps(benchmark))
        for rank_record in trace["per_rank"]:
            rank_record["peak_memory_mib"] = 1000.0
        trace_path.write_text(json.dumps(trace))

        wrong_world_issues, _ = verifier.check_topology_evidence(
            point, run_dir, expected_world=2
        )
        assert "topology_world_size_changed" in wrong_world_issues

        equivalence_path = outputs / "equivalence.json"
        manifest_path = outputs / "topology_manifest.json"
        equivalence_doc = json.loads(equivalence_path.read_text())
        manifest_doc = json.loads(manifest_path.read_text())
        duplicate_rank = {"rank": 3, "cp": 3}
        equivalence_doc["topology"]["rank_map"].append(duplicate_rank)
        manifest_doc["rank_map"].append(duplicate_rank)
        equivalence_path.write_text(json.dumps(equivalence_doc))
        manifest_path.write_text(json.dumps(manifest_doc))
        duplicate_issues, _ = verifier.check_topology_evidence(
            point, run_dir, expected_world=4
        )
        assert "topology_rank_map_invalid" in duplicate_issues
        equivalence_doc["topology"]["rank_map"].pop()
        manifest_doc["rank_map"].pop()

        del equivalence_doc["topology"]["process_groups"][0]["kind"]
        del manifest_doc["process_groups"][0]["kind"]
        equivalence_path.write_text(json.dumps(equivalence_doc))
        manifest_path.write_text(json.dumps(manifest_doc))
        group_issues, _ = verifier.check_topology_evidence(
            point, run_dir, expected_world=4
        )
        assert "topology_process_group_identity_missing" in group_issues
        equivalence_doc["topology"]["process_groups"][0]["kind"] = "cp"
        manifest_doc["process_groups"][0]["kind"] = "cp"
        equivalence_path.write_text(json.dumps(equivalence_doc))
        manifest_path.write_text(json.dumps(manifest_doc))

        manifest_doc["config_id"] = "stale_config"
        manifest_path.write_text(json.dumps(manifest_doc))
        identity_issues, _ = verifier.check_topology_evidence(
            point, run_dir, expected_world=4
        )
        assert "topology_manifest_config_id_mismatch" in identity_issues
        manifest_doc["config_id"] = config_id
        manifest_path.write_text(json.dumps(manifest_doc))

        trace_path = outputs / "topology_trace.json"
        trace_doc = json.loads(trace_path.read_text())
        del trace_doc["per_rank"][0]["participated"]
        trace_path.write_text(json.dumps(trace_doc))
        metric_issues, _ = verifier.check_topology_evidence(
            point, run_dir, expected_world=4
        )
        assert "topology_trace_per_rank_metrics_missing" in metric_issues
        trace_doc["per_rank"][0]["participated"] = True
        trace_path.write_text(json.dumps(trace_doc))

        (outputs / "topology_preflight.json").write_text(
            json.dumps(
                {
                    "config_id": config_id,
                    "run_id": run_id,
                    "status": "pass",
                    "world_size": 4,
                    "checks": [{"name": "coverage", "passed": False}],
                }
            )
        )
        preflight_issues, _ = verifier.check_topology_evidence(
            point, run_dir, expected_world=4
        )
        assert "topology_preflight_checks_missing_or_failed" in preflight_issues
        (outputs / "topology_preflight.json").write_text("[]")
        malformed_issues, _ = verifier.check_topology_evidence(
            point, run_dir, expected_world=4
        )
        assert "topology_preflight_missing" in malformed_issues
        (outputs / "topology_preflight.json").write_text(
            json.dumps(
                {
                    "config_id": config_id,
                    "run_id": run_id,
                    "status": "pass",
                    "world_size": 4,
                    "checks": [{"name": "coverage", "passed": True}],
                }
            )
        )

        trace_path = outputs / "topology_trace.json"
        trace = json.loads(trace_path.read_text())
        trace["collectives"] = [{"kind": "broadcast", "calls": 999}]
        trace_path.write_text(json.dumps(trace))
        trace_issues, _ = verifier.check_topology_evidence(point, run_dir, expected_world=4)
        assert "topology_trace_collectives_mismatch" in trace_issues
        trace["collectives"] = [{"kind": "all_to_all", "calls": 96}]
        trace_path.write_text(json.dumps(trace))

        manifest = json.loads(manifest_path.read_text())
        manifest["source_hashes"]["runtime/model.py"] = "b" * 64
        manifest_path.write_text(json.dumps(manifest))
        hash_issues, _ = verifier.check_topology_evidence(point, run_dir, expected_world=4)
        assert "topology_manifest_source_hash_mismatch" in hash_issues
        manifest["source_hashes"]["runtime/model.py"] = source_digest
        manifest_path.write_text(json.dumps(manifest))

        evidence["topology"]["no_silent_fallback"] = False
        (outputs / "equivalence.json").write_text(json.dumps(evidence))
        topology_issues, _ = verifier.check_topology_evidence(point, run_dir, expected_world=4)
        assert "topology_no_fallback_unproven" in topology_issues

        del evidence["config_dit_calls"]
        (outputs / "equivalence.json").write_text(json.dumps(evidence))
        correctness_issues, _ = verifier.check_correctness(point, run_dir)
        assert "dit_call_count_evidence_missing" in correctness_issues

        manifest_path.unlink()
        topology_issues, _ = verifier.check_topology_evidence(point, run_dir, expected_world=4)
        assert "topology_manifest_missing" in topology_issues

        (outputs / "equivalence.json").unlink()
        topology_issues, _ = verifier.check_topology_evidence(
            point, run_dir, expected_world=4
        )
        assert "topology_equivalence_artifact_missing" in topology_issues
        assert "topology_evidence_missing" in topology_issues


def test_spawn_topology_materializes_lingbot_prompt() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        baseline = tmp / "BASELINE.json"
        baseline.write_text(
            json.dumps(
                {
                    "model_id": "lingbot_video",
                    "total_s": 375.55,
                    "timing_scope": "load_excluded_request_wall",
                    "baseline_frames": "",
                }
            )
        )
        command = [
            sys.executable,
            "workflow_lite/bin/spawn_executor.py",
            "--model",
            "lingbot_video",
            "--tech",
            "topology",
            "--experiment-uid",
            "lingbot-topology_ta-9876",
            "--baseline",
            str(baseline),
            "--experiments-root",
            str(tmp / "experiments"),
            "--no-launch",
        ]
        proc = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        payload = json.loads(proc.stdout)
        assert payload["workflow_uid"] == "topology_ta"
        assert payload["correctness"] == "lossless"
        assert payload["launched"] is False

        worktree = Path(payload["worktree"])
        goal_dir = Path(payload["goal_dir"])
        goal = (goal_dir / "goal.md").read_text()
        context = json.loads((goal_dir / "context.json").read_text())
        assert "Multi-GPU topology optimization scope" in goal
        assert "Frozen baseline (do not re-run)" in goal
        assert '"total_s": 375.55' in goal
        assert context["workflow_uid"] == "topology_ta"
        assert context["aspect"] == "topology"
        assert context["technique"] == "topology"
        assert context["correctness_mode"] == "lossless"
        assert (worktree / "runtime/lingbot_video_baseline/gpu_infer.py").is_file()
        assert not (worktree / "runtime/lingbot_video_optimized").exists()

        metadata_path = tmp / "experiments/lingbot-topology_ta-9876/experiment.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["workflow_uid"] = "kernel_aw"
        metadata_path.write_text(json.dumps(metadata))
        mismatched = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert mismatched.returncode != 0
        assert "refusing mismatched experiment metadata" in mismatched.stderr

        metadata["workflow_uid"] = "topology_ta"
        metadata["worktree"] = str(tmp / "fake-worktree")
        metadata_path.write_text(json.dumps(metadata))
        noncanonical = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert noncanonical.returncode != 0
        assert "refusing non-canonical experiment paths" in noncanonical.stderr

        metadata["worktree"] = str(worktree)
        metadata["baseline"]["manifest"] = str(
            ROOT / "config/lingbot_video/baseline.toml"
        )
        metadata_path.write_text(json.dumps(metadata))
        escaped_baseline = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert escaped_baseline.returncode != 0
        assert "refusing mismatched baseline metadata" in escaped_baseline.stderr

        metadata["baseline"]["manifest"] = "config/lingbot_video/baseline.toml"
        metadata_path.write_text(json.dumps(metadata))
        launcher = worktree / "scripts/launch_config.py"
        launcher.unlink()
        launcher.mkdir()
        launcher_directory = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert launcher_directory.returncode != 0
        assert "refusing incomplete experiment runnable closure" in launcher_directory.stderr


def test_verifier_rejects_unowned_delivery_paths_and_unknown_tech() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        worktree = tmp / "worktree"
        (worktree / "runs").mkdir(parents=True)
        baseline = tmp / "BASELINE.json"
        baseline.write_text(
            json.dumps({"model_id": "lingbot_video", "total_s": 375.55, "world_size": 4})
        )
        (worktree / "DELIVERY.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "status": "complete",
                    "model_id": "lingbot_video",
                    "frontier_points": [
                        {"config_id": "stale", "run_dir": str(tmp / "outside-run")}
                    ],
                }
            )
        )
        base_command = [
            sys.executable,
            "workflow_lite/bin/verify_delivery.py",
            "--worktree",
            str(worktree),
            "--model",
            "lingbot_video",
            "--baseline",
            str(baseline),
        ]
        rejected = subprocess.run(
            base_command + ["--tech", "topology"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert rejected.returncode != 0
        payload = json.loads(rejected.stdout)
        assert "delivery_component_mismatch:None" in payload["issues"]
        assert any("run_dir_must_be_worktree_relative" in issue for issue in payload["issues"])

        unknown = subprocess.run(
            base_command + ["--tech", "typo"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert unknown.returncode != 0
        assert "invalid choice" in unknown.stderr


def main() -> int:
    tests = (
        test_registry_and_owned_scope,
        test_orchestrator_defaults_and_validates_techniques,
        test_override_baseline_freezes_selected_run_metrics,
        test_topology_verification_contract,
        test_spawn_topology_materializes_lingbot_prompt,
        test_verifier_rejects_unowned_delivery_paths_and_unknown_tech,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
