#!/usr/bin/env python3
"""Self-contained tests for launch_config single-flight guards."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/launch_config.py"


def load_module():
    spec = importlib.util.spec_from_file_location("launch_config", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def args() -> SimpleNamespace:
    return SimpleNamespace(mode="sbatch", confirm_submit=True, run_root="runs")


def readiness_args(mode: str, allow_unsupported_gpu: bool = False) -> SimpleNamespace:
    return SimpleNamespace(mode=mode, allow_unsupported_gpu=allow_unsupported_gpu)


def expect_block(fn, needle: str) -> None:
    try:
        fn()
    except SystemExit as exc:
        text = str(exc)
        assert needle in text, text
        return
    raise AssertionError("Expected SystemExit")


def test_scored_config_blocks_on_unrecorded_scored_run(module) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        module.repo_root = lambda: root
        write_json(root / "AGENT-STATUS.json", {"config": []})

        run_dir = root / "runs/c1"
        write_json(
            run_dir / "metadata.json",
            {
                "config_id": "c1",
                "kind": "methodology",
                "status": "completed",
                "run_dir": str(run_dir),
                "slurm_job_id": "123",
            },
        )

        expect_block(
            lambda: module.enforce_single_flight_or_exit(
                args(), {"id": "c2", "kind": "methodology"}
            ),
            "runs/c1",
        )


def test_duplicate_control_blocks(module) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        module.repo_root = lambda: root
        write_json(root / "AGENT-STATUS.json", {"config": []})

        run_dir = root / "runs/warm"
        write_json(
            run_dir / "metadata.json",
            {
                "config_id": "warm-control",
                "kind": "env_only",
                "status": "submitted",
                "run_dir": str(run_dir),
                "slurm_job_id": "456",
            },
        )

        expect_block(
            lambda: module.enforce_single_flight_or_exit(
                args(),
                {
                    "id": "warm-control",
                    "kind": "env_only",
                    "slurm": {"job_name": "warm-ctrl"},
                },
            ),
            "runs/warm",
        )


def test_baseline_does_not_block_on_scored_run(module) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        module.repo_root = lambda: root
        write_json(root / "AGENT-STATUS.json", {"config": []})

        run_dir = root / "runs/c1"
        write_json(
            run_dir / "metadata.json",
            {
                "config_id": "c1",
                "kind": "methodology",
                "status": "submitted",
                "run_dir": str(run_dir),
                "slurm_job_id": "789",
            },
        )

        module.enforce_single_flight_or_exit(
            args(), {"id": "baseline", "kind": "baseline"}
        )


def test_profile_does_not_block_scored_config(module) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        module.repo_root = lambda: root
        write_json(root / "AGENT-STATUS.json", {"config": []})

        run_dir = root / "runs/profile"
        write_json(
            run_dir / "metadata.json",
            {
                "config_id": "cosmos3_kwl_profile",
                "kind": "patch",
                "status": "completed",
                "run_dir": str(run_dir),
                "slurm_job_id": "999",
            },
        )

        module.enforce_single_flight_or_exit(
            args(), {"id": "kwl-next-config", "kind": "patch"}
        )


def test_sbatch_wrapper_forces_bash(module) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        launch = root / "launch.sh"
        launch.write_text("#!/usr/bin/env bash\n")
        job = module.write_sbatch_script(root, launch, {"job_name": "unit", "gpus_per_node": 4})
        text = job.read_text()
        assert text.startswith("#!/usr/bin/env bash")
        assert "#SBATCH --export=ALL" in text
        assert "set -euo pipefail" in text
        assert "exec /usr/bin/env bash" in text


def test_update_metadata_appends_status_history(module) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_json(
            root / "metadata.json",
            {
                "status": "prepared",
                "status_history": [{"status": "prepared", "at_utc": "t0"}],
            },
        )
        module.update_metadata(root, {"status": "submitted", "status_reason": "unit"})
        metadata = json.loads((root / "metadata.json").read_text())
        assert metadata["status"] == "submitted"
        assert metadata["status_history"][-1]["status"] == "submitted"
        assert metadata["status_history"][-1]["reason"] == "unit"


def test_unsupported_cosmos3_config_blocks_gpu_launch(module) -> None:
    data = {
        "id": {"name": "te_recipe_variant"},
        "run_script": "scripts/run_cosmos3_sglang.sh",
        "env": {"MODEL_REPO": "nvidia/Cosmos3-Super"},
    }

    expect_block(
        lambda: module.enforce_gpu_readiness_or_exit(readiness_args("sbatch"), data),
        "unsupported Cosmos3 GPU config",
    )


def test_route_label_config_blocks_gpu_launch(module) -> None:
    data = {
        "id": {"name": "semantic_permutation"},
        "run_script": "scripts/run_cosmos3_sglang.sh",
        "env": {"MODEL_REPO": "nvidia/Cosmos3-Super"},
    }

    expect_block(
        lambda: module.enforce_gpu_readiness_or_exit(readiness_args("local"), data),
        "is missing required module(s)",
    )


def test_payload_cache_config_allows_gpu_launch_after_consumer_wiring(module) -> None:
    data = {
        "id": {"name": "attention_broadcast"},
        "run_script": "scripts/run_cosmos3_sglang.sh",
        "env": {"MODEL_REPO": "nvidia/Cosmos3-Super"},
    }

    module.enforce_gpu_readiness_or_exit(readiness_args("sbatch"), data)


def test_nvfp4_config_allows_gpu_launch_after_online_quantizer_wiring(module) -> None:
    data = {
        "id": {"name": "conservative_ffn_nvfp4"},
        "run_script": "scripts/run_cosmos3_sglang.sh",
        "env": {"MODEL_REPO": "nvidia/Cosmos3-Super"},
    }

    module.enforce_gpu_readiness_or_exit(readiness_args("sbatch"), data)


def test_sparse_policy_config_allows_gpu_launch_after_runtime_consumer(module) -> None:
    data = {
        "id": {"name": "spatial_temporal_head_routing"},
        "run_script": "scripts/run_cosmos3_sglang.sh",
        "env": {"MODEL_REPO": "nvidia/Cosmos3-Super"},
    }

    module.enforce_gpu_readiness_or_exit(readiness_args("sbatch"), data)


def test_unsupported_cosmos3_config_allows_dry_run(module) -> None:
    data = {
        "id": {"name": "te_recipe_variant"},
        "run_script": "scripts/run_cosmos3_sglang.sh",
        "env": {"MODEL_REPO": "nvidia/Cosmos3-Super"},
    }

    module.enforce_gpu_readiness_or_exit(readiness_args("dry-run"), data)


def test_unsupported_cosmos3_config_allows_explicit_override(module) -> None:
    data = {
        "id": {"name": "semantic_permutation"},
        "run_script": "scripts/run_cosmos3_sglang.sh",
        "env": {"MODEL_REPO": "nvidia/Cosmos3-Super"},
    }

    module.enforce_gpu_readiness_or_exit(
        readiness_args("local", allow_unsupported_gpu=True), data
    )


def test_wired_cosmos3_config_allows_gpu_launch(module) -> None:
    data = {
        "id": {"name": "piecewise_pisa_env"},
        "run_script": "scripts/run_cosmos3_sglang.sh",
        "env": {"MODEL_REPO": "nvidia/Cosmos3-Super"},
    }

    module.enforce_gpu_readiness_or_exit(readiness_args("local"), data)


def main() -> int:
    module = load_module()
    tests = [
        test_scored_config_blocks_on_unrecorded_scored_run,
        test_duplicate_control_blocks,
        test_baseline_does_not_block_on_scored_run,
        test_profile_does_not_block_scored_config,
        test_sbatch_wrapper_forces_bash,
        test_update_metadata_appends_status_history,
        test_unsupported_cosmos3_config_blocks_gpu_launch,
        test_route_label_config_blocks_gpu_launch,
        test_payload_cache_config_allows_gpu_launch_after_consumer_wiring,
        test_nvfp4_config_allows_gpu_launch_after_online_quantizer_wiring,
        test_sparse_policy_config_allows_gpu_launch_after_runtime_consumer,
        test_unsupported_cosmos3_config_allows_dry_run,
        test_unsupported_cosmos3_config_allows_explicit_override,
        test_wired_cosmos3_config_allows_gpu_launch,
    ]
    for test in tests:
        test(module)
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
