"""LingBot-Video: the two arms stay separable, and a run resolves what it claims.

This file went stale rather than wrong. It was written against a `runtime/`
directory and flat `config/lingbot_video_*.toml` names, both of which the tree
reorganisation replaced with `models/lingbot_video/<arm>/` and
`config/lingbot_video/<arm>.toml`. Every one of its five tests then failed on a
missing path, and stayed red long enough to stop being read -- which is how the
finding below went unnoticed.

So the config list is derived from the directory now instead of being spelled
out. A config added or renamed is picked up; a config deleted stops being
asserted about. The previous version hardcoded four filenames, two of which
(`lingbot_video_cudnn_off`, `lingbot_video_fsdp4_reference`) no longer exist as
configs at all, and could not tell that apart from a typo.

Dropped with this rewrite: `test_lingbot_baseline_contract_does_not_copy_optimized_runtime`.
It asserted over the spelling of `[baseline.copy].include` -- that no entry
contains the string "lingbot_video_optimized" -- rather than over what the glob
list actually selects. It passes today while `include` carries
`models/lingbot_video/**`, which matches the optimized runtime it was written to
keep out. That contract is read only by the experiment materializer, which is no
longer part of what this repository publishes, so the test is not re-pointed
here; the glob is noted rather than quietly changed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config/lingbot_video"
ARMS = {"baseline": "models/lingbot_video/baseline",
        "optimized": "models/lingbot_video/optimized"}
CONFIGS = sorted(CONFIG_DIR.glob("*.toml"))


def load_toml(relative: str) -> dict:
    with (ROOT / relative).open("rb") as handle:
        return tomllib.load(handle)


def load_adapter_module():
    path = ROOT / "models/lingbot_video/baseline/gpu_infer.py"
    spec = importlib.util.spec_from_file_location("lingbot_video_gpu_infer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configs_exist_at_all() -> None:
    """Guards the derivation above: an empty glob would make the rest vacuous."""
    assert CONFIGS, f"no configs under {CONFIG_DIR}; the layout moved again"


def test_profile_and_eval_agree_on_the_workload() -> None:
    """The measured workload has to be one number, not one per file."""
    profile = load_toml("models/lingbot_video.toml")
    eval_profile = load_toml("evals/profiles/official_video_t2v_lingbot_video.toml")

    assert profile["official_config"] == eval_profile["official_config"]
    assert profile["official_config"]["num_gpus"] == 4
    assert profile["official_config"]["context_parallel_degree"] == 4


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.name)
def test_each_config_names_this_model_and_an_arm_that_exists(config: Path) -> None:
    """A config pointing at a runtime root that is not there is the failure that
    costs the most: it survives every dry-run check and surfaces after an
    allocation. Five GB200 configs did exactly this after their arms were
    flattened away."""
    with config.open("rb") as handle:
        data = tomllib.load(handle)

    assert data["model_profile"] == "lingbot_video"
    root = data["runtime"]["root"]
    assert root in ARMS.values(), f"{config.name}: unknown runtime root {root}"
    assert (ROOT / root).is_dir(), f"{config.name}: {root} does not exist"

    # The kernel a config selects has to match the arm it runs in: the baseline
    # sources do not contain the cudnn path at all (asserted below), so a
    # baseline-rooted config asking for cudnn would fail at run time, not here.
    kernel = data["env"]["LINGBOT_ATTN_KERNEL"]
    if root == ARMS["baseline"]:
        assert kernel == "fa2", f"{config.name}: baseline arm cannot select {kernel}"
    else:
        assert kernel in {"fa2", "cudnn"}, f"{config.name}: unknown kernel {kernel}"


def test_the_two_arms_are_physically_separate_sources() -> None:
    """Not 'configured differently' -- different files on disk.

    A baseline that shares a source file with the optimized arm is one stray
    environment variable away from measuring the optimized path and reporting it
    as the control.
    """
    baseline = ROOT / "models/lingbot_video/baseline/lingbot_src/lingbot_video"
    optimized = ROOT / "models/lingbot_video/optimized/lingbot_src/lingbot_video"

    baseline_transformer = (baseline / "transformer_lingbot_video.py").read_text()
    optimized_transformer = (optimized / "transformer_lingbot_video.py").read_text()
    baseline_runner = (baseline / "runner.py").read_text()

    assert "_cudnn_varlen_attention" not in baseline_transformer
    assert "LINGBOT_ATTN_KERNEL" not in baseline_transformer
    assert "_cudnn_varlen_attention" in optimized_transformer
    assert "LINGBOT_ATTN_KERNEL" in optimized_transformer
    assert "LINGBOT_PHASE_TIMING" in baseline_runner
    assert "LINGBOT_BCAST_WEIGHTS" not in baseline_runner

    baseline_adapter = (ROOT / "models/lingbot_video/baseline/gpu_infer.py").read_text()
    assert "lingbot_video_optimized" not in baseline_adapter
    assert "registered c5" not in baseline_adapter


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_vendored_sources_match_their_snapshot(arm: str) -> None:
    """The snapshot is what makes 'the baseline is upstream' checkable rather
    than asserted."""
    runtime_root = ROOT / ARMS[arm]
    snapshot = json.loads((runtime_root / "SOURCE_SNAPSHOT.json").read_text())
    for relative, expected in snapshot["core_sha256"].items():
        blob = (runtime_root / "lingbot_src" / relative).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == expected, f"{arm}: {relative} drifted"


def test_phase_parser_and_hot_sum_contract() -> None:
    adapter = load_adapter_module()
    phases = adapter.parse_phase_lines(
        [
            "PHASE base_denoise_done dt=89.10 total=325.10\n",
            "noise\n",
            "PHASE refiner_denoise_done dt=123.88 total=500.00\n",
        ]
    )

    assert phases["base_denoise_done"]["dt_s"] == 89.10
    assert phases["refiner_denoise_done"]["dt_s"] == 123.88
    # A hot sum with a missing part is None, never a partial total -- a partial
    # total is a number that looks measured and is not.
    assert round(adapter.sum_if_complete(89.10, 19.30, 123.88, 1.54), 2) == 233.82
    assert adapter.sum_if_complete(89.10, None) is None

    adapter.validate_topology(4, 4, True)
    adapter.validate_topology(4, 1, True)
    for invalid in ((8, 4, True), (4, 2, True), (4, 1, False)):
        with pytest.raises(SystemExit):
            adapter.validate_topology(*invalid)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.name)
def test_dry_run_persists_the_merged_profile_it_resolved(config: Path) -> None:
    """What a run records has to be what it resolved, not what the file said.

    Read back from the rendered bundle rather than compared to a hardcoded
    table: the expectations are the config's own values, so this cannot drift
    out of step with the configs the way the previous version did.
    """
    with config.open("rb") as handle:
        declared = tomllib.load(handle)
    relative = str(config.relative_to(ROOT))

    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "runs"
        subprocess.run(
            [sys.executable, "scripts/launch_config.py", relative,
             "--mode", "dry-run", "--strict-commit", "--run-root", str(run_root)],
            cwd=ROOT, text=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        run_dirs = list(run_root.iterdir())
        assert len(run_dirs) == 1, f"expected one run bundle, got {run_dirs}"
        run_dir = run_dirs[0]

        metadata = json.loads((run_dir / "metadata.json").read_text())
        manifest = tomllib.loads((run_dir / "manifest.resolved.toml").read_text())

    assert Path(metadata["config_manifest"]).name == config.name
    assert metadata["runtime_commit"].startswith("snapshot:")

    resolved = manifest["resolved_profile"]
    assert resolved["env"]["LINGBOT_ATTN_KERNEL"] == declared["env"]["LINGBOT_ATTN_KERNEL"]
    assert resolved["eval_profile"].endswith(".toml")
