#!/usr/bin/env python3
"""Self-contained tests for tools/vision/lpips_judge.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import lpips_judge


TOOLS_DIR = Path(__file__).resolve().parent
SCRIPT = TOOLS_DIR / "lpips_judge.py"


def write_blocking_module(directory: Path, name: str, exception: str = "ImportError") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.py").write_text(
        f"raise {exception}('simulated missing optional dependency')\n"
    )


def env_with_pythonpath(prefix: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(prefix) if not existing else f"{prefix}{os.pathsep}{existing}"
    return env


def write_tiny_frames(directory: Path) -> tuple[Path, Path, bool]:
    baseline = directory / "baseline.png"
    transfeat = directory / "transfeat.png"
    try:
        from PIL import Image  # type: ignore
    except Exception:
        baseline.write_bytes(b"placeholder baseline")
        transfeat.write_bytes(b"placeholder transfeat")
        return baseline, transfeat, False

    Image.new("RGB", (2, 2), (0, 0, 0)).save(baseline)
    Image.new("RGB", (2, 2), (1, 1, 1)).save(transfeat)
    return baseline, transfeat, True


def run_cli(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def assert_json_payload(data: dict[str, object]) -> None:
    assert data["metric"] == "lpips"
    assert data["status"] in {"ok", "unavailable"}
    assert isinstance(data["n"], int)
    if data["status"] == "ok":
        for key in ("per_frame", "mean", "median", "max", "notes"):
            assert key in data
        assert data["n"] == len(data["per_frame"])  # type: ignore[arg-type]
    else:
        assert data["n"] == 0
        assert isinstance(data["reason"], str)


def test_import_is_lightweight() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        blocker = Path(tmp)
        write_blocking_module(blocker, "lpips", "RuntimeError")
        write_blocking_module(blocker, "torch", "RuntimeError")
        proc = subprocess.run(
            [sys.executable, "-c", "import lpips_judge; print('import-ok')"],
            cwd=str(TOOLS_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env_with_pythonpath(blocker),
        )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "import-ok"


def test_cli_frame_pair_emits_valid_json() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        baseline, transfeat, _used_pillow = write_tiny_frames(root)
        blocker = root / "blocker"
        write_blocking_module(blocker, "lpips")
        out = root / "lpips.json"
        proc = run_cli(
            [
                "--baseline-frame",
                str(baseline),
                "--transfeat-frame",
                str(transfeat),
                "--out",
                str(out),
            ],
            env=env_with_pythonpath(blocker),
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == ""
        data = json.loads(out.read_text())
    assert_json_payload(data)


def test_unavailable_fallback_exits_zero_and_writes_stdout_json() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        baseline, transfeat, _used_pillow = write_tiny_frames(root)
        blocker = root / "blocker"
        write_blocking_module(blocker, "lpips")
        proc = run_cli(
            [
                "--baseline-frame",
                str(baseline),
                "--transfeat-frame",
                str(transfeat),
            ],
            env=env_with_pythonpath(blocker),
        )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["status"] == "unavailable"
    assert data["n"] == 0
    assert "lpips" in data["reason"]
    assert_json_payload(data)


def test_missing_torch_is_unavailable_without_scoring() -> None:
    def fake_import(name: str) -> object:
        if name == "lpips":
            return object()
        if name == "torch":
            raise ImportError("simulated missing torch")
        raise AssertionError(f"unexpected import: {name}")

    try:
        lpips_judge.load_lpips_modules(fake_import)
    except lpips_judge.MetricUnavailable as exc:
        assert "torch is not importable" in str(exc)
    else:
        raise AssertionError("missing torch should make LPIPS unavailable")


def test_success_payload_schema_without_optional_dependencies() -> None:
    payload = lpips_judge.success_payload([0.0, 0.5, 1.0], ["test-note"])
    assert payload["metric"] == "lpips"
    assert payload["status"] == "ok"
    assert payload["per_frame"] == [0.0, 0.5, 1.0]
    assert payload["mean"] == 0.5
    assert payload["median"] == 0.5
    assert payload["max"] == 1.0
    assert payload["n"] == 3
    assert payload["notes"] == ["test-note"]


def main() -> None:
    test_import_is_lightweight()
    test_cli_frame_pair_emits_valid_json()
    test_unavailable_fallback_exits_zero_and_writes_stdout_json()
    test_missing_torch_is_unavailable_without_scoring()
    test_success_payload_schema_without_optional_dependencies()
    print("lpips_judge tests passed")


if __name__ == "__main__":
    main()
