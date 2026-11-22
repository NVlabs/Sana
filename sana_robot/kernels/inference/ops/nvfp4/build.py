# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sana_robot.kernels.inference.ops.dispatch import SM110, SM120

CUTLASS_TAG = "v4.8.1"
CUTLASS_REPOSITORY = "https://github.com/NVIDIA/cutlass.git"
CACHE_DIRECTORY = Path.home() / ".cache" / "sana_robot"
SOURCE_DIRECTORY = Path(__file__).resolve().parent / "csrc"
CUDA_TARGETS = {SM110: "11.0a", SM120: "12.0a"}
BUILD_JOBS = {SM110: 2}
STALE_LOCK_SECONDS = 1800
EXTENSIONS = {}


def fetch_cutlass() -> Path:
    """Return the directory of CUTLASS v4.8.1 under ``~/.cache/sana_robot``, cloned from GitHub when it is not there yet."""
    root = CACHE_DIRECTORY / f"cutlass-{CUTLASS_TAG}"
    if not (root / "include" / "cutlass" / "cutlass.h").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", CUTLASS_TAG, "-q", CUTLASS_REPOSITORY, str(root)], check=True
        )
    return root


def load_extension(architecture: str):
    """Return the compiled NVFP4 matrix product of an architecture (``sm110`` or ``sm120``) as a Python module with ``gemm`` and ``configs``.

    The first call of a process compiles ``csrc/<architecture>`` against CUTLASS with ``nvcc`` and ``ninja`` for that
    architecture alone, one translation unit per tile configuration, or loads the build that
    ``~/.cache/sana_robot/torch_extensions`` already holds. The units compile in parallel, all at once, except on
    Jetson Thor: its memory is shared with the GPU, a unit takes about 5 GB to compile and the build usually runs with
    a policy already loaded, so two at a time (about 4 minutes); ``MAX_JOBS`` overrides both. The lock file of a build
    that was interrupted (older than ``STALE_LOCK_SECONDS``) is removed first, since every later build would wait on it.
    """
    if architecture in EXTENSIONS:
        return EXTENSIONS[architecture]
    cutlass = fetch_cutlass()
    from torch.utils import cpp_extension

    name = f"sana_robot_nvfp4_gemm_{architecture}"
    sources = SOURCE_DIRECTORY / architecture
    build_directory = CACHE_DIRECTORY / "torch_extensions" / name
    build_directory.mkdir(parents=True, exist_ok=True)
    lock = build_directory / "lock"
    if lock.exists() and time.time() - lock.stat().st_mtime > STALE_LOCK_SECONDS:
        print(f"removing the lock of an interrupted NVFP4 build: {lock}", file=sys.stderr, flush=True)
        lock.unlink(missing_ok=True)
    units = [str(sources / "binding.cu")] + sorted(str(path) for path in sources.glob("config_*.cu"))
    if not (build_directory / f"{name}.so").exists():
        print(
            f"compiling the NVFP4 matrix product for {architecture} (once, a few minutes)", file=sys.stderr, flush=True
        )
    interpreter_directory = Path(sys.executable).parent
    if shutil.which("ninja") is None and (interpreter_directory / "ninja").exists():
        os.environ["PATH"] = f"{interpreter_directory}{os.pathsep}{os.environ.get('PATH', '')}"
    previous_target = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = CUDA_TARGETS[architecture]
    os.environ.setdefault("MAX_JOBS", str(BUILD_JOBS.get(architecture, len(units))))
    try:
        extension = cpp_extension.load(
            name=name,
            sources=units,
            extra_include_paths=[str(sources), str(cutlass / "include"), str(cutlass / "tools" / "util" / "include")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-DNDEBUG",
                "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                "-lineinfo",
            ],
            build_directory=str(build_directory),
        )
    finally:
        if previous_target is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_target
    EXTENSIONS[architecture] = extension
    return extension
