# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Context Parallel process-group and runtime configuration.

All CP runtime knobs are sourced here to keep behavior config-first while
preserving temporary env-var fallback compatibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch.distributed as dist
from torch.distributed import ProcessGroup

_CP_GROUP: ProcessGroup | None = None
_WARNED_ENV_KEYS: set[str] = set()


@dataclass
class CpRuntimeConfig:
    """Runtime knobs for CP communication and diagnostics."""

    scan_backend: str | None = None
    allgather_impl: str | None = None
    halo_impl: str | None = None
    verify_scan: bool | None = None
    comm_trace: bool | None = None
    comm_trace_limit: int | None = None
    comm_world_barrier: bool | None = None
    comm_group_barrier: bool | None = None
    comm_cuda_sync: bool | None = None
    halo_trace: bool | None = None
    halo_trace_limit: int | None = None
    triton_block_fusion: bool | None = None


_CP_RUNTIME_CONFIG = CpRuntimeConfig()


def _warn_env_fallback_once(env_key: str, config_key: str) -> None:
    key = f"{env_key}->{config_key}"
    if key in _WARNED_ENV_KEYS:
        return
    _WARNED_ENV_KEYS.add(key)
    print(
        f"[CP-CONFIG] Using env fallback {env_key}; " f"please migrate to config key {config_key}.",
        flush=True,
    )


def _env_bool(env_key: str) -> bool | None:
    raw = os.environ.get(env_key)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _env_int(env_key: str) -> int | None:
    raw = os.environ.get(env_key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _normalized_choice(value: str | None, allowed: set[str], default: str) -> str:
    if value is None:
        return default
    norm = value.strip().lower()
    if norm in allowed:
        return norm
    return default


def set_cp_group(group: ProcessGroup | None) -> None:
    """Set the Context Parallel process group."""
    global _CP_GROUP
    _CP_GROUP = group


def get_cp_group() -> ProcessGroup | None:
    """Get the Context Parallel process group."""
    return _CP_GROUP


def cp_enabled() -> bool:
    """Return True when Context Parallel is active."""
    group = _CP_GROUP
    if group is None or not dist.is_available() or not dist.is_initialized():
        return False
    return dist.get_world_size(group) > 1


def get_cp_scan_backend() -> str:
    cfg = _CP_RUNTIME_CONFIG.scan_backend
    if cfg is not None:
        return _normalized_choice(cfg, {"torch", "triton"}, "torch")
    env_val = os.environ.get("CP_SCAN_BACKEND")
    if env_val is not None:
        _warn_env_fallback_once("CP_SCAN_BACKEND", "train.extra.cp.scan_backend")
    return _normalized_choice(env_val, {"torch", "triton"}, "torch")


def get_cp_allgather_impl() -> str:
    cfg = _CP_RUNTIME_CONFIG.allgather_impl
    if cfg is not None:
        return _normalized_choice(cfg, {"collective", "list", "p2p"}, "collective")
    env_val = os.environ.get("CP_ALLGATHER_IMPL")
    if env_val is not None:
        _warn_env_fallback_once("CP_ALLGATHER_IMPL", "train.extra.cp.allgather_impl")
    return _normalized_choice(env_val, {"collective", "list", "p2p"}, "collective")


def get_cp_halo_impl() -> str:
    cfg = _CP_RUNTIME_CONFIG.halo_impl
    if cfg is not None:
        return _normalized_choice(cfg, {"collective", "p2p"}, "collective")
    env_val = os.environ.get("CP_HALO_IMPL")
    if env_val is not None:
        _warn_env_fallback_once("CP_HALO_IMPL", "train.extra.cp.halo_impl")
    return _normalized_choice(env_val, {"collective", "p2p"}, "collective")


def get_cp_verify_scan() -> bool:
    if _CP_RUNTIME_CONFIG.verify_scan is not None:
        return bool(_CP_RUNTIME_CONFIG.verify_scan)
    env_val = _env_bool("CP_VERIFY_SCAN")
    if env_val is not None:
        _warn_env_fallback_once("CP_VERIFY_SCAN", "train.extra.cp.verify_scan")
        return env_val
    return False


def get_cp_comm_trace() -> bool:
    if _CP_RUNTIME_CONFIG.comm_trace is not None:
        return bool(_CP_RUNTIME_CONFIG.comm_trace)
    env_val = _env_bool("CP_COMM_TRACE")
    if env_val is not None:
        _warn_env_fallback_once("CP_COMM_TRACE", "train.extra.cp.comm_trace")
        return env_val
    return False


def get_cp_comm_trace_limit() -> int:
    if _CP_RUNTIME_CONFIG.comm_trace_limit is not None:
        return max(0, int(_CP_RUNTIME_CONFIG.comm_trace_limit))
    env_val = _env_int("CP_COMM_TRACE_LIMIT")
    if env_val is not None:
        _warn_env_fallback_once("CP_COMM_TRACE_LIMIT", "train.extra.cp.comm_trace_limit")
        return max(0, env_val)
    return 500


def get_cp_comm_world_barrier() -> bool:
    if _CP_RUNTIME_CONFIG.comm_world_barrier is not None:
        return bool(_CP_RUNTIME_CONFIG.comm_world_barrier)
    env_val = _env_bool("CP_COMM_WORLD_BARRIER")
    if env_val is not None:
        _warn_env_fallback_once("CP_COMM_WORLD_BARRIER", "train.extra.cp.comm_world_barrier")
        return env_val
    return False


def get_cp_comm_group_barrier() -> bool:
    if _CP_RUNTIME_CONFIG.comm_group_barrier is not None:
        return bool(_CP_RUNTIME_CONFIG.comm_group_barrier)
    env_val = _env_bool("CP_COMM_GROUP_BARRIER")
    if env_val is not None:
        _warn_env_fallback_once("CP_COMM_GROUP_BARRIER", "train.extra.cp.comm_group_barrier")
        return env_val
    return False


def get_cp_comm_cuda_sync() -> bool:
    if _CP_RUNTIME_CONFIG.comm_cuda_sync is not None:
        return bool(_CP_RUNTIME_CONFIG.comm_cuda_sync)
    env_val = _env_bool("CP_COMM_CUDA_SYNC")
    if env_val is not None:
        _warn_env_fallback_once("CP_COMM_CUDA_SYNC", "train.extra.cp.comm_cuda_sync")
        return env_val
    return False


def get_cp_halo_trace() -> bool:
    if _CP_RUNTIME_CONFIG.halo_trace is not None:
        return bool(_CP_RUNTIME_CONFIG.halo_trace)
    env_val = _env_bool("CP_HALO_TRACE")
    if env_val is not None:
        _warn_env_fallback_once("CP_HALO_TRACE", "train.extra.cp.halo_trace")
        return env_val
    return False


def get_cp_halo_trace_limit() -> int:
    if _CP_RUNTIME_CONFIG.halo_trace_limit is not None:
        return max(0, int(_CP_RUNTIME_CONFIG.halo_trace_limit))
    env_val = _env_int("CP_HALO_TRACE_LIMIT")
    if env_val is not None:
        _warn_env_fallback_once("CP_HALO_TRACE_LIMIT", "train.extra.cp.halo_trace_limit")
        return max(0, env_val)
    return 400


def get_cp_triton_block_fusion() -> bool:
    if _CP_RUNTIME_CONFIG.triton_block_fusion is not None:
        return bool(_CP_RUNTIME_CONFIG.triton_block_fusion)
    env_val = _env_bool("CP_TRITON_BLOCK_FUSION")
    if env_val is not None:
        _warn_env_fallback_once("CP_TRITON_BLOCK_FUSION", "train.extra.cp.triton_block_fusion")
        return env_val
    return False
