"""Context Parallel process-group and runtime configuration.

All CP runtime knobs are sourced here to keep behavior config-first while
retaining environment-variable fallbacks for existing launch scripts.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import torch.distributed as dist
from torch.distributed import ProcessGroup

_CP_GROUP: ProcessGroup | None = None
_WARNED_ENV_KEYS: set[str] = set()


@dataclass
class CpRuntimeConfig:
    """Runtime knobs for CP communication, validation, and memory policy."""

    scan_backend: str | None = None
    allgather_impl: str | None = None
    halo_impl: str | None = None
    # Enables fused Triton GDN blocks to use the CP scan path.
    triton_block_fusion: bool | None = None


_CP_RUNTIME_CONFIG = CpRuntimeConfig()


def _warn_env_fallback_once(env_key: str, config_key: str) -> None:
    key = f"{env_key}->{config_key}"
    if key in _WARNED_ENV_KEYS:
        return
    _WARNED_ENV_KEYS.add(key)
    warnings.warn(
        f"[CP-CONFIG] Using env fallback {env_key}; " f"please migrate to config key {config_key}.",
        stacklevel=2,
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


def _normalized_choice(value: str | None, allowed: set[str], default: str) -> str:
    if value is None:
        return default
    norm = value.strip().lower()
    if norm in allowed:
        return norm
    return default


def set_cp_runtime_config(config: CpRuntimeConfig) -> None:
    """Replace the CP runtime configuration."""
    global _CP_RUNTIME_CONFIG
    _CP_RUNTIME_CONFIG = config


def get_cp_runtime_config() -> CpRuntimeConfig:
    """Return the current CP runtime configuration."""
    return _CP_RUNTIME_CONFIG


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


def get_cp_world_size(default: int = 1) -> int:
    """Get CP world size from the registered CP group."""
    group = _CP_GROUP
    if group is None or not dist.is_available() or not dist.is_initialized():
        return default
    return dist.get_world_size(group)


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


def get_cp_triton_block_fusion() -> bool:
    """Enable the fused Triton block CP path.

    When True, ``ChunkCausalGDNTriton`` / ``BidirectionalGDNTriton`` (and
    the BothTriton variants) take a CP path that wraps the proven
    ``cp_frame_gdn_scan`` algorithm with fused Triton preprocessing/output
    projection kernels. CP execution of these Triton GDN blocks requires
    this flag to be enabled; ``False`` keeps the non-CP behavior unchanged.

    Toggleable via ``train.extra.cp.triton_block_fusion`` (preferred) or
    the legacy env var ``CP_TRITON_BLOCK_FUSION``.
    """
    if _CP_RUNTIME_CONFIG.triton_block_fusion is not None:
        return bool(_CP_RUNTIME_CONFIG.triton_block_fusion)
    env_val = _env_bool("CP_TRITON_BLOCK_FUSION")
    if env_val is not None:
        _warn_env_fallback_once("CP_TRITON_BLOCK_FUSION", "train.extra.cp.triton_block_fusion")
        return env_val
    return False


def init_context_parallel(cp_size: int = 1) -> None:
    """Initialize Context Parallel groups.

    Creates contiguous-rank CP groups of size *cp_size*.  Typically
    called alongside (or instead of) ``init_ulysses_sequence_parallel``
    in the training script.
    """
    set_cp_group(None)
    if cp_size <= 1:
        return
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before CP.")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size % cp_size != 0:
        raise ValueError(f"world_size={world_size} must be divisible by cp_size={cp_size}")

    for i in range(world_size // cp_size):
        start = i * cp_size
        ranks = list(range(start, start + cp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            set_cp_group(group)
