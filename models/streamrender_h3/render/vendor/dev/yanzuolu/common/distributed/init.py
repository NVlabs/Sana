"""Distributed process-group bootstrap."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from ..logging import get_logger
from ..seed import seed_everything

logger = get_logger()


def configure_distributed(config: Any) -> Any:
    """Initialize the default process group before anything else touches it.

    Only runs under a distributed launcher (torchrun sets WORLD_SIZE); plain
    single-process invocations skip initialization entirely.

    Decisions baked in:
    - bind the local CUDA device *before* creating the process group so NCCL
      communicators attach to the right GPU;
    - dual-backend default pg (``cpu:gloo,cuda:nccl``, the form recommended by
      ``dcp.async_save``) so CPU collectives never ride the NCCL stream;
    - ``device_id`` enables eager NCCL init and device-aware ``barrier()``;
    - initializing here -- before persistence builds the Checkpointer and
      before any training collective is in flight -- keeps later
      ``dist.new_group`` calls (e.g. the checkpointer's async gloo group) out
      of mid-training timing races.

    Environment knobs (config node ``distributed``): ``timeout_minutes`` (30),
    ``cudnn_benchmark`` (false), ``deterministic`` (false; also pins the cuBLAS
    workspace for reproducible GEMMs), ``cuda_alloc_conf``
    (``expandable_segments:True``; curbs fragmentation under video workloads'
    varying allocation sizes). Env vars use ``setdefault`` semantics -- an
    external ``export`` wins. They only take effect because this function is
    the first CUDA-touching code in the launch chain: torch reads the device
    order at first enumeration and the allocator config at first allocation,
    never at import.
    """
    dist_config = config.get("distributed", {})
    # The env knobs below are read by torch/CUDA lazily (device order at first
    # driver enumeration, allocator config at CUDA state init) -- they only
    # work if nothing has touched CUDA yet. Heavy libraries are all imported
    # lazily inside stage functions for exactly this reason; this assert fails
    # loud if a future import regresses that discipline.
    assert not torch.cuda.is_initialized(), "CUDA was initialized before configure_distributed; env knobs would be ignored"
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", dist_config.get("cuda_alloc_conf", "expandable_segments:True"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cudnn.benchmark = dist_config.get("cudnn_benchmark", False)
    if dist_config.get("deterministic", False):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    # Same seed on every rank, before anything constructs a model: replicated
    # parameters (LoRA init, from-scratch init) must be born bit-identical
    # across ranks -- sharding trusts local values and never re-syncs them.
    # Per-rank streams are derived later by the engine (common.seed.RandomState).
    seed_everything(config.get("seed", 1019))

    if "WORLD_SIZE" not in os.environ:
        return config

    device = None
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    timeout_minutes = dist_config.get("timeout_minutes", 30)
    backend = "cpu:gloo,cuda:nccl" if device is not None else "gloo"
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(minutes=timeout_minutes),
        device_id=device,
    )
    logger.info(
        "Distributed initialized: rank=%s/%s local_rank=%s backend=%s device=%s timeout=%smin",
        dist.get_rank(), dist.get_world_size(), os.environ.get("LOCAL_RANK", "0"), backend, device, timeout_minutes,
    )
    return config


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
