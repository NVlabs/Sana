# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime

import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import InitProcessGroupKwargs
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

PROCESS_GROUP_TIMEOUT = datetime.timedelta(seconds=5400)


def build_accelerator(block_name: str) -> Accelerator:
    """Build the Accelerator of a run: bf16 autocast and, on several GPUs, FSDP that shards the parameters inside a node and replicates them across nodes, one FSDP unit per policy layer."""
    plugin = FullyShardedDataParallelPlugin(
        sharding_strategy="HYBRID_SHARD",
        backward_prefetch="BACKWARD_PRE",
        forward_prefetch=False,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=[block_name],
        state_dict_type="FULL_STATE_DICT",
        use_orig_params=True,
        sync_module_states=True,
        cpu_offload=False,
        cpu_ram_efficient_loading=False,
    )
    handler = InitProcessGroupKwargs(timeout=PROCESS_GROUP_TIMEOUT)
    return Accelerator(mixed_precision="bf16", fsdp_plugin=plugin, log_with="wandb", kwargs_handlers=[handler])


def shard_process_group(model: torch.nn.Module):
    """Return the process group that holds the shards of one gradient, or None when the model is not sharded."""
    if isinstance(model, FSDP) and model.sharding_strategy != ShardingStrategy.NO_SHARD:
        return model.process_group
    return None
