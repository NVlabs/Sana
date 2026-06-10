"""Context Parallel for Gated Delta Net (training).

Splits the temporal sequence across GPUs and communicates only the
D x D recurrent state between ranks, eliminating the head-divisibility
constraint of Ulysses SP and reducing communication from O(T*S*H*D)
to O(D^2*H).
"""

from diffusion.distributed.context_parallel.config import (
    CpRuntimeConfig,
    cp_enabled,
    get_cp_group,
    get_cp_runtime_config,
    get_cp_world_size,
    init_context_parallel,
    set_cp_group,
    set_cp_runtime_config,
)
from diffusion.distributed.context_parallel.data_utils import (
    cp_broadcast_tensor,
    cp_build_frame_valid_mask,
    cp_reduce_loss,
    cp_right_pad_size,
    cp_right_pad_temporal,
    cp_split_temporal,
)
from diffusion.distributed.context_parallel.halo_exchange import cp_halo_exchange

__all__ = [
    "cp_broadcast_tensor",
    "cp_build_frame_valid_mask",
    "cp_enabled",
    "cp_halo_exchange",
    "cp_right_pad_size",
    "cp_right_pad_temporal",
    "cp_reduce_loss",
    "cp_split_temporal",
    "CpRuntimeConfig",
    "get_cp_group",
    "get_cp_runtime_config",
    "get_cp_world_size",
    "init_context_parallel",
    "set_cp_group",
    "set_cp_runtime_config",
]
