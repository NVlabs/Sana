"""All2All communication primitives for distributed attention.
The C++ kernels are exposed via ``torch.library.custom_op`` so that
``torch.compile`` (and CUDA Graph capture under ``mode="reduce-overhead"``)
can trace through them without a graph break. Each :class:`All2All`
instance registers itself in a class-level registry indexed by an integer
``comm_id``; the custom op takes that id plus an input tensor and dispatches
to the appropriate C++ runtime.
"""

import math
import weakref
from typing import Any, ClassVar

import torch
import torch.distributed as dist
from all2all_cpp import All2All as All2AllCpp
from torch.library import custom_op

# Output shapes are derived symbolically from the input tensor's shape under
# the assumption of *uniform* sharding (caller pads up-front so
# `total_tokens % world_size == 0`). `world_size` travels through the op as an int
# so it enters the traced graph: the fake needs it for the output shape, and since
# it is constant per process the Dynamo guard it installs never triggers a
# within-run recompile -- it only keys the compile cache by world size, so a graph
# compiled under one GPU count is never replayed under another. Per-rank token
# counts, which do vary per call, stay in the C++ runtime state (`set_rank_tokens`)
# and never travel through the op.


@custom_op("ltx_kernels::send_recv_heads", mutates_args=(), device_types="cuda")
def _send_recv_heads_op(
    x: torch.Tensor,
    comm_id: int,
    world_size: int,  # noqa: ARG001
    copy_out: bool,
) -> torch.Tensor:
    # copy_out=False returns a view into the IPC buffer (zero-copy). Safe under
    # cudagraph_trees because the IPC buffer is cudaMalloc'd inside All2All, not
    # in the static graph pool, so it isn't subject to graph-pool aliasing.
    return All2All._runtime_registry[comm_id].send_recv_heads(x, copy_out)


@_send_recv_heads_op.register_fake
def _send_recv_heads_fake(
    x: torch.Tensor,
    comm_id: int,  # noqa: ARG001
    world_size: int,
    copy_out: bool,  # noqa: ARG001
) -> torch.Tensor:
    return x.new_empty((x.shape[0], x.shape[1] * world_size, x.shape[2] // world_size, x.shape[3]))


@custom_op("ltx_kernels::gather_heads", mutates_args=(), device_types="cuda")
def _gather_heads_op(
    x: torch.Tensor,
    comm_id: int,
    world_size: int,  # noqa: ARG001
    copy_out: bool,
) -> torch.Tensor:
    return All2All._runtime_registry[comm_id].gather_heads(x, copy_out)


@_gather_heads_op.register_fake
def _gather_heads_fake(
    x: torch.Tensor,
    comm_id: int,  # noqa: ARG001
    world_size: int,
    copy_out: bool,  # noqa: ARG001
) -> torch.Tensor:
    return x.new_empty((x.shape[0], x.shape[1] // world_size, x.shape[2] * world_size, x.shape[3]))


class All2All:
    """IPC-based All2All communication for distributed head-parallel attention.
    This class manages GPU memory buffers and IPC handles to enable efficient
    cross-GPU communication for attention head redistribution.
    Args:
        rank: Local rank of this process.
        world_size: Total number of processes in the distributed group.
        seqlen: Maximum sequence length to allocate buffers for.
        hidden_dim: Hidden dimension size (num_heads * head_dim).
        num_sms: Number of SMs to use for kernel execution.
        tensor_dtype: Data type for tensors (e.g., torch.bfloat16).
        group: PyTorch distributed process group.
    """

    _next_id: ClassVar[int] = 0
    _runtime_registry: ClassVar[dict[int, "All2AllCpp"]] = {}

    @staticmethod
    def _release_comm(comm_id: int, runtime: "All2AllCpp") -> None:
        All2All._runtime_registry.pop(comm_id, None)
        runtime.destroy()

    def __init__(
        self,
        rank: int,
        world_size: int,
        seqlen: int,
        hidden_dim: int,
        num_sms: int,
        tensor_dtype: torch.dtype,
        group: torch.distributed.ProcessGroup | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.num_sms = num_sms
        self.tensor_dtype = tensor_dtype
        self.buffer_size = int(seqlen * hidden_dim * tensor_dtype.itemsize)

        # Initialize the C++ runtime. Omit timeout_seconds to keep the kernel's default
        # (DEFAULT_BARRIER_TIMEOUT_SECONDS); it can still be changed later via set_timeout_seconds.
        if timeout_seconds is None:
            self.runtime = All2AllCpp(rank, world_size, seqlen, hidden_dim, num_sms, tensor_dtype)
        else:
            if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
                raise ValueError(f"all2all timeout seconds must be finite and non-negative, got {timeout_seconds}")
            self.runtime = All2AllCpp(rank, world_size, seqlen, hidden_dim, num_sms, tensor_dtype, timeout_seconds)

        # Register in the class-level table so the custom ops can find us.
        # Auto-cleanup via weakref.finalize covers the case where destroy()
        # isn't called explicitly — without it the registry would keep the
        # runtime (and CUDA/IPC resources) alive for the process lifetime.
        self._comm_id = All2All._next_id
        All2All._next_id += 1
        All2All._runtime_registry[self._comm_id] = self.runtime
        self._finalizer = weakref.finalize(self, All2All._release_comm, self._comm_id, self.runtime)

        # Exchange IPC handles across all ranks
        ipc_handles: list[Any] = [None] * world_size
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        dist.all_gather_object(ipc_handles, local_ipc_handle, group)

        self.runtime.sync(ipc_handles)

    def set_rank_tokens(self, rank_num_tokens: list[int]) -> None:
        """Sets per-rank token counts on the C++ runtime."""
        self.runtime.set_rank_tokens(rank_num_tokens)

    def set_timeout_seconds(self, seconds: float) -> None:
        """Set the all2all barrier (deadlock-detection) timeout, in seconds.
        The C++ runtime converts to clock cycles using the device's peak SM clock. Raise it
        during the first ``torch.compile`` forward, where one rank's recompile can delay its
        all2all launch past the steady-state timeout; reset for steady state.
        """
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"all2all timeout seconds must be finite and non-negative, got {seconds}")
        self.runtime.set_timeout_seconds(seconds)

    def send_recv_heads(self, x: torch.Tensor, *, copy_out: bool = False) -> torch.Tensor:
        """Exchange attention heads across ranks (All2All pattern).
        Args:
            x: Input tensor of shape [batch, tokens, heads, head_dim].
            copy_out: If True, copy result to a new tensor instead of using buffer.
        Returns:
            Output tensor with redistributed heads.
        """
        return torch.ops.ltx_kernels.send_recv_heads(x, self._comm_id, self.world_size, copy_out)

    def gather_heads(self, x: torch.Tensor, *, copy_out: bool = False) -> torch.Tensor:
        """Gather heads back to original distribution (reverse All2All).
        Args:
            x: Input tensor with distributed heads.
            copy_out: If True, copy result to a new tensor instead of using buffer.
        Returns:
            Output tensor with gathered heads.
        """
        return torch.ops.ltx_kernels.gather_heads(x, self._comm_id, self.world_size, copy_out)

    def allgather(self, x: torch.Tensor, *, copy_out: bool = False) -> torch.Tensor:
        """Allgather operation across all ranks.
        Args:
            x: Input tensor to gather.
            copy_out: If True, copy result to a new tensor instead of using buffer.
        Returns:
            Gathered tensor from all ranks.
        """
        return self.runtime.allgather(x, copy_out)

    def destroy(self) -> None:
        """Release IPC handles and GPU memory buffers."""
        self._finalizer()
