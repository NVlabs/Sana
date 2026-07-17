from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

try:
    from torch.distributed._composable.fsdp import fully_shard
except Exception as exc:  # pragma: no cover - depends on the installed torch build
    fully_shard = None
    FSDP_IMPORT_ERROR = exc
else:
    FSDP_IMPORT_ERROR = None


@dataclass(frozen=True)
class FSDPInferenceInfo:
    enabled: bool
    world_size: int
    wrapped_blocks: int
    ignored_params: int


def init_fsdp_inference_mesh() -> DeviceMesh | None:
    if not dist.is_available() or not dist.is_initialized():
        return None
    world_size = dist.get_world_size()
    if world_size <= 1:
        return None
    return init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))


def _move_buffers_to_device(module: torch.nn.Module, device: torch.device) -> None:
    for submodule in module.modules():
        for name, buffer in tuple(submodule.named_buffers(recurse=False)):
            if buffer is not None and buffer.device != device:
                submodule._buffers[name] = buffer.to(device=device)


def _move_parameters_to_device(
    parameters: set[torch.nn.Parameter],
    device: torch.device,
) -> None:
    for param in parameters:
        if param.device != device:
            param.data = param.data.to(device=device)
        if param.grad is not None and param.grad.device != device:
            param.grad.data = param.grad.data.to(device=device)


def _current_cuda_device() -> torch.device | None:
    if not torch.cuda.is_available():
        return None
    return torch.device("cuda", torch.cuda.current_device())


def apply_fsdp_inference(
    transformer: torch.nn.Module,
    mesh: DeviceMesh | None,
) -> FSDPInferenceInfo:
    if mesh is None:
        return FSDPInferenceInfo(enabled=False, world_size=1, wrapped_blocks=0, ignored_params=0)
    if bool(getattr(transformer, "_lingbot_fsdp_inference_enabled", False)):
        blocks = getattr(transformer, "blocks", ())
        return FSDPInferenceInfo(
            enabled=True,
            world_size=int(mesh.size()),
            wrapped_blocks=len(blocks),
            ignored_params=int(getattr(transformer, "_lingbot_fsdp_inference_ignored_params", 0)),
        )

    if fully_shard is None:
        raise RuntimeError("PyTorch composable FSDP is not importable.") from FSDP_IMPORT_ERROR

    cuda_device = _current_cuda_device()
    if cuda_device is not None:
        _move_buffers_to_device(transformer, cuda_device)

    dtype_counts: dict[torch.dtype, int] = {}
    for param in transformer.parameters():
        dtype_counts[param.dtype] = dtype_counts.get(param.dtype, 0) + param.numel()
    if dtype_counts:
        primary_dtype = max(dtype_counts.items(), key=lambda item: item[1])[0]
        ignored_params = {param for param in transformer.parameters() if param.dtype != primary_dtype}
    else:
        ignored_params = set()
    if cuda_device is not None and ignored_params:
        _move_parameters_to_device(ignored_params, cuda_device)

    blocks: Any = getattr(transformer, "blocks", ())
    wrapped_blocks = 0
    for block in blocks:
        block_ignored_params = {
            param for param in block.parameters() if param in ignored_params
        }
        fully_shard(block, mesh=mesh, ignored_params=block_ignored_params)
        wrapped_blocks += 1
    fully_shard(transformer, mesh=mesh, ignored_params=ignored_params)
    gc.collect()
    if cuda_device is not None:
        torch.cuda.empty_cache()
    transformer._lingbot_fsdp_inference_enabled = True
    transformer._lingbot_fsdp_inference_ignored_params = len(ignored_params)
    return FSDPInferenceInfo(
        enabled=True,
        world_size=int(mesh.size()),
        wrapped_blocks=wrapped_blocks,
        ignored_params=len(ignored_params),
    )
