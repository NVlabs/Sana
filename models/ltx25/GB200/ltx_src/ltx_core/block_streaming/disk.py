"""Safetensors I/O and LoRA fusion for block streaming."""

from __future__ import annotations

from collections.abc import Iterator

import safetensors
import torch

from ltx_core.block_streaming.utils import allocate_layout_views, make_block_key
from ltx_core.loader.fuse_loras import LoraProduct
from ltx_core.loader.primitives import LoraStateDictWithStrength, StateDict
from ltx_core.loader.sd_ops import SDOps

_SAFETENSORS_DTYPE_TO_TORCH: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}


class DiskTensorReader:
    """Key-based tensor accessor over one or more safetensors files."""

    def __init__(self, paths: list[str]) -> None:
        self._handles: list[safetensors.safe_open] = []
        self._key_to_handle_idx: dict[str, int] = {}
        for path in paths:
            handle = safetensors.safe_open(path, framework="pt", device="cpu")
            handle_idx = len(self._handles)
            self._handles.append(handle)
            for sft_key in handle.keys():  # noqa: SIM118
                self._key_to_handle_idx[sft_key] = handle_idx

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._handles[self._key_to_handle_idx[key]].get_tensor(key)

    def close(self) -> None:
        self._handles.clear()
        self._key_to_handle_idx.clear()

    def __contains__(self, key: str) -> bool:
        return key in self._key_to_handle_idx

    def __iter__(self) -> Iterator[str]:
        return iter(self._key_to_handle_idx)


class DiskBlockReader:
    """Reads one block at a time from safetensors into provided buffers."""

    def __init__(
        self,
        reader: DiskTensorReader,
        block_key_map: dict[int, list[tuple[str, str]]],
        sd_ops: SDOps | None = None,
        blocks_prefix: str = "",
    ) -> None:
        self._reader = reader
        self._block_key_map = block_key_map
        self._sd_ops = sd_ops
        self._blocks_prefix = blocks_prefix

    def read_into(self, target: dict[str, torch.Tensor], block_idx: int) -> None:
        block_prefix = make_block_key(self._blocks_prefix, block_idx, "")
        for sft_key, param_name in self._block_key_map[block_idx]:
            tensor = self._reader.get_tensor(sft_key)
            if self._sd_ops is None:
                target[param_name].copy_(tensor)
                continue
            full_key = make_block_key(self._blocks_prefix, block_idx, param_name)
            for result in self._sd_ops.apply_to_key_value(full_key, tensor):
                if not result.new_key.startswith(block_prefix):
                    raise ValueError(
                        f"SDOps output key '{result.new_key}' is outside block {block_idx} "
                        f"(expected prefix '{block_prefix}'); cannot route to a per-block buffer."
                    )
                target[result.new_key[len(block_prefix) :]].copy_(result.new_value)

    def cleanup(self) -> None:
        self._reader.close()


class LoraSource:
    """Pinned-memory cache of matched LoRA A/B factors backed by a single buffer."""

    def __init__(self, path: str, sd_ops: SDOps | None, strength: float) -> None:
        self.strength = strength
        self._pinned_ab: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        a_keys: dict[str, str] = {}
        b_keys: dict[str, str] = {}
        with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
            for sft_key in handle.keys():  # noqa: SIM118
                model_key = sd_ops.apply_to_key(sft_key) if sd_ops is not None else sft_key
                if model_key is None:
                    continue
                if model_key.endswith(".lora_A.weight"):
                    a_keys[model_key[: -len(".lora_A.weight")]] = sft_key
                elif model_key.endswith(".lora_B.weight"):
                    b_keys[model_key[: -len(".lora_B.weight")]] = sft_key

            matched_prefixes = list(a_keys.keys() & b_keys.keys())

            # Build the layout from safetensors header metadata only — no tensor data is read.
            layout: dict[str, tuple[torch.Size, torch.dtype]] = {}
            for prefix in matched_prefixes:
                a_slice_view = handle.get_slice(a_keys[prefix])
                b_slice_view = handle.get_slice(b_keys[prefix])
                layout[f"{prefix}.A"] = (
                    torch.Size(a_slice_view.get_shape()),
                    _SAFETENSORS_DTYPE_TO_TORCH[a_slice_view.get_dtype()],
                )
                layout[f"{prefix}.B"] = (
                    torch.Size(b_slice_view.get_shape()),
                    _SAFETENSORS_DTYPE_TO_TORCH[b_slice_view.get_dtype()],
                )

            all_views = allocate_layout_views(layout, pin_memory=True)

            for prefix in matched_prefixes:
                a_view = all_views[f"{prefix}.A"]
                b_view = all_views[f"{prefix}.B"]
                a_view.copy_(handle.get_tensor(a_keys[prefix]))
                b_view.copy_(handle.get_tensor(b_keys[prefix]))
                self._pinned_ab[prefix] = (a_view, b_view)

    def as_state_dict_with_strength(self) -> LoraStateDictWithStrength:
        """Return a :class:`LoraStateDictWithStrength` view of the pinned A/B factors.
        Lets non-block fusion consume the already-loaded disk-streaming LoRA
        without re-reading the safetensors file or re-applying ``sd_ops``.
        """
        sd: dict[str, torch.Tensor] = {}
        for prefix, (a, b) in self._pinned_ab.items():
            sd[f"{prefix}.lora_A.weight"] = a
            sd[f"{prefix}.lora_B.weight"] = b
        size = sum(t.numel() * t.element_size() for t in sd.values())
        dtypes = {t.dtype for t in sd.values()}
        return LoraStateDictWithStrength(
            StateDict(sd=sd, device=torch.device("cpu"), size=size, dtype=dtypes),
            self.strength,
        )

    def get_ab(
        self,
        param_prefix: str,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> LoraProduct | None:
        """Return the :class:`LoraProduct` for *param_prefix*, or ``None``."""
        pair = self._pinned_ab.get(param_prefix)
        if pair is None:
            return None
        a, b = pair
        # Move A/B to a GPU-class target (CUDA/MPS) so the B@A aggregation runs on
        # the device; on a CPU target they stay put. non_blocking only helps CUDA.
        if device is not None and device.type in ("cuda", "mps"):
            a = a.to(device=device, non_blocking=device.type == "cuda")
            b = b.to(device=device, non_blocking=device.type == "cuda")
        if dtype is not None:
            a = a.to(dtype=dtype)
            b = b.to(dtype=dtype)
        return LoraProduct(a, b, self.strength)

    def cleanup(self) -> None:
        self._pinned_ab.clear()
