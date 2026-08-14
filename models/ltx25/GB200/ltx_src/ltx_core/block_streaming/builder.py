"""Builder that constructs a BlockStreamingWrapper from safetensors checkpoints."""

from __future__ import annotations

import copy
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Final, Generic

import safetensors
import torch
from torch import nn

from ltx_core.block_streaming import utils as bs_utils
from ltx_core.block_streaming.block_fetcher import BlockFetcher
from ltx_core.block_streaming.disk import DiskBlockReader, DiskTensorReader, LoraSource
from ltx_core.block_streaming.pool import BufferPool
from ltx_core.block_streaming.provider import WeightsProvider
from ltx_core.block_streaming.source import DiskWeightSource, PinnedBlock, PinnedWeightSource, WeightSource
from ltx_core.block_streaming.stream_sync import create_stream_sync
from ltx_core.block_streaming.utils import (
    carve_buffer,
    derive_layout,
    layout_nbytes,
    make_block_key,
    resolve_attr,
)
from ltx_core.block_streaming.wrapper import BlockStreamingWrapper
from ltx_core.devices import synchronize_device
from ltx_core.loader.fuse_loras import FuseRule, bf16_fuse_rule, fuse_lora_weights
from ltx_core.loader.helpers import create_meta_model, load_state_dict, read_model_config, read_model_metadata
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import (
    LoraPathStrengthAndSDOps,
    LoraStateDictWithStrength,
    ModelBuilderProtocol,
    StateDict,
    StateDictLoader,
    TensorLayout,
)
from ltx_core.loader.registry import ModelRegistry, Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.model_protocol import ModelConfigurator, ModelType

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)

DISK_CPU_SLOTS = 2
_DEFAULT_GPU_SLOTS = 2
_PREFETCH_DEPTH = 2


class StreamingModelBuilder(Generic[ModelType], ModelBuilderProtocol[ModelType]):
    """Immutable builder for :class:`BlockStreamingWrapper`.
    Reads block weights from safetensors on demand.  ``cpu_slots`` and
    ``gpu_slots`` control the memory/speed trade-off (see :meth:`build`).
    The builder is immutable (``with_*`` return modified copies) and exposes
    its state via read-only properties backed by private attributes.
    Args:
        model_class_configurator: Creates the model from a config dict.
        model_path: One or more ``.safetensors`` checkpoint paths.
        model_sd_ops: Key remapping applied to safetensors keys.
        module_ops: Module-level mutations for the meta model.
        loras: LoRA adapters fused into weights at load time.
        model_loader: Strategy for reading checkpoint metadata.
        registry: Shared cache for loaded state dicts.
        fuse_rule: Per-policy LoRA merge rule. Defaults to ``bf16_fuse_rule``;
            use ``fp8_cast_fuse_rule`` for fp8_cast streaming so the pinned
            buffers receive correctly-quantized weights.
        blocks_attr: Dotted path to the ``nn.ModuleList`` (e.g.
            ``"transformer_blocks"``).
        blocks_prefix: State-dict key prefix for block weights
            (e.g. ``"transformer_blocks"``).
        cpu_slots_count: Default number of pinned CPU buffer slots used by
            :meth:`build` when it is not given an explicit ``cpu_slots_count``.
            ``None`` = RAM streaming (all blocks pinned); a small value (e.g.
            ``DISK_CPU_SLOTS``) selects disk streaming. Lets a builder fully
            encode its offload behaviour so callers need not re-specify it.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_class_configurator: type[ModelConfigurator[ModelType]],
        model_path: str | tuple[str, ...],
        model_sd_ops: SDOps | None = None,
        module_ops: tuple[ModuleOps, ...] = (),
        loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
        model_loader: StateDictLoader | None = None,
        registry: Registry | None = None,
        fuse_rule: FuseRule = bf16_fuse_rule,
        blocks_attr: str = "",
        blocks_prefix: str = "",
        cpu_slots_count: int | None = None,
    ) -> None:
        # Read-only: typed with the covariant ModelType, so it must not be a mutable attribute.
        self._model_class_configurator: Final = model_class_configurator
        self._model_path = model_path
        self._model_sd_ops = model_sd_ops
        self._module_ops = module_ops
        self._loras = loras
        self._model_loader = model_loader if model_loader is not None else SafetensorsModelStateDictLoader()
        self._registry = registry if registry is not None else ModelRegistry(cache_models=True, cache_weights=False)
        self._fuse_rule = fuse_rule
        self._blocks_attr = blocks_attr
        self._blocks_prefix = blocks_prefix
        self._cpu_slots_count = cpu_slots_count

    @property
    def model_class_configurator(self) -> type[ModelConfigurator[ModelType]]:
        return self._model_class_configurator

    @property
    def model_path(self) -> str | tuple[str, ...]:
        return self._model_path

    @property
    def checkpoint(self) -> str | tuple[str, ...]:
        return self._model_path

    @property
    def model_sd_ops(self) -> SDOps | None:
        return self._model_sd_ops

    @property
    def module_ops(self) -> tuple[ModuleOps, ...]:
        return self._module_ops

    @property
    def loras(self) -> tuple[LoraPathStrengthAndSDOps, ...]:
        return self._loras

    @property
    def model_loader(self) -> StateDictLoader:
        return self._model_loader

    @property
    def registry(self) -> Registry:
        return self._registry

    @property
    def fuse_rule(self) -> FuseRule:
        return self._fuse_rule

    @property
    def blocks_attr(self) -> str:
        return self._blocks_attr

    @property
    def blocks_prefix(self) -> str:
        return self._blocks_prefix

    @property
    def cpu_slots_count(self) -> int | None:
        return self._cpu_slots_count

    @property
    def keeps_gpu_resident_weights(self) -> bool:
        # Block weights live in fixed GPU/CPU slots that are reused across builds.
        return True

    def with_sd_ops(self, sd_ops: SDOps | None) -> Self:
        clone = copy.copy(self)
        clone._model_sd_ops = sd_ops
        return clone

    def with_module_ops(self, module_ops: tuple[ModuleOps, ...]) -> Self:
        clone = copy.copy(self)
        clone._module_ops = module_ops
        return clone

    def with_loras(self, loras: tuple[LoraPathStrengthAndSDOps, ...]) -> Self:
        clone = copy.copy(self)
        clone._loras = loras
        return clone

    def with_registry(self, registry: Registry) -> Self:
        clone = copy.copy(self)
        clone._registry = registry
        return clone

    def with_lora_load_device(self, device: torch.device) -> Self:
        # Streaming fuses LoRAs into pinned CPU buffers; no other staging device is meaningful.
        raise NotImplementedError("StreamingModelBuilder loads LoRA weights on CPU only.")

    def with_fuse_rule(self, fuse_rule: FuseRule) -> Self:
        clone = copy.copy(self)
        clone._fuse_rule = fuse_rule
        return clone

    def model_config(self) -> dict:
        """Read model configuration from the checkpoint metadata."""
        return read_model_config(self.model_path, self.model_loader)

    def model_metadata(self) -> dict:
        """Read the full checkpoint metadata (including sibling keys beyond ``config``)."""
        return read_model_metadata(self.model_path, self.model_loader)

    def meta_model(self, metadata: dict, module_ops: tuple[ModuleOps, ...]) -> ModelType:
        """Create a model on the meta device and apply module operations."""
        return create_meta_model(self.model_class_configurator, metadata, module_ops)

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        cpu_slots_count: int | None = None,
        gpu_slots_count: int | None = None,
        **_kwargs: object,
    ) -> BlockStreamingWrapper:
        """Build and return a ready-to-use :class:`BlockStreamingWrapper`.
        Args:
            device: GPU device for compute. ``None`` defaults to ``cuda``.
            dtype: Weight dtype (e.g. ``torch.bfloat16``). Required.
            cpu_slots_count: Number of pinned CPU buffer slots. ``None`` falls
                back to the builder's configured ``cpu_slots_count``, and if that
                is also ``None``, to RAM streaming (all blocks pre-loaded with
                LoRA fusion).
            gpu_slots_count: Number of GPU buffer slots.
                ``None`` = ``_DEFAULT_GPU_SLOTS`` (2).
        """
        if not self.blocks_prefix:
            raise ValueError("blocks_prefix must be non-empty for streaming")
        if dtype is None:
            raise ValueError("StreamingModelBuilder.build requires an explicit dtype")
        device = device if device is not None else torch.device("cuda")

        metadata = self.model_metadata()
        meta_model: nn.Module = create_meta_model(self.model_class_configurator, metadata, self.module_ops)
        meta_model.eval()

        blocks = resolve_attr(meta_model, self.blocks_attr)

        checkpoint_paths = list(self.model_path) if isinstance(self.model_path, tuple) else [self.model_path]
        block_key_map, non_block_keys = _scan_checkpoint_keys(checkpoint_paths, self.model_sd_ops, self.blocks_prefix)
        expected_indices = set(range(len(blocks)))
        if set(block_key_map) != expected_indices:
            missing = sorted(expected_indices - set(block_key_map))
            extra = sorted(set(block_key_map) - expected_indices)
            raise ValueError(
                f"Block weights under prefix '{self.blocks_prefix}.' do not match the {len(blocks)} model blocks: "
                f"missing indices {missing}, unexpected indices {extra}"
            )

        cpu_slots_count = cpu_slots_count if cpu_slots_count is not None else self._cpu_slots_count
        cpu_slots_count = cpu_slots_count if cpu_slots_count is not None else len(blocks)
        gpu_slots_count = gpu_slots_count if gpu_slots_count is not None else _DEFAULT_GPU_SLOTS

        if cpu_slots_count >= len(blocks):
            lora_sd_and_strengths = self._load_lora_sds()
            source, lora_sources = self._build_pinned_source(
                blocks, dtype, cpu_slots_count, block_key_map, lora_sd_and_strengths
            )
            non_block_loras = lora_sd_and_strengths
        else:
            reader = DiskTensorReader(checkpoint_paths)
            source, lora_sources = self._build_disk_source(
                blocks, dtype, cpu_slots_count, reader, block_key_map, prefetch_depth=_PREFETCH_DEPTH
            )
            non_block_loras = [src.as_state_dict_with_strength() for src in lora_sources]

        self._load_non_block_weights(meta_model, non_block_keys, device, dtype, non_block_loras)

        sync = create_stream_sync(device)
        gpu_pool = BufferPool(source.slot_nbytes, gpu_slots_count, device, reuse_barrier=sync.reuse_barrier)
        provider = WeightsProvider(
            gpu_pool,
            sync,
            device,
            source,
            lora_sources,
            self.blocks_prefix,
            fuse_rule=self.fuse_rule,
        )
        return BlockStreamingWrapper(
            model=meta_model,
            blocks=blocks,
            provider=provider,
            target_device=device,
        )

    def _load_lora_sds(self) -> list[LoraStateDictWithStrength]:
        """Load each configured LoRA into a state dict for fusion (pinned path)."""
        return [
            LoraStateDictWithStrength(
                load_state_dict([lora.path], self.model_loader, self.registry, torch.device("cpu"), lora.sd_ops),
                lora.strength,
            )
            for lora in self.loras
        ]

    def _filtered_sd_ops(self, name_suffix: str, allowed_model_keys: frozenset[str]) -> SDOps:
        """``model_sd_ops`` restricted to *allowed_model_keys* (post-rename keys).
        The loader skips keys filtered to None before reading them, so a restricted
        load never materializes the excluded partition. The distinct ``name`` avoids
        a registry cache-id collision with the other partition.
        """
        base = self.model_sd_ops if self.model_sd_ops is not None else SDOps("streaming").with_matching()
        allowed = allowed_model_keys if base.allowed_keys is None else (allowed_model_keys & base.allowed_keys)
        return replace(base, name=f"{base.name}__{name_suffix}", allowed_keys=allowed)

    def _build_pinned_source(
        self,
        blocks: nn.ModuleList,
        dtype: torch.dtype,
        cpu_slots_count: int,
        block_key_map: dict[int, list[tuple[str, str]]],
        lora_sd_and_strengths: list[LoraStateDictWithStrength],
    ) -> tuple[WeightSource, list[LoraSource]]:
        """Pre-load each block into its own contiguous pinned CPU buffer with LoRA fusion."""
        for block_idx in block_key_map:
            if block_idx >= cpu_slots_count:
                raise ValueError(
                    f"Pinned source requires one CPU slot per block; "
                    f"got block index {block_idx} with only {cpu_slots_count} slots."
                )

        # One contiguous pinned buffer per block, carved into per-param views. The
        # views (flattened by full key) are filled in place; the source then keeps
        # only the contiguous buffer and the layout to re-carve it on read.
        pinned_buffers: dict[int, torch.Tensor] = {}
        block_layouts: dict[int, TensorLayout] = {}
        fill_views: dict[str, torch.Tensor] = {}
        for block_idx, entries in block_key_map.items():
            block_state = _block_state(blocks[block_idx])
            layout = derive_layout({param_name: block_state[param_name] for _sft_key, param_name in entries}, dtype)
            buffer = bs_utils.alloc_buffer(layout_nbytes(layout), torch.device("cpu"), pin_memory=True)
            views = carve_buffer(buffer, layout)
            pinned_buffers[block_idx] = buffer
            block_layouts[block_idx] = layout
            for param_name, view in views.items():
                fill_views[make_block_key(self.blocks_prefix, block_idx, param_name)] = view

        block_sd = load_state_dict(
            self.model_path,
            self.model_loader,
            self.registry,
            torch.device("cpu"),
            self._filtered_sd_ops("blocks", frozenset(fill_views)),
        )

        should_sync = False
        for key, fused in fuse_lora_weights(
            block_sd, lora_sd_and_strengths, fuse_rule=self.fuse_rule, preserve_input_device=False
        ):
            if key not in fill_views:
                raise ValueError(f"Block-restricted load produced {key!r}, which is not a pinned block weight")
            fill_views[key].copy_(fused, non_blocking=True)
            block_sd.sd[key] = None
            should_sync = True
        if should_sync:
            synchronize_device()

        # Fill remaining pinned keys from the source state dict.
        for key, view in fill_views.items():
            if block_sd.sd[key] is None:
                continue
            view.copy_(block_sd.sd[key])
            block_sd.sd[key] = None

        pinned = {idx: PinnedBlock(pinned_buffers[idx], block_layouts[idx]) for idx in pinned_buffers}
        return PinnedWeightSource(pinned), []

    def _build_disk_source(
        self,
        blocks: nn.ModuleList,
        dtype: torch.dtype,
        cpu_slots_count: int,
        reader: DiskTensorReader,
        block_key_map: dict[int, list[tuple[str, str]]],
        prefetch_depth: int,
    ) -> tuple[WeightSource, list[LoraSource]]:
        """Create a DiskWeightSource backed by a DiskBlockReader.
        Pool slots are sized to the largest block and carved per block on read, so
        heterogeneous blocks (e.g. layers with differing attention layouts) share
        one pool. Pool capacity is ``cpu_slots_count + prefetch_depth`` so the
        lookahead loop in ``DiskWeightSource.get`` never evicts its own target.
        Layouts come from the meta model; assumes module_ops keep the meta param
        dtype in sync with the post-sd_ops checkpoint dtype.
        """
        block_layouts = _block_layouts(blocks, block_key_map, dtype)
        slot_nbytes = max(layout_nbytes(layout) for layout in block_layouts.values())

        cpu_pool = BufferPool(
            slot_nbytes,
            cpu_slots_count + prefetch_depth,
            torch.device("cpu"),
            reuse_barrier=lambda event: event.synchronize(),
            pin_memory=True,
        )
        block_reader = DiskBlockReader(
            reader=reader,
            block_key_map=block_key_map,
            sd_ops=self.model_sd_ops,
            blocks_prefix=self.blocks_prefix,
        )
        fetcher = BlockFetcher(block_reader)
        source = DiskWeightSource(
            cpu_pool,
            fetcher,
            block_layouts,
            blocks_number=len(blocks),
            prefetch_depth=prefetch_depth,
        )
        lora_sources = [LoraSource(lora.path, lora.sd_ops, lora.strength) for lora in self.loras]

        return source, lora_sources

    @torch.inference_mode()
    def _load_non_block_weights(
        self,
        model: nn.Module,
        non_block_keys: list[tuple[str, str]],
        device: torch.device,
        dtype: torch.dtype,
        lora_sd_and_strengths: list[LoraStateDictWithStrength],
    ) -> None:
        """Load the non-block weights onto *device* and fuse LoRAs -- both paths.
        Reads through the loader with ``model_sd_ops`` restricted to the
        non-block keys, so ``sd_ops`` (incl. kv-ops such as Gemma's ``lm_head``
        duplication) is applied exactly once and block tensors are never read.
        """
        non_block_sd_ops = self._filtered_sd_ops("non_block", frozenset(mk for _sft_key, mk in non_block_keys))
        loaded = load_state_dict(self.model_path, self.model_loader, self.registry, device, non_block_sd_ops)
        non_block_sd = {key: tensor.to(dtype=dtype) for key, tensor in loaded.sd.items()}

        if lora_sd_and_strengths:
            non_block_state = StateDict(sd=non_block_sd, device=device, size=0, dtype={dtype})
            for key, fused in fuse_lora_weights(
                non_block_state,
                lora_sd_and_strengths,
                fuse_rule=self.fuse_rule,
                preserve_input_device=True,
            ):
                non_block_sd[key] = fused

        model.load_state_dict(non_block_sd, strict=False, assign=True)


def _block_state(block: nn.Module) -> dict[str, torch.Tensor]:
    """Streamed-eligible tensors of a block: parameters then buffers.
    Block streaming swaps both params and checkpoint-backed buffers (e.g. Gemma4's
    per-layer ``layer_scalar``), so the layout, pinned packing, and meta-ordering
    all consult parameters and buffers together. Non-checkpoint (computed) buffers
    are harmless here -- only keys present in ``block_key_map`` are ever streamed.
    """
    return {**dict(block.named_parameters()), **dict(block.named_buffers())}


def _block_layouts(
    blocks: nn.ModuleList,
    block_key_map: dict[int, list[tuple[str, str]]],
    dtype: torch.dtype,
) -> dict[int, TensorLayout]:
    """Per-block layout of the streamed tensors, taken from the meta model.
    Blocks may differ in shape and even in which tensors they have (e.g. Gemma4's
    full-attention layers drop ``v_proj``), so each block gets its own layout in
    ``block_key_map`` order. The pinned packing, the disk reader, and the GPU carve
    all key off this same per-block layout, so the provider's contiguous H2D copy
    is valid for any entry order (no cross-block ordering required).
    """
    layouts: dict[int, TensorLayout] = {}
    for idx, entries in block_key_map.items():
        state = _block_state(blocks[idx])
        layouts[idx] = derive_layout({param_name: state[param_name] for _sft_key, param_name in entries}, dtype)
    return layouts


def _scan_checkpoint_keys(
    checkpoint_paths: list[str],
    sd_ops: SDOps | None,
    blocks_prefix: str,
) -> tuple[dict[int, list[tuple[str, str]]], list[tuple[str, str]]]:
    """Partition checkpoint keys into per-block and non-block lists.
    Opens the safetensors files for header-only key enumeration; no tensor data
    is read.
    """
    block_key_map: dict[int, list[tuple[str, str]]] = {}
    non_block_keys: list[tuple[str, str]] = []
    prefix_dot = blocks_prefix + "."
    for path in checkpoint_paths:
        with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
            for sft_key in handle.keys():  # noqa: SIM118
                model_key = sd_ops.apply_to_key(sft_key) if sd_ops else sft_key
                if model_key is None:
                    continue
                if model_key.startswith(prefix_dot):
                    rest = model_key[len(prefix_dot) :]
                    idx_str, _, param_name = rest.partition(".")
                    try:
                        block_idx = int(idx_str)
                    except ValueError:
                        non_block_keys.append((sft_key, model_key))
                        continue
                    block_key_map.setdefault(block_idx, []).append((sft_key, param_name))
                else:
                    non_block_keys.append((sft_key, model_key))
    return block_key_map, non_block_keys
