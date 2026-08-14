from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Final, Generic

import torch
from torch import nn

from ltx_core.loader.fuse_loras import FuseRule, apply_loras, bf16_fuse_rule
from ltx_core.loader.helpers import (
    as_path_list,
    create_meta_model,
    load_state_dict,
    read_model_config,
    read_model_metadata,
)
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import (
    LoRAAdaptableProtocol,
    LoraPathStrengthAndSDOps,
    LoraStateDictWithStrength,
    ModelBuilderProtocol,
    StateDict,
    StateDictLoader,
)
from ltx_core.loader.registry import ModelRegistry, Registry, module_registry_key
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.model_protocol import ModelConfigurator, ModelType

if TYPE_CHECKING:
    from typing_extensions import Self

logger: logging.Logger = logging.getLogger(__name__)


def _check_uninitialized(model: nn.Module) -> list[str]:
    """Return names of any parameters/buffers still on meta device."""
    names = []
    for name, param in model.named_parameters():
        if str(param.device) == "meta":
            names.append(name)
    for name, buf in model.named_buffers():
        if str(buf.device) == "meta":
            names.append(name)
    return names


# dtypes that ``build(..., dtype=)`` may cast. Quantized payloads (uint8 NVFP4,
# float8, etc.) must not be rewritten — SDOps policies emit them at load time.
_DTYPE_CASTABLE = frozenset(
    {
        torch.float32,
        torch.float64,
        torch.float16,
        torch.bfloat16,
    }
)


def _cast_floating_sd(sd: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    def _cast(value: torch.Tensor) -> torch.Tensor:
        if value.dtype not in _DTYPE_CASTABLE:
            return value
        # Keep FP32 scalars (e.g. NVFP4 ``weight_scale_2`` / alpha) — bf16 would
        # corrupt the per-tensor decode-scale convention.
        if value.ndim == 0 and value.dtype == torch.float32:
            return value
        return value.to(dtype=dtype)

    return {key: _cast(value) for key, value in sd.items()}


def _load_model_weights(
    meta_model: nn.Module,
    model_path: str | tuple[str, ...],
    loras: tuple[LoraPathStrengthAndSDOps, ...],
    loader: StateDictLoader,
    registry: Registry,
    device: torch.device,
    dtype: torch.dtype | None,
    model_sd_ops: SDOps | None = None,
    lora_load_device: torch.device | None = None,
    fuse_rule: FuseRule = bf16_fuse_rule,
) -> None:
    """Load base weights and fuse LoRAs into *meta_model* in-place."""
    if lora_load_device is None:
        lora_load_device = device

    model_sd = load_state_dict(model_path, loader, registry, device, model_sd_ops)

    lora_strengths = [lora.strength for lora in loras]
    if not lora_strengths or (min(lora_strengths) == 0 and max(lora_strengths) == 0):
        sd = model_sd.sd
        if dtype is not None:
            sd = _cast_floating_sd(sd, dtype)
        meta_model.load_state_dict(sd, strict=False, assign=True)
        return

    lora_state_dicts = [load_state_dict([lora.path], loader, registry, lora_load_device, lora.sd_ops) for lora in loras]
    lora_sd_and_strengths = [
        LoraStateDictWithStrength(sd, strength) for sd, strength in zip(lora_state_dicts, lora_strengths, strict=True)
    ]
    # Retained registry entries are clean cache — fuse into a fresh SD. Ephemeral loads (get miss)
    # may be fused in place. Always leave fused tensors on the fusion device (no D2H bounce when
    # ``model_sd`` is CPU-retained).
    destination_sd = None if registry.get(as_path_list(model_path), model_sd_ops) is not None else model_sd
    final_sd = apply_loras(
        model_sd=model_sd,
        lora_sd_and_strengths=lora_sd_and_strengths,
        fuse_rule=fuse_rule,
        destination_sd=destination_sd,
        preserve_input_device=False,
    )
    fused_sd = final_sd.sd
    if dtype is not None:
        fused_sd = _cast_floating_sd(fused_sd, dtype)
    meta_model.load_state_dict(fused_sd, strict=False, assign=True)


class SingleGPUModelBuilder(Generic[ModelType], ModelBuilderProtocol[ModelType], LoRAAdaptableProtocol):
    """
    Builder for PyTorch models residing on a single GPU.
    The builder is immutable: ``with_*``/``lora`` return modified copies. The
    ``ModelBuilderProtocol`` surface is exposed via read-only properties backed
    by private attributes.
    Attributes:
        model_class_configurator: Class responsible for constructing the model from a config dict.
        model_path: Path (or tuple of shard paths) to the model's `.safetensors` checkpoint(s).
        model_sd_ops: Optional state-dict operations applied when loading the model weights.
        module_ops: Sequence of module-level mutations applied to the meta model before weight loading.
        loras: Sequence of LoRA adapters (path, strength, optional sd_ops) to fuse into the model.
        model_loader: Strategy for loading state dicts from disk. Defaults to
            :class:`SafetensorsModelStateDictLoader`.
        registry: Cache for model shells / state dicts. Defaults to a shell-only
            :class:`ModelRegistry` (``cache_models=True``, ``cache_weights=False``).
        lora_load_device: Device used when loading LoRA weight tensors from disk. Defaults to
            ``torch.device("cpu")``, which keeps LoRA weights in CPU memory and transfers them to
            the target GPU sequentially during fusion, reducing peak GPU memory usage compared to
            loading all LoRA weights directly onto the GPU at once.
        fuse_rule: Per-policy LoRA merge rule. Defaults to ``bf16_fuse_rule``;
    """

    def __init__(
        self,
        model_class_configurator: type[ModelConfigurator[ModelType]],
        model_path: str | tuple[str, ...],
        model_sd_ops: SDOps | None = None,
        module_ops: tuple[ModuleOps, ...] = (),
        loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
        model_loader: StateDictLoader | None = None,
        registry: Registry | None = None,
        lora_load_device: torch.device | None = None,
        fuse_rule: FuseRule = bf16_fuse_rule,
    ) -> None:
        # Read-only: typed with the covariant ModelType, so it must not be a mutable attribute.
        self._model_class_configurator: Final = model_class_configurator
        self._model_path = model_path
        self._model_sd_ops = model_sd_ops
        self._module_ops = module_ops
        self._loras = loras
        self._model_loader = model_loader if model_loader is not None else SafetensorsModelStateDictLoader()
        self._registry = registry if registry is not None else ModelRegistry(cache_models=True, cache_weights=False)
        self._lora_load_device = lora_load_device if lora_load_device is not None else torch.device("cpu")
        self._fuse_rule = fuse_rule

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
    def registry(self) -> Registry:
        return self._registry

    @property
    def keeps_gpu_resident_weights(self) -> bool:
        # Registry may retain a CPU SD; each build H2Ds into fresh GPU storages.
        return False

    @property
    def model_path(self) -> str | tuple[str, ...]:
        return self._model_path

    @property
    def checkpoint(self) -> str | tuple[str, ...]:
        return self._model_path

    @property
    def model_loader(self) -> StateDictLoader:
        return self._model_loader

    @property
    def lora_load_device(self) -> torch.device:
        return self._lora_load_device

    @property
    def fuse_rule(self) -> FuseRule:
        return self._fuse_rule

    def lora(self, lora_path: str, strength: float, sd_ops: SDOps) -> Self:
        clone = copy.copy(self)
        clone._loras = (*self._loras, LoraPathStrengthAndSDOps(lora_path, strength, sd_ops))
        return clone

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
        clone = copy.copy(self)
        clone._lora_load_device = device
        return clone

    def with_fuse_rule(self, fuse_rule: FuseRule) -> Self:
        clone = copy.copy(self)
        clone._fuse_rule = fuse_rule
        return clone

    def model_config(self) -> dict:
        return read_model_config(self._model_path, self._model_loader)

    def model_metadata(self) -> dict:
        return read_model_metadata(self._model_path, self._model_loader)

    def meta_model(self, metadata: dict, module_ops: tuple[ModuleOps, ...]) -> ModelType:
        struct_key = module_registry_key(
            configurator=self._model_class_configurator,
            config=metadata,
            module_ops_names=tuple(op.name for op in module_ops),
        )
        cached = self._registry.get_model(struct_key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        model = create_meta_model(self._model_class_configurator, metadata, module_ops)
        return self._registry.add_model(struct_key, model)  # type: ignore[return-value]

    def load_sd(
        self, paths: list[str], registry: Registry, device: torch.device | None, sd_ops: SDOps | None = None
    ) -> StateDict:
        return load_state_dict(paths, self._model_loader, registry, device, sd_ops)

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: object,  # noqa: ARG002
    ) -> ModelType:
        device = torch.device("cuda") if device is None else device
        metadata = self.model_metadata()
        meta_model = self.meta_model(metadata, self._module_ops)

        _load_model_weights(
            meta_model=meta_model,
            model_path=self._model_path,
            loras=self._loras,
            loader=self._model_loader,
            registry=self._registry,
            device=device,
            dtype=dtype,
            model_sd_ops=self._model_sd_ops,
            lora_load_device=self._lora_load_device,
            fuse_rule=self._fuse_rule,
        )

        uninitialized = _check_uninitialized(meta_model)
        if uninitialized:
            logger.warning(f"Uninitialized parameters or buffers: {uninitialized}")
            return meta_model
        return meta_model.to(device)
