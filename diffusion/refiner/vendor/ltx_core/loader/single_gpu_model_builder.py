import logging
from dataclasses import dataclass, field, replace
from typing import Generic

import torch

from diffusion.refiner.vendor.ltx_core.loader.fuse_loras import apply_loras
from diffusion.refiner.vendor.ltx_core.loader.module_ops import ModuleOps
from diffusion.refiner.vendor.ltx_core.loader.primitives import (
    LoRAAdaptableProtocol,
    LoraPathStrengthAndSDOps,
    LoraStateDictWithStrength,
    ModelBuilderProtocol,
    StateDict,
    StateDictLoader,
)
from diffusion.refiner.vendor.ltx_core.loader.registry import DummyRegistry, Registry
from diffusion.refiner.vendor.ltx_core.loader.sd_ops import SDOps
from diffusion.refiner.vendor.ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from diffusion.refiner.vendor.ltx_core.model.model_protocol import ModelConfigurator, ModelType

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SingleGPUModelBuilder(Generic[ModelType], ModelBuilderProtocol[ModelType], LoRAAdaptableProtocol):
    """
    Builder for PyTorch models residing on a single GPU.
    Attributes:
        model_class_configurator: Class responsible for constructing the model from a config dict.
        model_path: Path (or tuple of shard paths) to the model's `.safetensors` checkpoint(s).
        model_sd_ops: Optional state-dict operations applied when loading the model weights.
        module_ops: Sequence of module-level mutations applied to the meta model before weight loading.
        loras: Sequence of LoRA adapters (path, strength, optional sd_ops) to fuse into the model.
        model_loader: Strategy for loading state dicts from disk. Defaults to
            :class:`SafetensorsModelStateDictLoader`.
        registry: Cache for already-loaded state dicts. Defaults to :class:`DummyRegistry` (no caching).
        lora_load_device: Device used when loading LoRA weight tensors from disk. Defaults to
            ``torch.device("cpu")``, which keeps LoRA weights in CPU memory and transfers them to
            the target GPU sequentially during fusion, reducing peak GPU memory usage compared to
            loading all LoRA weights directly onto the GPU at once.
    """

    model_class_configurator: type[ModelConfigurator[ModelType]]
    model_path: str | tuple[str, ...]
    model_sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = field(default_factory=tuple)
    loras: tuple[LoraPathStrengthAndSDOps, ...] = field(default_factory=tuple)
    model_loader: StateDictLoader = field(default_factory=SafetensorsModelStateDictLoader)
    registry: Registry = field(default_factory=DummyRegistry)
    lora_load_device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def lora(self, lora_path: str, strength: float = 1.0, sd_ops: SDOps | None = None) -> "SingleGPUModelBuilder":
        return replace(self, loras=(*self.loras, LoraPathStrengthAndSDOps(lora_path, strength, sd_ops)))

    def with_sd_ops(self, sd_ops: SDOps | None) -> "SingleGPUModelBuilder":
        return replace(self, model_sd_ops=sd_ops)

    def with_module_ops(self, module_ops: tuple[ModuleOps, ...]) -> "SingleGPUModelBuilder":
        return replace(self, module_ops=module_ops)

    def with_loras(self, loras: tuple[LoraPathStrengthAndSDOps, ...]) -> "SingleGPUModelBuilder":
        return replace(self, loras=loras)

    def with_registry(self, registry: Registry) -> "SingleGPUModelBuilder":
        return replace(self, registry=registry)

    def with_lora_load_device(self, device: torch.device) -> "SingleGPUModelBuilder":
        return replace(self, lora_load_device=device)

    def model_config(self) -> dict:
        first_shard_path = self.model_path[0] if isinstance(self.model_path, tuple) else self.model_path
        return self.model_loader.metadata(first_shard_path)

    def meta_model(self, config: dict, module_ops: tuple[ModuleOps, ...]) -> ModelType:
        with torch.device("meta"):
            model = self.model_class_configurator.from_config(config)
        for module_op in module_ops:
            if module_op.matcher(model):
                model = module_op.mutator(model)
        return model

    def load_sd(
        self, paths: list[str], registry: Registry, device: torch.device | None, sd_ops: SDOps | None = None
    ) -> StateDict:
        state_dict = registry.get(paths, sd_ops)
        if state_dict is None:
            state_dict = self.model_loader.load(paths, sd_ops=sd_ops, device=device)
            registry.add(paths, sd_ops=sd_ops, state_dict=state_dict)
        return state_dict

    def _return_model(self, meta_model: ModelType, device: torch.device) -> ModelType:
        uninitialized_params = [name for name, param in meta_model.named_parameters() if str(param.device) == "meta"]
        uninitialized_buffers = [name for name, buffer in meta_model.named_buffers() if str(buffer.device) == "meta"]
        if uninitialized_params or uninitialized_buffers:
            logger.warning(f"Uninitialized parameters or buffers: {uninitialized_params + uninitialized_buffers}")
            return meta_model
        retval = meta_model.to(device)
        return retval

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: object,  # noqa: ARG002
    ) -> ModelType:
        device = torch.device("cuda") if device is None else device
        config = self.model_config()
        meta_model = self.meta_model(config, self.module_ops)
        model_paths = list(self.model_path) if isinstance(self.model_path, tuple) else [self.model_path]
        model_state_dict = self.load_sd(model_paths, sd_ops=self.model_sd_ops, registry=self.registry, device=device)

        lora_strengths = [lora.strength for lora in self.loras]
        if not lora_strengths or (min(lora_strengths) == 0 and max(lora_strengths) == 0):
            sd = model_state_dict.sd
            if dtype is not None:
                sd = {key: value.to(dtype=dtype) for key, value in model_state_dict.sd.items()}
            meta_model.load_state_dict(sd, strict=False, assign=True)
            return self._return_model(meta_model, device)

        lora_state_dicts = [
            self.load_sd([lora.path], sd_ops=lora.sd_ops, registry=self.registry, device=self.lora_load_device)
            for lora in self.loras
        ]
        lora_sd_and_strengths = [
            LoraStateDictWithStrength(sd, strength)
            for sd, strength in zip(lora_state_dicts, lora_strengths, strict=True)
        ]
        final_sd = apply_loras(
            model_sd=model_state_dict,
            lora_sd_and_strengths=lora_sd_and_strengths,
            dtype=dtype,
            destination_sd=model_state_dict if isinstance(self.registry, DummyRegistry) else None,
        )
        meta_model.load_state_dict(final_sd.sd, strict=False, assign=True)
        return self._return_model(meta_model, device)
