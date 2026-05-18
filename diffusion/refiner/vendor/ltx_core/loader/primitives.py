from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

import torch

from diffusion.refiner.vendor.ltx_core.loader.module_ops import ModuleOps
from diffusion.refiner.vendor.ltx_core.loader.sd_ops import SDOps
from diffusion.refiner.vendor.ltx_core.model.model_protocol import ModelType

if TYPE_CHECKING:
    from diffusion.refiner.vendor.ltx_core.loader.registry import Registry


@dataclass(frozen=True)
class StateDict:
    """
    Immutable container for a PyTorch state dictionary.
    Contains:
    - sd: Dictionary of tensors (weights, buffers, etc.)
    - device: Device where tensors are stored
    - size: Total memory footprint in bytes
    - dtype: Set of tensor dtypes present
    """

    sd: dict
    device: torch.device
    size: int
    dtype: set[torch.dtype]

    def footprint(self) -> tuple[int, torch.device]:
        return self.size, self.device


class StateDictLoader(Protocol):
    """
    Protocol for loading state dictionaries from various sources.
    Implementations must provide:
    - metadata: Extract model metadata from a single path
    - load: Load state dict from path(s) and apply SDOps transformations
    """

    def metadata(self, path: str) -> dict:
        """
        Load metadata from path
        """

    def load(self, path: str | list[str], sd_ops: SDOps | None = None, device: torch.device | None = None) -> StateDict:
        """
        Load state dict from path or paths (for sharded model storage) and apply sd_ops
        """


class ModelBuilderProtocol(Protocol[ModelType]):
    """
    Protocol for building PyTorch models from configuration dictionaries.
    Implementations must provide:
    - meta_model: Create a model from configuration dictionary and apply module operations
    - build: Create and initialize a model from state dictionary and apply dtype transformations
    """

    model_sd_ops: SDOps | None
    module_ops: tuple[ModuleOps, ...]
    loras: tuple[LoraPathStrengthAndSDOps, ...]
    registry: Registry

    def meta_model(self, config: dict, module_ops: list[ModuleOps] | None = None) -> ModelType:
        """
        Create a model on the meta device from a configuration dictionary.
        This decouples model creation from weight loading, allowing the model
        architecture to be instantiated without allocating memory for parameters.
        Args:
            config: Model configuration dictionary.
            module_ops: Optional list of module operations to apply (e.g., quantization).
        Returns:
            Model instance on meta device (no actual memory allocated for parameters).
        """
        ...

    def with_sd_ops(self, sd_ops: SDOps | None) -> ModelBuilderProtocol[ModelType]:
        """Return a copy of this builder with the given state-dict key remapping ops."""
        ...

    def with_module_ops(self, module_ops: tuple[ModuleOps, ...]) -> ModelBuilderProtocol[ModelType]:
        """Return a copy of this builder with the given module operations (e.g. quantization)."""
        ...

    def with_loras(self, loras: tuple[LoraPathStrengthAndSDOps, ...]) -> ModelBuilderProtocol[ModelType]:
        """Return a copy of this builder with the given LoRAs to fuse at build time."""
        ...

    def with_registry(self, registry: Registry) -> ModelBuilderProtocol[ModelType]:
        """Return a copy of this builder using the given weight registry for allocation."""
        ...

    def with_lora_load_device(self, device: torch.device) -> ModelBuilderProtocol[ModelType]:
        """Return a copy of this builder that loads LoRA weights onto the given device."""
        ...

    def build(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None, **kwargs: object
    ) -> ModelType:
        """
        Build the model
        Args:
            device: Target device for the model
            dtype: Target dtype for the model, if None, uses the dtype of the model_path model
        Returns:
            Model instance
        """
        ...

    def model_config(self) -> dict:
        """Return the model configuration dictionary extracted from the checkpoint metadata."""
        ...


class LoRAAdaptableProtocol(Protocol):
    """
    Protocol for models that can be adapted with LoRAs.
    Implementations must provide:
    - lora: Add a LoRA to the model
    """

    def lora(self, lora_path: str, strength: float) -> LoRAAdaptableProtocol:
        pass


class LoraPathStrengthAndSDOps(NamedTuple):
    """
    Tuple containing a LoRA path, strength, and SDOps for applying to the LoRA state dict.
    """

    path: str
    strength: float
    sd_ops: SDOps


class LoraStateDictWithStrength(NamedTuple):
    """
    Tuple containing a LoRA state dict and strength for applying to the model.
    """

    state_dict: StateDict
    strength: float
