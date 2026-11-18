from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypeVar

import torch

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import SDOps

if TYPE_CHECKING:
    from typing_extensions import Self

    from ltx_core.loader.fuse_loras import FuseRule
    from ltx_core.loader.registry import Registry

BuiltType = TypeVar("BuiltType", covariant=True)  # noqa: PLC0105


# Per-key shape and dtype description for a flat collection of tensors.
TensorLayout = dict[str, tuple[torch.Size, torch.dtype]]


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
        ...

    def load(self, path: str | list[str], sd_ops: SDOps | None = None, device: torch.device | None = None) -> StateDict:
        """
        Load state dict from path or paths (for sharded model storage) and apply sd_ops
        """
        ...


class BuilderProtocol(Protocol[BuiltType]):
    """Protocol for model builders that produce a model via ``build()``."""

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> BuiltType: ...

    @property
    def registry(self) -> "Registry": ...

    def with_registry(self, registry: "Registry") -> "Self":
        """Return a copy of this builder using the given weight registry for allocation."""
        ...


class ModelBuilderProtocol(BuilderProtocol[BuiltType], Protocol[BuiltType]):
    """
    Protocol for building PyTorch models from configuration dictionaries.
    Implementations must provide:
    - build: Create and initialize a model from state dictionary and apply dtype transformations
    """

    @property
    def checkpoint(self) -> str | tuple[str, ...]:
        """Path(s) to the checkpoint this builder loads from (for logging/diagnostics)."""
        ...

    @property
    def keeps_gpu_resident_weights(self) -> bool:
        """Whether rebuilds rebind the same GPU weight storages (safe for CUDA-graph capture)."""
        ...

    @property
    def model_sd_ops(self) -> SDOps | None: ...

    @property
    def module_ops(self) -> tuple[ModuleOps, ...]: ...

    @property
    def loras(self) -> tuple["LoraPathStrengthAndSDOps", ...]: ...

    def with_sd_ops(self, sd_ops: SDOps | None) -> "Self":
        """Return a copy of this builder with the given state-dict key remapping ops."""
        ...

    def with_module_ops(self, module_ops: tuple[ModuleOps, ...]) -> "Self":
        """Return a copy of this builder with the given module operations (e.g. quantization)."""
        ...

    def with_loras(self, loras: tuple["LoraPathStrengthAndSDOps", ...]) -> "Self":
        """Return a copy of this builder with the given LoRAs to fuse at build time."""
        ...

    def with_lora_load_device(self, device: torch.device) -> "Self":
        """Return a copy of this builder that loads LoRA weights onto the given device."""
        ...

    def with_fuse_rule(self, fuse_rule: "FuseRule") -> "Self":
        """Return a copy of this builder with the given LoRA fuse rule (e.g. from a quantization policy)."""
        ...

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> BuiltType:
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

    def lora(self, lora_path: str, strength: float, sd_ops: SDOps) -> "LoRAAdaptableProtocol": ...


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
