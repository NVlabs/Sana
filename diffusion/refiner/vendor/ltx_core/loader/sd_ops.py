from dataclasses import dataclass, replace
from typing import NamedTuple, Protocol

import torch


@dataclass(frozen=True, slots=True)
class ContentReplacement:
    """
    Represents a content replacement operation.
    Used to replace a specific content with a replacement in a state dict key.
    """

    content: str
    replacement: str


@dataclass(frozen=True, slots=True)
class ContentMatching:
    """
    Represents a content matching operation.
    Used to match a specific prefix and suffix in a state dict key.
    """

    prefix: str = ""
    suffix: str = ""


class KeyValueOperationResult(NamedTuple):
    """
    Represents the result of a key-value operation.
    Contains the new key and value after the operation has been applied.
    """

    new_key: str
    new_value: torch.Tensor


class KeyValueOperation(Protocol):
    """
    Protocol for key-value operations.
    Used to apply operations to a specific key and value in a state dict.
    """

    def __call__(self, tensor_key: str, tensor_value: torch.Tensor) -> list[KeyValueOperationResult]: ...


@dataclass(frozen=True, slots=True)
class SDKeyValueOperation:
    """
    Represents a key-value operation.
    Used to apply operations to a specific key and value in a state dict.
    """

    key_matcher: ContentMatching
    kv_operation: KeyValueOperation


@dataclass(frozen=True, slots=True)
class SDOps:
    """Immutable class representing state dict key operations."""

    name: str
    mapping: tuple[
        ContentReplacement | ContentMatching | SDKeyValueOperation, ...
    ] = ()  # Immutable tuple of (key, value) pairs
    allowed_keys: frozenset[str] | None = None

    def with_replacement(self, content: str, replacement: str) -> "SDOps":
        """Create a new SDOps instance with the specified replacement added to the mapping."""

        new_mapping = (*self.mapping, ContentReplacement(content, replacement))
        return replace(self, mapping=new_mapping)

    def with_matching(self, prefix: str = "", suffix: str = "") -> "SDOps":
        """Create a new SDOps instance with the specified prefix and suffix matching added to the mapping."""

        new_mapping = (*self.mapping, ContentMatching(prefix, suffix))
        return replace(self, mapping=new_mapping)

    def with_additional_allowed_keys(self, keys: frozenset[str]) -> "SDOps":
        """Create a new SDOps instance that only passes keys present in *keys* (post-replacement).
        If allowed_keys already exists, the sets are merged via union.
        """
        merged = frozenset(keys) | self.allowed_keys if self.allowed_keys is not None else frozenset(keys)
        return replace(self, allowed_keys=merged)

    def with_kv_operation(
        self,
        operation: KeyValueOperation,
        key_prefix: str = "",
        key_suffix: str = "",
    ) -> "SDOps":
        """Create a new SDOps instance with the specified value operation added to the mapping."""
        key_matcher = ContentMatching(key_prefix, key_suffix)
        sd_kv_operation = SDKeyValueOperation(key_matcher, operation)
        new_mapping = (*self.mapping, sd_kv_operation)
        return replace(self, mapping=new_mapping)

    def apply_to_key(self, key: str) -> str | None:
        """Apply the mapping to the given name."""
        matchers = [content for content in self.mapping if isinstance(content, ContentMatching)]
        valid = any(key.startswith(f.prefix) and key.endswith(f.suffix) for f in matchers)
        if not valid:
            return None

        for replacement in self.mapping:
            if not isinstance(replacement, ContentReplacement):
                continue
            if replacement.content in key:
                key = key.replace(replacement.content, replacement.replacement)

        if self.allowed_keys is not None and key not in self.allowed_keys:
            return None

        return key

    def apply_to_key_value(self, key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        """Apply the value operation to the given name and associated value."""
        for operation in self.mapping:
            if not isinstance(operation, SDKeyValueOperation):
                continue
            if key.startswith(operation.key_matcher.prefix) and key.endswith(operation.key_matcher.suffix):
                return operation.kv_operation(key, value)
        return [KeyValueOperationResult(key, value)]


# Predefined SDOps instances
LTXV_LORA_COMFY_RENAMING_MAP = (
    SDOps("LTXV_LORA_COMFY_PREFIX_MAP").with_matching().with_replacement("diffusion_model.", "")
)

LTXV_LORA_COMFY_TARGET_MAP = (
    SDOps("LTXV_LORA_COMFY_TARGET_MAP")
    .with_matching()
    .with_replacement("diffusion_model.", "")
    .with_replacement(".lora_A.weight", ".weight")
    .with_replacement(".lora_B.weight", ".weight")
)
