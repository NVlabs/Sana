"""Multi-GPU Gemma text encoder builder.
Replaces the text encoder builder on the ``PromptEncoder`` block with an
:class:`AccelerateGemmaBuilder` that uses ``device_map="auto"`` on the
source rank and a broadcast stub elsewhere.
On the source rank the first ``build()`` loads via HuggingFace
``from_pretrained`` and caches the full state dict (including non-persistent
buffers) in the provided :class:`Registry`. Model shells are reused from the
registry when the structural key matches: weights are reassigned from the SD,
while non-persistent buffers are left on the shell (``dispose`` does not meta
them). A weights-only hit (no shell) rebuilds a meta model from the cached SD,
assigns both persistent and non-persistent tensors, then reinstalls accelerate
dispatch hooks. The outer wrapper is not cached.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from accelerate import dispatch_model
from transformers import AutoModelForImageTextToText

from ltx_core.loader.primitives import BuilderProtocol, StateDict
from ltx_core.loader.registry import Registry, module_registry_key
from ltx_core.multigpu.gemma.accelerate_wrapper import AccelerateGemmaWrapper
from ltx_core.multigpu.gemma.loader import load_gemma_with_device_map
from ltx_core.text_encoders.gemma import gemma_model_config
from ltx_core.text_encoders.gemma.encoders.base_encoder import LTXGemmaTextEncoder

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


def _assign_non_persistent(model: torch.nn.Module, non_persistent_sd: dict[str, torch.Tensor]) -> None:
    for name, tensor in non_persistent_sd.items():
        parent_path, attr = name.rsplit(".", 1)
        module = model
        for part in parent_path.split("."):
            module = getattr(module, part)
        setattr(module, attr, tensor)


def _split_persistent(
    model: torch.nn.Module, sd: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    expected_keys = set(model.state_dict().keys())
    persistent = {k: v for k, v in sd.items() if k in expected_keys}
    non_persistent = {k: v for k, v in sd.items() if k not in expected_keys}
    return persistent, non_persistent


def _state_dict_with_non_persistent(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """``state_dict()`` plus non-persistent buffers (needed for weights-only rebuild)."""
    sd = model.state_dict()
    for name, buf in model.named_buffers():
        if name not in sd:
            sd[name] = buf
    return sd


class AccelerateGemmaBuilder(BuilderProtocol):
    """Builder that loads Gemma with ``device_map="auto"`` on the source rank.
    Conforms to the builder interface expected by ``PromptEncoder``:
    ``build(device, dtype) -> model``.  Non-source ranks get a lightweight
    :class:`AccelerateGemmaWrapper` that receives embeddings via broadcast.
    """

    def __init__(
        self,
        gemma_root_path: str,
        gemma_group: dist.ProcessGroup | None,
        broadcast_group: dist.ProcessGroup | None,
        registry: Registry,
        *,
        src_rank: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._gemma_root_path = gemma_root_path
        self._gemma_group = gemma_group
        self._broadcast_group = broadcast_group
        self._registry = registry
        self._src_rank = src_rank
        self._is_src = dist.get_rank() == src_rank
        self._dtype = dtype
        # Cached on the src rank after first build (non-tensor objects).
        self._config: object | None = None
        self._hf_device_map: dict[str, int | str] | None = None
        self._tokenizer: object | None = None
        self._processor: object | None = None

    @property
    def registry(self) -> Registry:
        return self._registry

    def with_registry(self, registry: Registry) -> Self:
        clone = copy.copy(self)
        clone._registry = registry
        return clone

    def model_metadata(self) -> dict:
        """Full metadata with a ``config`` sub-dict (same shape as safetensors loaders)."""
        return {"config": gemma_model_config(self._gemma_root_path)}

    def model_config(self) -> dict:
        return self.model_metadata().get("config", {})

    def _module_registry_key(self) -> str:
        return module_registry_key(
            configurator=AccelerateGemmaBuilder,
            config={"accelerate_gemma": str(Path(self._gemma_root_path).resolve())},
            module_ops_names=(),
        )

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **_kwargs: Any,  # noqa: ANN401
    ) -> AccelerateGemmaWrapper:
        dtype = dtype or self._dtype

        encoder = self._build_encoder(dtype) if self._is_src else None

        return AccelerateGemmaWrapper(
            encoder=encoder,
            broadcast_group=self._broadcast_group,
            src_rank=dist.get_group_rank(self._broadcast_group, self._src_rank),
            dtype=dtype,
            device=device,
        )

    def _build_encoder(self, dtype: torch.dtype) -> LTXGemmaTextEncoder:
        struct = self._module_registry_key()
        cached_sd = self._registry.get([self._gemma_root_path], None)
        cached_model = self._registry.get_model(struct)

        if cached_model is not None:
            if cached_sd is None:
                raise RuntimeError(
                    "Gemma state dict missing from registry but cached encoder is set; "
                    "registry must outlive the builder."
                )
            encoder = cached_model  # type: ignore[assignment]
            encoder.model.load_state_dict(cached_sd.sd, strict=False, assign=True)
            encoder.dtype = dtype
            return encoder

        if cached_sd is not None:
            if self._config is None or self._hf_device_map is None:
                raise RuntimeError(
                    "Gemma state dict is in the registry but this builder has no HF config/"
                    "device_map/tokenizer/processor. Rebuild requires a builder that previously "
                    "loaded from disk, or use cache_models=True so the encoder shell is reused."
                )
            logger.info("Rebuilding Gemma from cached state dict (no disk I/O).")
            encoder = self._rebuild_from_cache(cached_sd, dtype)
        else:
            encoder = load_gemma_with_device_map(self._gemma_root_path, dtype)

            self._config = encoder.model.config
            self._hf_device_map = encoder.model.hf_device_map
            self._tokenizer = encoder.tokenizer
            self._processor = encoder.processor

            # Include non-persistent buffers so weights-only rebuild can restore them
            # onto a fresh meta model (there is no shell to keep them on).
            sd = _state_dict_with_non_persistent(encoder.model)
            total_size = sum(t.nelement() * t.element_size() for t in sd.values())
            dtypes = {t.dtype for t in sd.values()}
            self._registry.add(
                [self._gemma_root_path],
                None,
                StateDict(sd=sd, device=torch.device("meta"), size=total_size, dtype=dtypes),
            )
            logger.info("Cached Gemma state dict in registry (%d entries).", len(sd))

        return self._registry.add_model(struct, encoder)  # type: ignore[return-value]

    def _rebuild_from_cache(self, cached: StateDict, dtype: torch.dtype) -> LTXGemmaTextEncoder:
        with torch.device("meta"):
            model = AutoModelForImageTextToText.from_config(self._config)

        persistent_sd, non_persistent_sd = _split_persistent(model, cached.sd)
        model.load_state_dict(persistent_sd, strict=True, assign=True)
        _assign_non_persistent(model, non_persistent_sd)
        dispatch_model(model, self._hf_device_map)

        return LTXGemmaTextEncoder(
            model=model,
            tokenizer=self._tokenizer,
            processor=self._processor,
            dtype=dtype,
        )
