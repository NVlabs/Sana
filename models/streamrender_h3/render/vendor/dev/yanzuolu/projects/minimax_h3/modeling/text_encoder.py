# SPDX-License-Identifier: Apache-2.0
"""Expose Qwen3-VL as what MiniMax-H3 actually needs: one hidden layer.

The released pipeline conditions the DiT on an intermediate decoder layer of
Qwen3-VL, not on the head's logits. The HuggingFace class cannot hand that over:
``Qwen3VLForConditionalGeneration.forward`` returns only loss, logits,
past_key_values and rope_deltas, and ``output_hidden_states=True`` is swallowed
by its ``**kwargs``, so ``.hidden_states`` is always None. The layer output has
to be taken where it is produced.

Wrapping that here rather than at the call site keeps the depth with the weights
it indexes into: ``hidden_layer`` is only meaningful against this checkpoint's
layer count, and the assertion that bounds it can only be written where the
layers exist.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, Qwen3VLForConditionalGeneration


class MiniMaxH3TextEncoder(torch.nn.Module):
    """Qwen3-VL restricted to ``(input_ids, attention_mask) -> hidden``."""

    def __init__(self, model: torch.nn.Module, hidden_layer: int) -> None:
        super().__init__()
        self.model = model
        layers = self._layers(model)
        assert 1 <= hidden_layer <= len(layers), (
            f"hidden_layer must be in [1, {len(layers)}], got {hidden_layer}"
        )
        self.hidden_layer = int(hidden_layer)
        # The released directory ships the full 64-layer Qwen3-VL, but the layers
        # above the one H3 reads cannot influence it -- the official encoder
        # simply declares num_hidden_layers = 50 and never builds them. Dropping
        # them here is not an optimization we invented: keeping them shards ~14
        # unused layers of a 32B model onto every rank of a run that is already
        # memory-bound.
        del layers[hidden_layer:]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        hidden_layer: int,
        **kwargs: Any,
    ) -> "MiniMaxH3TextEncoder":
        """Build through ``FromPretrainedInstantiate``; kwargs reach the head."""
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        return cls(model, hidden_layer)

    @staticmethod
    def _layers(model: torch.nn.Module) -> torch.nn.ModuleList:
        return model.model.language_model.layers

    def forward(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """``[B, L]`` ids -> ``[B, L, D]`` at ``hidden_layer``.

        The FULL head is what gets called, never the inner ``model``. FSDP's root
        unit owns the embeddings and the final norm, so calling a submodule
        directly would skip its pre-forward unshard and read sharded parameters
        -- a wrong answer rather than an error.
        """
        # HF numbers hidden_states[0] as the embedding output, so entry k is the
        # output of layer k - 1.
        layer = self._layers(self.model)[self.hidden_layer - 1]
        captured: list[torch.Tensor] = []
        handle = layer.register_forward_hook(
            lambda module, args, output: captured.append(
                output[0] if isinstance(output, tuple) else output
            )
        )
        try:
            self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            handle.remove()
        assert len(captured) == 1, f"hook fired {len(captured)} times, expected once"
        return captured[0]


def build_text_encoder(
    pretrained_model_name_or_path: str,
    *,
    hidden_layer: int,
    **kwargs: Any,
) -> MiniMaxH3TextEncoder:
    """Build the encoder's structure from config alone, reading no weights.

    The counterpart to ``from_pretrained`` for the 1x-memory route.
    ``from_pretrained`` fuses construction with materialization, which forces the
    weights to arrive before FSDP has sharded anything -- so every rank reads a
    slice of every tensor out of the same files through mmap page faults. Under
    this route the shells stay on meta until ``fully_shard`` has decided who owns
    what, and a DCP load then streams each rank its own chunk.

    A plain callable rather than a classmethod because that is all
    ``DefaultModelInstantiate`` needs: it resolves ``class_name`` with ``getattr``
    and calls it inside ``init_empty_weights``. The path travels in ``args``, the
    same convention ``FromPretrainedInstantiate`` uses, so the weights stage sees
    a completed construction and naturally no-ops.

        models:
          text_encoder:
            module: dev.yanzuolu.projects.minimax_h3.modeling.text_encoder
            class_name: build_text_encoder
            meta_init: true
            args:
              pretrained_model_name_or_path: .../text_encoder
              hidden_layer: 50
            placement:
              fsdp: { wrap_modules: [Qwen3VLTextDecoderLayer, Qwen3VLVisionBlock] }
              plugins:
                - module: dev.yanzuolu.common.plugin.dcp_weights
                  class_name: ShardedDCPWeights
                  path: .../h3_text_encoder.dcp

    ``num_hidden_layers`` is narrowed here rather than left to ``__init__``'s
    ``del``: the released directory declares 64 and H3 reads 50, and building the
    other 14 only to drop them is 14/64 of a 32 B decoder. Under the old route
    their weights were read off disk first as well. The ``del`` downstream then
    finds nothing to remove, and the bound assertion still holds at 50 <= 50.
    """
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    config.text_config.num_hidden_layers = hidden_layer
    return MiniMaxH3TextEncoder(Qwen3VLForConditionalGeneration(config), hidden_layer)


EntryClass = MiniMaxH3TextEncoder

__all__ = ["MiniMaxH3TextEncoder"]
