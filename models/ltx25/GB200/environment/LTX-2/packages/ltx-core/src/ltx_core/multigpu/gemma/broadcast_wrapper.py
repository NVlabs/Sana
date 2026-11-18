"""Shared base for the multi-GPU Gemma wrappers: src-rank gating + NCCL broadcast."""

from __future__ import annotations

import torch
import torch.distributed as dist

from ltx_core.model.disposable import Disposable
from ltx_core.text_encoders.gemma.encoders.base_encoder import LTXGemmaTextEncoder


class BroadcastGemmaWrapper(torch.nn.Module, Disposable):
    """Encoder/group plumbing, prompt enhancement, and result broadcast.
    Subclasses implement ``encode`` (the stub below raises).
    Args:
        encoder: The encoder; real on ``src_rank``, may be ``None`` elsewhere.
        broadcast_group: NCCL group covering ranks that need the embeddings.
        src_rank: Rank *within* ``broadcast_group`` that holds the real encoder and runs
            the sampling-based ``enhance_*`` methods; its results are broadcast to every
            other rank in the group. Builders derive it from a global driver rank via
            ``dist.get_group_rank``.
        dtype: Target dtype for output tensors.
        device: Target device for output tensors.
    """

    def __init__(
        self,
        encoder: LTXGemmaTextEncoder | None,
        broadcast_group: dist.ProcessGroup | None,
        src_rank: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if device is None and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        self._encoder = encoder
        self._group = broadcast_group
        self._src_rank = src_rank
        self._rank = dist.get_rank(broadcast_group)
        self._dtype = dtype
        self._device = device

    def encode(self, prompts: list[str]) -> list[tuple[tuple[torch.Tensor, ...], torch.Tensor]]:
        """Encode a batch of prompts to per-prompt hidden states; implemented by subclasses."""
        raise NotImplementedError

    def enhance_t2v(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        system_prompt: str | None = None,
        seed: int = 10,
        static_cache: bool = False,
    ) -> str:
        result = None
        if self._rank == self._src_rank:
            result = self._encoder.enhance_t2v(
                prompt,
                max_new_tokens=max_new_tokens,
                system_prompt=system_prompt,
                seed=seed,
                static_cache=static_cache,
            )
        return self._broadcast_str(result)

    def enhance_i2v(
        self,
        prompt: str,
        image: torch.Tensor,
        max_new_tokens: int | None = None,
        system_prompt: str | None = None,
        seed: int = 10,
        static_cache: bool = False,
    ) -> str:
        result = None
        if self._rank == self._src_rank:
            result = self._encoder.enhance_i2v(
                prompt,
                image,
                max_new_tokens=max_new_tokens,
                system_prompt=system_prompt,
                seed=seed,
                static_cache=static_cache,
            )
        return self._broadcast_str(result)

    def _broadcast_str(self, value: str | None) -> str:
        obj_list: list[str | None] = [value]
        dist.broadcast_object_list(obj_list, src=self._src_rank, group=self._group)
        result = obj_list[0]
        assert result is not None, "broadcast returned None; check src_rank/broadcast_group"
        return result

    def _broadcast_encoder_output(
        self,
        hidden_states: tuple[torch.Tensor, ...] | None,
        attention_mask: torch.Tensor | None,
        src_rank: int,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Broadcast hidden states + attention mask via NCCL from ``src_rank``."""
        if self._rank == src_rank:
            meta = [{"hs_shapes": [h.shape for h in hidden_states], "mask_shape": attention_mask.shape}]
        else:
            meta = [None]
        dist.broadcast_object_list(meta, src=src_rank, group=self._group)
        info = meta[0]

        if self._rank != src_rank:
            hidden_states = tuple(torch.empty(s, device=self._device, dtype=self._dtype) for s in info["hs_shapes"])
            attention_mask = torch.empty(info["mask_shape"], device=self._device, dtype=torch.long)
        else:
            hidden_states = tuple(h.to(device=self._device, dtype=self._dtype) for h in hidden_states)
            attention_mask = attention_mask.to(device=self._device)

        for h in hidden_states:
            dist.broadcast(h, src=src_rank, group=self._group)
        dist.broadcast(attention_mask, src=src_rank, group=self._group)

        return hidden_states, attention_mask
