"""MiniMax H3 tokenizer contract."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoTokenizer


class MiniMaxH3Tokenizer:
    """Thin wrapper around the checkpoint tokenizer used by MiniMax H3."""

    def __init__(self, pretrained_model_name_or_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
    ) -> MiniMaxH3Tokenizer:
        return cls(pretrained_model_name_or_path)

    def __call__(
        self,
        prompt: str,
        *,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded.input_ids
        if return_mask:
            return input_ids, torch.ones_like(input_ids)
        return input_ids

    def encode(self, prompt: str) -> tuple[torch.Tensor, int]:
        input_ids = self(prompt)[0]
        return input_ids, int(input_ids.numel())

    def decode(self, token_ids: Any, **kwargs: Any) -> str:
        return self.tokenizer.decode(token_ids, **kwargs)


__all__ = ["MiniMaxH3Tokenizer"]
