"""Exact FFN chunking used by the BF16 RTX 5090 profile."""

from __future__ import annotations


class FeedForwardChunking:
    def __init__(self, chunk_tokens: int = 16384, min_tokens: int = 65536) -> None:
        self.chunk_tokens = chunk_tokens
        self.min_tokens = min_tokens
        self.original_forward = None
        self.feed_forward_class = None

    def install(self) -> None:
        import torch
        from ltx_core.model.transformer.feed_forward import FeedForward

        original_forward = FeedForward.forward
        owner = self

        def chunked_forward(module, x):
            if x.shape[-2] < owner.min_tokens:
                return original_forward(module, x)
            outputs = [
                original_forward(module, chunk)
                for chunk in torch.split(x, owner.chunk_tokens, dim=-2)
            ]
            return torch.cat(outputs, dim=-2)

        self.original_forward = original_forward
        self.feed_forward_class = FeedForward
        FeedForward.forward = chunked_forward

    def uninstall(self) -> None:
        if self.original_forward is not None:
            self.feed_forward_class.forward = self.original_forward
            self.original_forward = None
            self.feed_forward_class = None
