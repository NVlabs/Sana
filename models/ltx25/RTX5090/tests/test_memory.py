from __future__ import annotations

import torch
from ltx_core.model.transformer.feed_forward import FeedForward

from models.ltx25.RTX5090.memory import FeedForwardChunking


def test_feed_forward_chunking_is_exact() -> None:
    torch.manual_seed(7)
    module = FeedForward(dim=16, dim_out=16, mult=2)
    inputs = torch.randn(2, 13, 16)
    expected = module(inputs)
    chunking = FeedForwardChunking(chunk_tokens=5, min_tokens=10)
    try:
        chunking.install()
        actual = module(inputs)
    finally:
        chunking.uninstall()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
