# SPDX-License-Identifier: Apache-2.0
"""Logits and sample ownership for joint chunk and sample discrimination."""

from typing import NamedTuple

import torch


class MiniMaxH3DiscriminatorOutput(NamedTuple):
    """Chunk logits [N, 1], sample logits [B, 1], and chunk sample indices [N]."""

    chunk_logits: torch.Tensor
    sample_logits: torch.Tensor
    chunk_sample_indices: torch.Tensor


__all__ = ["MiniMaxH3DiscriminatorOutput"]
