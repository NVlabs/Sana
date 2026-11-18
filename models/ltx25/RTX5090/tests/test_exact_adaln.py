from __future__ import annotations

from collections import Counter

import pytest
import torch

from models.ltx25.RTX5090.build_exact_adaln import load_prefixed_state
from models.ltx25.RTX5090.exact_adaln import ExactScheduleAdaLN


class KeysOnlyCheckpoint:
    def __init__(self) -> None:
        self.tensors = {
            "video.linear.weight": torch.tensor([1.0]),
            "video.linear.bias": torch.tensor([2.0]),
            "audio.linear.weight": torch.tensor([3.0]),
        }

    def keys(self):
        return self.tensors.keys()

    def get_tensor(self, key: str) -> torch.Tensor:
        return self.tensors[key]


def test_prefixed_state_accepts_noniterable_safe_open_handle() -> None:
    state = load_prefixed_state(KeysOnlyCheckpoint(), "video")

    assert set(state) == {"linear.weight", "linear.bias"}
    torch.testing.assert_close(state["linear.weight"], torch.tensor([1.0]))


def test_exact_schedule_selects_the_matching_row() -> None:
    calls = Counter()
    module = ExactScheduleAdaLN(
        "video_base",
        torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
        torch.tensor([[[5.0]], [[6.0]]]),
        torch.tensor([100.0, 200.0]),
        4,
        calls,
    )

    projected, embedded = module(torch.full((4,), 200.0))

    torch.testing.assert_close(projected, torch.tensor([[3.0, 4.0]]))
    torch.testing.assert_close(embedded, torch.tensor([[6.0]]))
    assert calls[("video_base", 1)] == 1


def test_exact_schedule_rejects_wrong_tokens_and_nonuniform_steps() -> None:
    module = ExactScheduleAdaLN(
        "video_base",
        torch.zeros(2, 1, 2),
        torch.zeros(2, 1, 1),
        torch.tensor([100.0, 200.0]),
        4,
        Counter(),
    )
    with pytest.raises(RuntimeError, match="expects 4 tokens"):
        module(torch.full((3,), 100.0))
    with pytest.raises(RuntimeError, match="uniform timestep"):
        module(torch.tensor([100.0, 100.0, 200.0, 100.0]))
