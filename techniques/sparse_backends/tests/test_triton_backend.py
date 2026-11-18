"""Numerical regressions for the SM80 Triton pointer backend."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("thresh_type", ["diag", "exact"])
@pytest.mark.parametrize("sink_tokens", [0, 951], ids=["no_sink", "prefix_sink"])
@pytest.mark.parametrize(
    "tail_only", [False, True], ids=["constant_values", "tail_only"]
)
def test_sm80_approximate_tail_normalization(thresh_type, sink_tokens, tail_only):
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("requires SM80 to exercise the Triton pointer backend")

    # Keep the CPU-only CI suite collectable without Triton installed.
    from sol_attn import get_sol_attn_backend, sol_attn

    assert get_sol_attn_backend("cuda") == "triton"

    # 32 blocks of 64 tokens fill one routing group. The last token starts
    # the next group and must contribute with length 1, rather than 64.
    tokens = 2049
    q = torch.zeros((1, tokens, 1, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.ones_like(q)
    expected_value = 1.0
    if tail_only:
        v.zero_()
        v[:, -1] = 64
        expected_value = 64 / tokens

    # Zero Q/K give zero scores and positive thresholds in both modes. For
    # the first query block, the tail is neither a neighbor nor in the prefix
    # sink, so it takes the approximate branch. Both branches must produce
    # the uniform average of V for this input, including the singleton tail.
    actual = sol_attn(
        q,
        k,
        v,
        tau=1.0,
        thresh_type=thresh_type,
        sink_start=0,
        sink_tokens=sink_tokens,
    )
    expected = torch.full_like(v, expected_value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
