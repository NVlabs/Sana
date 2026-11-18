"""Numerical and replay contracts for the SM100 accumulator/mask fast paths."""

import pytest
import torch
from sol_attn import interface, sol_attn
from test_exact_block_mask import hybrid_oracle

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="requires the SM100/SM103 CuTe kernel",
)


def inputs(tokens, batch=1, heads=2, seed=71):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        torch.randn(
            batch,
            tokens,
            heads,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        for _ in range(3)
    )


@pytest.mark.parametrize("threshold", ["diag", "exact"])
@pytest.mark.parametrize("tokens", [63, 64, 65, 127, 128, 129, 255, 256, 257, 513])
@torch.inference_mode()
def test_tile_boundaries_match_independent_oracle(tokens, threshold):
    q, k, v = inputs(
        tokens, batch=2 if tokens == 257 else 1, heads=3 if tokens == 129 else 2
    )
    mask = torch.zeros((tokens + 63) // 64, device="cuda", dtype=torch.bool)
    mask[0] = mask[-1] = True
    if len(mask) > 5:
        mask[4] = True
    actual = sol_attn(q, k, v, tau=1.5, thresh_type=threshold, exact_block_mask=mask)
    expected = hybrid_oracle(q, k, v, mask, 1.5, threshold)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("tokens", [4095, 4096, 4097])
@pytest.mark.parametrize(
    "kind", ["random", "mixed_maxima", "zero_q", "zero_v", "tiny_v", "subnormal_v"]
)
@torch.inference_mode()
def test_long_exact_stream_matches_dense(tokens, kind):
    q, k, v = inputs(tokens, seed=153)
    if kind == "mixed_maxima":
        # Different Q rows see different sequences of maxima and identity scales.
        q[:, ::2].mul_(3)
        q[:, 1::4].zero_()
        k[:, 64::128].mul_(3)
    elif kind == "zero_q":
        q.zero_()
    elif kind == "zero_v":
        v.zero_()
    elif kind == "tiny_v":
        v.mul_(2.0**-120)
    elif kind == "subnormal_v":
        v.mul_(2.0**-130)
    mask = torch.ones((tokens + 63) // 64, device="cuda", dtype=torch.bool)
    actual = sol_attn(q, k, v, exact_block_mask=mask)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2),
        v.float().transpose(1, 2),
    ).transpose(1, 2)
    assert torch.isfinite(actual).all()
    if kind in ("tiny_v", "subnormal_v"):
        # Scale the comparison, not the input to the kernel: a normal absolute
        # tolerance would accept an accidentally all-zero tiny result.
        scale = 2.0 ** (120 if kind == "tiny_v" else 130)
        # BF16 subnormals are spaced by 2**-133, hence rounding alone can
        # contribute 0.0625 after scaling by 2**130. Account for the known
        # output quantization separately from the attention error tolerance.
        rounding = 2.0**-134 * scale if kind == "subnormal_v" else 0.0
        torch.testing.assert_close(
            actual.double() * scale,
            expected.double() * scale,
            atol=0.02 + rounding,
            rtol=0.02,
        )
        assert torch.count_nonzero(actual) > 0
    else:
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("threshold", ["diag", "exact"])
@torch.inference_mode()
def test_graph_reads_updated_inputs_and_masks_without_recompile(threshold):
    q, k, v = inputs(513, batch=2, heads=3)
    mask = torch.zeros(9, device="cuda", dtype=torch.bool)
    mask[0] = mask[-1] = True
    options = dict(tau=1.5, thresh_type=threshold, exact_block_mask=mask)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        sol_attn(q, k, v, **options)
        stream.synchronize()
        compiled = set(interface._compiled)
        with torch.cuda.graph(graph, stream=stream):
            captured = sol_attn(q, k, v, **options)
    torch.cuda.current_stream().wait_stream(stream)
    for update in (False, True):
        if update:
            q.copy_(q.flip(1))
            k.mul_(0.75)
            v.neg_()
            mask.zero_()
            mask[2] = mask[5] = True
        graph.replay()
        expected = sol_attn(q, k, v, **options)
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
        oracle = hybrid_oracle(q, k, v, mask, 1.5, threshold)
        torch.testing.assert_close(captured.float(), oracle, atol=0.02, rtol=0.02)
        assert set(interface._compiled) == compiled


@torch.inference_mode()
def test_absent_empty_and_updated_mask_preserve_sink_contract():
    q, k, v = inputs(513, batch=2, heads=1)
    mask = torch.zeros(9, device="cuda", dtype=torch.bool)
    options = dict(tau=1e6, sink_start=128, sink_tokens=64)
    absent = sol_attn(q, k, v, **options)
    empty = sol_attn(q, k, v, **options, exact_block_mask=mask)
    torch.testing.assert_close(absent, empty, atol=0, rtol=0)
    mask[4] = mask[8] = True
    actual = sol_attn(q, k, v, **options, exact_block_mask=mask)
    # A sink protects only KV columns; exact_block_mask protects Q rows too.
    expected = hybrid_oracle(q, k, v, mask, 1e6, "diag", sink_start=128, sink_tokens=64)
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
