# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch


@pytest.fixture
def short_conv_api():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA and FLA's Triton short-convolution backend")
    from fla.modules import ShortConvolution

    from diffusion.model.ops.fused_streaming import _cached_temporal_short_conv

    return ShortConvolution, _cached_temporal_short_conv


def _values(shape, offset=0):
    count = 1
    for size in shape:
        count *= size
    return ((torch.arange(count, dtype=torch.float64) + offset) % 7 - 3).reshape(shape)


def _reference(current, history, weight):
    """Independent tap sums: causal past, plus future only within this chunk."""
    batches, frames, channels = current.shape
    kernel = weight.shape[-1]
    full = torch.cat((history, current), dim=1)
    out = torch.zeros(batches, frames, channels, dtype=torch.float64)
    for frame in range(frames):
        position = history.shape[1] + frame
        for lag in range(kernel):
            if position >= lag:
                out[:, frame] += full[:, position - lag] * weight[:, 0, kernel - 1 - lag]
        for lead in range(1, kernel):
            if frame + lead < frames:
                out[:, frame] += current[:, frame + lead] * weight[:, 0, kernel - 1 - lead]
    return out


def _expected_cache(history, kernel):
    if kernel == 1:
        return None
    expected = torch.zeros(history.shape[0], kernel - 1, history.shape[2], dtype=history.dtype)
    count = min(history.shape[1], kernel - 1)
    if count:
        expected[:, -count:] = history[:, -count:]
    return expected


def _make_conv(short_convolution, channels, kernel, dtype):
    conv = short_convolution(channels, kernel, bias=False, activation=None, backend="triton").to("cuda", dtype)
    assert conv.backend == "triton"
    weight = ((torch.arange(channels * kernel, dtype=torch.float64) % 5) - 2).reshape(channels, 1, kernel)
    with torch.no_grad():
        conv.weight.copy_(weight)
    return conv.eval(), weight


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True], ids=["contiguous", "qkv-view"])
@pytest.mark.parametrize(
    "kernel, chunks, initial_frames",
    [
        (4, [2, 2], 0),
        (4, [4, 2, 2], 0),
        (4, [4, 3, 3], 0),
        (4, [1, 1, 2], 0),
        (3, [1, 1], 0),
        (4, [2, 2], 1),
        (1, [1, 2], 0),
    ],
)
@torch.inference_mode()
def test_cached_short_conv_matches_causal_reference(short_conv_api, dtype, packed, kernel, chunks, initial_frames):
    short_convolution, cached_conv = short_conv_api
    batch, height, width, channels = (2, 1, 2, 16) if packed else (1, 2, 3, 8)
    spatial = height * width
    conv, weight = _make_conv(short_convolution, channels, kernel, dtype)
    history = _values((batch * spatial, initial_frames, channels), offset=2)
    cache = history.to("cuda", dtype) if initial_frames else None
    for index, frames in enumerate(chunks):
        current = _values((batch * spatial, frames, channels), offset=3 * index + 1)
        tokens = current.reshape(batch, spatial, frames, channels).permute(0, 2, 1, 3).reshape(batch, -1, channels)
        if packed:
            storage = torch.empty(batch, frames * spatial, 3, channels, device="cuda", dtype=dtype)
            x = storage[:, :, 1, :]
            x.copy_(tokens)
            assert x.stride(1) == 3 * channels
        else:
            x = tokens.to("cuda", dtype).contiguous()
        before_x = x.clone()
        before_cache = cache.clone() if cache is not None else None
        readonly, ignored = cached_conv(x, conv, (frames, height, width), cache, False)
        actual, new_cache = cached_conv(x, conv, (frames, height, width), cache, True)
        assert ignored is None
        torch.testing.assert_close(readonly, actual, rtol=0, atol=0)
        torch.testing.assert_close(x, before_x, rtol=0, atol=0)
        if cache is not None:
            torch.testing.assert_close(cache, before_cache, rtol=0, atol=0)
        expected = _reference(current, history, weight)
        expected = expected.reshape(batch, spatial, frames, channels).permute(0, 2, 1, 3).reshape_as(tokens)
        torch.testing.assert_close(actual.cpu(), expected.to(dtype), rtol=0, atol=0)
        history = torch.cat((history, current), dim=1)
        expected_cache = _expected_cache(history, kernel)
        if expected_cache is None:
            assert new_cache is None
        else:
            assert not new_cache.requires_grad
            torch.testing.assert_close(new_cache.cpu(), expected_cache.to(dtype), rtol=0, atol=0)
            assert new_cache.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
            if cache is not None:
                assert new_cache.untyped_storage().data_ptr() != cache.untyped_storage().data_ptr()
        cache = new_cache


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_cached_short_conv_graph_replay_reads_current_inputs(short_conv_api, dtype):
    short_convolution, cached_conv = short_conv_api
    conv, weight = _make_conv(short_convolution, 8, 4, dtype)
    x = torch.zeros(1, 2, 8, device="cuda", dtype=dtype)
    cache = torch.zeros(1, 3, 8, device="cuda", dtype=dtype)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            cached_conv(x, conv, (2, 1, 1), cache, True)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, new_cache = cached_conv(x, conv, (2, 1, 1), cache, True)
    for index in range(2):
        current = _values((1, 2, 8), offset=2 * index)
        history = _values((1, 3, 8), offset=3 * index + 1)
        x.copy_(current)
        cache.copy_(history)
        graph.replay()
        expected = _reference(current, history, weight).to(dtype)
        expected_cache = _expected_cache(torch.cat((history, current), dim=1), 4).to(dtype)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(new_cache.cpu(), expected_cache, rtol=0, atol=0)
        torch.testing.assert_close(x.cpu(), current.to(dtype), rtol=0, atol=0)
        torch.testing.assert_close(cache.cpu(), history.to(dtype), rtol=0, atol=0)
    assert new_cache.untyped_storage().data_ptr() != cache.untyped_storage().data_ptr()
    assert new_cache.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
