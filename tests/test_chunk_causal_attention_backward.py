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
import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from diffusion.model.nets.sana_blocks import ChunkCausalAttention


def _dense_chunk_reference(module, x, shape, freqs, chunk_index):
    """Use a dense causal mask, without recurrent or cumulative states."""
    batch, tokens, channels = x.shape
    heads, dim = module.heads, module.dim
    q, k, v = module.qkv(x).reshape(batch, tokens, 3, channels).unbind(2)
    dtype = q.dtype
    q = module.kernel_func(module.q_norm(q)).reshape(batch, tokens, heads, dim).transpose(1, 2)
    k = module.kernel_func(module.k_norm(k)).reshape(batch, tokens, heads, dim).transpose(1, 2)
    v = v.reshape(batch, tokens, heads, dim).transpose(1, 2)

    def rotate(tensor):
        pairs = tensor.double().reshape(batch, heads, tokens, dim // 2, 2)
        real, imag = pairs.unbind(-1)
        rotated = torch.stack((real * freqs.real - imag * freqs.imag, real * freqs.imag + imag * freqs.real), dim=-1)
        return rotated.flatten(-2).to(tensor.dtype)

    q_rotated, k_rotated = rotate(q), rotate(k)
    if module.fp32_attention:
        q_rotated, k_rotated, v = q_rotated.float(), k_rotated.float(), v.float()
    frames, height, width = shape
    starts = chunk_index if chunk_index is not None else [0]
    chunk_ids = torch.bucketize(torch.arange(frames), torch.tensor(starts[1:]), right=True)
    chunk_ids = chunk_ids.repeat_interleave(height * width)
    visible = chunk_ids[:, None] >= chunk_ids[None, :]
    numerator_scores = (q_rotated @ k_rotated.transpose(-1, -2)).masked_fill(~visible, 0)
    denominator_scores = (q @ k.transpose(-1, -2)).masked_fill(~visible, 0)
    out = (numerator_scores @ v) / (denominator_scores.sum(dim=-1, keepdim=True) + module.eps)
    return module.proj(out.to(dtype).transpose(1, 2).reshape(batch, tokens, channels))


@pytest.mark.parametrize("chunk_index", [None, [0, 2], [0, 1, 3]], ids=["single", "two", "three"])
@pytest.mark.parametrize("use_checkpoint", [False, True], ids=["direct", "checkpoint"])
@pytest.mark.parametrize("fp32_attention", [False, True], ids=["fp64", "fp32-norm"])
def test_chunk_attention_backward_matches_dense_reference(chunk_index, use_checkpoint, fp32_attention):
    torch.manual_seed(7)
    dtype = torch.float32 if fp32_attention else torch.float64
    module = ChunkCausalAttention(8, 8, heads=2, use_bias=True, qk_norm=fp32_attention, eps=1e-6).to(dtype)
    module.fp32_attention = fp32_attention
    with torch.no_grad():
        module.qkv.weight.mul_(0.25)
        module.qkv.bias[:16].fill_(0.5)
    reference_module = copy.deepcopy(module)
    shape = (4, 2, 1)
    x = (torch.randn(2, 8, 8, dtype=dtype) * 0.2).requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    angles = torch.linspace(0.01, 0.3, 16, dtype=torch.float64).reshape(1, 1, 8, 2)
    freqs = torch.polar(torch.ones_like(angles), angles)

    def forward(value):
        return module(value, HW=shape, rotary_emb=freqs, chunk_index=chunk_index)

    actual = checkpoint(forward, x, use_reentrant=False) if use_checkpoint else forward(x)
    expected = _dense_chunk_reference(reference_module, reference_x, shape, freqs, chunk_index)
    rtol, atol = (1e-4, 2e-6) if fp32_attention else (1e-9, 1e-10)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    probe = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad((actual * probe).sum(), (x, *module.parameters()))
    expected_gradients = torch.autograd.grad((expected * probe).sum(), (reference_x, *reference_module.parameters()))
    names = ["input", *dict(module.named_parameters())]
    for name, actual_gradient, expected_gradient in zip(names, actual_gradients, expected_gradients):
        assert torch.isfinite(actual_gradient).all(), name
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=rtol, atol=atol, msg=lambda msg: f"{name}: {msg}"
        )
