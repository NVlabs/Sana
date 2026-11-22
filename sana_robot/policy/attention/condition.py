# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sana_robot.policy.attention.gdn import GatedDeltaNet
from sana_robot.policy.attention.softmax import GatedSoftmaxAttention, rotate

CONDITION_CHUNK = 0
PREDICTION_CHUNK = 1
(PREFILL, TARGET) = ("prefill", "target")


class ConditionChunkLayout:
    """Split of the token sequence into the condition chunk (observation latent and state) and the prediction chunk (future latents and actions), with the permutation that puts the condition chunk first."""

    def __init__(self, chunk_ids: torch.Tensor) -> None:
        host_ids = chunk_ids.detach().to("cpu", torch.long)
        sizes = torch.bincount(host_ids, minlength=PREDICTION_CHUNK + 1)
        order = torch.argsort(host_ids, stable=True)
        self.num_tokens = int(host_ids.numel())
        self.bounds = (0, *torch.cumsum(sizes, 0).tolist())
        self.order_host = order
        self.order = order.to(chunk_ids.device)
        self.restore = torch.argsort(order).to(chunk_ids.device)

    def slices(self) -> list[tuple[int, int]]:
        """Return the (start, end) token range of every chunk in chunk-first order."""
        return list(zip(self.bounds[:-1], self.bounds[1:]))


class ConditionPrefix:
    """What the condition chunk leaves for the prediction chunk in every self-attention layer: a softmax layer's keys and values, a Gated DeltaNet layer's state."""

    def __init__(self) -> None:
        self._layers = {}

    def store(self, module, value) -> None:
        self._layers[id(module)] = value

    def load(self, module):
        return self._layers[id(module)]

    def __len__(self) -> int:
        return len(self._layers)


class ConditionPass(ConditionChunkLayout):
    """One of the two passes of prefix caching: the condition chunk alone, which fills a ``ConditionPrefix`` (``PREFILL``), or the prediction chunk alone, which reads it (``TARGET``). ``keep`` indexes the tokens of the pass in the full sequence."""

    def __init__(self, layout: ConditionChunkLayout, phase: str, cache: ConditionPrefix) -> None:
        chunk = CONDITION_CHUNK if phase == PREFILL else PREDICTION_CHUNK
        (start, end) = (layout.bounds[chunk], layout.bounds[chunk + 1])
        self.keep = layout.order[start:end]
        self.keep_host = layout.order_host[start:end]
        self.full_tokens = layout.num_tokens
        self.num_tokens = end - start
        self.bounds = (0, self.num_tokens)
        self.order = torch.arange(self.num_tokens, device=self.keep.device)
        self.restore = self.order
        self.phase = phase
        self.cache = cache

    def kept(self, offset: int, length: int) -> tuple[int, int]:
        """Return the (offset, length) inside this pass of the full-sequence token span [offset, offset + length)."""
        start = int((self.keep_host < int(offset)).sum())
        return (start, int((self.keep_host < int(offset) + int(length)).sum()) - start)


def condition_chunk_layout(frame_ids: torch.Tensor, action_rows: int, device=None) -> ConditionChunkLayout:
    """Return the layout of video tokens with latent frame ids ``frame_ids`` followed by the state row and the action rows: frame 0 and the state are the condition, the rest the prediction."""
    video = (torch.as_tensor(frame_ids, dtype=torch.long) > 0).long()
    robot = torch.full((1 + int(action_rows),), PREDICTION_CHUNK, dtype=torch.long)
    robot[0] = CONDITION_CHUNK
    return ConditionChunkLayout(torch.cat((video.cpu(), robot)).to(device))


class _ChunkMeans(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, slices):
        ctx.slices = slices
        return torch.stack([x[:, start:end].mean(dim=1) for (start, end) in slices], dim=1)

    @staticmethod
    def backward(ctx, grad):
        parts = [
            grad[:, chunk : chunk + 1].div(end - start).expand(-1, end - start, -1)
            for (chunk, (start, end)) in enumerate(ctx.slices)
        ]
        return (torch.cat(parts, dim=1), None)


def chunk_means(x: torch.Tensor, slices: list[tuple[int, int]]) -> torch.Tensor:
    """Return the per-chunk token means [B, chunks, C] of chunk-first [B, N, C] tokens; the backward writes each token once."""
    return _ChunkMeans.apply(x, slices)


def _compute_dtype(reference: torch.Tensor) -> torch.dtype:
    return torch.float64 if reference.dtype == torch.float64 else torch.float32


def condition_chunk_write(q, k, v, beta):
    """Return the output of the condition chunk and the state it writes. ``q``, ``k``, ``v`` are [B, heads, N, head_dim] and ``beta`` [B, heads, N]; the state is ``(scaled keys * beta)^T values``."""
    dtype = _compute_dtype(q)
    (q, k, v, beta) = (tensor.to(dtype) for tensor in (q, k, v, beta))
    key_scale = (q.shape[-1] * q.shape[2]) ** (-0.5)
    state = (k * key_scale * beta[..., None]).transpose(-1, -2) @ v
    return (q @ state, state)


def prediction_chunk_read(q, k, v, beta, decay, state):
    """Return the output of the prediction chunk: its queries read its own tokens plus the condition ``state`` scaled by ``decay`` [B, heads]."""
    dtype = _compute_dtype(q)
    (q, k, v, beta, decay) = (tensor.to(dtype) for tensor in (q, k, v, beta, decay))
    key_scale = (q.shape[-1] * q.shape[2]) ** (-0.5)
    own = (k * key_scale * beta[..., None]).transpose(-1, -2) @ v
    return q @ (own + state * decay[:, :, None, None])


class ConditionSoftmaxAttention(GatedSoftmaxAttention):
    """Softmax attention of SANA-Robot-base: the condition chunk reads itself and the prediction chunk reads both."""

    def forward(self, x, rotary_emb, layout: ConditionChunkLayout):
        (q, k, v) = self.project(x)
        dtype = v.dtype
        (q, k) = (rotate(q, rotary_emb), rotate(k, rotary_emb))
        out = self._attend_chunks(q, k, v, layout)
        return self.gate_and_project(out, x, dtype)

    def _attend_chunks(self, q, k, v, layout: ConditionChunkLayout):
        """Attend every chunk to the chunks up to itself; a prefix-caching pass stores or reads the condition's keys and values."""
        if isinstance(layout, ConditionPass):
            if layout.phase == TARGET:
                (keys, values) = layout.cache.load(self)
                return self.attend(q, torch.cat((keys, k), dim=1), torch.cat((values, v), dim=1))
            layout.cache.store(self, (k, v))
        (q, k, v) = (tensor.index_select(1, layout.order) for tensor in (q, k, v))
        out = torch.cat(
            [self.attend(q[:, start:end], k[:, :end], v[:, :end]) for (start, end) in layout.slices()], dim=1
        )
        return out.index_select(1, layout.restore)


class ConditionGatedDeltaNet(GatedDeltaNet):
    """Gated DeltaNet of SANA-Robot-base: the condition chunk writes the state, the prediction chunk reads it decayed together with its own tokens."""

    def forward(self, x, rotary_emb, layout: ConditionChunkLayout):
        (batch, tokens, channels) = x.shape
        (q, k, v) = self.qkv(x).reshape(batch, tokens, 3, channels).unbind(2)
        dtype = q.dtype

        def heads_first(tensor):
            return tensor.transpose(-1, -2).reshape(batch, self.heads, self.dim, tokens)

        def chunk_first(tensor):
            return tensor.transpose(-1, -2).index_select(2, layout.order)

        q = chunk_first(self._rotate(heads_first(self.q_norm(q)), rotary_emb))
        k = chunk_first(self._rotate(heads_first(self.k_norm(k)), rotary_emb))
        v = chunk_first(heads_first(v))
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2).index_select(2, layout.order)
        slices = layout.slices()
        if isinstance(layout, ConditionPass) and layout.phase == PREFILL:
            (out, state) = condition_chunk_write(q, k, v, beta)
            layout.cache.store(self, state)
        else:
            decay = self._prediction_decay(x.index_select(1, layout.order), slices)
            if isinstance(layout, ConditionPass):
                out = prediction_chunk_read(q, k, v, beta, decay, layout.cache.load(self))
            else:
                ((_, split), _) = slices

                def chunk(tensor, start, end):
                    return tensor[:, :, start:end]

                (condition_out, state) = condition_chunk_write(*(chunk(tensor, 0, split) for tensor in (q, k, v, beta)))
                prediction_out = prediction_chunk_read(
                    *(chunk(tensor, split, tokens) for tensor in (q, k, v, beta)), decay, state
                )
                out = torch.cat((condition_out, prediction_out), dim=2)
        out = out.to(dtype).index_select(2, layout.restore).transpose(-1, -2)
        out = self.o_norm(out).reshape(batch, channels, tokens).transpose(1, 2)
        return self.proj(out * torch.sigmoid(self.output_gate(x)))

    def _prediction_decay(self, x: torch.Tensor, slices: list[tuple[int, int]]) -> torch.Tensor:
        """Return the decay [B, heads] the prediction chunk applies to the condition state, from the mean gate logit of its tokens."""
        gate = chunk_means(self.gate_proj(x), slices).float()
        return self.decay_from_gate(gate)[:, -1]

    @staticmethod
    def _rotate(features: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Apply rotary phases to [B, heads, head_dim, N] features in float64 complex arithmetic."""
        pairs = torch.view_as_complex(features.permute(0, 1, 3, 2).to(torch.float64).unflatten(3, (-1, 2)))
        rotated = torch.view_as_real(pairs * freqs).flatten(3, 4).permute(0, 1, 3, 2)
        return rotated.type_as(features)
