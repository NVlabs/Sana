# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import weakref

import torch

from sana_robot.kernels.inference.ops.dispatch import SM90, SM110, SM120, device_architecture
from sana_robot.policy.attention.causal_cross_attention import ChunkTextCrossAttention, ChunkTextSpans
from sana_robot.policy.attention.cross_attention import MultiHeadCrossAttention, PackedGroupedText

SCRATCH_KEY_ROWS = 16


def cross_norm_kernels(device: torch.device):
    """Return the module that holds the q / k norm kernel of text cross-attention for the device's architecture (``supports`` and ``cross_qk_norm``), or None for a device without one."""
    architecture = device_architecture(device)
    if architecture == SM110:
        from sana_robot.kernels.inference.ops.cross_qk_norm import sm110 as kernels
    elif architecture in (SM90, SM120):
        from sana_robot.kernels.inference.ops.cross_qk_norm import sm120 as kernels
    else:
        return None
    return kernels


class TextCrossAttentionKernels:
    """The inference forms of a text cross-attention over a packed text, each in use once its install function has set it up: the keys and values of a packed text kept between forwards (``caption_cache``) and the query and key RMS norms as one in-place kernel launch (``norm_kernel``)."""

    caption_cache = None
    norm_kernel = None

    def cached_keys_values(self, text):
        """Return the normalized keys and the values kept for a packed text tensor, or None at its first forward."""
        entry = self.caption_cache.get(id(text))
        if entry is None or entry[0]() is not text:
            return None
        return entry[1]

    def keep_keys_values(self, text, keys, values) -> None:
        """Keep the normalized keys and the values of a packed text tensor for as long as that tensor lives. They must come from an eager forward: a captured CUDA graph would compute them into memory of its own at every replay."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "a text cross-attention met a packed text for the first time inside a CUDA graph capture; "
                "run the forward eagerly once before capturing it"
            )
        cache = self.caption_cache
        cache[id(text)] = (weakref.ref(text), (keys, values))
        weakref.finalize(text, cache.pop, id(text), None)

    def attend_packed_text(self, x, text, spans, leading_shape: tuple):
        """Attend every token span to its span of the packed text in one variable-length attention call. ``leading_shape`` is what the module's own forward puts ahead of [text tokens, channels] when it normalizes the keys."""
        from flash_attn import flash_attn_varlen_func

        channels = x.shape[-1]
        head_shape = (-1, self.num_heads, self.head_dim)
        queries = self.q_linear(x)
        fused_norm = self.norm_kernel is not None and queries.dtype == torch.bfloat16
        norm_arguments = (self.q_norm.weight, self.k_norm.weight, self.q_norm.eps)
        caching = self.caption_cache is not None
        keys_values = self.cached_keys_values(text) if caching else None
        if keys_values is None:
            if fused_norm:
                (keys, values) = self.kv_linear(text).view(-1, 2, channels).unbind(1)
                query_rows = queries.view(-1, channels)
                self.norm_kernel(query_rows.clone() if caching else query_rows, keys, *norm_arguments)
            else:
                (keys, values) = self.kv_linear(text).view(*leading_shape, -1, 2, channels).unbind(-2)
                keys = self.k_norm(keys)
            keys_values = (keys.view(head_shape), values.reshape(head_shape))
            if caching:
                self.keep_keys_values(text, *keys_values)
        if not fused_norm:
            queries = self.q_norm(queries)
        elif caching:
            self.norm_kernel(queries.view(-1, channels), self.norm_scratch_keys, *norm_arguments)
        (keys, values) = keys_values
        out = flash_attn_varlen_func(
            queries.view(head_shape),
            keys.to(queries.dtype),
            values.to(queries.dtype),
            cu_seqlens_q=spans.cu_seqlens_q,
            cu_seqlens_k=spans.cu_seqlens_k,
            max_seqlen_q=int(spans.max_seqlen_q),
            max_seqlen_k=int(spans.max_seqlen_k),
            dropout_p=0.0,
        )
        return self.proj(out.reshape(x.shape))


class FusedChunkTextCrossAttention(TextCrossAttentionKernels, ChunkTextCrossAttention):
    """``ChunkTextCrossAttention`` with the inference forms of ``TextCrossAttentionKernels``; training, a forward with gradients and a text with shared rows run the original forward."""

    def forward(self, x, text, text_spans: ChunkTextSpans):
        if self.training or torch.is_grad_enabled() or text_spans.kv_index is not None:
            return super().forward(x, text, text_spans)
        return self.attend_packed_text(x, text, text_spans, leading_shape=(1,))


class FusedMultiHeadCrossAttention(TextCrossAttentionKernels, MultiHeadCrossAttention):
    """``MultiHeadCrossAttention`` with the inference forms of ``TextCrossAttentionKernels`` for a packed grouped text; text groups that arrive unpacked run the original forward."""

    def _attend_packed(self, x, text, packed: PackedGroupedText):
        if self.training or torch.is_grad_enabled():
            return super()._attend_packed(x, text, packed)
        return self.attend_packed_text(x, text, packed, leading_shape=())


FUSED_CLASSES = {
    ChunkTextCrossAttention: FusedChunkTextCrossAttention,
    MultiHeadCrossAttention: FusedMultiHeadCrossAttention,
}


def fused_cross_attentions(policy) -> list:
    """Return the text cross-attention of every layer in its fused class, swapping the original modules in place."""
    modules = []
    for block in policy.blocks:
        module = block.cross_attn
        module.__class__ = FUSED_CLASSES.get(type(module), type(module))
        if isinstance(module, TextCrossAttentionKernels):
            modules.append(module)
    return modules


def install_caption_cache(policy) -> None:
    """Project and normalize the keys and values of a caption once: every text cross-attention keeps them at the first forward of a packed text and reuses them at every later one, bitwise equal to projecting them again.

    Args:
        policy: SANA-Robot-AR or SANA-Robot-base in eval mode on its CUDA device with its final weights, before any
            CUDA graph is captured. The cache is keyed by the packed text tensor, so the policy's memo of the packed
            text (``packed_text_cache``) is switched on: a caption tensor then keeps one packed text, and a new
            caption or a new packed text is projected again. The first forward of a packed text must run eagerly.
            A policy on another device stays as it is.
    """
    if next(policy.parameters()).device.type != "cuda":
        return
    if getattr(policy, "packed_text_cache", None) is None:
        policy.packed_text_cache = {}
    for module in fused_cross_attentions(policy):
        module.caption_cache = {}


def install_fused_cross_norm(policy) -> None:
    """Run the query norm and the key norm of every text cross-attention as one in-place kernel launch.

    Args:
        policy: SANA-Robot-AR or SANA-Robot-base in eval mode on its CUDA device with its final bf16 weights. The
            kernel serves the forwards over a packed text. With the caption cache installed it normalizes the keys
            at the first forward of a packed text and afterwards only the queries, next to ``SCRATCH_KEY_ROWS`` rows
            of zeros, the fewest key rows the sm120 kernel takes (the buffer ``norm_scratch_keys``, not saved in
            checkpoints). A device without a kernel leaves the policy as it is.
    """
    device = next(policy.parameters()).device
    kernels = cross_norm_kernels(device)
    if kernels is None or not kernels.supports(policy.hidden_size):
        return
    for module in fused_cross_attentions(policy):
        scratch = torch.zeros(SCRATCH_KEY_ROWS, policy.hidden_size, dtype=torch.bfloat16, device=device)
        module.register_buffer("norm_scratch_keys", scratch, persistent=False)
        module.norm_kernel = kernels.cross_qk_norm
