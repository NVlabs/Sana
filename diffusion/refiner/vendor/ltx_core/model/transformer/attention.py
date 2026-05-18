from enum import Enum
from typing import Protocol

import torch

from diffusion.refiner.vendor.ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb

memory_efficient_attention = None
flash_attn_interface = None
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None
try:
    # FlashAttention3 and XFormersAttention cannot be used together
    if memory_efficient_attention is None:
        import flash_attn_interface
except ImportError:
    flash_attn_interface = None


class AttentionCallable(Protocol):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...


class PytorchAttention(AttentionCallable):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        if mask is not None:
            # add a batch dimension if there isn't already one
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a heads dimension if there isn't already one
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class XFormersAttention(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_efficient_attention is None:
            raise RuntimeError("XFormersAttention was selected but `xformers` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        # xformers expects [B, M, H, K]
        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            # add a singleton batch dimension
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a singleton heads dimension
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            # pad to a multiple of 8
            pad = 8 - mask.shape[-1] % 8
            # the xformers docs says that it's allowed to have a mask of shape (1, Nq, Nk)
            # but when using separated heads, the shape has to be (B, H, Nq, Nk)
            # in flux, this matrix ends up being over 1GB
            # here, we create a mask with the same batch/head size as the input mask (potentially singleton or full)
            mask_out = torch.empty(
                [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device
            )

            mask_out[..., : mask.shape[-1]] = mask
            # doesn't this remove the padding again??
            mask = mask_out[..., : mask.shape[-1]]
            mask = mask.expand(b, heads, -1, -1)

        out = memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, attn_bias=mask, p=0.0)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention3(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flash_attn_interface is None:
            raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            raise NotImplementedError("Mask is not supported for FlashAttention3")

        out = flash_attn_interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_3 = "flash_attention_3"
    DEFAULT = "default"

    def to_callable(self) -> AttentionCallable:
        """Resolve to a concrete callable. Use this at module init time so that
        torch.compile can trace through the attention call without graph breaks."""
        if self is AttentionFunction.PYTORCH:
            return PytorchAttention()
        elif self is AttentionFunction.XFORMERS:
            return XFormersAttention()
        elif self is AttentionFunction.FLASH_ATTENTION_3:
            return FlashAttention3()
        else:
            # Default behavior: XFormers if installed else - PyTorch
            return XFormersAttention() if memory_efficient_attention is not None else PytorchAttention()


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        attention_function: AttentionCallable | AttentionFunction = AttentionFunction.DEFAULT,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = (
            attention_function.to_callable()
            if isinstance(attention_function, AttentionFunction)
            else attention_function
        )

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        # Optional per-head gating
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
        kv_cache_prefix: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # Snapshot of self-attn vs cross-attn signal BEFORE we rebind context to x.
        # Ulysses context-parallel is applied only to self-attention on the video
        # sequence dimension; cross-attention to text (attn2) runs with full text
        # K/V per rank and local query shard, which needs no all-to-all.
        _is_self_attn_for_cp = context is None
        """Multi-head attention with optional RoPE, perturbation masking, and per-head gating.
        When ``perturbation_mask`` is all zeros, the expensive query/key path
        (linear projections, RMSNorm, RoPE) is skipped entirely and only the
        value projection is used as a pass-through.
        Args:
            x: Query input tensor of shape ``(B, T, query_dim)``.
            context: Key/value context tensor of shape ``(B, S, context_dim)``.
                Falls back to ``x`` (self-attention) when *None*.
            mask: Optional attention mask. Interpretation depends on the attention
                backend (additive bias for xformers/PyTorch SDPA).
            pe: Rotary positional embeddings applied to both ``q`` and ``k``.
            k_pe: Separate rotary positional embeddings for ``k`` only. When
                *None*, ``pe`` is reused for keys.
            perturbation_mask: Optional mask in ``[0, 1]`` that
                blends the attention output with the raw value projection:
                ``out = attn_out * mask + v * (1 - mask)``.
                **1** keeps the full attention output, **0** bypasses attention
                and passes the value projection through unchanged.
                *None* or all-ones means standard attention; all-zeros skips
                the query/key path entirely for efficiency.
            all_perturbed: Whether all perturbations are active for this block.
        Returns:
            Output tensor of shape ``(B, T, query_dim)``.
        """
        context = x if context is None else context
        use_attention = not all_perturbed

        v = self.to_v(context)

        if not use_attention:
            out = v
        else:
            q = self.to_q(x)
            k = self.to_k(context)

            q = self.q_norm(q)
            k = self.k_norm(k)

            # KV cache capture: save pre-RoPE (post-norm) K,V for streaming.
            # Storing pre-RoPE allows re-encoding with correct positions when
            # the cache is injected into a different window.
            if getattr(self, "_kv_cache_capture", False):
                self._cached_kv = (k.detach(), v.detach())

            _k_pe = pe if k_pe is None else k_pe
            if pe is not None:
                q = apply_rotary_emb(q, pe, self.rope_type)
                k = apply_rotary_emb(k, _k_pe, self.rope_type)

            # Teacher-forcing: capture post-RoPE K, V for context caching.
            # Post-RoPE means absolute 3D positions are already baked into K,
            # so the target pass can directly concatenate without RoPE re-application.
            if getattr(self, "_tf_capture_kv", False):
                self._tf_cached_kv = (k.detach().clone(), v.detach().clone())

            # Teacher-forcing: inject cached post-RoPE K, V as prefix.
            # Target chunk queries attend to context K/V + self K/V.
            _tf_prefix = getattr(self, "_tf_kv_prefix", None)
            if _tf_prefix is not None:
                _tf_k, _tf_v = _tf_prefix
                k = torch.cat([_tf_k.to(k.dtype), k], dim=1)
                v = torch.cat([_tf_v.to(v.dtype), v], dim=1)
                if mask is not None:
                    pl = _tf_k.shape[1]
                    # 0.0 = unrestricted in additive log-space (target sees all context)
                    prefix_cols = torch.zeros(
                        mask.shape[0],
                        mask.shape[1],
                        mask.shape[2],
                        pl,
                        device=mask.device,
                        dtype=mask.dtype,
                    )
                    mask = torch.cat([prefix_cols, mask], dim=-1)

            # Keep original v (without prefix) for perturbation_mask blending
            v_orig = v

            # KV cache inject: prepend cached K, V from previous streaming window.
            # Cached K is pre-RoPE — apply current window's RoPE for the overlap
            # positions so that positional encoding is consistent.
            _kv_prefix = kv_cache_prefix or getattr(self, "_kv_cache_prefix", None)
            if _kv_prefix is not None:
                cached_k, cached_v = _kv_prefix
                prefix_len = cached_k.shape[1]
                # Re-apply RoPE with current window's overlap positions.
                # pe layout: SPLIT → (B, H, T, D) with T at dim 2;
                #            INTERLEAVED → (B, T, D) with T at dim 1.
                if pe is not None:
                    t_dim = 2 if _k_pe[0].ndim == 4 else 1
                    prefix_pe = (
                        _k_pe[0].narrow(t_dim, 0, prefix_len),
                        _k_pe[1].narrow(t_dim, 0, prefix_len),
                    )
                    cached_k = apply_rotary_emb(cached_k.to(k.dtype), prefix_pe, self.rope_type)
                k = torch.cat([cached_k.to(k.dtype), k], dim=1)  # (B, S_cache + T, H*D)
                v = torch.cat([cached_v.to(v.dtype), v], dim=1)
                # Expand mask: prefix tokens share temporal position with the
                # first `prefix_len` tokens of the current window (overlap region),
                # so duplicate those mask columns for the prefix.
                if mask is not None:
                    prefix_mask_cols = mask[..., :prefix_len]
                    mask = torch.cat([prefix_mask_cols, mask], dim=-1)

            # Ulysses context parallel: when self-attention is requested and a cp
            # group is active, split heads across the cp group while gathering the
            # full sequence for the attention call, then reverse so the output is
            # in local-seq layout again.
            _use_cp_attn = False
            if _is_self_attn_for_cp:
                from diffusion.refiner.ltx_core.context_parallel import (
                    cp_enabled,
                    get_cp_group,
                    get_cp_size,
                    head_to_seq,
                    seq_to_head,
                )

                _use_cp_attn = cp_enabled()

            if _use_cp_attn:
                if mask is not None:
                    # Self-attention masks + CP is not wired up yet (bidirectional
                    # training uses mask=None, so we don't need it for smoke).
                    raise NotImplementedError("Ulysses CP with a non-None self-attention mask is not implemented")
                cp_size = get_cp_size()
                _b, _t_loc, _hd = q.shape
                _h_full = self.heads
                _d_head = self.dim_head
                if _h_full % cp_size != 0:
                    raise ValueError(f"cp_size={cp_size} does not divide heads={_h_full}")
                _h_loc = _h_full // cp_size
                _t_full = _t_loc * cp_size

                # q, k, v are [B, T_loc, H*D]; reshape to [B, T_loc, H, D] for all-to-all.
                q4 = q.view(_b, _t_loc, _h_full, _d_head)
                k4 = k.view(_b, _t_loc, _h_full, _d_head)
                v4 = v.view(_b, _t_loc, _h_full, _d_head)

                # seq→head: [B, T_loc, H_full, D] -> [B, T_full, H_loc, D]
                q4 = seq_to_head(q4)
                k4 = seq_to_head(k4)
                v4 = seq_to_head(v4)

                # Back to [B, T_full, H_loc*D] for the attention callable, which
                # reshapes to [B, T_full, H_loc, D] internally.
                q_full = q4.reshape(_b, _t_full, _h_loc * _d_head)
                k_full = k4.reshape(_b, _t_full, _h_loc * _d_head)
                v_full = v4.reshape(_b, _t_full, _h_loc * _d_head)

                out_full = self.attention_function(q_full, k_full, v_full, _h_loc, None)
                # Back to [B, T_full, H_loc, D] -> head→seq -> [B, T_loc, H_full, D]
                out4 = out_full.view(_b, _t_full, _h_loc, _d_head)
                out4 = head_to_seq(out4)
                out = out4.reshape(_b, _t_loc, _h_full * _d_head)
            else:
                out = self.attention_function(q, k, v, self.heads, mask)  # (B, T, H*D)

            if perturbation_mask is not None:
                out = out * perturbation_mask + v_orig * (1 - perturbation_mask)

        # Apply per-head gating if enabled
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)  # (B, T, H)
            b, t, _ = out.shape
            # Reshape to (B, T, H, D) for per-head gating
            out = out.view(b, t, self.heads, self.dim_head)
            # Apply gating: 2 * sigmoid(x) so that zero-init gives identity (2 * 0.5 = 1.0)
            gates = 2.0 * torch.sigmoid(gate_logits)  # (B, T, H)
            out = out * gates.unsqueeze(-1)  # (B, T, H, D) * (B, T, H, 1)
            # Reshape back to (B, T, H*D)
            out = out.view(b, t, self.heads * self.dim_head)

        return self.to_out(out)
