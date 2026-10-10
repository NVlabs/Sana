"""Variable-length attention wrappers with optional kernel acceleration.

Supports packed ``[L, N, C]`` and batched ``[B, L, N, C]`` layouts.
FlashAttention 4 is preferred on SM100 and SM110 devices, FlashAttention 3 is
preferred on SM90, and FlashAttention 2 is used otherwise. Calls can fall back
to scaled dot-product attention when no backend is usable. The packed fallback
preserves separate query and key sequence boundaries for self-attention,
cross-attention, and cached reads.

The module provides both ``nn.Module`` and functional interfaces and preserves
the input dtype on output.
"""
import os
import warnings

import torch
import torch.nn as nn

try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_hopper
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_hopper
    FLASH_ATTN_3_AVAILABLE = bool(int(os.environ.get("FLASH_ATTN_3_AVAILABLE", "1")))
except ModuleNotFoundError as error:
    if error.name != "flash_attn_interface":
        raise
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn as flash_attn_module
except ModuleNotFoundError as error:
    if error.name != "flash_attn":
        raise
    FLASH_ATTN_2_AVAILABLE = False
else:
    flash_attn_func = getattr(flash_attn_module, "flash_attn_func", None)
    flash_attn_varlen_func = getattr(
        flash_attn_module, "flash_attn_varlen_func", None
    )
    FLASH_ATTN_2_AVAILABLE = (
        flash_attn_varlen_func is not None
        and bool(int(os.environ.get("FLASH_ATTN_2_AVAILABLE", "1")))
    )

try:
    from flash_attn.cute import flash_attn_func as flash_attn_func_blackwell
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_varlen_func_blackwell
    FLASH_ATTN_4_AVAILABLE = bool(int(os.environ.get("FLASH_ATTN_4_AVAILABLE", "1")))
except ModuleNotFoundError as error:
    missing = error.name or ""
    if missing not in ("flash_attn", "flash_attn.cute", "cutlass") and not missing.startswith(
        "cutlass."
    ):
        raise
    FLASH_ATTN_4_AVAILABLE = False

_FA4_NUM_SPLITS = int(os.environ.get("H3_FA4_NUM_SPLITS", "1"))
if _FA4_NUM_SPLITS < 0:
    raise ValueError("H3_FA4_NUM_SPLITS must be non-negative (0 selects auto)")


def _flash_attn_fallback_name():
    if FLASH_ATTN_2_AVAILABLE:
        return "flash attention 2"
    return "scaled dot-product attention"


def _fa4_shape_supported(capability, q_head_dim, v_head_dim):
    if capability == (9, 0):
        return (
            8 <= q_head_dim <= 256
            and 8 <= v_head_dim <= 256
            and q_head_dim % 8 == 0
            and v_head_dim % 8 == 0
        )
    if capability[0] not in (10, 11):
        return False
    standard = (
        8 <= q_head_dim <= 128
        and 8 <= v_head_dim <= 128
        and q_head_dim % 8 == 0
        and v_head_dim % 8 == 0
    )
    special = (q_head_dim, v_head_dim) in ((192, 128), (256, 256), (64, 512))
    return standard or special


def _select_flash_attn_version(
    device, dropout_p, version, q_head_dim, v_head_dim, warn=True
):
    if version not in (None, 2, 3, 4):
        raise ValueError(f"unsupported flash attention version: {version}")
    capability = None
    if version in (None, 4) and device.type == 'cuda':
        capability = torch.cuda.get_device_capability(device)
    if version == 4:
        reason = None
        if not FLASH_ATTN_4_AVAILABLE:
            reason = "Flash attention 4 is not available"
        elif dropout_p != 0:
            reason = "Flash attention 4 does not support dropout"
        elif capability is None or not _fa4_shape_supported(
            capability, q_head_dim, v_head_dim
        ):
            reason = (
                f"Flash attention 4 does not support q/v head dimensions "
                f"{q_head_dim}/{v_head_dim} on {device}"
            )
        if reason is None:
            return 4
        if warn:
            warnings.warn(
                f"{reason}, use {_flash_attn_fallback_name()} instead."
            )
        return 2 if FLASH_ATTN_2_AVAILABLE else None
    if version == 3:
        if FLASH_ATTN_3_AVAILABLE:
            return 3
        if warn:
            warnings.warn(
                "Flash attention 3 is not available, use "
                f"{_flash_attn_fallback_name()} instead."
            )
        return 2 if FLASH_ATTN_2_AVAILABLE else None
    if version == 2:
        return 2 if FLASH_ATTN_2_AVAILABLE else None
    if capability is None:
        return None
    if (
        capability[0] in (10, 11)
        and FLASH_ATTN_4_AVAILABLE
        and dropout_p == 0
        and _fa4_shape_supported(capability, q_head_dim, v_head_dim)
    ):
        return 4
    if capability == (9, 0) and FLASH_ATTN_3_AVAILABLE:
        return 3
    if FLASH_ATTN_2_AVAILABLE:
        return 2
    return None


# ===== PACKED SDPA FALLBACK BEGIN =====
# FlashAttention kernels require CUDA; this path preserves packed sequence
# boundaries, scaling, masking, and input dtype when no backend is usable.
def _bottom_right_mask(q_len, k_len, *, causal, window_size, device):
    left, right = window_size
    has_window = (
        left is not None and left >= 0
        or right is not None and right >= 0
    )
    if not has_window and (not causal or q_len == k_len):
        return None
    q_positions = torch.arange(q_len, device=device) + k_len - q_len
    k_positions = torch.arange(k_len, device=device)
    mask = torch.ones((q_len, k_len), dtype=torch.bool, device=device)
    if causal:
        mask &= k_positions[None, :] <= q_positions[:, None]
    if left is not None and left >= 0:
        mask &= k_positions[None, :] >= q_positions[:, None] - left
    if right is not None and right >= 0:
        mask &= k_positions[None, :] <= q_positions[:, None] + right
    return mask


def _sdpa_batched(
    q,
    k,
    v,
    *,
    dropout_p,
    softmax_scale,
    q_scale,
    causal,
    window_size,
    dtype,
):
    if q.ndim == 4 and not causal:
        raise ValueError("batched SDPA requires causal=True")
    out_dtype = q.dtype
    q_len, k_len = q.size(1), k.size(1)
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    if q_scale is not None:
        q = q * q_scale
    attn_mask = _bottom_right_mask(
        q_len, k_len, causal=causal, window_size=window_size, device=q.device
    )
    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal and attn_mask is None,
        scale=softmax_scale,
    )
    return out.transpose(1, 2).contiguous().type(out_dtype)


def _sdpa_varlen(
    q,
    k,
    v,
    *,
    q_bounds,
    k_bounds,
    dropout_p,
    softmax_scale,
    q_scale,
    causal,
    window_size,
):
    """Attention over packed [L, N, C] rows, one document per [start, stop).

    Query and key bounds are separate, so this serves cross-attention and cached
    causal reads as well as self-attention. A shared bound list would slice k and
    v with query offsets whenever the layouts differ, including text
    cross-attention (``k_lens = text_lens``) and cached reads
    (``k_lens = q_lens + history``).
    """
    out = torch.empty_like(q)
    pairs = zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:])
    for q_start, q_stop, k_start, k_stop in pairs:
        if q_start == q_stop:
            continue
        q_segment = q[q_start:q_stop].transpose(0, 1)
        if q_scale is not None:
            q_segment = q_segment * q_scale
        q_len, k_len = q_stop - q_start, k_stop - k_start
        attn_mask = _bottom_right_mask(
            q_len,
            k_len,
            causal=causal,
            window_size=window_size,
            device=q.device,
        )
        segment = torch.nn.functional.scaled_dot_product_attention(
            q_segment,
            k[k_start:k_stop].transpose(0, 1),
            v[k_start:k_stop].transpose(0, 1),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=causal and attn_mask is None,
            scale=softmax_scale,
        ).transpose(0, 1)
        out[q_start:q_stop].copy_(segment)
    return out
# ===== PACKED SDPA FALLBACK END =====


def _flash_attention_batched(
    q, k, v, *, dropout_p, softmax_scale, q_scale, causal, window_size,
    deterministic, dtype, version, impl=None,
):
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256
    out_dtype = q.dtype
    q, k, v = (x if x.dtype in half_dtypes else x.to(dtype) for x in (q, k, v))
    q, k = q.to(v.dtype), k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale
    kwargs = dict(q=q, k=k, v=v, softmax_scale=softmax_scale,
                  causal=causal, deterministic=deterministic)
    if version == 4:
        kwargs.update(window_size=(None, None) if window_size == (-1, -1) else window_size,
                      num_splits=1, return_lse=False)
    elif version == 2:
        kwargs.update(dropout_p=dropout_p, window_size=window_size)
    if impl is not None:
        output = impl(version=version, **kwargs)
    elif version == 4:
        output = flash_attn_func_blackwell(**kwargs)
    elif version == 3:
        output = flash_attn_func_hopper(**kwargs)
    else:
        output = flash_attn_func(**kwargs)
    output = output[0] if isinstance(output, tuple) else output
    return output.type(out_dtype)


class _FlashAttentionVarlen(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def tflops(self, args, kwargs, output) -> float:
        causal = kwargs.get("causal", False)
        if output.ndim == 4:
            b, lq, h, d = output.shape
            seqlens_q = torch.full((b,), lq, device=output.device)
            seqlens_k = torch.full((b,), kwargs["k"].size(1), device=output.device)
        else:
            cu_seqlens_q = kwargs["cu_seqlens_q"]
            cu_seqlens_k = kwargs["cu_seqlens_k"]
            _, h, d = output.shape
            seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        if causal:
            min_s = torch.min(seqlens_q, seqlens_k)
            square_part = ((1 + min_s) * min_s) / 2
            full_part = torch.relu(seqlens_q - seqlens_k) * min_s
            valid = (full_part / 1e12 + square_part / 1e12).sum()
        else:
            valid = ((seqlens_q / 1e6) * (seqlens_k / 1e6)).sum()
        return h * (4 * d * valid)

    def forward(self, *args, **kwargs):
        version = kwargs.pop("version", None)
        q = kwargs["q"] if "q" in kwargs else args[0]
        # apply attention
        if version == 4:
            assert FLASH_ATTN_4_AVAILABLE
            func = flash_attn_func_blackwell if q.ndim == 4 else flash_attn_varlen_func_blackwell
            output = func(*args, **kwargs)
        elif version == 3:
            assert FLASH_ATTN_3_AVAILABLE
            func = flash_attn_func_hopper if q.ndim == 4 else flash_attn_varlen_func_hopper
            output = func(*args, **kwargs)
        elif version == 2:
            assert FLASH_ATTN_2_AVAILABLE
            func = flash_attn_func if q.ndim == 4 else flash_attn_varlen_func
            output = func(*args, **kwargs)
        else:
            raise ValueError(f"unsupported flash attention version: {version}")
        return output[0] if isinstance(output, tuple) else output


class FlashAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._flash_attn_impl = _FlashAttentionVarlen()

    def forward(
        self,
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
    ):
        """
        q:              [B, Lq, Nq, C1] or packed [L, Nq, C1].
        k:              [B, Lk, Nk, C1] or packed [L, Nk, C1].
        v:              [B, Lk, Nk, C2] or packed [L, Nk, C2]. Nq must be divisible by Nk.
        q_lens:         [B].
        k_lens:         [B].
        dropout_p:      float. Dropout probability.
        softmax_scale:  float. The scaling of QK^T before applying softmax.
        causal:         bool. Whether to apply causal attention mask.
        window_size:    (left right). If not (-1, -1), apply sliding window local attention.
        deterministic:  bool. If True, slightly slower and uses more memory.
        dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
        """
        flash_attn_version = _select_flash_attn_version(
            q.device, dropout_p, version, q.size(-1), v.size(-1)
        )
        # ===== PACKED SDPA DISPATCH BEGIN =====
        # Without a usable backend, use SDPA while preserving packed sequence
        # boundaries, scale, masking mode, and dtype.
        if flash_attn_version is None:
            if q.ndim == 4:
                if q_lens is not None or k_lens is not None:
                    warnings.warn(
                        'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
                    )
                return _sdpa_batched(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    q_scale=q_scale,
                    causal=causal,
                    window_size=window_size,
                    dtype=dtype,
                )
            assert q.ndim == 3, "the SDPA fallback expects packed or batched attention"
            assert q_lens is not None, "q_lens is required for packed attention"
            k_lens_effective = q_lens if k_lens is None else k_lens
            if len(k_lens_effective) != len(q_lens):
                raise ValueError(
                    f"q_lens has {len(q_lens)} documents but k_lens has "
                    f"{len(k_lens_effective)}; they describe the same packing"
                )
            q_bounds = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0).tolist()
            k_bounds = (
                torch.cat([k_lens_effective.new_zeros([1]), k_lens_effective])
                .cumsum(0)
                .tolist()
            )
            return _sdpa_varlen(
                q, k, v,
                q_bounds=q_bounds,
                k_bounds=k_bounds,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
            )
        # ===== PACKED SDPA DISPATCH END =====
        if q.ndim == k.ndim == v.ndim == 4 and q_lens is None and k_lens is None:
            return _flash_attention_batched(
                q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale,
                q_scale=q_scale, causal=causal, window_size=window_size,
                deterministic=deterministic, dtype=dtype, version=flash_attn_version,
                impl=self._flash_attn_impl,
            )
        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        assert q.device.type == 'cuda' and q.size(-1) <= 256

        # params
        # b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
        out_dtype = q.dtype

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        # preprocess query
        q_is_packed = q.ndim == 3
        if q_is_packed:  # packed
            assert q_lens is not None, "q_lens must be provided when q is packed."
            b = len(q_lens)
            # Move the lengths to the host before reducing. Python's max()
            # iterates a CUDA tensor element by element and every comparison
            # forces a device->host sync, so a packed call cost B-1 pipeline
            # drains -- ~23% of the training step on a 16-layer backbone under
            # gradient checkpointing. One small D2H copy replaces all of them.
            lq = int(q_lens.cpu().max())
            q = half(q)
        else:
            b, lq = q.size(0), q.size(1)
            if q_lens is None:
                q = half(q.flatten(0, 1))
                q_lens = torch.tensor(
                    [lq] * b, dtype=torch.int32).to(
                        device=q.device, non_blocking=True)
            else:
                q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))
        # q_lens = torch.tensor(q_lens, dtype=torch.int32, device=q.device)
        # out_dtype = q.dtype

        # preprocess key, value
        kv_is_packed = k.ndim == 3
        if kv_is_packed:
            assert k_lens is not None, "k_lens must be provided when k is packed."
            lk = int(k_lens.cpu().max())
            k = half(k)
            v = half(v)
        else:
            lk = k.size(1)
            if k_lens is None:
                k = half(k.flatten(0, 1))
                v = half(v.flatten(0, 1))
                k_lens = torch.tensor(
                    [lk] * b, dtype=torch.int32).to(
                        device=k.device, non_blocking=True)
            else:
                k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
                v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if q_scale is not None:
            q = q * q_scale

        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
            0, dtype=torch.int32).to(q.device, non_blocking=True)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
            0, dtype=torch.int32).to(q.device, non_blocking=True)

        # apply attention
        if flash_attn_version == 4:
            kwargs = dict(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic,
                window_size=(None, None) if window_size == (-1, -1) else window_size,
                num_splits=_FA4_NUM_SPLITS,
                return_lse=False)
        elif flash_attn_version == 3:
            # Note: dropout_p, window_size are not supported in FA3 now.
            kwargs = dict(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic)
        else:
            kwargs = dict(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic)
        x = self._flash_attn_impl(version=flash_attn_version, **kwargs)

        # output
        if q_is_packed:
            return x.type(out_dtype)
        else:
            return x.unflatten(0, (b, lq)).type(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    flash_attn_version = _select_flash_attn_version(
        q.device, dropout_p, version, q.size(-1), v.size(-1)
    )
    if flash_attn_version is None:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        return _sdpa_batched(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            dtype=dtype,
        )

    if q.ndim == k.ndim == v.ndim == 4 and q_lens is None and k_lens is None:
        return _flash_attention_batched(
            q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale,
            q_scale=q_scale, causal=causal, window_size=window_size,
            deterministic=deterministic, dtype=dtype, version=flash_attn_version,
        )

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
        0, dtype=torch.int32).to(q.device, non_blocking=True)
    cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
        0, dtype=torch.int32).to(q.device, non_blocking=True)

    # apply attention
    if flash_attn_version == 4:
        output = flash_attn_varlen_func_blackwell(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
            window_size=(None, None) if window_size == (-1, -1) else window_size,
            num_splits=_FA4_NUM_SPLITS,
            return_lse=False)
    elif flash_attn_version == 3:
        # Note: dropout_p, window_size are not supported in FA3 now.
        output = flash_attn_varlen_func_hopper(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)
    else:
        output = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic)
    x = output[0] if isinstance(output, tuple) else output
    x = x.unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )
