"""Shim: expose FlashAttention-2's varlen API under the `flash_attn_interface`
(FA3) module name that LingBot-Video imports.

FA3 (`flash_attn_interface`) has no aarch64+Blackwell wheel and isn't on PyPI; it
only builds from the flash-attention `hopper/` source tree. FA2 (2.8.3, prebuilt in
the nunchaku_blackwell env and reused via a py3.11 clone) exposes
`flash_attn_varlen_func` with a signature identical to the code's FA3 call:
    flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q, max_seqlen_k, ..., causal=False)
so it is a drop-in. FA2 returns the output tensor directly; the caller already
handles both tuple and tensor returns.

If FA2 is unavailable (e.g. the py3.12 env), importing this module raises, and
LingBot-Video's top-level `try/except` falls back to flash_attn_varlen_func_v3=None
(the SDPA path) — i.e. safe no-op.
"""
from flash_attn import flash_attn_varlen_func as _fa2_flash_attn_varlen_func


def flash_attn_varlen_func(*args, **kwargs):
    # Drop any FA3-only kwargs FA2 doesn't accept (none used by LingBot-Video today).
    return _fa2_flash_attn_varlen_func(*args, **kwargs)
