"""Process-local SM120 fix for FlashAttention-4 (4.0.0b19-b22) varlen forward, used by the Sol-H3-VAE decoder.

On SM120/SM121 (RTX 5090, DGX Spark) surplus scheduler work tiles could read sequence metadata past the batch.
fa4_sm120_flash_fwd.py is FA4's flash_fwd.py with that guard added (BSD-3-Clause, licenses/flash-attention.txt);
the installed FA4 package and its valid-tile computation stay unchanged. Other GPUs are left alone.
"""


def install():
    import torch

    if torch.cuda.get_device_capability()[0] != 12:
        return False
    from flash_attn.cute import interface
    from flash_attn.cute.cache_utils import JITCache
    from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120
    from fa4_sm120_flash_fwd import FlashAttentionForwardSm80

    if getattr(interface.FlashAttentionForwardSm120, "_solh3_guarded", False):
        return True

    class GuardedFlashAttentionForwardSm120(FlashAttentionForwardSm120):
        _solh3_guarded = True
        kernel = FlashAttentionForwardSm80.kernel

    interface.FlashAttentionForwardSm120 = GuardedFlashAttentionForwardSm120
    # Never reuse an unguarded binary from an existing in-memory/disk cache.
    interface._flash_attn_fwd.compile_cache = JITCache()
    return True
