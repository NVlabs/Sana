"""Blockwise FP8/FP6 quantization kernels.
Importing this subpackage pulls in the compiled ``ops_cpp`` / ``blockwise_cpp``
extensions (via :mod:`.functional` and :mod:`.linear`) and ``triton``, so it
raises :class:`ImportError` on hosts where the kernels were not built. Callers
that want a soft dependency should guard the import (e.g. ``pytest.importorskip``
or the lazy gate in ``ltx_core.quantization.blockwise``).
"""

from ltx_kernels.blockwise.functional import (
    blockwise_dequantize,
    blockwise_quantize_adanorm_triton,
    blockwise_quantize_rms_fma_triton,
    fp6_blockwise_quantize_weights_torch,
    fp6_pack_tensor,
    fp6_unpack_tensor,
    fp8_blockwise_quantize_weights_torch,
    gated_attention_triton,
    rms_norm_rope,
    rms_norm_split_rope,
)
from ltx_kernels.blockwise.linear import BlockwiseFP6Linear, BlockwiseFP8Linear

__all__ = [
    "BlockwiseFP6Linear",
    "BlockwiseFP8Linear",
    "blockwise_dequantize",
    "blockwise_quantize_adanorm_triton",
    "blockwise_quantize_rms_fma_triton",
    "fp6_blockwise_quantize_weights_torch",
    "fp6_pack_tensor",
    "fp6_unpack_tensor",
    "fp8_blockwise_quantize_weights_torch",
    "gated_attention_triton",
    "rms_norm_rope",
    "rms_norm_split_rope",
]
