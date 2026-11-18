"""Shared Triton helpers for the H3 communication quantization kernels."""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _encode_ue5m3_int8(values, GROUPS: tl.constexpr):
    """Return biased int8 values and one UE5M3 scale code per group of 32."""
    values = tl.reshape(values.to(tl.float32), GROUPS, 32)
    maximum = tl.max(tl.abs(values), axis=1)
    target_scale = maximum * (1.0 / 127.0)
    scale_bits = target_scale.to(tl.int32, bitcast=True)
    exponent = ((scale_bits >> 23) & 255) - 127
    mantissa_bits = scale_bits & 0x7FFFFF
    # Round the positive scale upward to the next 3-bit mantissa so the group
    # maximum remains representable. Eight mantissa steps carry into the
    # exponent. The unsigned 5-bit exponent is biased by 15 on the wire.
    mantissa = (mantissa_bits + 0xFFFFF) >> 20
    carry = mantissa == 8
    exponent = exponent + carry.to(tl.int32)
    mantissa = tl.where(carry, 0, mantissa)
    below = exponent < -15
    above = exponent > 16
    exponent = tl.maximum(-15, tl.minimum(16, exponent))
    mantissa = tl.where(below, 0, tl.where(above, 7, mantissa))
    encoded_scale = ((exponent + 15) << 3) | mantissa
    decoded_bits = ((exponent + 127) << 23) | (mantissa << 20)
    decoded_scale = decoded_bits.to(tl.float32, bitcast=True)
    inverse_scale = libdevice.fast_dividef(1.0, decoded_scale)
    scaled = values * inverse_scale[:, None]
    rounded = tl.where(scaled >= 0.0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
    quantized = tl.maximum(-127.0, tl.minimum(127.0, rounded)).to(tl.int32)
    return tl.reshape(quantized + 127, GROUPS * 32), encoded_scale
