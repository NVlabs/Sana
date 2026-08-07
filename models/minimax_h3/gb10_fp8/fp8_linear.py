"""`Fp8Linear`: one `torch._scaled_mm` per projection, with the checkpoint's own scheme.

Quantisation in this file is per-layer and the checkpoint says so. Each quantised `Linear`
carries a `comfy_quant` blob that decodes to JSON:

    qkv_proj / out_proj / fc1  {"format": "float8_e4m3fn"}
    mlp.fc2                    {"format": "float8_e4m3fn", "full_precision_matrix_mult": true}

so `fc2` is weight-only — it ships no `input_scale`, and its post-SwiGLU activations, which
have a wide range, are not meant to be quantised at all. This honours that flag rather than
assuming one scheme for the whole model.
"""

from __future__ import annotations

import torch
from torch import nn

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0

from fusions import QUANTIZERS

class Fp8Linear(nn.Module):
    """`nn.Linear` whose weight stays FP8 E4M3 on the device.

    With an `input_scale` the activation is quantised too and the GEMM runs on the FP8
    tensor cores through `torch._scaled_mm` (measured 1.71x over BF16 on GB10 at H3's
    `qkv_proj` shape). Without one — `full_precision_matrix_mult`, i.e. `fc2` — the weight is
    dequantised once at load and the GEMM runs in the compute dtype, which is what the
    checkpoint asks for and costs 154 MB per block.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor | None,
        bias: torch.Tensor | None,
        compute_dtype: torch.dtype = torch.bfloat16,
        quantizer: str = "eager",
    ) -> None:
        super().__init__()
        self.compute_dtype = compute_dtype
        self.quantized_activations = input_scale is not None
        self._quantize = QUANTIZERS[quantizer]

        if self.quantized_activations:
            # `_scaled_mm` wants B column-major; `weight` is `[out, in]` row-major, so `.t()`
            # is already the layout it needs and no copy happens here.
            self.register_buffer("weight", weight, persistent=False)
            self.register_buffer("weight_scale", weight_scale.float().reshape(()), persistent=False)
            self.register_buffer("input_scale", input_scale.float().reshape(()), persistent=False)
        else:
            dequantized = weight.to(compute_dtype) * weight_scale.to(compute_dtype)
            self.register_buffer("weight", dequantized, persistent=False)
            self.weight_scale = None
            self.input_scale = None

        self.register_buffer(
            "bias", None if bias is None else bias.to(compute_dtype), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantized_activations:
            return nn.functional.linear(x.to(self.weight.dtype), self.weight, self.bias)

        shape = x.shape
        flat = x.reshape(-1, shape[-1]).to(self.compute_dtype)
        quantized = self._quantize(flat, self.input_scale)
        out = torch._scaled_mm(
            quantized,
            self.weight.t(),
            scale_a=self.input_scale,
            scale_b=self.weight_scale,
            out_dtype=self.compute_dtype,
        )
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], out.shape[-1])

    def extra_repr(self) -> str:
        kind = "w8a8" if self.quantized_activations else "w8a16"
        return f"{tuple(self.weight.shape)}, {kind}"


