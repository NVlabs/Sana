"""Blockwise FP8/FP6 quantized linear layers and the FP8 GEMM custom op.
The FP8 GEMM kernels live in the ``blockwise_cpp`` extension.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from ops_cpp import fp6_pack, fp6_unpack

from ltx_kernels.arch import get_device_arch
from ltx_kernels.blockwise.functional import (
    blockwise_quantize_triton,
    blockwise_quantize_weights,
    fp6_blockwise_quantize_weights_torch,
)

# Lazily-initialized, architecture-specific FP8 GEMM callable. Importing this
# module must not require a CUDA device or the compiled extension; the kernel is
# only resolved on first use (matching the lazy-import gate in ltx-core's
# blockwise.__init__).
_fp8_gemm = None


def get_fp8_gemm_nt():
    """Runtime GPU architecture dispatch for FP8 GEMM."""
    arch = get_device_arch()
    if arch in ["ada", "blackwell"]:
        # Ada/Blackwell use SM89 kernels (GeForce path)
        from blockwise_cpp import fp8_gemm_nt_sm89

        def _func(a, b, d, bias=None, c=None, num_sms=132, use_fast_accum=True):
            return fp8_gemm_nt_sm89(a, b, d, bias=bias, use_fast_accum=use_fast_accum)

        return _func
    elif arch == "hopper":
        # Hopper uses SM90 kernels (H100 path)
        from blockwise_cpp import fp8_gemm_nt_sm90

        def _func(a, b, d, bias=None, c=None, num_sms=132, use_fast_accum=True):
            return fp8_gemm_nt_sm90(a, b, d, bias=bias, c=c, num_sms=num_sms)

        return _func
    else:
        raise RuntimeError(f"Unsupported GPU architecture: {arch}")


def _fp8_gemm_dispatch(a, b, d, bias=None, c=None, num_sms=132, use_fast_accum=True):
    global _fp8_gemm
    if _fp8_gemm is None:
        _fp8_gemm = get_fp8_gemm_nt()
    return _fp8_gemm(a, b, d, bias=bias, c=c, num_sms=num_sms, use_fast_accum=use_fast_accum)


@torch.library.custom_op("blockwise::fp8_gemm", mutates_args=())
def blockwise_fp8_gemm(
    a: List[torch.Tensor],
    b: List[torch.Tensor],
    bias: Optional[torch.Tensor],
    use_fast_accum: bool,
) -> torch.Tensor:
    d = torch.empty(a[0].shape[0], b[0].shape[0], dtype=torch.bfloat16, device=b[0].device)
    _fp8_gemm_dispatch(a, b, d, bias=bias, use_fast_accum=use_fast_accum)
    return d


@blockwise_fp8_gemm.register_fake
def blockwise_fp8_gemm_fake(
    a: List[torch.Tensor],
    b: List[torch.Tensor],
    bias: Optional[torch.Tensor],
    use_fast_accum: bool,
) -> torch.Tensor:
    o = torch.empty(a[0].shape[0], b[0].shape[0], dtype=torch.bfloat16, device=b[0].device)
    return o


def is_16bit(x) -> bool:
    return x.dtype == torch.float16 or x.dtype == torch.bfloat16


class BlockwiseGemmLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: tuple[torch.Tensor, torch.Tensor],
        w: tuple[torch.Tensor],
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:

        is_16bit_a = is_16bit(a[0])
        if is_16bit_a:
            a = blockwise_quantize_triton(a[0])

        b, n, h = a[0].shape
        out_h, _ = w[0].shape
        new_a = (a[0].view(-1, h), a[1])
        d = blockwise_fp8_gemm(new_a, w, bias, use_fast_accum=True)
        return d.view(b, n, out_h)


def blockwise_linear_func(a, b, bias=None):
    return BlockwiseGemmLinearFunc.apply(a, b, bias)


class BlockwiseLinear(nn.Module):
    """Base class for blockwise quantized linear layers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,  # for compatibility with nn.Linear
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if bias:
            # FP32 to match the SM89/SM90 GEMM bias contract (the kernels take float*).
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    @property
    def fp8weight(self):
        """Should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the weight property")

    def forward(self, x):
        if not isinstance(x, tuple):
            x = (x, None)
        return blockwise_linear_func(x, (self.fp8weight, self.weight_scale), self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, transform_weights=True):
        """Should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement from_linear method")


class BlockwiseFP8Linear(BlockwiseLinear):
    """Blockwise quantized linear layer using FP8 weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)

        assert in_features % 128 == 0, f"in_features must be divisible by 128, got {in_features}"
        assert out_features % 128 == 0, f"out_features must be divisible by 128, got {out_features}"

        self.weight = nn.Parameter(
            torch.empty(
                out_features, in_features, device=device, dtype=torch.float8_e4m3fn
            ),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.ones(out_features//128, in_features//128, device=device, dtype=torch.float32),
            requires_grad=False,
        )

    @property
    def fp8weight(self):
        """Return FP8 weight directly."""
        return self.weight

    @classmethod
    def from_linear(cls, linear: nn.Linear, transform_weights=True):
        layer = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            device=linear.weight.device,
        )

        if transform_weights:
            w_fp8, w_scales = blockwise_quantize_weights(linear.weight.data.cuda())
        else:
            w_fp8 = torch.empty(linear.out_features, linear.in_features, dtype=torch.float8_e4m3fn, device=linear.weight.device)
            w_scales = torch.ones(
                w_fp8.shape[0]//128,
                w_fp8.shape[1]//128,
                device=w_fp8.device,
                dtype=torch.float32,
            )

        layer.weight.data = w_fp8
        layer.weight_scale.data = w_scales

        if linear.bias is not None:
            layer.bias.data = linear.bias.data.to(torch.float32)

        return layer


class BlockwiseFP6Linear(BlockwiseLinear):
    """Blockwise quantized linear layer using FP6 weights with 6-bit packing."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        assert in_features % 4 == 0, f"in_features must be divisible by 4, got {in_features}"
        assert in_features % 8 == 0, f"in_features must be divisible by 8 for fp6_pack, got {in_features}"
        assert in_features % 128 == 0, f"in_features must be divisible by 128, got {in_features}"
        assert out_features % 128 == 0, f"out_features must be divisible by 128, got {out_features}"

        super().__init__(in_features, out_features, bias, device, dtype)

        # Store packed weight: [out_features, (in_features // 4)*3]
        self.weight = nn.Parameter(
            torch.empty(
                out_features, (in_features // 4)*3, device=device, dtype=torch.uint8
            ),
            requires_grad=False,
        )

        # Scales are based on unpacked dimensions
        self.weight_scale = nn.Parameter(
            torch.ones(out_features//128, in_features//128, device=device, dtype=torch.float32),
            requires_grad=False,
        )

    @property
    def fp8weight(self):
        """Unpack the 6-bit weights to 8-bit format on-the-fly."""
        # Unpack from [out_features, (in_features // 4)*3] to [out_features, in_features]
        unpacked = fp6_unpack(self.weight, self.in_features)
        # Convert uint8 back to float8_e4m3fn view
        return unpacked.view(torch.float8_e4m3fn)

    @classmethod
    def from_linear(cls, linear: nn.Linear, transform_weights=True):
        layer = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            device=linear.weight.device,
        )

        if transform_weights:
            # Quantize to FP6 (actually FP8 with restricted range)
            w_fp6, w_scales = fp6_blockwise_quantize_weights_torch(linear.weight.data.cuda())

            # Pack the FP6 weights from [out, in] uint8 to [out, in*3/4] uint8
            w_fp6_uint8 = w_fp6.view(torch.uint8)
            w_packed = fp6_pack(w_fp6_uint8)

            layer.weight.data = w_packed
            layer.weight_scale.data = w_scales
        else:
            # If not transforming, assume weights are already in packed format
            layer.weight.data = torch.empty(linear.out_features, (linear.in_features // 4) * 3, dtype=torch.uint8, device=linear.weight.device)
            layer.weight_scale.data = torch.ones(
                linear.weight.shape[0]//128,
                linear.in_features//128,
                device=linear.weight.device,
                dtype=torch.float32,
            )

        if linear.bias is not None:
            layer.bias.data = linear.bias.data.to(torch.float32)

        return layer
