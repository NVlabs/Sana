# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sana_robot.kernels.inference.install.gdn import (
    FusedCausalGatedDeltaNet,
    FusedConditionGatedDeltaNet,
    install_fused_gdn,
)
from sana_robot.kernels.inference.install.softmax import (
    FusedCausalSoftmaxAttention,
    FusedConditionSoftmaxAttention,
    install_packed_qkv_gate,
)
from sana_robot.kernels.inference.install.swiglu import FusedSwiGLU, install_fused_swiglu
from sana_robot.kernels.inference.ops.dispatch import SM120, device_architecture
from sana_robot.kernels.inference.ops.fp8 import portable, sm120


def quantized_products_serve(x: torch.Tensor) -> bool:
    """Return whether the FP8 or NVFP4 matrix products serve a call: bf16 tokens outside autograd."""
    return x.dtype == torch.bfloat16 and not torch.is_grad_enabled()


def install_packed_projections(policy) -> None:
    """Install the packed bf16 projections that the FP8 and NVFP4 weights are quantized from, where the kernels of the device left them out: the input projection of the Gated DeltaNet layers, the qkv and output gate projection of the softmax layers and the gate and up projection of the MLPs."""
    install_fused_gdn(policy)
    install_packed_qkv_gate(policy)
    install_fused_swiglu(policy)


def fp8_projection(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias) -> torch.Tensor:
    """Return ``x @ weight^T + bias`` [..., out] in bf16 for bf16 ``x`` [..., in] and an E4M3 weight [out, in] with its scale, as one FP8 matrix product over all rows of ``x``."""
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    return portable.fp8_linear(rows, weight, weight_scale, bias).reshape(*x.shape[:-1], weight.shape[0])


class Fp8InputProjection:
    """The packed input projection of a fused Gated DeltaNet layer as one FP8 matrix product on ``fp8_input_weight``; a call that FP8 does not serve multiplies the bf16 packed weight."""

    def project_input(self, x: torch.Tensor) -> torch.Tensor:
        if not quantized_products_serve(x):
            return super().project_input(x)
        return fp8_projection(x, self.fp8_input_weight, self.fp8_input_scale, self.packed_input_bias)


class Fp8QkvGate:
    """The packed qkv and output gate projection of a fused softmax layer as one FP8 matrix product on ``fp8_qkv_gate_weight``; a call that FP8 does not serve multiplies the bf16 packed weight."""

    def packed_qkv_gate(self, x: torch.Tensor) -> torch.Tensor:
        if not quantized_products_serve(x):
            return super().packed_qkv_gate(x)
        return fp8_projection(x, self.fp8_qkv_gate_weight, self.fp8_qkv_gate_scale, self.packed_qkv_gate_bias)


class Fp8CausalGatedDeltaNet(Fp8InputProjection, FusedCausalGatedDeltaNet):
    """``FusedCausalGatedDeltaNet`` whose packed input projection runs in FP8."""


class Fp8ConditionGatedDeltaNet(Fp8InputProjection, FusedConditionGatedDeltaNet):
    """``FusedConditionGatedDeltaNet`` whose packed input projection runs in FP8."""


class Fp8CausalSoftmaxAttention(Fp8QkvGate, FusedCausalSoftmaxAttention):
    """``FusedCausalSoftmaxAttention`` whose packed qkv and output gate projection runs in FP8."""


class Fp8ConditionSoftmaxAttention(Fp8QkvGate, FusedConditionSoftmaxAttention):
    """``FusedConditionSoftmaxAttention`` whose packed qkv and output gate projection runs in FP8."""


class Fp8SwiGLU(FusedSwiGLU):
    """``FusedSwiGLU`` whose gate and up projections run as one FP8 matrix product on ``fp8_gate_up_weight`` with the activation, and whose down projection is an ``Fp8Linear``."""

    def fp8_gate_up(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Return ``silu(gate_proj(x)) * up_proj(x)`` [1, M, F] in bf16 for contiguous bf16 ``x`` [1, M, K], computed in FP8 by the kernel of the device; the bf16 packed ``weight`` is not read."""
        return self.fp8_gate_up_kernel(x[0], self.fp8_gate_up_weight, self.fp8_gate_up_scale, bias)[None]


class Fp8Linear(torch.nn.Linear):
    """``Linear`` whose matrix product runs in FP8 on ``fp8_weight`` for one bf16 sample at inference; every other call runs the original layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or x.dim() != 3 or x.shape[0] != 1 or not quantized_products_serve(x):
            return super().forward(x)
        return fp8_projection(x, self.fp8_weight, self.fp8_scale, self.bias)


FP8_GATED_DELTA_NETS = {
    FusedCausalGatedDeltaNet: Fp8CausalGatedDeltaNet,
    FusedConditionGatedDeltaNet: Fp8ConditionGatedDeltaNet,
}
FP8_SOFTMAX_LAYERS = {
    FusedCausalSoftmaxAttention: Fp8CausalSoftmaxAttention,
    FusedConditionSoftmaxAttention: Fp8ConditionSoftmaxAttention,
}


def register_fp8_weight(module: torch.nn.Module, name: str, weight: torch.Tensor) -> None:
    """Register on a module the E4M3 form of a bf16 weight [out, in] as two buffers that are not saved in checkpoints: ``<name>_weight`` [out, in] in E4M3 and ``<name>_scale`` [1] in float32, the largest absolute value of the weight over 448. The weight is about ``<name>_weight * <name>_scale``."""
    (quantized, scale) = portable.quantize_fp8(weight.detach().contiguous())
    module.register_buffer(f"{name}_weight", quantized, persistent=False)
    module.register_buffer(f"{name}_scale", scale, persistent=False)


def install_fp8(policy) -> None:
    """Run the four large matrix products of every layer in FP8 (E4M3) at inference on one bf16 sample.

    The products are the packed input projection of a Gated DeltaNet layer, the packed qkv and output gate
    projection of a softmax layer, and the gate / up projection and the down projection of the MLP. Each weight is
    quantized here once and the tokens at every call, each with one scale for the whole tensor; a product returns
    bf16, and its bias, the activation and every other operation stay in bf16. The outputs differ from the bf16
    policy by the FP8 rounding of these operands. Training, autograd and batches of several samples run the bf16
    layers, and the policy stays as it is on a device without the inference kernels.

    Args:
        policy: ``SanaRobotAR`` or ``SanaRobotBase`` on its CUDA device with bf16 weights, in eval mode, after
            ``install_kernels`` and before any session is built. The packed projections the weights are quantized
            from are installed here when the kernels of the device left them out
            (``install_packed_projections``). A policy takes ``install_fp8`` or ``install_nvfp4``, not both.
    """
    architecture = device_architecture(next(policy.parameters()).device)
    if architecture is None:
        return
    install_packed_projections(policy)
    for module in policy.modules():
        module_type = type(module)
        if module_type in FP8_GATED_DELTA_NETS:
            if portable.supports(*module.packed_input_weight.shape):
                register_fp8_weight(module, "fp8_input", module.packed_input_weight)
                module.__class__ = FP8_GATED_DELTA_NETS[module_type]
        elif module_type in FP8_SOFTMAX_LAYERS:
            weight = module._buffers.get("packed_qkv_gate_weight")
            if weight is not None and portable.supports(*weight.shape):
                register_fp8_weight(module, "fp8_qkv_gate", weight)
                module.__class__ = FP8_SOFTMAX_LAYERS[module_type]
        elif module_type is FusedSwiGLU and portable.supports(*module.packed_gate_up_weight.shape):
            (packed_features, in_features) = module.packed_gate_up_weight.shape
            one_kernel = architecture == SM120 and sm120.supports(in_features, packed_features // 2)
            register_fp8_weight(module, "fp8_gate_up", module.packed_gate_up_weight)
            module.fp8_gate_up_kernel = (sm120 if one_kernel else portable).fp8_swiglu_gate_up
            module.__class__ = Fp8SwiGLU
            module.gate_up_kernel = module.fp8_gate_up
            down_proj = module.down_proj
            if type(down_proj) is torch.nn.Linear and portable.supports(*down_proj.weight.shape):
                register_fp8_weight(down_proj, "fp8", down_proj.weight)
                down_proj.__class__ = Fp8Linear
