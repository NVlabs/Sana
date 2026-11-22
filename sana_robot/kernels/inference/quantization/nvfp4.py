# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sana_robot.kernels.inference.install.gdn import FusedCausalGatedDeltaNet, FusedConditionGatedDeltaNet
from sana_robot.kernels.inference.install.softmax import FusedCausalSoftmaxAttention, FusedConditionSoftmaxAttention
from sana_robot.kernels.inference.install.swiglu import FusedSwiGLU
from sana_robot.kernels.inference.ops.dispatch import SM110, SM120, device_architecture
from sana_robot.kernels.inference.quantization.fp8 import install_packed_projections, quantized_products_serve


def nvfp4_kernels(device: torch.device):
    """Return the module that holds the NVFP4 kernels of the device's architecture (``supports``, ``load``, ``quantize``, ``quantize_weight``, ``nvfp4_linear`` and ``swiglu``), or None for a device without them."""
    architecture = device_architecture(device)
    if architecture == SM110:
        from sana_robot.kernels.inference.ops.nvfp4 import sm110 as kernels
    elif architecture == SM120:
        from sana_robot.kernels.inference.ops.nvfp4 import sm120 as kernels
    else:
        return None
    return kernels


def nvfp4_projection(linear, x, weight, block_scales, weight_scale, bias, amax=None) -> torch.Tensor:
    """Return ``x @ W^T + bias`` [..., out] in bf16 for bf16 ``x`` [..., in] and an NVFP4 weight (packed codes [out, in / 2], block scales, tensor scale), as one NVFP4 matrix product over all rows of ``x`` by the device's ``linear``; ``amax`` is the largest absolute value of ``x`` when the caller knows it."""
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    return linear(rows, weight, block_scales, weight_scale, bias, amax).reshape(*x.shape[:-1], weight.shape[0])


class Nvfp4InputProjection:
    """The packed input projection of a fused Gated DeltaNet layer as one NVFP4 matrix product on ``nvfp4_input_weight``; a call that NVFP4 does not serve multiplies the bf16 packed weight."""

    def project_input(self, x: torch.Tensor) -> torch.Tensor:
        if not quantized_products_serve(x):
            return super().project_input(x)
        return nvfp4_projection(
            self.nvfp4_linear,
            x,
            self.nvfp4_input_weight,
            self.nvfp4_input_block_scales,
            self.nvfp4_input_scale,
            self.packed_input_bias,
        )


class Nvfp4QkvGate:
    """The packed qkv and output gate projection of a fused softmax layer as one NVFP4 matrix product on ``nvfp4_qkv_gate_weight``; a call that NVFP4 does not serve multiplies the bf16 packed weight."""

    def packed_qkv_gate(self, x: torch.Tensor) -> torch.Tensor:
        if not quantized_products_serve(x):
            return super().packed_qkv_gate(x)
        return nvfp4_projection(
            self.nvfp4_linear,
            x,
            self.nvfp4_qkv_gate_weight,
            self.nvfp4_qkv_gate_block_scales,
            self.nvfp4_qkv_gate_scale,
            self.packed_qkv_gate_bias,
        )


class Nvfp4CausalGatedDeltaNet(Nvfp4InputProjection, FusedCausalGatedDeltaNet):
    """``FusedCausalGatedDeltaNet`` whose packed input projection runs in NVFP4."""


class Nvfp4ConditionGatedDeltaNet(Nvfp4InputProjection, FusedConditionGatedDeltaNet):
    """``FusedConditionGatedDeltaNet`` whose packed input projection runs in NVFP4."""


class Nvfp4CausalSoftmaxAttention(Nvfp4QkvGate, FusedCausalSoftmaxAttention):
    """``FusedCausalSoftmaxAttention`` whose packed qkv and output gate projection runs in NVFP4."""


class Nvfp4ConditionSoftmaxAttention(Nvfp4QkvGate, FusedConditionSoftmaxAttention):
    """``FusedConditionSoftmaxAttention`` whose packed qkv and output gate projection runs in NVFP4."""


class Nvfp4Linear(torch.nn.Linear):
    """``Linear`` whose matrix product runs in NVFP4 on ``nvfp4_weight`` for one bf16 sample at inference; every other call runs the original layer. ``input_peak`` is (tensor, largest absolute value of that tensor) when the layer before has computed it for the next input, which then is quantized without a pass over it."""

    input_peak = (None, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (measured, peak) = self.input_peak
        self.input_peak = (None, None)
        if self.training or x.dim() != 3 or x.shape[0] != 1 or not quantized_products_serve(x):
            return super().forward(x)
        return nvfp4_projection(
            self.nvfp4_linear,
            x,
            self.nvfp4_weight,
            self.nvfp4_block_scales,
            self.nvfp4_scale,
            self.bias,
            peak if measured is x else None,
        )


class Nvfp4SwiGLU(FusedSwiGLU):
    """``FusedSwiGLU`` whose gate and up projections run as one NVFP4 matrix product on ``nvfp4_gate_up_weight``, followed by the activation kernel, and whose down projection is an ``Nvfp4Linear``."""

    def nvfp4_gate_up(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Return ``silu(gate_proj(x)) * up_proj(x)`` [1, M, F] in bf16 for contiguous bf16 ``x`` [1, M, K], computed in NVFP4, and hand its largest absolute value to an NVFP4 down projection; the bf16 packed ``weight`` is not read."""
        gate_up = self.nvfp4_linear(
            x[0], self.nvfp4_gate_up_weight, self.nvfp4_gate_up_block_scales, self.nvfp4_gate_up_scale, bias
        )
        (hidden, peak) = self.nvfp4_swiglu(gate_up)
        hidden = hidden[None]
        if isinstance(self.down_proj, Nvfp4Linear):
            self.down_proj.input_peak = (hidden, peak)
        return hidden


NVFP4_GATED_DELTA_NETS = {
    FusedCausalGatedDeltaNet: Nvfp4CausalGatedDeltaNet,
    FusedConditionGatedDeltaNet: Nvfp4ConditionGatedDeltaNet,
}
NVFP4_SOFTMAX_LAYERS = {
    FusedCausalSoftmaxAttention: Nvfp4CausalSoftmaxAttention,
    FusedConditionSoftmaxAttention: Nvfp4ConditionSoftmaxAttention,
}


def register_nvfp4_weight(module: torch.nn.Module, name: str, weight: torch.Tensor, kernels) -> None:
    """Register on a module the NVFP4 form of a bf16 weight [out, in] as three buffers that are not saved in checkpoints, and the matrix product of the device as ``module.nvfp4_linear``: ``<name>_weight`` [out, in / 2] in uint8 with two E2M1 codes per byte (element 2j of a row in the low nibble), ``<name>_block_scales`` in uint8 with one E4M3 scale per 16 consecutive elements of a row in the tiled layout of the kernels (``scale_bytes``), and ``<name>_scale`` [1] in float32, the tensor scale. An element of the weight is about its code times its block scale times ``<name>_scale``. The device's ``quantize_weight`` chooses the scales: Four Over Six block scales on Jetson Thor, RTX 5090 and RTX PRO 6000."""
    (packed, block_scales, scale) = kernels.quantize_weight(weight.detach().contiguous())
    module.register_buffer(f"{name}_weight", packed, persistent=False)
    module.register_buffer(f"{name}_block_scales", block_scales, persistent=False)
    module.register_buffer(f"{name}_scale", scale, persistent=False)
    module.nvfp4_linear = kernels.nvfp4_linear


def install_nvfp4(policy) -> None:
    """Run the four large matrix products of every layer in NVFP4 at inference on one bf16 sample, on Jetson Thor, RTX 5090 and RTX PRO 6000.

    The products are those of ``install_fp8``. NVFP4 stores an element as a 4-bit E2M1 code with one E4M3 scale per 16
    elements and one float32 scale per tensor. Each weight is quantized here once and the tokens at every call; a
    product returns bf16, and its bias, the activation and every other operation stay in bf16. A weight takes Four
    Over Six block scales: every block of 16 elements puts its largest magnitude on the E2M1 value 6 or on 4,
    whichever gives the lower squared error, which costs nothing at inference. The outputs differ from the bf16 policy
    by the NVFP4 rounding of these operands. Training, autograd and batches of several samples run the bf16 layers. The first install on a machine clones CUTLASS v4.8.1 into ``~/.cache/sana_robot`` and compiles the
    matrix product with ``nvcc`` and ``ninja``. The policy stays as it is on a device without NVFP4 kernels (H100).

    Args:
        policy: ``SanaRobotAR`` or ``SanaRobotBase`` on its CUDA device with bf16 weights, in eval mode, after
            ``install_kernels`` and before any session is built. The packed projections the weights are quantized
            from are installed here when the kernels of the device left them out
            (``install_packed_projections``). A policy takes ``install_fp8`` or ``install_nvfp4``, not both.
    """
    kernels = nvfp4_kernels(next(policy.parameters()).device)
    if kernels is None:
        return
    kernels.load()
    install_packed_projections(policy)
    for module in policy.modules():
        module_type = type(module)
        if module_type in NVFP4_GATED_DELTA_NETS:
            if kernels.supports(*module.packed_input_weight.shape):
                register_nvfp4_weight(module, "nvfp4_input", module.packed_input_weight, kernels)
                module.__class__ = NVFP4_GATED_DELTA_NETS[module_type]
        elif module_type in NVFP4_SOFTMAX_LAYERS:
            weight = module._buffers.get("packed_qkv_gate_weight")
            if weight is not None and kernels.supports(*weight.shape):
                register_nvfp4_weight(module, "nvfp4_qkv_gate", weight, kernels)
                module.__class__ = NVFP4_SOFTMAX_LAYERS[module_type]
        elif module_type is FusedSwiGLU and kernels.supports(*module.packed_gate_up_weight.shape):
            register_nvfp4_weight(module, "nvfp4_gate_up", module.packed_gate_up_weight, kernels)
            module.nvfp4_swiglu = kernels.swiglu
            module.__class__ = Nvfp4SwiGLU
            module.gate_up_kernel = module.nvfp4_gate_up
            down_proj = module.down_proj
            if type(down_proj) is torch.nn.Linear and kernels.supports(*down_proj.weight.shape):
                register_nvfp4_weight(down_proj, "nvfp4", down_proj.weight, kernels)
                down_proj.__class__ = Nvfp4Linear
