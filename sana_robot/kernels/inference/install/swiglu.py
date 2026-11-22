# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sana_robot.kernels.inference.ops.dispatch import SM90, SM110, SM120, device_architecture
from sana_robot.policy.backbone.swiglu import SwiGLU


def swiglu_kernels(device: torch.device):
    """Return the module that holds the fused gate / up kernel of the device's architecture (``supports`` and ``swiglu_gate_up``), or None for a device without one."""
    architecture = device_architecture(device)
    if architecture == SM90:
        from sana_robot.kernels.inference.ops.swiglu import sm90 as kernels
    elif architecture == SM110:
        from sana_robot.kernels.inference.ops.swiglu import sm110 as kernels
    elif architecture == SM120:
        from sana_robot.kernels.inference.ops.swiglu import sm120 as kernels
    else:
        return None
    return kernels


class FusedSwiGLU(SwiGLU):
    """``SwiGLU`` whose gate projection, up projection and activation run as one kernel on the packed weight ``packed_gate_up_weight``, for one sample in bf16 at inference; every other call runs the original layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.packed_gate_up_weight
        half_precision = weight.dtype == torch.bfloat16 and (
            x.dtype == torch.bfloat16
            or (
                x.dtype == torch.float32
                and torch.is_autocast_enabled(x.device.type)
                and torch.get_autocast_dtype(x.device.type) == torch.bfloat16
            )
        )
        if self.training or torch.is_grad_enabled() or x.dim() != 3 or x.shape[0] != 1 or not half_precision:
            return super().forward(x)
        return self.down_proj(self.gate_up_kernel(x.to(torch.bfloat16).contiguous(), weight, self.packed_gate_up_bias))


def install_fused_swiglu(policy) -> None:
    """Run the gate projection, the up projection and the activation of every SwiGLU layer as one kernel.

    Args:
        policy: SANA-Robot-AR or SANA-Robot-base in eval mode on its CUDA device with its final bf16 weights. Every
            ``SwiGLU`` the device's kernel serves becomes a ``FusedSwiGLU`` with two buffers that are not saved in
            checkpoints: ``packed_gate_up_weight`` [2F, K] in bf16, the rows of ``gate_proj.weight`` followed by the
            rows of ``up_proj.weight``, and ``packed_gate_up_bias`` [2F] in bf16, the gate bias followed by the up
            bias. A device without a kernel leaves the policy as it is.
    """
    kernels = swiglu_kernels(next(policy.parameters()).device)
    if kernels is None:
        return
    for block in policy.blocks:
        mlp = block.mlp
        if type(mlp) is not SwiGLU or not kernels.supports(mlp.gate_proj.in_features, mlp.gate_proj.out_features):
            continue
        weight = torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight)).detach().to(torch.bfloat16).contiguous()
        bias = torch.cat((mlp.gate_proj.bias, mlp.up_proj.bias)).detach().to(torch.bfloat16).contiguous()
        mlp.register_buffer("packed_gate_up_weight", weight, persistent=False)
        mlp.register_buffer("packed_gate_up_bias", bias, persistent=False)
        mlp.gate_up_kernel = kernels.swiglu_gate_up
        mlp.__class__ = FusedSwiGLU
