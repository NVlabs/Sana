# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sana_robot.kernels.inference.install.blockglue import install_blockglue, install_fused_final_layer
from sana_robot.kernels.inference.install.cross_attention import install_caption_cache, install_fused_cross_norm
from sana_robot.kernels.inference.install.gdn import install_fused_gdn
from sana_robot.kernels.inference.install.softmax import (
    install_fused_qk_rope,
    install_fused_softmax_read,
    install_packed_qkv_gate,
)
from sana_robot.kernels.inference.install.swiglu import install_fused_swiglu
from sana_robot.kernels.inference.ops.dispatch import SM90, SM110, SM120, device_architecture
from sana_robot.kernels.inference.quantization.fp8 import install_fp8
from sana_robot.kernels.inference.quantization.nvfp4 import install_nvfp4
from sana_robot.policy.sana_robot_ar import SanaRobotAR

PRECISION_BF16 = "bf16"
PRECISION_FP8 = "fp8"
PRECISION_NVFP4 = "nvfp4"

AR_KERNELS = {
    SM90: (
        install_fused_gdn,
        install_fused_cross_norm,
        install_caption_cache,
        install_fused_softmax_read,
        install_blockglue,
    ),
    SM110: (
        install_fused_gdn,
        install_packed_qkv_gate,
        install_fused_softmax_read,
        install_fused_cross_norm,
        install_caption_cache,
        install_blockglue,
        install_fused_final_layer,
    ),
    SM120: (
        install_fused_gdn,
        install_packed_qkv_gate,
        install_fused_swiglu,
        install_fused_cross_norm,
        install_caption_cache,
        install_blockglue,
        install_fused_final_layer,
    ),
}
BASE_KERNELS = {
    SM90: (
        install_fused_gdn,
        install_fused_qk_rope,
        install_packed_qkv_gate,
        install_fused_swiglu,
        install_fused_cross_norm,
        install_caption_cache,
        install_blockglue,
        install_fused_final_layer,
    ),
    SM110: (
        install_fused_gdn,
        install_fused_qk_rope,
        install_packed_qkv_gate,
        install_fused_cross_norm,
        install_caption_cache,
        install_blockglue,
        install_fused_final_layer,
    ),
    SM120: (
        install_fused_gdn,
        install_fused_qk_rope,
        install_packed_qkv_gate,
        install_fused_swiglu,
        install_fused_cross_norm,
        install_caption_cache,
        install_blockglue,
        install_fused_final_layer,
    ),
}


def install_kernels(policy, precision: str = PRECISION_BF16) -> None:
    """Put the inference kernels of the policy's GPU on the policy.

    Every GPU architecture has its own set of kernels for each policy: H100 (sm90), Jetson Thor (sm110), and RTX
    5090 and RTX PRO 6000 (sm120). The kernels replace sublayers of the policy for inference on one sample and
    leave training and the checkpoint keys as they are. A policy on another device stays as it is.

    Args:
        policy: The policy on its CUDA device with bf16 weights, in eval mode, before any session is built.
        precision: The precision of the large matrix products (the Gated DeltaNet input projection, the softmax
            qkv and gate projection and both projections of the MLP): ``bf16``; ``fp8``, E4M3 with one scale per
            tensor; or ``nvfp4``, E2M1 with one E4M3 scale per 16 elements, on Jetson Thor, RTX 5090 and RTX PRO
            6000 only. The low-bit products are faster and change the actions more than bf16 rounding does; a
            device without the chosen one stays bf16.
    """
    architecture = device_architecture(next(policy.parameters()).device)
    kernels = AR_KERNELS if isinstance(policy, SanaRobotAR) else BASE_KERNELS
    for install in kernels.get(architecture, ()):
        install(policy)
    if precision == PRECISION_FP8:
        install_fp8(policy)
    elif precision == PRECISION_NVFP4:
        install_nvfp4(policy)
