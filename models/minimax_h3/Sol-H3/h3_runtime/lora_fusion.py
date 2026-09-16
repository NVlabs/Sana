"""Keep LoRA's three GEMMs; consume their sum inside the next kernel.

Weights are never merged. Each consumer must round ``base + delta`` to the
projection dtype *before* doing its existing arithmetic. This removes a wide
temporary and its write/read pair without changing GEMM reduction order.
"""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F


class LoRAFusion:
    """Inference-only, reversible specialization for one fixed scale-1 adapter."""

    def __init__(self):
        self.enabled = True
        self.audit = False
        self.calls = Counter()
        self.targets = []

    def uninstall(self):
        self.enabled = False
        for module in self.targets:
            if getattr(module, "_sol_lora_fusion", None) is self:
                del module._sol_lora_fusion
        self.targets.clear()


def _supported(module) -> bool:
    if not hasattr(module, "lora_A") or not hasattr(module, "get_base_layer"):
        return False
    adapters = list(module.active_adapters)
    if len(adapters) != 1 or adapters[0] not in module.lora_A or adapters[0] not in module.lora_B:
        return False
    adapter = adapters[0]
    if module.merged or module.disable_adapters or module.lora_variant:
        return False
    if module.scaling[adapter] != 1 or module.training:
        return False
    if not isinstance(module.lora_dropout[adapter], torch.nn.Identity):
        return False
    base = module.get_base_layer()
    a, b = module.lora_A[adapter], module.lora_B[adapter]
    if not all(type(m) is torch.nn.Linear for m in (base, a, b)):
        return False
    return (
        base.weight.dtype == a.weight.dtype == b.weight.dtype == torch.bfloat16
        and a.bias is None and b.bias is None
        and not any(m._forward_hooks or m._forward_pre_hooks for m in (module, base, a, b))
    )


def install(transformer) -> LoRAFusion:
    """Validate first, then attach the specialization to the main block linears.

    Projections outside the main transformer blocks retain native PEFT.
    Unsupported adapter configurations fail explicitly instead of silently
    presenting a benchmark with inactive fusions.
    """
    state = LoRAFusion()
    for block in transformer.transformer_blocks:
        targets = [block.attn.to_q, block.attn.to_k, block.attn.to_v,
                   block.attn.to_out[0], block.ff.net[0].proj, block.ff.net[2]]
        for module in targets:
            if hasattr(module, "_sol_lora_fusion"):
                raise ValueError("LoRA fusion is already installed")
            if not _supported(module):
                raise ValueError("LoRA fusion requires frozen BF16 LoRA scale-1 Linear branches")
        if not isinstance(block.attn.to_out[1], torch.nn.Dropout) or block.attn.to_out[1].training:
            raise ValueError("LoRA output fusion requires inference-mode attention dropout")
        state.targets.extend(targets)
    if not state.targets:
        raise ValueError("No LoRA transformer blocks found")
    for module in state.targets:
        module._sol_lora_fusion = state
    return state


def split_linear(module, x):
    """Return ``(base, delta)`` or ``(native_output, None)`` when inactive.

    Native fallback preserves adapter switches, module hooks, input casts and
    training semantics. Returning a pair is private to the fused consumers;
    the public PEFT forward and its tensor return type are never replaced.
    """
    state = getattr(module, "_sol_lora_fusion", None)
    if (state is None or not state.enabled or torch.is_grad_enabled()
            or not _supported(module) or x.dtype != torch.bfloat16 or not x.is_cuda):
        return module(x), None
    adapter = module.active_adapters[0]
    base = module.get_base_layer()
    a, b = module.lora_A[adapter], module.lora_B[adapter]
    y = F.linear(x, base.weight, base.bias)
    delta = F.linear(F.linear(x, a.weight), b.weight)
    if state.audit:
        state.calls["split_linear"] += 1
    return y, delta


def mark(module, consumer):
    state = getattr(module, "_sol_lora_fusion", None)
    if state is not None and state.audit:
        state.calls[consumer] += 1
