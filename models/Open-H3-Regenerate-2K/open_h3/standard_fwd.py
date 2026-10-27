"""standard_fwd: the Diffusers MiniMax-H3 transformer forward, dense attention, BF16 weights.

This is the unaccelerated reference path on one GPU: the original block forward, dense cuDNN SDPA attention and a
per-request LoRA merge when switching between the HyperFlow and refine stages.
"""
import types


def install(model):
    model.set_attention_backend("_native_cudnn")
    return types.SimpleNamespace(name="standard", plan=None, graphs=None, mx_layers=[], resident=False)
