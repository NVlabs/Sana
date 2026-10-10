"""Inference-only AdaLN and per-forward attention metadata reuse.

Ported from the paired/cumulative experiments, without benchmark globals.
Values are guarded by exact equality; banks reset between sessions.
FA4 and persistent final-layout KV are implemented in the vendored runtime.
"""
from contextlib import contextmanager
import torch


class OptimizationState:
    def __init__(self):
        self.inputs = []
        self.projections = {}
        self.current_inputs = {}
        self.metadata = {}

    def reset(self):
        self.inputs.clear()
        self.projections.clear()
        self.begin_forward()

    def begin_forward(self):
        self.current_inputs.clear()
        self.metadata.clear()


STATE = OptimizationState()
_installed = False


def install():
    global _installed
    if _installed:
        return
    from dev.yanzuolu.projects.minimax_h3.modeling.transformer.model import MiniMaxH3AdalnProj
    from dev.yanzuolu.utils import flash_attn as fa
    original_projection = MiniMaxH3AdalnProj.forward
    original_attention = fa.FlashAttention.forward

    def projection(self, value):
        if torch.is_grad_enabled() or self.training:
            return original_projection(self, value)
        token = id(value)
        if token not in STATE.current_inputs:
            key = next((i for i, saved in enumerate(STATE.inputs)
                        if value.shape == saved.shape and value.dtype == saved.dtype
                        and value.device == saved.device and torch.equal(value, saved)), None)
            if key is None:
                key = len(STATE.inputs)
                STATE.inputs.append(value.detach().clone())
            STATE.current_inputs[token] = (value, key)
        key = (id(self), STATE.current_inputs[token][1])
        if key not in STATE.projections:
            STATE.projections[key] = tuple(x.detach() for x in original_projection(self, value))
        return STATE.projections[key]

    def attention(self, q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                  softmax_scale=None, q_scale=None, causal=False, window_size=(-1, -1),
                  deterministic=False, dtype=torch.bfloat16, version=None):
        args = (q, k, v, q_lens, k_lens, dropout_p, softmax_scale, q_scale,
                causal, window_size, deterministic, dtype, version)
        if (torch.is_grad_enabled() or q.ndim != 3 or k.ndim != 3
                or q_lens is None or k_lens is None or dropout_p != 0):
            return original_attention(self, *args)
        selected = fa._select_flash_attn_version(q.device, dropout_p, version, q.shape[-1], v.shape[-1])
        if selected != 4:
            return original_attention(self, *args)
        key = (id(q_lens), id(k_lens), q.device)
        if key not in STATE.metadata:
            cu_q = torch.cat((q_lens.new_zeros(1), q_lens)).cumsum(0, dtype=torch.int32).to(q.device)
            cu_k = torch.cat((k_lens.new_zeros(1), k_lens)).cumsum(0, dtype=torch.int32).to(q.device)
            STATE.metadata[key] = (q_lens, k_lens, cu_q, cu_k,
                                   int(q_lens.cpu().max()), int(k_lens.cpu().max()))
        _, _, cu_q, cu_k, max_q, max_k = STATE.metadata[key]
        output_dtype = q.dtype
        v = v if v.dtype in (torch.float16, torch.bfloat16) else v.to(dtype)
        q, k = q.to(v.dtype), k.to(v.dtype)
        if q_scale is not None:
            q = q * q_scale
        result = self._flash_attn_impl(
            q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            seqused_q=None, seqused_k=None, max_seqlen_q=max_q, max_seqlen_k=max_k,
            softmax_scale=softmax_scale, causal=causal, deterministic=deterministic,
            window_size=(None, None) if window_size == (-1, -1) else window_size,
            num_splits=fa._FA4_NUM_SPLITS, return_lse=False, version=4)
        return result.to(output_dtype)

    MiniMaxH3AdalnProj.forward = projection
    fa.FlashAttention.forward = attention
    _installed = True
