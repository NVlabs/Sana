"""Regional torch.compile for the HyperFlow MiniMax-H3 DiT (after Sol-H3's regional.py).

One compiled region per block kind (MiniMaxH3TransformerBlock / MiniMaxH3TokenRefinerBlock): the class-level forward
is compiled, so diffusers' group-offload hooks on each instance still run eagerly around it and the 50 blocks share one
graph. Inside a region, the attention kernel call is an opaque custom op — Sol-Attn / SageAttention / SDPA and HyperFlow's
step- and layer-dependent dense/sparse choice run eagerly in Python — while projections, QK-norm, RoPE, AdaLN modulation,
RMSNorm, the FP8 activation quantisation, SwiGLU and the residuals are fused by inductor. mode="default": the GEMMs stay
cuBLASLt (`torch._scaled_mm`); max-autotune's Triton scaled-mm template asserts on SM120 per-tensor scales.
"""
import weakref

import torch

# index → processor (weakly held, so a DiT swapped out of host RAM is freed), so the op can reach the Python
# dense/sparse logic without tracing it.
_PROCESSORS = []


# The processor index is a CPU scalar *tensor*, not a Python int: dynamo would specialise on 52 distinct ints and
# recompile per block (hitting the cache limit and falling back to eager); a tensor value is never guarded on.
@torch.library.custom_op("solh3::attention", mutates_args=(), device_types=("cuda", "cpu"))
def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return _PROCESSORS[int(index)]()._solh3_kernel(query, key, value).contiguous()


@_attention.register_fake
def _(query, key, value, index):
    return torch.empty_like(query)


def _make_call(processor_cls):
    """Processor __call__ with the kernel step routed through the custom op (body identical to the originals)."""
    from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
    original = processor_cls.__call__

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if attention_mask is not None:  # never on H3 (one packed document), kept for the processor contract
            return original(self, attn, hidden_states, rotary_emb, attention_mask)
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)
        hidden_states = torch.ops.solh3.attention(query, key, value, self._solh3_index)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)
    return __call__


def _kernel_for(processor):
    """The exact attention call the original processor would make for these q/k/v (eager, outside the graph)."""
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    sparse = getattr(processor, "use_sparse", None)

    def kernel(query, key, value):
        if sparse is not None:
            from hyperflow_h3.blocks import current_step
            if processor.use_sparse(current_step()) and processor._kernel_ready():
                processor.calls_sparse += 1
                return processor._sparse(query, key, value)
            processor.calls_dense += 1
        return dispatch_attention_fn(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                                     backend=processor._attention_backend, parallel_config=processor._parallel_config)
    return kernel


def install(transformer):
    """Route every attention through the opaque op and compile the two block kinds' forwards once each."""
    from diffusers.models.transformers.transformer_minimax_h3 import (MiniMaxH3TokenRefinerBlock,
                                                                      MiniMaxH3TransformerBlock)
    count = 0
    for module in transformer.modules():
        if type(module).__name__ != "MiniMaxH3Attention":
            continue
        processor = module.processor
        cls = type(processor)
        if not cls.__dict__.get("_solh3_patched", False):
            cls.__call__ = _make_call(cls)
            cls._solh3_patched = True
        processor._solh3_index = torch.tensor(len(_PROCESSORS))  # CPU int64 scalar
        processor._solh3_kernel = _kernel_for(processor)
        _PROCESSORS.append(weakref.ref(processor))
        count += 1
    for cls in (MiniMaxH3TransformerBlock, MiniMaxH3TokenRefinerBlock):
        if not getattr(cls, "_solh3_compiled", False):
            # emulate_precision_casts: keep eager's per-op BF16 rounding inside fused kernels (as Sol-H3's regional
            # compile does), so fusion changes speed, not the trajectory.
            cls.forward = torch.compile(cls.forward, dynamic=False, options={"emulate_precision_casts": True})
            cls._solh3_compiled = True
    return count
