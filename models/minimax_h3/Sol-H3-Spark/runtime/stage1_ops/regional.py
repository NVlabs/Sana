"""Compile 52 native H3 regions around opaque SM121 VSA attention."""
from types import MethodType, SimpleNamespace
from typing import Optional


def _layer_number(tensor, count):
    # CPU metadata only: never introduce one GPU .item/sync per attention call.
    if tensor.device.type != "cpu" or tensor.ndim != 0:
        raise RuntimeError("regional identity must be a CPU scalar tensor")
    value = int(tensor.item())
    if not 0 <= value < count:
        raise RuntimeError("regional identity is outside the installed model")
    return value


def install(model, state, compile_kwargs=None):
    """Wrap actual installed forwards, then use native generic/region helpers.

    The caller explicitly opts in, on one fresh Triton or cuDNN-BSA VSA model. Fake kernels
    do not increment counters. CPU tensor layer identities avoid50 distinct
    Python-int specializations; they are never read by the compiled wrapper.
    """
    import torch
    from fastvideo.models.loader import fsdp_load
    from fastvideo.attention.layer import _attention_compile_explicitly_disabled
    from . import vsa as policy

    audit = state.get("official_vsa", {})
    backend = audit.get("selected_attention_backend")
    if (audit.get("fastvideo_commit") != policy.FASTVIDEO_COMMIT
            or audit.get("vsa_source_sha256") != policy.VSA_SOURCE_SHA
            or backend != "cudnn_bsa"
            or audit.get("active_compression_gates") != 50
            or audit.get("vsa_forward_calls") != 0 or "vsa_regional_compile" in state):
        raise RuntimeError("install once after the original fresh VSA backend audit, before any forward")
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("this explicit regional adaptation requires SM121")
    if _attention_compile_explicitly_disabled():
        raise RuntimeError("explicit attention compiler-disable cannot enter fullgraph regions")
    bodies, texts = [], []
    for module in model.modules():
        if type(module).__name__ != "MiniMaxH3Attention":
            continue
        impl = module.distributed_attention.attn_impl
        if module.to_gate_compress is None:
            texts.append(impl)
        else:
            if type(impl).__name__ != "MiniMaxH3VSAImpl" or module._gate_compress_active is not True:
                raise RuntimeError("native active VSA compression gate missing")
            bodies.append(impl)
    bodies.sort(key=lambda impl: impl.layer_idx)
    if len(texts) != 2 or [impl.layer_idx for impl in bodies] != list(range(50)):
        raise RuntimeError("expected 50 actual VSA body layers and two original dense text layers")
    body_originals = [impl.forward for impl in bodies]
    text_originals = [impl.forward for impl in texts]
    receipt = {"status": "INSTALLED_NOT_RUNTIME_VALIDATED", "selected_backend": backend,
        "boundary": "entire original VSA forward opaque; native projections/norm/FFN regions compile",
        "sparse_math_changed": False, "compression_gates_retained": 50,
        "layer_identity": "CPU int64 scalar Tensor; no GPU scalar read",
        "body_runtime_calls": 0, "text_runtime_calls": 0,
        "inner_vsa_compiled": False, "fullgraph": True}

    @torch.library.custom_op("sol_h3_spark_vsa::body", mutates_args=(), device_types="cuda")
    def body_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gate: torch.Tensor,
                variable_sizes: torch.Tensor, layer: torch.Tensor,
                prefix_tiles: int, video_tiles: int, step: int) -> torch.Tensor:
        metadata = SimpleNamespace(variable_block_sizes=variable_sizes,
            num_prefix_tiles=prefix_tiles, num_video_tiles=video_tiles,
            VSA_sparsity=0.9, tile_elems=64, exempt=True, dense_layers=(), current_timestep=step)
        result = body_originals[_layer_number(layer, 50)](q, k, v, gate, metadata)
        receipt["body_runtime_calls"] += 1
        return result.contiguous()

    @body_op.register_fake
    def body_fake(q, k, v, gate, variable_sizes, layer, prefix_tiles, video_tiles, step):
        return q.new_empty((*q.shape[:-1], v.shape[-1]))

    @torch.library.custom_op("sol_h3_spark_vsa::text", mutates_args=(), device_types="cuda")
    def text_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: torch.Tensor,
                mask: Optional[torch.Tensor], is_causal: bool) -> torch.Tensor:
        metadata = SimpleNamespace(attn_mask=mask, is_causal=is_causal) if mask is not None else None
        result = text_originals[_layer_number(layer, 2)](q, k, v, metadata)
        receipt["text_runtime_calls"] += 1
        return result.contiguous()

    @text_op.register_fake
    def text_fake(q, k, v, layer, mask, is_causal):
        return q.new_empty((*q.shape[:-1], v.shape[-1]))

    def body_forward(self, query, key, value, gate_compress, attn_metadata):
        if (attn_metadata.VSA_sparsity != 0.9 or attn_metadata.tile_elems != 64
                or attn_metadata.exempt is not True or attn_metadata.dense_layers
                or gate_compress is None):
            raise RuntimeError("regional integration cannot change official tile64/prefix/sparsity/compression policy")
        return body_op(query, key, value, gate_compress, attn_metadata.variable_block_sizes,
            self._sol_h3_cpu_layer, attn_metadata.num_prefix_tiles,
            attn_metadata.num_video_tiles, attn_metadata.current_timestep)

    def text_forward(self, query, key, value, attn_metadata):
        return text_op(query, key, value, self._sol_h3_cpu_layer,
            getattr(attn_metadata, "attn_mask", None), getattr(attn_metadata, "is_causal", False))

    for entries, forward in ((bodies, body_forward), (texts, text_forward)):
        for index, impl in enumerate(entries):
            impl._sol_h3_cpu_layer = torch.tensor(index, device="cpu", dtype=torch.int64)
            impl.forward = MethodType(forward, impl)
    # Deliberately not prepare_for_regional_compile(): that hook probes SM100a.
    model.prepare_for_compile()
    enabled = fsdp_load._enable_regional_attention_compile(model)
    compiled = fsdp_load._compile_model_regions(model, compile_kwargs or {})
    if (enabled, compiled) != (52, 52):
        raise RuntimeError(f"expected native 52 attention modules/regions, got {(enabled, compiled)}")
    receipt.update(attention_modules_enabled=enabled, regions_wrapped=compiled)
    state["vsa_regional_compile"] = receipt
    audit["compile_policy"] = "explicit SM121 opaque VSA + original 52 fullgraph regions; runtime gate required"
    return receipt
