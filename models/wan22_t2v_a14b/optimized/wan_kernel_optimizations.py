"""Guarded, experiment-local exact optimizations for Wan CP4 inference."""

from __future__ import annotations

import os
import importlib.util
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import torch


_COUNTERS = {
    "packed_qkv_a2a_calls": 0,
    "packed_qkv_input_collectives": 0,
    "baseline_qkv_input_collectives_equivalent": 0,
    "qkv_input_collectives_eliminated": 0,
    "output_collectives_unchanged": 0,
    "qkv_elements_exchanged": 0,
    "fallback_calls": 0,
    "async_qkv_a2a_calls": 0,
    "async_qkv_input_collectives": 0,
    "async_qkv_elements_exchanged": 0,
    "rope_cache_hits": 0,
    "rope_cache_misses": 0,
    "rope_cached_elements_reused": 0,
    "text_projection_cache_hits": 0,
    "text_projection_cache_misses": 0,
    "cross_kv_cache_hits": 0,
    "cross_kv_cache_misses": 0,
    "cross_kv_cached_elements_reused": 0,
    "direct_output_a2a_calls": 0,
    "direct_output_a2a_elements": 0,
    "reusable_a2a_buffer_hits": 0,
    "reusable_a2a_buffer_misses": 0,
}
_PACKED_QKV_A2A_INSTALLED = False
_ORIGINAL_ULYSSES_FORWARD = None
_FUSED_QKV_PROJECTIONS_INSTALLED = False
_COMPILED_BLOCK_GLUE_INSTALLED = False
_COMPILED_BLOCK_GLUE_ACTIVATION: dict[str, Any] = {}
_COMPILED_MODULATE_DISPATCH = None
_COMPILED_GATED_RESIDUAL_DISPATCH = None
_COMPILED_FFN_INSTALLED = False
_COMPILED_FFN_ACTIVATION: dict[str, Any] = {}
_COMPILED_FFN_DISPATCH = None
_COMPILED_FFN_EPILOGUE_INSTALLED = False
_COMPILED_FFN_EPILOGUE_ACTIVATION: dict[str, Any] = {}
_COMPILED_FFN_EPILOGUE_DISPATCH = None
_ORIGINAL_BLOCK_FORWARDS: dict[int, Any] = {}
_COMPILED_QK_ROPE_INSTALLED = False
_COMPILED_QK_ROPE_ACTIVATION: dict[str, Any] = {}
_COMPILED_QK_ROPE_DISPATCH = None
_COMPILED_QKV_NORM_ROPE_INSTALLED = False
_COMPILED_QKV_NORM_ROPE_ACTIVATION: dict[str, Any] = {}
_COMPILED_QKV_NORM_ROPE_DISPATCH = None
_ORIGINAL_WAN_PROCESSOR_CALL = None
_NATIVE_FLASH_SELF_ATTENTION_INSTALLED = False
_NATIVE_CUDNN_SELF_ATTENTION_INSTALLED = False
_COMPILED_NATIVE_SDPA_INSTALLED = False
_COMPILED_NATIVE_SDPA_ACTIVATION: dict[str, Any] = {}
_COMPILED_NATIVE_SDPA_DISPATCH = None
_ORIGINAL_NATIVE_ATTENTION_FORWARD_OP = None
_ASYNC_QKV_A2A_INSTALLED = False
_ORIGINAL_ASYNC_ULYSSES_FORWARD = None
_INVARIANT_ROPE_CACHE_INSTALLED = False
_INVARIANT_ROPE_CACHE_ACTIVATION: dict[str, Any] = {}
_INVARIANT_CONDITIONING_CACHE_INSTALLED = False
_INVARIANT_CONDITIONING_CACHE_ACTIVATION: dict[str, Any] = {}
_ORIGINAL_WAN_GET_QKV_PROJECTIONS = None
_DIRECT_OUTPUT_A2A_ENABLED = False
_REUSABLE_A2A_BUFFERS_ENABLED = False
_REUSABLE_A2A_SOURCE_BUFFERS_ENABLED = False
_REUSABLE_A2A_BUFFERS: dict[tuple[Any, ...], torch.Tensor] = {}
_PISA_ATTENTION_INSTALLED = False
_PISA_ATTENTION_ACTIVATION: dict[str, Any] = {}
_PISA_MODULE: Any = None
_PISA_CONTEXT = threading.local()
_PISA_ATTN_LAYERS: dict[int, int] = {}
_ORIGINAL_PISA_PROCESSOR_CALL = None
_ORIGINAL_PISA_NATIVE_ATTENTION_FORWARD_OP = None
_PISA_STEP_TRACKING_INSTALLED = False
_PISA_STEP_MODEL_STATES: dict[str, dict[str, Any]] = {}
_PISA_PROMPT_INDEX = 0
_PISA_GLOBAL_FORWARD_CALLS = 0


def get_optimization_counters() -> dict[str, int]:
    return dict(_COUNTERS)


def install_pisa_attention(pipe: Any, output_dir: Path) -> dict[str, Any]:
    """Install the local Piecewise/PISA kernel at Wan's dense self-attn op.

    The archived PISA implementation is the repo-local Triton implementation
    previously integrated for Sana.  Wan's CP path presents attention to its
    forward op as ``[B, S, H_local, D]``; PISA consumes ``[B, H, S, D]``, so
    this adapter only transposes around the real PISA call.  The processor
    context identifies ``attn1`` modules, which keeps cross-attention dense and
    remains correct when EasyCache skips complete blocks.
    """

    global _PISA_ATTENTION_INSTALLED
    global _PISA_ATTENTION_ACTIVATION
    global _PISA_MODULE
    global _ORIGINAL_PISA_PROCESSOR_CALL
    global _ORIGINAL_PISA_NATIVE_ATTENTION_FORWARD_OP
    if _PISA_ATTENTION_INSTALLED:
        return _PISA_ATTENTION_ACTIVATION

    root = Path(os.environ.get("AUTOVIDEO_REPO_ROOT", str(Path(__file__).resolve().parents[3])))
    source = root / (
        "output/archive/20260701T200248Z/experiments/"
        "sana-integrator_ia-0001/worktree/state/integration-source-snapshots/"
        "pisa/external/sana_standalone/dev/junsongc/diffusion/model/nets/"
        "pisa_attention.py"
    )
    if not source.is_file():
        raise RuntimeError(f"local PISA implementation is missing: {source}")

    # Keep the archived adapter's tested configuration surface, while allowing
    # Wan config to use WAN22_* names in their manifests.
    aliases = {
        "WAN22_PISA_DENSITY": "SANA_PISA_DENSITY",
        "WAN22_PISA_DENSITY_RULES": "SANA_PISA_DENSITY_RULES",
        "WAN22_PISA_DENSE_LAYERS": "SANA_PISA_DENSE_LAYERS",
        "WAN22_PISA_PISA_LAYERS": "SANA_PISA_PISA_LAYERS",
        "WAN22_PISA_DENSE_STEPS": "SANA_PISA_DENSE_STEPS",
        "WAN22_PISA_PISA_STEPS": "SANA_PISA_PISA_STEPS",
        "WAN22_PISA_BLOCK_SIZE": "SANA_PISA_BLOCK_SIZE",
        "WAN22_PISA_KERNEL_NUM_STAGES": "SANA_PISA_KERNEL_NUM_STAGES",
        "WAN22_PISA_APPROX_REMAINDER": "SANA_PISA_APPROX_REMAINDER",
        "WAN22_STEPS": "SANA_PISA_NUM_STEPS",
    }
    for wan_name, pisa_name in aliases.items():
        if os.environ.get(wan_name) is not None:
            os.environ[pisa_name] = os.environ[wan_name]
    os.environ["SANA_PISA_ENABLED"] = "1"
    os.environ.setdefault("SANA_PISA_APPROX_REMAINDER", "1")
    os.environ.setdefault("SANA_PISA_NUM_STEPS", "40")
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    stats_path = Path(output_dir) / f"pisa_stats_rank{rank}.json"
    os.environ["SANA_PISA_STATS_PATH"] = str(stats_path)

    module_name = "wan22_local_pisa_attention"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load local PISA implementation: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _PISA_MODULE = module

    from diffusers.models import attention_dispatch as dispatch
    from diffusers.models.transformers import transformer_wan

    for model_name, layer_offset in (("transformer", 0), ("transformer_2", 40)):
        model = getattr(pipe, model_name, None)
        for layer_index, block in enumerate(getattr(model, "blocks", ())):
            attn = getattr(block, "attn1", None)
            if attn is not None:
                _PISA_ATTN_LAYERS[id(attn)] = layer_offset + layer_index
    if not _PISA_ATTN_LAYERS:
        raise RuntimeError("no Wan self-attention modules found for PISA")

    _ORIGINAL_PISA_PROCESSOR_CALL = transformer_wan.WanAttnProcessor.__call__

    def pisa_processor_call(self, attn, *args, **kwargs):
        previous = getattr(_PISA_CONTEXT, "layer_index", None)
        _PISA_CONTEXT.layer_index = _PISA_ATTN_LAYERS.get(id(attn))
        try:
            return _ORIGINAL_PISA_PROCESSOR_CALL(self, attn, *args, **kwargs)
        finally:
            _PISA_CONTEXT.layer_index = previous

    transformer_wan.WanAttnProcessor.__call__ = pisa_processor_call
    _ORIGINAL_PISA_NATIVE_ATTENTION_FORWARD_OP = dispatch._native_attention_forward_op

    def pisa_native_attention_forward_op(
        ctx,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        return_lse=False,
        _save_ctx=True,
        _parallel_config=None,
    ):
        layer_index = getattr(_PISA_CONTEXT, "layer_index", None)
        if (
            layer_index is None
            or return_lse
            or attn_mask is not None
            or is_causal
            or dropout_p != 0.0
        ):
            return _ORIGINAL_PISA_NATIVE_ATTENTION_FORWARD_OP(
                ctx,
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                return_lse,
                _save_ctx,
                _parallel_config,
            )

        q_bh = query.permute(0, 2, 1, 3).contiguous()
        k_bh = key.permute(0, 2, 1, 3).contiguous()
        v_bh = value.permute(0, 2, 1, 3).contiguous()
        attention_scale = scale if scale is not None else query.shape[-1] ** -0.5

        def dense_fn():
            return _ORIGINAL_PISA_NATIVE_ATTENTION_FORWARD_OP(
                ctx,
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                return_lse,
                _save_ctx,
                _parallel_config,
            )

        output_bh = _PISA_MODULE.sana_pisa_attention(
            q_bh,
            k_bh,
            v_bh,
            scale=attention_scale,
            layer_index=int(layer_index),
            dense_fn=lambda: dense_fn().permute(0, 2, 1, 3).contiguous(),
        )
        return output_bh.permute(0, 2, 1, 3).contiguous()

    dispatch._native_attention_forward_op = pisa_native_attention_forward_op

    # Single-GPU (no context parallel) never routes through _native_attention_forward_op,
    # so also intercept dispatch_attention_fn itself (the entry the WanAttnProcessor calls).
    # Only PISA the non-CP real self-attention of a hooked attn1 layer; the 4-GPU CP path
    # (parallel_config set) and cross-attention pass straight through unchanged.
    _ORIGINAL_PISA_DISPATCH_ATTENTION_FN = transformer_wan.dispatch_attention_fn

    def pisa_dispatch_attention_fn(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
        scale=None, enable_gqa=False, attention_kwargs=None, *,
        backend=None, parallel_config=None,
    ):
        layer_index = getattr(_PISA_CONTEXT, "layer_index", None)
        if (
            parallel_config is not None
            or layer_index is None
            or attn_mask is not None
            or is_causal
            or dropout_p != 0.0
            or key.shape[1] != query.shape[1]
        ):
            return _ORIGINAL_PISA_DISPATCH_ATTENTION_FN(
                query, key, value, attn_mask, dropout_p, is_causal, scale,
                enable_gqa, attention_kwargs, backend=backend,
                parallel_config=parallel_config,
            )
        q_bh = query.permute(0, 2, 1, 3).contiguous()
        k_bh = key.permute(0, 2, 1, 3).contiguous()
        v_bh = value.permute(0, 2, 1, 3).contiguous()
        attention_scale = scale if scale is not None else query.shape[-1] ** -0.5

        def dense_fn():
            return _ORIGINAL_PISA_DISPATCH_ATTENTION_FN(
                query, key, value, attn_mask, dropout_p, is_causal, scale,
                enable_gqa, attention_kwargs, backend=backend,
                parallel_config=parallel_config,
            )

        output_bh = _PISA_MODULE.sana_pisa_attention(
            q_bh, k_bh, v_bh, scale=attention_scale, layer_index=int(layer_index),
            dense_fn=lambda: dense_fn().permute(0, 2, 1, 3).contiguous(),
        )
        return output_bh.permute(0, 2, 1, 3).contiguous()

    transformer_wan.dispatch_attention_fn = pisa_dispatch_attention_fn
    _PISA_ATTENTION_ACTIVATION = {
        "installed": True,
        "backend": "local_archived_piecewise_pisa_triton",
        "source": str(source),
        "source_commit": getattr(module, "AUTHORITATIVE_COMMIT", None),
        "source_sha256": getattr(module, "AUTHORITATIVE_SHA256", None),
        "shape_adapter": "Wan [B,S,H,D] <-> PISA [B,H,S,D]",
        "attention_scope": "video_self_attention_only",
        "cross_attention": "dense_unchanged",
        "layer_count": len(_PISA_ATTN_LAYERS),
        "dense_fn_fallback": True,
        "stats_path": str(stats_path),
    }
    _PISA_ATTENTION_INSTALLED = True
    return _PISA_ATTENTION_ACTIVATION


def install_pisa_step_tracking(pipe: Any) -> dict[str, Any]:
    """Bind PISA policy steps to real Wan transformer forward boundaries.

    Wan 2.2 selects one of its two experts for each CFG forward. With CFG
    enabled, two expert forwards belong to one denoising step; a shared clock
    therefore remains correct when the model switches from high-noise to
    low-noise, and is unaffected by block Cache replay.
    """

    global _PISA_STEP_TRACKING_INSTALLED
    if _PISA_STEP_TRACKING_INSTALLED:
        return {
            "installed": True,
            "already_installed": True,
            "models": sorted(_PISA_STEP_MODEL_STATES),
        }
    if _PISA_MODULE is None:
        raise RuntimeError("PISA step tracking requires install_pisa_attention first")
    if not hasattr(_PISA_MODULE, "set_pisa_context"):
        raise RuntimeError("loaded PISA adapter has no explicit context API")

    for model_name in ("transformer", "transformer_2"):
        model = getattr(pipe, model_name, None)
        if model is None:
            continue
        original = model.forward
        state: dict[str, Any] = {
            "model": model_name,
            "forward_calls": 0,
            "original_forward": original,
        }

        def tracked_forward(*args, _original=original, _state=state, **kwargs):
            global _PISA_GLOBAL_FORWARD_CALLS
            step_index = int(_PISA_GLOBAL_FORWARD_CALLS // 2)
            _PISA_GLOBAL_FORWARD_CALLS += 1
            _state["forward_calls"] += 1
            _PISA_MODULE.set_pisa_context(step_index, _PISA_PROMPT_INDEX)
            try:
                return _original(*args, **kwargs)
            finally:
                _PISA_MODULE.set_pisa_context(None, _PISA_PROMPT_INDEX)

        model.forward = tracked_forward
        _PISA_STEP_MODEL_STATES[model_name] = state

    if not _PISA_STEP_MODEL_STATES:
        raise RuntimeError("no Wan transformer models found for PISA step tracking")
    _PISA_STEP_TRACKING_INSTALLED = True
    return {
        "installed": True,
        "clock": "global_cfg_forward_pair_clock",
        "cache_safe": True,
        "models": sorted(_PISA_STEP_MODEL_STATES),
    }


def reset_pisa_step_tracking(prompt_index: int = 0) -> None:
    """Reset the explicit PISA step clock at the start of every prompt."""

    global _PISA_PROMPT_INDEX
    global _PISA_GLOBAL_FORWARD_CALLS
    if not _PISA_STEP_TRACKING_INSTALLED:
        return
    _PISA_PROMPT_INDEX = int(prompt_index)
    _PISA_GLOBAL_FORWARD_CALLS = 0
    for state in _PISA_STEP_MODEL_STATES.values():
        state["forward_calls"] = 0
    _PISA_MODULE.reset_pisa_layer_counters()
    _PISA_MODULE.set_pisa_context(None, _PISA_PROMPT_INDEX)


def pisa_step_tracking_summary() -> dict[str, Any]:
    """Return forward-call counts for post-run verification."""

    summary: dict[str, Any] = {
        "global_forward_calls": int(_PISA_GLOBAL_FORWARD_CALLS),
    }
    summary.update({
        name: {
            "forward_calls": int(state["forward_calls"]),
            "expected_step_count": int(state["forward_calls"]),
        }
        for name, state in sorted(_PISA_STEP_MODEL_STATES.items())
    })
    return summary


def install_fused_qkv_projections(pipe: Any) -> dict[str, Any]:
    """Pack Wan projection weights into exact self-QKV and cross-KV linears.

    Concatenating the output-row blocks of independent linear projections and
    chunking the result is the same matrix multiplication as evaluating each
    row block separately.  The original modules remain present, so this is a
    guarded inference dispatch change rather than a checkpoint mutation.
    """

    global _FUSED_QKV_PROJECTIONS_INSTALLED
    if _FUSED_QKV_PROJECTIONS_INSTALLED:
        return {"installed": True, "already_installed": True}

    model_stats: dict[str, Any] = {}
    total_self_qkv = 0
    total_cross_kv = 0
    for model_name in ("transformer", "transformer_2"):
        model = getattr(pipe, model_name, None)
        if model is None:
            continue
        blocks = list(getattr(model, "blocks", ()))
        if not blocks:
            raise RuntimeError(f"{model_name} exposes no Wan transformer blocks")

        self_count = 0
        cross_count = 0
        for block in blocks:
            self_attn = block.attn1
            cross_attn = block.attn2
            self_attn.fuse_projections()
            cross_attn.fuse_projections()
            if not getattr(self_attn, "fused_projections", False) or not hasattr(
                self_attn, "to_qkv"
            ):
                raise RuntimeError(f"failed to fuse {model_name} self-attention QKV")
            if not getattr(cross_attn, "fused_projections", False) or not hasattr(
                cross_attn, "to_kv"
            ):
                raise RuntimeError(f"failed to fuse {model_name} cross-attention KV")
            self_count += 1
            cross_count += 1

        model_stats[model_name] = {
            "block_count": len(blocks),
            "self_qkv_modules": self_count,
            "cross_kv_modules": cross_count,
        }
        total_self_qkv += self_count
        total_cross_kv += cross_count

    if total_self_qkv == 0 or total_cross_kv == 0:
        raise RuntimeError("no Wan QKV projection modules were fused")

    _FUSED_QKV_PROJECTIONS_INSTALLED = True
    return {
        "installed": True,
        "implementation": "diffusers.WanAttention.fuse_projections",
        "models": model_stats,
        "self_qkv_modules": total_self_qkv,
        "cross_kv_modules": total_cross_kv,
        "source_projection_parameters_retained": True,
    }


def install_compiled_block_glue(pipe: Any) -> dict[str, Any]:
    """Compile exact FP32 modulation and gated-residual expression groups.

    Native layer normalization, attention, projections, and FFN modules remain
    unchanged.  TorchInductor only fuses the pointwise casts, multiply/adds,
    and final BF16 store that the eager Wan block evaluates separately.
    """

    global _COMPILED_BLOCK_GLUE_INSTALLED
    global _COMPILED_BLOCK_GLUE_ACTIVATION
    global _COMPILED_MODULATE_DISPATCH
    global _COMPILED_GATED_RESIDUAL_DISPATCH

    if _COMPILED_BLOCK_GLUE_INSTALLED:
        return _COMPILED_BLOCK_GLUE_ACTIVATION

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    cache_root = (
        (Path(os.environ.get("AUTOVIDEO_REPO_ROOT", str(Path(__file__).resolve().parents[3]))) / "output/orchestrated/wan14b-20260706-155110/integrator")
        / "caches"
        / "wan14b_compiled_block_glue_v1"
        / f"rank_{rank}"
    )
    cache_preexisting = cache_root.exists() and any(cache_root.iterdir())
    inductor_cache = cache_root / "torchinductor"
    triton_cache = cache_root / "triton"
    temp_root = cache_root / "tmp"
    xdg_cache = cache_root / "xdg"
    for path in (inductor_cache, triton_cache, temp_root, xdg_cache):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    os.environ["TMPDIR"] = str(temp_root)
    os.environ["TEMP"] = str(temp_root)
    os.environ["TMP"] = str(temp_root)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)

    def modulate_fp32_to_bf16(
        normalized: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return (normalized * (1.0 + scale) + shift).to(torch.bfloat16)

    def gated_residual_fp32_to_bf16(
        hidden_states: torch.Tensor,
        branch: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        return (hidden_states.float() + branch.float() * gate).to(torch.bfloat16)

    compile_options = {
        "fullgraph": True,
        "dynamic": False,
        "mode": "max-autotune-no-cudagraphs",
    }
    compiled_modulate = torch.compile(modulate_fp32_to_bf16, **compile_options)
    compiled_gated_residual = torch.compile(
        gated_residual_fp32_to_bf16, **compile_options
    )

    _COMPILED_BLOCK_GLUE_ACTIVATION = {
        "installed": True,
        "compiler": "torch.compile/TorchInductor",
        "fullgraph": True,
        "dynamic": False,
        "mode": "max-autotune-no-cudagraphs",
        "cache_root": str(cache_root),
        "cache_preexisting": cache_preexisting,
        "cold_compile_plus_first_dispatch_s": {},
        "target_dtype": "torch.bfloat16",
    }

    def first_modulate(*args):
        global _COMPILED_MODULATE_DISPATCH
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = compiled_modulate(*args)
        torch.cuda.synchronize()
        _COMPILED_BLOCK_GLUE_ACTIVATION[
            "cold_compile_plus_first_dispatch_s"
        ]["modulation"] = time.perf_counter() - started
        _COMPILED_MODULATE_DISPATCH = compiled_modulate
        return result

    def first_gated_residual(*args):
        global _COMPILED_GATED_RESIDUAL_DISPATCH
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = compiled_gated_residual(*args)
        torch.cuda.synchronize()
        _COMPILED_BLOCK_GLUE_ACTIVATION[
            "cold_compile_plus_first_dispatch_s"
        ]["gated_residual"] = time.perf_counter() - started
        _COMPILED_GATED_RESIDUAL_DISPATCH = compiled_gated_residual
        return result

    _COMPILED_MODULATE_DISPATCH = first_modulate
    _COMPILED_GATED_RESIDUAL_DISPATCH = first_gated_residual

    def compiled_block_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.dtype != torch.bfloat16:
            raise RuntimeError(
                "compiled Wan block glue is certified for the official BF16 workload"
            )
        if temb.ndim == 4:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        norm_hidden_states = _COMPILED_MODULATE_DISPATCH(
            self.norm1(hidden_states.float()), scale_msa, shift_msa
        )
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        hidden_states = _COMPILED_GATED_RESIDUAL_DISPATCH(
            hidden_states, attn_output, gate_msa
        )

        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states, encoder_hidden_states, None, None
        )
        hidden_states = hidden_states + attn_output

        norm_hidden_states = _COMPILED_MODULATE_DISPATCH(
            self.norm3(hidden_states.float()), c_scale_msa, c_shift_msa
        )
        if _COMPILED_FFN_EPILOGUE_DISPATCH is not None:
            ffn_input = self.ffn.net[0].proj
            ffn_output = self.ffn.net[2]
            return _COMPILED_FFN_EPILOGUE_DISPATCH(
                hidden_states,
                norm_hidden_states,
                ffn_input.weight,
                ffn_input.bias,
                ffn_output.weight,
                ffn_output.bias,
                c_gate_msa,
            )
        if _COMPILED_FFN_DISPATCH is not None:
            ffn_input = self.ffn.net[0].proj
            ffn_output = self.ffn.net[2]
            ff_output = _COMPILED_FFN_DISPATCH(
                norm_hidden_states,
                ffn_input.weight,
                ffn_input.bias,
                ffn_output.weight,
                ffn_output.bias,
            )
        else:
            ff_output = self.ffn(norm_hidden_states)
        hidden_states = _COMPILED_GATED_RESIDUAL_DISPATCH(
            hidden_states, ff_output, c_gate_msa
        )
        return hidden_states

    model_stats: dict[str, Any] = {}
    patched_blocks = 0
    for model_name in ("transformer", "transformer_2"):
        model = getattr(pipe, model_name, None)
        if model is None:
            continue
        blocks = list(getattr(model, "blocks", ()))
        for block in blocks:
            _ORIGINAL_BLOCK_FORWARDS[id(block)] = block.forward
            block.forward = types.MethodType(compiled_block_forward, block)
        model_stats[model_name] = {"patched_blocks": len(blocks)}
        patched_blocks += len(blocks)
    if patched_blocks == 0:
        raise RuntimeError("no Wan transformer blocks were patched for compiled glue")

    _COMPILED_BLOCK_GLUE_ACTIVATION["models"] = model_stats
    _COMPILED_BLOCK_GLUE_ACTIVATION["patched_blocks"] = patched_blocks
    _COMPILED_BLOCK_GLUE_ACTIVATION["source_block_forwards_retained"] = len(
        _ORIGINAL_BLOCK_FORWARDS
    )
    _COMPILED_BLOCK_GLUE_INSTALLED = True
    return _COMPILED_BLOCK_GLUE_ACTIVATION


def install_compiled_ffn(pipe: Any) -> dict[str, Any]:
    """Compile Wan's exact Linear -> GELU(tanh) -> Linear FFN path."""

    global _COMPILED_FFN_INSTALLED
    global _COMPILED_FFN_ACTIVATION
    global _COMPILED_FFN_DISPATCH
    if _COMPILED_FFN_INSTALLED:
        return _COMPILED_FFN_ACTIVATION

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    cache_root = (
        (Path(os.environ.get("AUTOVIDEO_REPO_ROOT", str(Path(__file__).resolve().parents[3]))) / "output/orchestrated/wan14b-20260706-155110/integrator")
        / "caches"
        / "wan14b_compiled_ffn_v1"
        / f"rank_{rank}"
    )
    cache_preexisting = cache_root.exists() and any(cache_root.iterdir())
    inductor_cache = cache_root / "torchinductor"
    triton_cache = cache_root / "triton"
    temp_root = cache_root / "tmp"
    xdg_cache = cache_root / "xdg"
    for path in (inductor_cache, triton_cache, temp_root, xdg_cache):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    os.environ["TMPDIR"] = str(temp_root)
    os.environ["TEMP"] = str(temp_root)
    os.environ["TMP"] = str(temp_root)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)

    def fused_ffn(
        hidden_states: torch.Tensor,
        input_weight: torch.Tensor,
        input_bias: torch.Tensor,
        output_weight: torch.Tensor,
        output_bias: torch.Tensor,
    ) -> torch.Tensor:
        projected = torch.nn.functional.linear(
            hidden_states, input_weight, input_bias
        )
        projected = torch.nn.functional.gelu(projected, approximate="tanh")
        return torch.nn.functional.linear(projected, output_weight, output_bias)

    compiled_ffn = torch.compile(
        fused_ffn,
        fullgraph=True,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )
    _COMPILED_FFN_ACTIVATION = {
        "installed": True,
        "compiler": "torch.compile/TorchInductor",
        "fullgraph": True,
        "dynamic": False,
        "mode": "max-autotune-no-cudagraphs",
        "cache_root": str(cache_root),
        "cache_preexisting": cache_preexisting,
        "cold_compile_plus_first_dispatch_s": None,
        "formula": "linear -> gelu(approximate=tanh) -> linear",
    }

    def first_ffn(*args):
        global _COMPILED_FFN_DISPATCH
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = compiled_ffn(*args)
        torch.cuda.synchronize()
        _COMPILED_FFN_ACTIVATION[
            "cold_compile_plus_first_dispatch_s"
        ] = time.perf_counter() - started
        _COMPILED_FFN_DISPATCH = compiled_ffn
        return result

    _COMPILED_FFN_DISPATCH = first_ffn
    model_stats: dict[str, Any] = {}
    total_modules = 0
    for model_name in ("transformer", "transformer_2"):
        model = getattr(pipe, model_name, None)
        if model is None:
            continue
        blocks = list(getattr(model, "blocks", ()))
        for block in blocks:
            net = getattr(getattr(block, "ffn", None), "net", None)
            if (
                net is None
                or len(net) < 3
                or not hasattr(net[0], "proj")
                or not isinstance(net[2], torch.nn.Linear)
            ):
                raise RuntimeError(
                    f"unsupported Wan FFN structure in {model_name}: {type(net)}"
                )
        model_stats[model_name] = {"ffn_modules": len(blocks)}
        total_modules += len(blocks)
    if total_modules == 0:
        raise RuntimeError("no Wan FFN modules found for compiled FFN")

    _COMPILED_FFN_ACTIVATION["models"] = model_stats
    _COMPILED_FFN_ACTIVATION["ffn_modules"] = total_modules
    _COMPILED_FFN_INSTALLED = True
    return _COMPILED_FFN_ACTIVATION


def install_compiled_ffn_epilogue(pipe: Any) -> dict[str, Any]:
    """Compile Wan FFN and its gated residual into one exact epilogue graph."""

    global _COMPILED_FFN_EPILOGUE_INSTALLED
    global _COMPILED_FFN_EPILOGUE_ACTIVATION
    global _COMPILED_FFN_EPILOGUE_DISPATCH
    if _COMPILED_FFN_EPILOGUE_INSTALLED:
        return _COMPILED_FFN_EPILOGUE_ACTIVATION

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    cache_root = (
        (Path(os.environ.get("AUTOVIDEO_REPO_ROOT", str(Path(__file__).resolve().parents[3]))) / "output/orchestrated/wan14b-20260706-155110/integrator")
        / "caches"
        / "wan14b_compiled_ffn_epilogue_v1"
        / f"rank_{rank}"
    )
    cache_preexisting = cache_root.exists() and any(cache_root.iterdir())
    inductor_cache = cache_root / "torchinductor"
    triton_cache = cache_root / "triton"
    temp_root = cache_root / "tmp"
    xdg_cache = cache_root / "xdg"
    for path in (inductor_cache, triton_cache, temp_root, xdg_cache):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    os.environ["TMPDIR"] = str(temp_root)
    os.environ["TEMP"] = str(temp_root)
    os.environ["TMP"] = str(temp_root)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)

    def fused_ffn_epilogue(
        hidden_states: torch.Tensor,
        norm_hidden_states: torch.Tensor,
        input_weight: torch.Tensor,
        input_bias: torch.Tensor,
        output_weight: torch.Tensor,
        output_bias: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        projected = torch.nn.functional.linear(
            norm_hidden_states, input_weight, input_bias
        )
        projected = torch.nn.functional.gelu(projected, approximate="tanh")
        branch = torch.nn.functional.linear(projected, output_weight, output_bias)
        return (hidden_states.float() + branch.float() * gate).to(torch.bfloat16)

    compiled_epilogue = torch.compile(
        fused_ffn_epilogue,
        fullgraph=True,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )
    _COMPILED_FFN_EPILOGUE_ACTIVATION = {
        "installed": True,
        "compiler": "torch.compile/TorchInductor",
        "fullgraph": True,
        "dynamic": False,
        "mode": "max-autotune-no-cudagraphs",
        "cache_root": str(cache_root),
        "cache_preexisting": cache_preexisting,
        "cold_compile_plus_first_dispatch_s": None,
        "formula": "linear -> gelu(approximate=tanh) -> linear -> gated_residual",
        "residual_dtype": "fp32_accumulate_then_bfloat16_store",
    }

    def first_epilogue(*args):
        global _COMPILED_FFN_EPILOGUE_DISPATCH
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = compiled_epilogue(*args)
        torch.cuda.synchronize()
        _COMPILED_FFN_EPILOGUE_ACTIVATION[
            "cold_compile_plus_first_dispatch_s"
        ] = time.perf_counter() - started
        _COMPILED_FFN_EPILOGUE_DISPATCH = compiled_epilogue
        return result

    _COMPILED_FFN_EPILOGUE_DISPATCH = first_epilogue
    model_stats: dict[str, Any] = {}
    total_modules = 0
    for model_name in ("transformer", "transformer_2"):
        model = getattr(pipe, model_name, None)
        if model is None:
            continue
        blocks = list(getattr(model, "blocks", ()))
        for block in blocks:
            net = getattr(getattr(block, "ffn", None), "net", None)
            if (
                net is None
                or len(net) < 3
                or not hasattr(net[0], "proj")
                or not isinstance(net[2], torch.nn.Linear)
            ):
                raise RuntimeError(
                    f"unsupported Wan FFN structure in {model_name}: {type(net)}"
                )
        model_stats[model_name] = {"ffn_modules": len(blocks)}
        total_modules += len(blocks)
    if total_modules == 0:
        raise RuntimeError("no Wan FFN modules found for compiled FFN epilogue")

    _COMPILED_FFN_EPILOGUE_ACTIVATION["models"] = model_stats
    _COMPILED_FFN_EPILOGUE_ACTIVATION["ffn_modules"] = total_modules
    _COMPILED_FFN_EPILOGUE_INSTALLED = True
    return _COMPILED_FFN_EPILOGUE_ACTIVATION


def install_compiled_qk_rope(pipe: Any) -> dict[str, Any]:
    """Compile the exact pairwise real-valued RoPE rotation for self Q/K."""

    global _COMPILED_QK_ROPE_INSTALLED
    global _COMPILED_QK_ROPE_ACTIVATION
    global _COMPILED_QK_ROPE_DISPATCH
    global _ORIGINAL_WAN_PROCESSOR_CALL

    if _COMPILED_QK_ROPE_INSTALLED:
        return _COMPILED_QK_ROPE_ACTIVATION

    worktree_root = (Path(os.environ.get("AUTOVIDEO_REPO_ROOT", str(Path(__file__).resolve().parents[3]))) / "output/orchestrated/wan14b-20260706-155110/integrator")
    existing_inductor = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if existing_inductor and Path(existing_inductor).resolve().is_relative_to(
        worktree_root
    ):
        cache_root = Path(existing_inductor).resolve().parent
    else:
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        cache_root = (
            worktree_root
            / "caches"
            / "wan14b_compiled_qk_rope_v1"
            / f"rank_{rank}"
        )
        inductor_cache = cache_root / "torchinductor"
        triton_cache = cache_root / "triton"
        temp_root = cache_root / "tmp"
        xdg_cache = cache_root / "xdg"
        for path in (inductor_cache, triton_cache, temp_root, xdg_cache):
            path.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
        os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
        os.environ["TMPDIR"] = str(temp_root)
        os.environ["TEMP"] = str(temp_root)
        os.environ["TMP"] = str(temp_root)
        os.environ["XDG_CACHE_HOME"] = str(xdg_cache)

    cache_preexisting = bool(existing_inductor)

    def rotate_qk(
        query: torch.Tensor,
        key: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = freqs_cos[..., 0::2]
        sin = freqs_sin[..., 1::2]

        def rotate(hidden_states: torch.Tensor) -> torch.Tensor:
            pairs = hidden_states.unflatten(-1, (-1, 2))
            x1 = pairs[..., 0]
            x2 = pairs[..., 1]
            even = x1 * cos - x2 * sin
            odd = x1 * sin + x2 * cos
            return torch.stack((even, odd), dim=-1).flatten(-2).to(
                hidden_states.dtype
            )

        return rotate(query), rotate(key)

    compiled_rotate_qk = torch.compile(
        rotate_qk,
        fullgraph=True,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )

    _COMPILED_QK_ROPE_ACTIVATION = {
        "installed": True,
        "compiler": "torch.compile/TorchInductor",
        "fullgraph": True,
        "dynamic": False,
        "mode": "max-autotune-no-cudagraphs",
        "cache_root": str(cache_root),
        "cache_preexisting": cache_preexisting,
        "cold_compile_plus_first_dispatch_s": None,
        "rotation": "pairwise_real_RoPE_for_self_attention_Q_and_K",
    }

    def first_rotate_qk(*args):
        global _COMPILED_QK_ROPE_DISPATCH
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = compiled_rotate_qk(*args)
        torch.cuda.synchronize()
        _COMPILED_QK_ROPE_ACTIVATION[
            "cold_compile_plus_first_dispatch_s"
        ] = time.perf_counter() - started
        _COMPILED_QK_ROPE_DISPATCH = compiled_rotate_qk
        return result

    _COMPILED_QK_ROPE_DISPATCH = first_rotate_qk

    from diffusers.models.transformers import transformer_wan as wan_module

    _ORIGINAL_WAN_PROCESSOR_CALL = wan_module.WanAttnProcessor.__call__

    def compiled_processor_call(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if (
            encoder_hidden_states is not None
            or rotary_emb is None
            or attn.add_k_proj is not None
            or kwargs
        ):
            return _ORIGINAL_WAN_PROCESSOR_CALL(
                self,
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                rotary_emb,
                **kwargs,
            )

        if (
            _COMPILED_QKV_NORM_ROPE_DISPATCH is not None
            and getattr(attn, "fused_projections", False)
            and hasattr(attn, "to_qkv")
        ):
            qkv = attn.to_qkv(hidden_states)
            query, key, value = _COMPILED_QKV_NORM_ROPE_DISPATCH(
                qkv,
                attn.norm_q.weight,
                attn.norm_k.weight,
                rotary_emb[0],
                rotary_emb[1],
            )
        else:
            query, key, value = wan_module._get_qkv_projections(
                attn, hidden_states, encoder_hidden_states
            )
            query = attn.norm_q(query)
            key = attn.norm_k(key)
            query = query.unflatten(2, (attn.heads, -1))
            key = key.unflatten(2, (attn.heads, -1))
            value = value.unflatten(2, (attn.heads, -1))
            query, key = _COMPILED_QK_ROPE_DISPATCH(
                query, key, rotary_emb[0], rotary_emb[1]
            )

        hidden_states = wan_module.dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    wan_module.WanAttnProcessor.__call__ = compiled_processor_call
    patched_self_attention_modules = sum(
        len(getattr(model, "blocks", ()))
        for model in (
            getattr(pipe, "transformer", None),
            getattr(pipe, "transformer_2", None),
        )
        if model is not None
    )
    if patched_self_attention_modules == 0:
        raise RuntimeError("no Wan self-attention modules found for compiled QK RoPE")
    _COMPILED_QK_ROPE_ACTIVATION[
        "patched_self_attention_modules"
    ] = patched_self_attention_modules
    _COMPILED_QK_ROPE_ACTIVATION["cross_attention_fallback_unchanged"] = True
    _COMPILED_QK_ROPE_INSTALLED = True
    return _COMPILED_QK_ROPE_ACTIVATION


def install_compiled_qkv_norm_rope(pipe: Any) -> dict[str, Any]:
    """Compile Q/K RMSNorm, head reshape, and pairwise RoPE after fused QKV.

    This keeps the existing attention backend and Ulysses communication
    unchanged.  It only replaces the eager attention front end with one
    TorchInductor graph after ``fuse_projections`` has eagerly packed self Q/K/V.
    Cross-attention and unfused/fallback processors remain on the native path.
    """

    global _COMPILED_QKV_NORM_ROPE_INSTALLED
    global _COMPILED_QKV_NORM_ROPE_ACTIVATION
    global _COMPILED_QKV_NORM_ROPE_DISPATCH

    if _COMPILED_QKV_NORM_ROPE_INSTALLED:
        return _COMPILED_QKV_NORM_ROPE_ACTIVATION

    reference_attn = None
    for model_name in ("transformer", "transformer_2"):
        model = getattr(pipe, model_name, None)
        if model is None:
            continue
        for block in getattr(model, "blocks", ()):
            config = getattr(block, "attn1", None)
            if config is not None:
                reference_attn = config
                break
        if reference_attn is not None:
            break
    if reference_attn is None:
        raise RuntimeError("no Wan self-attention module found for compiled QKV front end")
    if not getattr(reference_attn, "fused_projections", False) or not hasattr(
        reference_attn, "to_qkv"
    ):
        raise RuntimeError(
            "compiled QKV front end requires fused_qkv_projections to be enabled first"
        )

    heads = int(reference_attn.heads)
    inner_dim = int(reference_attn.norm_q.normalized_shape[0])
    q_eps = float(reference_attn.norm_q.eps)
    k_eps = float(reference_attn.norm_k.eps)
    model_stats: dict[str, Any] = {}
    self_modules = 0
    for model_name in ("transformer", "transformer_2"):
        model = getattr(pipe, model_name, None)
        if model is None:
            continue
        blocks = list(getattr(model, "blocks", ()))
        for block in blocks:
            attn = block.attn1
            if not getattr(attn, "fused_projections", False) or not hasattr(
                attn, "to_qkv"
            ):
                raise RuntimeError(
                    f"{model_name} contains an unfused self-attention projection"
                )
            if int(attn.heads) != heads or int(attn.norm_q.normalized_shape[0]) != inner_dim:
                raise RuntimeError("Wan self-attention shapes differ across transformer stages")
            if float(attn.norm_q.eps) != q_eps or float(attn.norm_k.eps) != k_eps:
                raise RuntimeError("Wan self-attention RMSNorm eps differs across stages")
        model_stats[model_name] = {"self_attention_modules": len(blocks)}
        self_modules += len(blocks)
    if self_modules == 0:
        raise RuntimeError("no Wan self-attention modules found for compiled QKV front end")

    def qkv_norm_reshape_rope(
        qkv: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = qkv.chunk(3, dim=-1)
        query = F.rms_norm(query, (inner_dim,), q_norm_weight, q_eps)
        key = F.rms_norm(key, (inner_dim,), k_norm_weight, k_eps)
        query = query.unflatten(2, (heads, -1))
        key = key.unflatten(2, (heads, -1))
        value = value.unflatten(2, (heads, -1))

        cos = freqs_cos[..., 0::2]
        sin = freqs_sin[..., 1::2]

        def rotate(hidden: torch.Tensor) -> torch.Tensor:
            pairs = hidden.unflatten(-1, (-1, 2))
            x1 = pairs[..., 0]
            x2 = pairs[..., 1]
            even = x1 * cos - x2 * sin
            odd = x1 * sin + x2 * cos
            return torch.stack((even, odd), dim=-1).flatten(-2).to(hidden.dtype)

        return rotate(query), rotate(key), value

    compile_options = {
        "fullgraph": True,
        "dynamic": False,
        "mode": "max-autotune-no-cudagraphs",
    }
    compiled_front_end = torch.compile(qkv_norm_reshape_rope, **compile_options)
    existing_inductor = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    cache_root = (
        str(Path(existing_inductor).resolve().parent)
        if existing_inductor
        else None
    )
    _COMPILED_QKV_NORM_ROPE_ACTIVATION = {
        "installed": True,
        "compiler": "torch.compile/TorchInductor",
        "fullgraph": True,
        "dynamic": False,
        "mode": "max-autotune-no-cudagraphs",
        "cache_root": cache_root,
        "cold_compile_plus_first_dispatch_s": None,
        "fused_operations": [
            "eager_fused_self_qkv_linear_boundary",
            "q_norm_rms_norm",
            "k_norm_rms_norm",
            "qkv_head_reshape",
            "pairwise_real_rope_qk",
        ],
        "heads": heads,
        "head_dim": inner_dim // heads,
        "self_attention_modules": self_modules,
        "models": model_stats,
        "cross_attention_fallback_unchanged": True,
    }

    def first_front_end(*args):
        global _COMPILED_QKV_NORM_ROPE_DISPATCH
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = compiled_front_end(*args)
        torch.cuda.synchronize()
        _COMPILED_QKV_NORM_ROPE_ACTIVATION[
            "cold_compile_plus_first_dispatch_s"
        ] = time.perf_counter() - started
        _COMPILED_QKV_NORM_ROPE_DISPATCH = compiled_front_end
        return result

    _COMPILED_QKV_NORM_ROPE_DISPATCH = first_front_end
    _COMPILED_QKV_NORM_ROPE_INSTALLED = True
    return _COMPILED_QKV_NORM_ROPE_ACTIVATION


def install_native_flash_self_attention(pipe: Any) -> dict[str, Any]:
    """Select PyTorch native Flash SDPA for exact dense self-attention."""

    global _NATIVE_FLASH_SELF_ATTENTION_INSTALLED
    if _NATIVE_FLASH_SELF_ATTENTION_INSTALLED:
        return {"installed": True, "already_installed": True}

    model_stats: dict[str, Any] = {}
    total_modules = 0
    previous_backends: set[str] = set()
    for model_name in ("transformer", "transformer_2"):
        model = getattr(pipe, model_name, None)
        if model is None:
            continue
        blocks = list(getattr(model, "blocks", ()))
        for block in blocks:
            processor = block.attn1.processor
            previous_backends.add(str(getattr(processor, "_attention_backend", None)))
            processor._attention_backend = "_native_flash"
        model_stats[model_name] = {"self_attention_modules": len(blocks)}
        total_modules += len(blocks)
    if total_modules == 0:
        raise RuntimeError("no Wan self-attention processors found for native Flash")

    _NATIVE_FLASH_SELF_ATTENTION_INSTALLED = True
    return {
        "installed": True,
        "backend": "_native_flash",
        "implementation": "torch native Flash scaled_dot_product_attention",
        "previous_backends": sorted(previous_backends),
        "self_attention_modules": total_modules,
        "models": model_stats,
        "cross_attention_backend_unchanged": True,
        "dense_attention_preserved": True,
    }


def install_native_cudnn_self_attention(pipe: Any) -> dict[str, Any]:
    """Select PyTorch native cuDNN SDPA for exact dense self-attention."""

    global _NATIVE_CUDNN_SELF_ATTENTION_INSTALLED
    if _NATIVE_CUDNN_SELF_ATTENTION_INSTALLED:
        return {"installed": True, "already_installed": True}

    model_stats: dict[str, Any] = {}
    total_modules = 0
    previous_backends: set[str] = set()
    for model_name in ("transformer", "transformer_2"):
        model = getattr(pipe, model_name, None)
        if model is None:
            continue
        blocks = list(getattr(model, "blocks", ()))
        for block in blocks:
            processor = block.attn1.processor
            previous_backends.add(str(getattr(processor, "_attention_backend", None)))
            processor._attention_backend = "_native_cudnn"
        model_stats[model_name] = {"self_attention_modules": len(blocks)}
        total_modules += len(blocks)
    if total_modules == 0:
        raise RuntimeError("no Wan self-attention processors found for native cuDNN")

    _NATIVE_CUDNN_SELF_ATTENTION_INSTALLED = True
    return {
        "installed": True,
        "backend": "_native_cudnn",
        "implementation": "torch native cuDNN scaled_dot_product_attention",
        "previous_backends": sorted(previous_backends),
        "self_attention_modules": total_modules,
        "models": model_stats,
        "cross_attention_backend_unchanged": True,
        "dense_attention_preserved": True,
    }


def install_compiled_native_sdpa() -> dict[str, Any]:
    """Compile native dense SDPA while preserving the CP attention contract."""

    global _COMPILED_NATIVE_SDPA_INSTALLED
    global _COMPILED_NATIVE_SDPA_ACTIVATION
    global _COMPILED_NATIVE_SDPA_DISPATCH
    global _ORIGINAL_NATIVE_ATTENTION_FORWARD_OP
    if _COMPILED_NATIVE_SDPA_INSTALLED:
        return _COMPILED_NATIVE_SDPA_ACTIVATION

    from diffusers.models import attention_dispatch as dispatch

    _ORIGINAL_NATIVE_ATTENTION_FORWARD_OP = dispatch._native_attention_forward_op
    worktree_root = (Path(os.environ.get("AUTOVIDEO_REPO_ROOT", str(Path(__file__).resolve().parents[3]))) / "output/orchestrated/wan14b-20260706-155110/integrator")
    existing_inductor = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    cache_root = (
        worktree_root
        / "caches"
        / "wan14b_compiled_native_sdpa_v1"
        / f"rank_{rank}"
    )
    if existing_inductor and Path(existing_inductor).resolve().is_relative_to(
        worktree_root
    ):
        cache_root = Path(existing_inductor).resolve().parent
    inductor_cache = cache_root / "torchinductor"
    triton_cache = cache_root / "triton"
    temp_root = cache_root / "tmp"
    xdg_cache = cache_root / "xdg"
    cache_preexisting = cache_root.exists() and any(cache_root.iterdir())
    for path in (inductor_cache, triton_cache, temp_root, xdg_cache):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    os.environ["TMPDIR"] = str(temp_root)
    os.environ["TEMP"] = str(temp_root)
    os.environ["TMP"] = str(temp_root)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)

    def compiled_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout_p: float,
        is_causal: bool,
        scale: float | None,
        enable_gqa: bool,
    ) -> torch.Tensor:
        output = torch.nn.functional.scaled_dot_product_attention(
            query=query.permute(0, 2, 1, 3),
            key=key.permute(0, 2, 1, 3),
            value=value.permute(0, 2, 1, 3),
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        return output.permute(0, 2, 1, 3)

    compiled_dispatch = torch.compile(
        compiled_sdpa,
        fullgraph=True,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )
    _COMPILED_NATIVE_SDPA_ACTIVATION = {
        "installed": True,
        "compiler": "torch.compile/TorchInductor",
        "fullgraph": True,
        "dynamic": False,
        "mode": "max-autotune-no-cudagraphs",
        "cache_root": str(cache_root),
        "cache_preexisting": cache_preexisting,
        "cold_compile_plus_first_dispatch_s": None,
        "attention_formula": "dense scaled_dot_product_attention",
        "return_lse_contract": "fallback_to_native_original_when_true",
        "dense_attention_preserved": True,
    }

    def compiled_native_forward_op(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
        return_lse: bool = False,
        _save_ctx: bool = True,
        _parallel_config=None,
    ):
        if return_lse or torch.is_grad_enabled():
            return _ORIGINAL_NATIVE_ATTENTION_FORWARD_OP(
                ctx,
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                return_lse,
                _save_ctx,
                _parallel_config,
            )
        if _save_ctx:
            ctx.save_for_backward(query, key, value)
            ctx.attn_mask = attn_mask
            ctx.dropout_p = dropout_p
            ctx.is_causal = is_causal
            ctx.scale = scale
            ctx.enable_gqa = enable_gqa
        global _COMPILED_NATIVE_SDPA_DISPATCH
        if _COMPILED_NATIVE_SDPA_DISPATCH is None:
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = compiled_dispatch(
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
            )
            torch.cuda.synchronize()
            _COMPILED_NATIVE_SDPA_ACTIVATION[
                "cold_compile_plus_first_dispatch_s"
            ] = time.perf_counter() - started
            _COMPILED_NATIVE_SDPA_DISPATCH = compiled_dispatch
            return output
        return _COMPILED_NATIVE_SDPA_DISPATCH(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
        )

    dispatch._native_attention_forward_op = compiled_native_forward_op
    _COMPILED_NATIVE_SDPA_INSTALLED = True
    return _COMPILED_NATIVE_SDPA_ACTIVATION


def install_async_qkv_ulysses_all_to_all() -> dict[str, Any]:
    """Enqueue the three exact Ulysses input exchanges before waiting."""

    global _ASYNC_QKV_A2A_INSTALLED, _ORIGINAL_ASYNC_ULYSSES_FORWARD
    if _ASYNC_QKV_A2A_INSTALLED:
        return {"installed": True, "already_installed": True}
    if not torch.distributed.is_initialized():
        raise RuntimeError("async QKV Ulysses requires an initialized process group")
    if torch.distributed.get_world_size() != 4:
        raise RuntimeError("async QKV Ulysses is certified for world_size=4")

    from diffusers.models import attention_dispatch as dispatch

    _ORIGINAL_ASYNC_ULYSSES_FORWARD = dispatch.TemplatedUlyssesAttention.forward

    def async_forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
        dropout_p: float,
        is_causal: bool,
        scale: float | None,
        enable_gqa: bool,
        return_lse: bool,
        forward_op,
        backward_op,
        _parallel_config=None,
    ):
        ulysses_mesh = _parallel_config.context_parallel_config._ulysses_mesh
        world_size = _parallel_config.context_parallel_config.ulysses_degree
        if world_size != 4 or query.shape[2] % world_size:
            _COUNTERS["fallback_calls"] += 1
            return _ORIGINAL_ASYNC_ULYSSES_FORWARD(
                ctx,
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                return_lse,
                forward_op,
                backward_op,
                _parallel_config,
            )

        group = ulysses_mesh.get_group()
        ctx.forward_op = forward_op
        ctx.backward_op = backward_op
        ctx._parallel_config = _parallel_config

        batch, query_sequence, heads, head_dim = query.shape
        key_sequence = key.shape[1]
        local_heads = heads // world_size

        def destination_major(
            tensor: torch.Tensor, local_sequence: int, role: str
        ) -> torch.Tensor:
            layout = tensor.reshape(
                batch, local_sequence, world_size, local_heads, head_dim
            ).permute(2, 1, 0, 3, 4)
            if _REUSABLE_A2A_SOURCE_BUFFERS_ENABLED:
                source = _reusable_a2a_buffer(layout, f"{role}_source")
                source.copy_(layout)
                return source
            return layout.contiguous()

        inputs = (
            destination_major(query, query_sequence, "query"),
            destination_major(key, key_sequence, "key"),
            destination_major(value, key_sequence, "value"),
        )
        outputs = tuple(
            _reusable_a2a_buffer(tensor, role)
            for tensor, role in zip(inputs, ("query", "key", "value"), strict=True)
        )
        works = [
            torch.distributed.all_to_all_single(
                output.flatten(),
                tensor.flatten(),
                group=group,
                async_op=True,
            )
            for output, tensor in zip(outputs, inputs, strict=True)
        ]
        for work in works:
            work.wait()
        query, key, value = (
            tensor.flatten(0, 1).permute(1, 0, 2, 3).contiguous()
            for tensor in outputs
        )

        _COUNTERS["async_qkv_a2a_calls"] += 1
        _COUNTERS["async_qkv_input_collectives"] += 3
        _COUNTERS["baseline_qkv_input_collectives_equivalent"] += 3
        _COUNTERS["output_collectives_unchanged"] += 1
        _COUNTERS["async_qkv_elements_exchanged"] += sum(
            int(tensor.numel()) for tensor in inputs
        )

        out = forward_op(
            ctx,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            _save_ctx=True,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse, *_ = out

        out = out.reshape(
            batch, world_size, query_sequence, local_heads, head_dim
        ).permute(1, 3, 0, 2, 4).contiguous()
        if _DIRECT_OUTPUT_A2A_ENABLED:
            received_out = _reusable_a2a_buffer(out, "direct_output")
            torch.distributed.all_to_all_single(
                received_out.flatten(), out.flatten(), group=group
            )
            out = received_out
            _COUNTERS["direct_output_a2a_calls"] += 1
            _COUNTERS["direct_output_a2a_elements"] += int(out.numel())
        else:
            out = dispatch._all_to_all_single(out, group)
        out = out.flatten(0, 1).permute(1, 2, 0, 3).contiguous()

        if return_lse:
            lse = lse.reshape(
                batch, world_size, query_sequence, local_heads
            ).permute(1, 3, 0, 2).contiguous()
            lse = dispatch._all_to_all_single(lse, group)
            lse = lse.flatten(0, 1).permute(1, 2, 0).contiguous()
        else:
            lse = None
        return (out, lse) if return_lse else out

    dispatch.TemplatedUlyssesAttention.forward = staticmethod(async_forward)
    _ASYNC_QKV_A2A_INSTALLED = True
    return {
        "installed": True,
        "world_size": 4,
        "input_collectives_per_block": 3,
        "input_collectives_enqueued_before_wait": 3,
        "message_sizes_and_layouts": "unchanged_from_baseline",
        "output_collectives_per_block": 1,
        "fallback_supported": True,
    }


def install_reusable_ulysses_a2a_buffers() -> dict[str, Any]:
    """Reuse inference-only A2A receive buffers across identical block shapes.

    Ulysses attention materializes three receive tensors and, for direct output
    exchange, one more receive tensor on every executed block.  The tensors
    are fully consumed before the next attention call and the official path is
    inference-only, so caching those allocations by shape/device/dtype is
    exact while reducing allocator churn.  Source packing remains unchanged.
    """

    global _REUSABLE_A2A_BUFFERS_ENABLED
    _REUSABLE_A2A_BUFFERS_ENABLED = True
    return {
        "installed": True,
        "inference_only": True,
        "cached_buffer_kinds": [
            "query",
            "key",
            "value",
            "packed_qkv",
            "direct_output",
        ],
        "cache_key": "role_device_dtype_shape",
        "source_packing_unchanged": True,
    }


def install_reusable_ulysses_a2a_source_buffers() -> dict[str, Any]:
    """Reuse source-side destination-major input buffers for async QKV A2A.

    This is separate from receive-buffer reuse because source buffers are
    overwritten by a copy before each collective and are only safe for the
    inference-only async path after the prior collective has completed.
    """

    global _REUSABLE_A2A_SOURCE_BUFFERS_ENABLED
    _REUSABLE_A2A_SOURCE_BUFFERS_ENABLED = True
    return {
        "installed": True,
        "inference_only": True,
        "requires": "async_qkv_ulysses_a2a",
        "cached_buffer_kinds": [
            "query_source",
            "key_source",
            "value_source",
        ],
        "source_layout_unchanged": True,
        "communication_volume_unchanged": True,
    }


def _reusable_a2a_buffer(tensor: torch.Tensor, role: str) -> torch.Tensor:
    if not _REUSABLE_A2A_BUFFERS_ENABLED or torch.is_grad_enabled():
        return torch.empty_like(tensor)
    key = (
        role,
        tensor.device.type,
        tensor.device.index,
        tensor.dtype,
        tuple(tensor.shape),
    )
    cached = _REUSABLE_A2A_BUFFERS.get(key)
    if cached is not None:
        _COUNTERS["reusable_a2a_buffer_hits"] += 1
        return cached
    # A source destination-major tensor is a permuted view.  Keep every
    # reusable collective buffer contiguous so ``flatten()`` remains a view
    # and never introduces an implicit temporary copy.
    cached = torch.empty_like(tensor, memory_format=torch.contiguous_format)
    _REUSABLE_A2A_BUFFERS[key] = cached
    _COUNTERS["reusable_a2a_buffer_misses"] += 1
    return cached


def install_direct_ulysses_output_all_to_all() -> dict[str, Any]:
    """Use a direct preallocated collective for the exact output exchange."""

    global _DIRECT_OUTPUT_A2A_ENABLED
    _DIRECT_OUTPUT_A2A_ENABLED = True
    return {
        "installed": True,
        "implementation": "torch.distributed.all_to_all_single",
        "preallocated_receive_tensor": True,
        "message_layout_and_volume": "unchanged",
        "compatible_input_paths": [
            "async_qkv_ulysses_a2a",
            "packed_qkv_ulysses_a2a",
        ],
        "output_collectives_per_block": 1,
    }


def install_invariant_rope_cache(pipe: Any) -> dict[str, Any]:
    """Cache each model's shape-only Wan rotary position tensor pair."""

    global _INVARIANT_ROPE_CACHE_INSTALLED
    global _INVARIANT_ROPE_CACHE_ACTIVATION
    if _INVARIANT_ROPE_CACHE_INSTALLED:
        return _INVARIANT_ROPE_CACHE_ACTIVATION

    model_stats: dict[str, Any] = {}
    installed_models = 0
    for model_name in ("transformer", "transformer_2"):
        model = getattr(pipe, model_name, None)
        if model is None:
            continue
        rope = model.rope
        original_forward = rope.forward
        cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        stats = {
            "cache_entries": 0,
            "hits": 0,
            "misses": 0,
            "input_values_read": False,
        }

        def cached_rope_forward(
            self,
            hidden_states: torch.Tensor,
            *,
            _original=original_forward,
            _cache=cache,
            _stats=stats,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            device = hidden_states.device
            key = (
                tuple(hidden_states.shape),
                device.type,
                device.index,
                str(hidden_states.dtype),
            )
            cached = _cache.get(key)
            if cached is None:
                cached = _original(hidden_states)
                _cache[key] = cached
                _stats["misses"] += 1
                _stats["cache_entries"] = len(_cache)
                _COUNTERS["rope_cache_misses"] += 1
            else:
                _stats["hits"] += 1
                _COUNTERS["rope_cache_hits"] += 1
                _COUNTERS["rope_cached_elements_reused"] += sum(
                    int(tensor.numel()) for tensor in cached
                )
            return cached

        rope.forward = types.MethodType(cached_rope_forward, rope)
        model_stats[model_name] = stats
        installed_models += 1
    if installed_models == 0:
        raise RuntimeError("no Wan rotary-position modules found for invariant cache")

    _INVARIANT_ROPE_CACHE_ACTIVATION = {
        "installed": True,
        "models": model_stats,
        "installed_models": installed_models,
        "key": "latent_shape_device_dtype",
        "invariance_proof": "WanRotaryPosEmbed.forward reads hidden_states.shape only and fixed registered frequency buffers; it never reads latent values or timestep",
        "consumers_are_read_only": True,
    }
    _INVARIANT_ROPE_CACHE_INSTALLED = True
    return _INVARIANT_ROPE_CACHE_ACTIVATION


def install_invariant_conditioning_cache(pipe: Any) -> dict[str, Any]:
    """Cache exact text projections and cross-attention K/V by input identity."""

    global _INVARIANT_CONDITIONING_CACHE_INSTALLED
    global _INVARIANT_CONDITIONING_CACHE_ACTIVATION
    global _ORIGINAL_WAN_GET_QKV_PROJECTIONS
    if _INVARIANT_CONDITIONING_CACHE_INSTALLED:
        return _INVARIANT_CONDITIONING_CACHE_ACTIVATION

    text_stats: dict[str, Any] = {}
    model_count = 0
    for model_name in ("transformer", "transformer_2"):
        model = getattr(pipe, model_name, None)
        if model is None:
            continue
        text_projection = model.condition_embedder.text_embedder
        original_forward = text_projection.forward
        cache: dict[int, tuple[torch.Tensor, int, torch.Tensor]] = {}
        stats = {"entries": 0, "hits": 0, "misses": 0}

        def cached_text_forward(
            self,
            hidden_states: torch.Tensor,
            *args,
            _original=original_forward,
            _cache=cache,
            _stats=stats,
            **kwargs,
        ) -> torch.Tensor:
            if args or kwargs:
                return _original(hidden_states, *args, **kwargs)
            identity = id(hidden_states)
            version = int(getattr(hidden_states, "_version", 0))
            entry = _cache.get(identity)
            if entry is not None and entry[0] is hidden_states and entry[1] == version:
                _stats["hits"] += 1
                _COUNTERS["text_projection_cache_hits"] += 1
                return entry[2]
            projected = _original(hidden_states)
            _cache[identity] = (hidden_states, version, projected)
            _stats["entries"] = len(_cache)
            _stats["misses"] += 1
            _COUNTERS["text_projection_cache_misses"] += 1
            return projected

        text_projection.forward = types.MethodType(
            cached_text_forward, text_projection
        )
        text_stats[model_name] = stats
        model_count += 1
    if model_count == 0:
        raise RuntimeError("no Wan text projection modules found for conditioning cache")

    from diffusers.models.transformers import transformer_wan as wan_module

    _ORIGINAL_WAN_GET_QKV_PROJECTIONS = wan_module._get_qkv_projections
    attention_caches: dict[
        int,
        tuple[Any, dict[int, tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]]],
    ] = {}
    kv_stats = {"attention_modules": 0, "entries": 0, "hits": 0, "misses": 0}

    def cached_get_qkv_projections(
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if encoder_hidden_states is None or not attn.is_cross_attention:
            return _ORIGINAL_WAN_GET_QKV_PROJECTIONS(
                attn, hidden_states, encoder_hidden_states
            )

        query = attn.to_q(hidden_states)
        attn_identity = id(attn)
        module_entry = attention_caches.get(attn_identity)
        if module_entry is None or module_entry[0] is not attn:
            module_cache: dict[
                int, tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]
            ] = {}
            attention_caches[attn_identity] = (attn, module_cache)
            kv_stats["attention_modules"] = len(attention_caches)
        else:
            module_cache = module_entry[1]

        identity = id(encoder_hidden_states)
        version = int(getattr(encoder_hidden_states, "_version", 0))
        cached = module_cache.get(identity)
        if cached is not None and cached[0] is encoder_hidden_states and cached[1] == version:
            key, value = cached[2], cached[3]
            kv_stats["hits"] += 1
            _COUNTERS["cross_kv_cache_hits"] += 1
            _COUNTERS["cross_kv_cached_elements_reused"] += int(
                key.numel() + value.numel()
            )
        else:
            if getattr(attn, "fused_projections", False):
                key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
            else:
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
            module_cache[identity] = (
                encoder_hidden_states,
                version,
                key,
                value,
            )
            kv_stats["entries"] += 1
            kv_stats["misses"] += 1
            _COUNTERS["cross_kv_cache_misses"] += 1
        return query, key, value

    wan_module._get_qkv_projections = cached_get_qkv_projections
    _INVARIANT_CONDITIONING_CACHE_ACTIVATION = {
        "installed": True,
        "text_projection": text_stats,
        "cross_kv": kv_stats,
        "key": "strong_tensor_identity_plus_mutation_version",
        "input_lifetime_pinned_by_cache": True,
        "query_projection_unchanged": True,
        "cross_key_normalization_unchanged": True,
        "cross_attention_unchanged": True,
    }
    _INVARIANT_CONDITIONING_CACHE_INSTALLED = True
    return _INVARIANT_CONDITIONING_CACHE_ACTIVATION


def install_packed_qkv_ulysses_all_to_all() -> dict[str, Any]:
    """Pack three equal Ulysses Q/K/V exchanges into one equal-volume call.

    The packed tensor is destination-major: each rank's outgoing chunk contains
    Q, K, and V for the same destination head group.  The receive tensor is
    unpacked along that Q/K/V axis before the unchanged dense attention call.
    """

    global _PACKED_QKV_A2A_INSTALLED, _ORIGINAL_ULYSSES_FORWARD
    if _PACKED_QKV_A2A_INSTALLED:
        return {"installed": True, "already_installed": True}

    if not torch.distributed.is_initialized():
        raise RuntimeError("packed QKV Ulysses all-to-all requires an initialized process group")
    if torch.distributed.get_world_size() != 4:
        raise RuntimeError("packed QKV Ulysses all-to-all is certified for world_size=4")

    from diffusers.models import attention_dispatch as dispatch

    _ORIGINAL_ULYSSES_FORWARD = dispatch.TemplatedUlyssesAttention.forward

    def packed_forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
        dropout_p: float,
        is_causal: bool,
        scale: float | None,
        enable_gqa: bool,
        return_lse: bool,
        forward_op,
        backward_op,
        _parallel_config=None,
    ):
        ulysses_mesh = _parallel_config.context_parallel_config._ulysses_mesh
        world_size = _parallel_config.context_parallel_config.ulysses_degree
        if world_size != 4 or query.shape[2] % world_size:
            _COUNTERS["fallback_calls"] += 1
            return _ORIGINAL_ULYSSES_FORWARD(
                ctx,
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                return_lse,
                forward_op,
                backward_op,
                _parallel_config,
            )

        group = ulysses_mesh.get_group()
        ctx.forward_op = forward_op
        ctx.backward_op = backward_op
        ctx._parallel_config = _parallel_config

        batch, local_sequence, heads, head_dim = query.shape
        local_heads = heads // world_size

        def destination_major(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch, local_sequence, world_size, local_heads, head_dim
            ).permute(2, 1, 0, 3, 4)

        # [destination, qkv, local_sequence, batch, local_heads, head_dim].
        # torch.stack performs the same three layout materializations as the
        # baseline, but in one destination-major output allocation.
        packed_qkv = torch.stack(
            (
                destination_major(query),
                destination_major(key),
                destination_major(value),
            ),
            dim=1,
        )
        packed_received = _reusable_a2a_buffer(packed_qkv, "packed_qkv")
        torch.distributed.all_to_all_single(
            packed_received.flatten(), packed_qkv.flatten(), group=group
        )
        packed_qkv = packed_received
        query, key, value = packed_qkv.unbind(1)
        # Each unbound view is [source, sequence, batch, local_heads, dim].
        # Permute before flattening so the source/sequence merge and the final
        # batch-major materialization happen in one copy (not two).
        query, key, value = (
            tensor.permute(2, 0, 1, 3, 4).flatten(1, 2).contiguous()
            for tensor in (query, key, value)
        )

        _COUNTERS["packed_qkv_a2a_calls"] += 1
        _COUNTERS["packed_qkv_input_collectives"] += 1
        _COUNTERS["baseline_qkv_input_collectives_equivalent"] += 3
        _COUNTERS["qkv_input_collectives_eliminated"] += 2
        _COUNTERS["output_collectives_unchanged"] += 1
        _COUNTERS["qkv_elements_exchanged"] += int(packed_qkv.numel())

        out = forward_op(
            ctx,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            _save_ctx=True,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse, *_ = out

        out = out.reshape(
            batch, world_size, local_sequence, local_heads, head_dim
        ).permute(1, 3, 0, 2, 4).contiguous()
        if _DIRECT_OUTPUT_A2A_ENABLED:
            received_out = _reusable_a2a_buffer(out, "direct_output")
            torch.distributed.all_to_all_single(
                received_out.flatten(), out.flatten(), group=group
            )
            out = received_out
            _COUNTERS["direct_output_a2a_calls"] += 1
            _COUNTERS["direct_output_a2a_elements"] += int(out.numel())
        else:
            out = dispatch._all_to_all_single(out, group)
        out = out.flatten(0, 1).permute(1, 2, 0, 3).contiguous()

        if return_lse:
            lse = lse.reshape(
                batch, world_size, local_sequence, local_heads
            ).permute(1, 3, 0, 2).contiguous()
            lse = dispatch._all_to_all_single(lse, group)
            lse = lse.flatten(0, 1).permute(1, 2, 0).contiguous()
        else:
            lse = None

        return (out, lse) if return_lse else out

    dispatch.TemplatedUlyssesAttention.forward = staticmethod(packed_forward)
    _PACKED_QKV_A2A_INSTALLED = True
    return {
        "installed": True,
        "world_size": 4,
        "baseline_qkv_collectives_per_block": 3,
        "packed_qkv_collectives_per_block": 1,
        "output_collectives_per_block": 1,
        "fallback_supported": True,
    }
