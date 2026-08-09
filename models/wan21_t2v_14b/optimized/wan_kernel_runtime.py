"""Experiment-local exact Wan transformer optimizations and timing ledger.

The module is inert when ``WAN22_KERNEL_STACK`` is empty. Every installed
method preserves the scheduler, token set, attention density, block count,
precision floor, and conditional/unconditional DiT calls. Diagnostic paired
DiT calls run outside generation and never contribute to its output.
"""

from __future__ import annotations

import json
import os
import statistics
import time
import types
import weakref
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


KNOWN_METHODS = {
    "regional_compile",
    "qkv_fusion",
    "invariant_cache",
    "invariant_cache_v2",
    "cross_kv_cache",
    "bf16_block_glue",
    "bf16_output_glue",
    "native_cudnn_attention",
    "native_flash_attention",
}


def _bf16_layer_norm(module: Any, inputs: torch.Tensor) -> torch.Tensor:
    if not hasattr(module, "normalized_shape"):
        return module(inputs)
    return F.layer_norm(
        inputs,
        module.normalized_shape,
        getattr(module, "_wan_weight_16", None),
        getattr(module, "_wan_bias_16", None),
        module.eps,
    )


def _bf16_block_forward(
    block: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    rotary_emb: torch.Tensor,
) -> torch.Tensor:
    dtype = hidden_states.dtype
    modulation = block._wan_scale_shift_table_16 + temb.to(dtype)
    if temb.ndim == 4:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = modulation.chunk(
            6, dim=2
        )
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        c_shift_msa = c_shift_msa.squeeze(2)
        c_scale_msa = c_scale_msa.squeeze(2)
        c_gate_msa = c_gate_msa.squeeze(2)
    else:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = modulation.chunk(
            6, dim=1
        )

    norm_hidden_states = _bf16_layer_norm(block.norm1, hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
    attn_output = block.attn1(norm_hidden_states, None, None, rotary_emb)
    hidden_states = hidden_states + attn_output * gate_msa

    norm_hidden_states = _bf16_layer_norm(block.norm2, hidden_states)
    attn_output = block.attn2(norm_hidden_states, encoder_hidden_states, None, None)
    hidden_states = hidden_states + attn_output

    norm_hidden_states = _bf16_layer_norm(block.norm3, hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + c_scale_msa) + c_shift_msa
    ff_output = block.ffn(norm_hidden_states)
    hidden_states = hidden_states + ff_output * c_gate_msa
    return hidden_states


def _bf16_output_norm_forward(module: Any, inputs: torch.Tensor) -> torch.Tensor:
    """Execute the final per-DiT normalization in the allowed BF16 precision."""
    return _bf16_layer_norm(module, inputs.to(torch.bfloat16))


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def timing_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "total_ms": sum(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p25_ms": _percentile(values, 0.25),
        "p75_ms": _percentile(values, 0.75),
        "min_ms": min(values),
        "max_ms": max(values),
    }


class WanKernelRuntime:
    """Apply the cumulative exact stack and record complete DiT timings."""

    def __init__(self, pipe: Any, output_dir: Path):
        self.pipe = pipe
        self.model = pipe.transformer
        self.output_dir = Path(output_dir)
        self.stack = tuple(
            item.strip()
            for item in os.environ.get("WAN22_KERNEL_STACK", "").split(",")
            if item.strip()
        )
        unknown = sorted(set(self.stack) - KNOWN_METHODS)
        if unknown:
            raise RuntimeError(f"Unknown WAN22 kernel methods: {unknown}")
        self.active_transfeat = os.environ.get("WAN22_KERNEL_ACTIVE_CANDIDATE", "").strip()
        self.ledger_enabled = _enabled("WAN22_KERNEL_LEDGER") or bool(self.stack)
        self.pair_enabled = _enabled("WAN22_KERNEL_PAIR_BENCH") and bool(self.active_transfeat)
        self.cudagraph_mark_step = _enabled("WAN22_CUDAGRAPH_MARK_STEP")
        self.current_pass: dict[str, Any] | None = None
        self.completed_passes: list[dict[str, Any]] = []
        self.last_inputs: dict[str, Any] | None = None
        self.activation: dict[str, Any] = {
            "stack": list(self.stack),
            "active_transfeat": self.active_transfeat or None,
            "methods": [],
            "initialization_s": 0.0,
        }
        self._compiled_impls: list[Any] = []
        self._original_forward = self.model.forward
        self.invariant_cache_enabled = bool(
            {"invariant_cache", "invariant_cache_v2"}.intersection(self.stack)
        )
        self.invariant_cache_stats: dict[str, dict[str, int]] = {}
        self.cross_kv_cache_enabled = "cross_kv_cache" in self.stack
        self._bf16_block_original_forwards: list[Any] = []
        self._bf16_block_forwards: list[Any] = []
        self._bf16_output_original_norm_forward: Any | None = None
        self._bf16_output_norm_forward: Any | None = None
        self._bf16_output_scale_shift_32: torch.Tensor | None = None
        self._bf16_output_scale_shift_16: torch.Tensor | None = None
        self._native_attention_processors: list[tuple[Any, Any]] = []

        started = time.perf_counter()
        self._apply_stack()
        self.activation["initialization_s"] = time.perf_counter() - started
        if self.ledger_enabled:
            self._install_ledger()
        self._write("kernel_activation.json", self.activation)

    @property
    def active(self) -> bool:
        return bool(self.stack) or self.ledger_enabled

    def _apply_stack(self) -> None:
        if "invariant_cache_v2" in self.stack:
            started = time.perf_counter()
            self._install_invariant_caches_v2()
            self.activation["methods"].append(
                {
                    "id": "invariant_cache_v2",
                    "initialization_s": time.perf_counter() - started,
                    "cached_exact_values": [
                        "official-shape RoPE cos/sin tensors",
                        "positive/negative text projection by live tensor identity/version",
                        "complete timestep embedding/activation/projection shared by conditional and unconditional calls",
                        "patch embedding shared by conditional and unconditional calls",
                    ],
                    "scope": "single live request tensors; weak-reference identity and version guarded",
                }
            )
        elif "invariant_cache" in self.stack:
            started = time.perf_counter()
            self._install_invariant_caches()
            self.activation["methods"].append(
                {
                    "id": "invariant_cache",
                    "initialization_s": time.perf_counter() - started,
                    "cached_exact_values": [
                        "official-shape RoPE cos/sin tensors",
                        "positive/negative text projection by live tensor identity/version",
                        "timestep embedding/projection shared by conditional and unconditional calls",
                        "patch embedding shared by conditional and unconditional calls",
                    ],
                    "scope": "single live request tensors; weak-reference identity and version guarded",
                }
            )

        # Projection packing must precede compilation so the compiled graph sees
        # the canonical ON operator topology when methods are composed.
        if "qkv_fusion" in self.stack:
            started = time.perf_counter()
            self.model.fuse_qkv_projections()
            self.activation["methods"].append(
                {
                    "id": "qkv_fusion",
                    "initialization_s": time.perf_counter() - started,
                    "self_attention": "three concatenated affine maps -> one affine map + exact chunk",
                    "cross_attention": "two concatenated K/V affine maps -> one affine map + exact chunk",
                }
            )

        if "cross_kv_cache" in self.stack:
            if "qkv_fusion" not in self.stack or "invariant_cache_v2" not in self.stack:
                raise RuntimeError(
                    "cross_kv_cache requires qkv_fusion and invariant_cache_v2"
                )
            started = time.perf_counter()
            self._install_cross_attention_kv_cache()
            self.activation["methods"].append(
                {
                    "id": "cross_kv_cache",
                    "initialization_s": time.perf_counter() - started,
                    "block_count": len(self.model.blocks),
                    "cached_exact_values": "packed cross-attention K/V affine output and normalized K per block and conditioning branch",
                    "scope": "live positive/negative conditioning tensor identity/version",
                }
            )

        if "bf16_block_glue" in self.stack:
            started = time.perf_counter()
            self._install_bf16_block_glue()
            self.activation["methods"].append(
                {
                    "id": "bf16_block_glue",
                    "initialization_s": time.perf_counter() - started,
                    "block_count": len(self.model.blocks),
                    "precision": "bfloat16",
                    "scope": "AdaLN modulation, LayerNorm execution, attention/FFN gates, and residual updates inside repeated blocks",
                }
            )

        if "bf16_output_glue" in self.stack:
            started = time.perf_counter()
            self._install_bf16_output_glue()
            self.activation["methods"].append(
                {
                    "id": "bf16_output_glue",
                    "initialization_s": time.perf_counter() - started,
                    "precision": "bfloat16",
                    "scope": "final per-DiT output LayerNorm, scale/shift modulation, and adjacent casts",
                }
            )

        if "native_cudnn_attention" in self.stack:
            started = time.perf_counter()
            self._install_native_cudnn_attention()
            self.activation["methods"].append(
                {
                    "id": "native_cudnn_attention",
                    "initialization_s": time.perf_counter() - started,
                    "backend": "_native_cudnn",
                    "attention_module_count": len(self._native_attention_processors),
                    "scope": "dense self- and cross-attention SDPA primitive selection",
                }
            )

        if "native_flash_attention" in self.stack:
            if "native_cudnn_attention" in self.stack:
                raise RuntimeError("Only one forced native attention backend may be active")
            started = time.perf_counter()
            self._install_native_flash_attention()
            self.activation["methods"].append(
                {
                    "id": "native_flash_attention",
                    "initialization_s": time.perf_counter() - started,
                    "backend": "_native_flash",
                    "attention_module_count": len(self._native_attention_processors),
                    "scope": "dense self- and cross-attention SDPA primitive selection",
                }
            )

        if "regional_compile" in self.stack:
            mode = os.environ.get("WAN22_COMPILE_MODE", "default")
            fullgraph = _enabled("WAN22_COMPILE_FULLGRAPH")
            started = time.perf_counter()
            self.model.compile_repeated_blocks(
                fullgraph=fullgraph,
                dynamic=False,
                mode=mode,
            )
            registration_s = time.perf_counter() - started
            self._compiled_impls = [
                getattr(block, "_compiled_call_impl", None) for block in self.model.blocks
            ]
            if not all(impl is not None for impl in self._compiled_impls):
                raise RuntimeError(
                    "Regional compilation did not install a compiled callable on every Wan block"
                )
            self.activation["methods"].append(
                {
                    "id": "regional_compile",
                    "mode": mode,
                    "fullgraph": fullgraph,
                    "dynamic": False,
                    "block_count": len(self._compiled_impls),
                    "registration_s": registration_s,
                    "cold_compile_location": "first excluded warmup generation",
                }
            )

    @staticmethod
    def _tensor_root(tensor: torch.Tensor) -> torch.Tensor:
        root = tensor
        seen: set[int] = set()
        while isinstance(getattr(root, "_base", None), torch.Tensor):
            if id(root) in seen:
                break
            seen.add(id(root))
            root = root._base
        return root

    @staticmethod
    def _tensor_version(tensor: torch.Tensor) -> int | None:
        try:
            return int(tensor._version)
        except RuntimeError:
            return None

    def _install_identity_cache(self, module: Any, label: str, capacity: int) -> None:
        original = module.forward
        entries: dict[tuple[Any, ...], tuple[weakref.ReferenceType, Any]] = {}
        order: list[tuple[Any, ...]] = []
        stats = self.invariant_cache_stats.setdefault(
            label, {"hits": 0, "misses": 0, "bypassed": 0}
        )

        def cached_forward(*args, **kwargs):
            if not self.invariant_cache_enabled:
                stats["bypassed"] += 1
                return original(*args, **kwargs)
            if len(args) != 1 or kwargs or not isinstance(args[0], torch.Tensor):
                stats["bypassed"] += 1
                return original(*args, **kwargs)
            tensor = args[0]
            root = self._tensor_root(tensor)
            key = (
                id(root),
                self._tensor_version(root),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                str(tensor.dtype),
                tensor.device.type,
                tensor.device.index,
            )
            entry = entries.get(key)
            if entry is not None and entry[0]() is root:
                stats["hits"] += 1
                if key in order:
                    order.remove(key)
                order.append(key)
                return entry[1]
            stats["misses"] += 1
            result = original(*args, **kwargs)
            entries[key] = (weakref.ref(root), result)
            if key in order:
                order.remove(key)
            order.append(key)
            while len(order) > capacity:
                entries.pop(order.pop(0), None)
            return result

        module.forward = cached_forward

    def _install_invariant_caches(self) -> None:
        rope = self.model.rope
        original_rope = rope.forward
        rope_entries: dict[tuple[Any, ...], Any] = {}
        rope_stats = self.invariant_cache_stats.setdefault(
            "rope", {"hits": 0, "misses": 0, "bypassed": 0}
        )

        def cached_rope(hidden_states: torch.Tensor):
            if not self.invariant_cache_enabled:
                rope_stats["bypassed"] += 1
                return original_rope(hidden_states)
            key = (
                tuple(hidden_states.shape),
                str(hidden_states.dtype),
                hidden_states.device.type,
                hidden_states.device.index,
                self._tensor_version(rope.freqs_cos),
                self._tensor_version(rope.freqs_sin),
            )
            if key in rope_entries:
                rope_stats["hits"] += 1
                return rope_entries[key]
            rope_stats["misses"] += 1
            result = original_rope(hidden_states)
            rope_entries.clear()
            rope_entries[key] = result
            return result

        rope.forward = cached_rope
        condition = self.model.condition_embedder
        self._install_identity_cache(self.model.patch_embedding, "patch_embedding", capacity=1)
        self._install_identity_cache(condition.timesteps_proj, "timesteps_proj", capacity=1)
        self._install_identity_cache(condition.time_embedder, "time_embedder", capacity=1)
        self._install_identity_cache(condition.act_fn, "time_act", capacity=1)
        self._install_identity_cache(condition.time_proj, "time_proj", capacity=1)
        self._install_identity_cache(condition.text_embedder, "text_embedder", capacity=2)

    def _install_invariant_caches_v2(self) -> None:
        rope = self.model.rope
        original_rope = rope.forward
        rope_entries: dict[tuple[Any, ...], Any] = {}
        rope_stats = self.invariant_cache_stats.setdefault(
            "rope", {"hits": 0, "misses": 0, "bypassed": 0}
        )

        def cached_rope(hidden_states: torch.Tensor):
            if not self.invariant_cache_enabled:
                rope_stats["bypassed"] += 1
                return original_rope(hidden_states)
            key = (
                tuple(hidden_states.shape),
                str(hidden_states.dtype),
                hidden_states.device.type,
                hidden_states.device.index,
                self._tensor_version(rope.freqs_cos),
                self._tensor_version(rope.freqs_sin),
            )
            if key in rope_entries:
                rope_stats["hits"] += 1
                return rope_entries[key]
            rope_stats["misses"] += 1
            result = original_rope(hidden_states)
            rope_entries.clear()
            rope_entries[key] = result
            return result

        rope.forward = cached_rope
        self._install_identity_cache(self.model.patch_embedding, "patch_embedding", capacity=1)

        condition = self.model.condition_embedder
        original_condition = condition.forward
        time_entries: dict[tuple[Any, ...], tuple[weakref.ReferenceType, Any]] = {}
        text_entries: dict[tuple[Any, ...], tuple[weakref.ReferenceType, Any]] = {}
        image_entries: dict[tuple[Any, ...], tuple[weakref.ReferenceType, Any]] = {}
        time_order: list[tuple[Any, ...]] = []
        text_order: list[tuple[Any, ...]] = []
        image_order: list[tuple[Any, ...]] = []
        time_stats = self.invariant_cache_stats.setdefault(
            "condition_time", {"hits": 0, "misses": 0, "bypassed": 0}
        )
        text_stats = self.invariant_cache_stats.setdefault(
            "condition_text", {"hits": 0, "misses": 0, "bypassed": 0}
        )
        image_stats = self.invariant_cache_stats.setdefault(
            "condition_image", {"hits": 0, "misses": 0, "bypassed": 0}
        )

        def tensor_key(tensor: torch.Tensor) -> tuple[tuple[Any, ...], torch.Tensor]:
            root = self._tensor_root(tensor)
            key = (
                id(root),
                self._tensor_version(root),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                str(tensor.dtype),
                tensor.device.type,
                tensor.device.index,
            )
            return key, root

        def lookup(
            entries: dict[tuple[Any, ...], tuple[weakref.ReferenceType, Any]],
            order: list[tuple[Any, ...]],
            key: tuple[Any, ...],
            root: torch.Tensor,
            stats: dict[str, int],
        ) -> Any | None:
            entry = entries.get(key)
            if entry is None or entry[0]() is not root:
                stats["misses"] += 1
                return None
            stats["hits"] += 1
            if key in order:
                order.remove(key)
            order.append(key)
            return entry[1]

        def insert(
            entries: dict[tuple[Any, ...], tuple[weakref.ReferenceType, Any]],
            order: list[tuple[Any, ...]],
            key: tuple[Any, ...],
            root: torch.Tensor,
            value: Any,
            capacity: int,
        ) -> None:
            entries[key] = (weakref.ref(root), value)
            if key in order:
                order.remove(key)
            order.append(key)
            while len(order) > capacity:
                entries.pop(order.pop(0), None)

        def cached_condition(
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            encoder_hidden_states_image: torch.Tensor | None = None,
            timestep_seq_len: int | None = None,
        ):
            if (
                not self.invariant_cache_enabled
                or not isinstance(timestep, torch.Tensor)
                or not isinstance(encoder_hidden_states, torch.Tensor)
            ):
                time_stats["bypassed"] += 1
                text_stats["bypassed"] += 1
                if encoder_hidden_states_image is not None:
                    image_stats["bypassed"] += 1
                return original_condition(
                    timestep,
                    encoder_hidden_states,
                    encoder_hidden_states_image,
                    timestep_seq_len=timestep_seq_len,
                )

            time_key, time_root = tensor_key(timestep)
            time_key = time_key + (
                timestep_seq_len,
                str(encoder_hidden_states.dtype),
                encoder_hidden_states.device.type,
                encoder_hidden_states.device.index,
            )
            time_value = lookup(
                time_entries, time_order, time_key, time_root, time_stats
            )
            if time_value is None:
                projected = condition.timesteps_proj(timestep)
                if timestep_seq_len is not None:
                    projected = projected.unflatten(0, (-1, timestep_seq_len))
                embedder_dtype = next(iter(condition.time_embedder.parameters())).dtype
                if projected.dtype != embedder_dtype and embedder_dtype != torch.int8:
                    projected = projected.to(embedder_dtype)
                temb = condition.time_embedder(projected).type_as(encoder_hidden_states)
                timestep_proj = condition.time_proj(condition.act_fn(temb))
                time_value = (temb, timestep_proj)
                insert(time_entries, time_order, time_key, time_root, time_value, capacity=1)
            temb, timestep_proj = time_value

            text_key, text_root = tensor_key(encoder_hidden_states)
            text_value = lookup(
                text_entries, text_order, text_key, text_root, text_stats
            )
            if text_value is None:
                text_value = condition.text_embedder(encoder_hidden_states)
                insert(text_entries, text_order, text_key, text_root, text_value, capacity=2)

            image_value = None
            if encoder_hidden_states_image is not None:
                if not isinstance(encoder_hidden_states_image, torch.Tensor):
                    image_stats["bypassed"] += 1
                    image_value = condition.image_embedder(encoder_hidden_states_image)
                else:
                    image_key, image_root = tensor_key(encoder_hidden_states_image)
                    image_value = lookup(
                        image_entries,
                        image_order,
                        image_key,
                        image_root,
                        image_stats,
                    )
                    if image_value is None:
                        image_value = condition.image_embedder(encoder_hidden_states_image)
                        insert(
                            image_entries,
                            image_order,
                            image_key,
                            image_root,
                            image_value,
                            capacity=1,
                        )

            return temb, timestep_proj, text_value, image_value

        condition.forward = cached_condition

    def _install_cross_attention_kv_cache(self) -> None:
        condition = self.model.condition_embedder
        original_condition = condition.forward
        records: list[tuple[Any, Any, Any]] = []
        stats = self.invariant_cache_stats.setdefault(
            "cross_attention_kv", {"hits": 0, "misses": 0, "bypassed": 0}
        )

        for block in self.model.blocks:
            attn = block.attn2
            if not getattr(attn, "is_cross_attention", False) or not hasattr(attn, "to_kv"):
                raise RuntimeError("Expected fused cross-attention K/V projection in every Wan block")
            projection = attn.to_kv
            norm = attn.norm_k
            original_projection = projection.forward
            original_norm = norm.forward
            records.append((attn, original_projection, original_norm))

            def cached_projection(hidden_states, _attn=attn, _original=original_projection):
                if not self.cross_kv_cache_enabled:
                    return _original(hidden_states)
                return _attn._wan_cross_packed_kv

            def cached_norm(key, _attn=attn, _original=original_norm):
                if not self.cross_kv_cache_enabled:
                    return _original(key)
                return _attn._wan_cross_normalized_key

            projection.forward = cached_projection
            norm.forward = cached_norm

        entries: dict[
            tuple[Any, ...], tuple[weakref.ReferenceType, list[tuple[torch.Tensor, torch.Tensor]]]
        ] = {}
        order: list[tuple[Any, ...]] = []

        def cached_condition(
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            encoder_hidden_states_image: torch.Tensor | None = None,
            timestep_seq_len: int | None = None,
        ):
            result = original_condition(
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                timestep_seq_len=timestep_seq_len,
            )
            if not self.cross_kv_cache_enabled:
                stats["bypassed"] += 1
                return result

            text_value = result[2]
            root = self._tensor_root(encoder_hidden_states)
            key = (
                id(root),
                self._tensor_version(root),
                tuple(encoder_hidden_states.shape),
                tuple(encoder_hidden_states.stride()),
                str(encoder_hidden_states.dtype),
                encoder_hidden_states.device.type,
                encoder_hidden_states.device.index,
                str(text_value.dtype),
                text_value.device.type,
                text_value.device.index,
            )
            entry = entries.get(key)
            if entry is not None and entry[0]() is root:
                stats["hits"] += 1
                values = entry[1]
                if key in order:
                    order.remove(key)
                order.append(key)
            else:
                stats["misses"] += 1
                values = []
                for _, original_projection, original_norm in records:
                    packed_kv = original_projection(text_value)
                    key_value, _ = packed_kv.chunk(2, dim=-1)
                    normalized_key = original_norm(key_value)
                    values.append((packed_kv, normalized_key))
                entries[key] = (weakref.ref(root), values)
                if key in order:
                    order.remove(key)
                order.append(key)
                while len(order) > 2:
                    entries.pop(order.pop(0), None)

            for (attn, _, _), (packed_kv, normalized_key) in zip(records, values):
                attn._wan_cross_packed_kv = packed_kv
                attn._wan_cross_normalized_key = normalized_key
            return result

        condition.forward = cached_condition

    def _install_bf16_block_glue(self) -> None:
        for block in self.model.blocks:
            self._bf16_block_original_forwards.append(block.forward)
            block._wan_scale_shift_table_16 = block.scale_shift_table.detach().to(torch.bfloat16)
            for norm in (block.norm1, block.norm2, block.norm3):
                if not hasattr(norm, "normalized_shape"):
                    continue
                norm._wan_weight_16 = (
                    norm.weight.detach().to(torch.bfloat16) if norm.weight is not None else None
                )
                norm._wan_bias_16 = (
                    norm.bias.detach().to(torch.bfloat16) if norm.bias is not None else None
                )
            replacement = types.MethodType(_bf16_block_forward, block)
            self._bf16_block_forwards.append(replacement)
            block.forward = replacement

    def _install_bf16_output_glue(self) -> None:
        norm = self.model.norm_out
        self._bf16_output_original_norm_forward = norm.forward
        self._bf16_output_norm_forward = types.MethodType(_bf16_output_norm_forward, norm)
        self._bf16_output_scale_shift_32 = self.model.scale_shift_table.detach().clone()
        self._bf16_output_scale_shift_16 = self._bf16_output_scale_shift_32.to(torch.bfloat16)
        self._set_bf16_output_glue(True)

    def _set_bf16_output_glue(self, enabled: bool) -> None:
        if self._bf16_output_original_norm_forward is None:
            return
        self.model.norm_out.forward = (
            self._bf16_output_norm_forward
            if enabled
            else self._bf16_output_original_norm_forward
        )
        value = (
            self._bf16_output_scale_shift_16
            if enabled
            else self._bf16_output_scale_shift_32
        )
        if value is None:
            raise RuntimeError("BF16 output-glue scale/shift tensors were not initialized")
        self.model.scale_shift_table.data = value

    def _install_native_cudnn_attention(self) -> None:
        for block in self.model.blocks:
            for attention in (block.attn1, block.attn2):
                processor = attention.processor
                original = getattr(processor, "_attention_backend", None)
                self._native_attention_processors.append((processor, original))
        self._set_native_cudnn_attention(True)

    def _set_native_cudnn_attention(self, enabled: bool) -> None:
        for processor, original in self._native_attention_processors:
            processor._attention_backend = "_native_cudnn" if enabled else original

    def _install_native_flash_attention(self) -> None:
        for block in self.model.blocks:
            for attention in (block.attn1, block.attn2):
                processor = attention.processor
                original = getattr(processor, "_attention_backend", None)
                self._native_attention_processors.append((processor, original))
        self._set_native_flash_attention(True)

    def _set_native_flash_attention(self, enabled: bool) -> None:
        for processor, original in self._native_attention_processors:
            processor._attention_backend = "_native_flash" if enabled else original

    def _install_ledger(self) -> None:
        original = self.model.forward

        def wrapped_forward(*args, **kwargs):
            if self.cudagraph_mark_step:
                torch.compiler.cudagraph_mark_step_begin()
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            timestep = kwargs.get("timestep", args[1] if len(args) > 1 else None)
            encoder_hidden_states = kwargs.get(
                "encoder_hidden_states", args[2] if len(args) > 2 else None
            )
            self.last_inputs = {
                "hidden_states": hidden_states,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_hidden_states_image": kwargs.get("encoder_hidden_states_image"),
                "return_dict": kwargs.get("return_dict", True),
                "attention_kwargs": kwargs.get("attention_kwargs"),
            }
            current = self.current_pass
            if current is None:
                return original(*args, **kwargs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            current["calls"].append((start, end))
            return result

        self.model.forward = wrapped_forward

    def start_pass(self, tag: str) -> None:
        if not self.ledger_enabled:
            return
        if self.current_pass is not None:
            raise RuntimeError(f"Unfinished DiT timing pass: {self.current_pass['tag']}")
        self.current_pass = {"tag": tag, "calls": []}

    def finish_pass(self, tag: str) -> dict[str, Any]:
        if not self.ledger_enabled:
            return {}
        if self.current_pass is None or self.current_pass["tag"] != tag:
            raise RuntimeError(f"DiT timing pass mismatch for {tag!r}")
        torch.cuda.synchronize()
        values = [float(start.elapsed_time(end)) for start, end in self.current_pass["calls"]]
        summary = {
            "tag": tag,
            "timing_scope": "complete_DiT_forward_cuda_event",
            "call_count": len(values),
            "stats": timing_stats(values),
            "calls_ms": values,
        }
        self.completed_passes.append(summary)
        self.current_pass = None
        self._write("dit_call_ledger.json", self.summary())
        return summary

    def _set_active_transfeat(self, enabled: bool) -> None:
        if self.active_transfeat == "regional_compile":
            if not self._compiled_impls:
                raise RuntimeError("regional_compile pair requested without compiled block callables")
            for block, compiled in zip(self.model.blocks, self._compiled_impls):
                block._compiled_call_impl = compiled if enabled else None
            return
        if self.active_transfeat == "qkv_fusion":
            # Keep packed modules resident so the ON compiled graph retains the
            # same module/parameter identities. OFF selects the original Q/K/V
            # paths and disables every compiled block; ON reselects packed
            # projections and restores the already-warmed compiled callables.
            # This yields a true composed-stack OFF/ON pair without recompiling
            # or letting diagnostic outputs affect generation.
            if not enabled and self._compiled_impls:
                for block in self.model.blocks:
                    block._compiled_call_impl = None
            fused_module_count = 0
            for module in self.model.modules():
                if hasattr(module, "fused_projections") and (
                    hasattr(module, "to_qkv") or hasattr(module, "to_kv")
                ):
                    module.fused_projections = enabled
                    fused_module_count += 1
            if fused_module_count == 0:
                raise RuntimeError("qkv_fusion toggle found no packed attention modules")
            if enabled and self._compiled_impls:
                for block, compiled in zip(self.model.blocks, self._compiled_impls):
                    block._compiled_call_impl = compiled
            return
        if self.active_transfeat in {"invariant_cache", "invariant_cache_v2"}:
            self.invariant_cache_enabled = enabled
            return
        if self.active_transfeat == "cross_kv_cache":
            self.cross_kv_cache_enabled = enabled
            return
        if self.active_transfeat == "bf16_output_glue":
            self._set_bf16_output_glue(enabled)
            return
        if self.active_transfeat == "native_cudnn_attention":
            self._set_native_cudnn_attention(enabled)
            return
        if self.active_transfeat == "native_flash_attention":
            self._set_native_flash_attention(enabled)
            return
        raise RuntimeError(f"No paired toggle for active transfeat {self.active_transfeat!r}")

    def set_composed_stack(self, enabled: bool) -> None:
        """Toggle every retained method for a full-generation integration pair."""
        if not enabled and self._compiled_impls:
            for block in self.model.blocks:
                block._compiled_call_impl = None

        if "qkv_fusion" in self.stack:
            fused_module_count = 0
            for module in self.model.modules():
                if hasattr(module, "fused_projections") and (
                    hasattr(module, "to_qkv") or hasattr(module, "to_kv")
                ):
                    module.fused_projections = enabled
                    fused_module_count += 1
            if fused_module_count == 0:
                raise RuntimeError("Composed-stack toggle found no packed attention modules")

        if {"invariant_cache", "invariant_cache_v2"}.intersection(self.stack):
            self.invariant_cache_enabled = enabled
        if "cross_kv_cache" in self.stack:
            self.cross_kv_cache_enabled = enabled

        if self._bf16_block_forwards:
            selected = (
                self._bf16_block_forwards if enabled else self._bf16_block_original_forwards
            )
            for block, forward in zip(self.model.blocks, selected):
                block.forward = forward

        if self._bf16_output_original_norm_forward is not None:
            self._set_bf16_output_glue(enabled)

        if self._native_attention_processors:
            if "native_flash_attention" in self.stack:
                self._set_native_flash_attention(enabled)
            else:
                self._set_native_cudnn_attention(enabled)

        if enabled and self._compiled_impls:
            for block, compiled in zip(self.model.blocks, self._compiled_impls):
                block._compiled_call_impl = compiled

    def _time_direct_dit(self) -> float:
        if self.last_inputs is None:
            raise RuntimeError("No real DiT inputs were captured during warmup")
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            result = self.model(**self.last_inputs)
        end.record()
        torch.cuda.synchronize()
        elapsed = float(start.elapsed_time(end))
        del result
        return elapsed

    def run_paired_dit_benchmark(self) -> dict[str, Any] | None:
        """Pair OFF/ON on one captured real call; never compare tensor outputs."""
        if not self.pair_enabled:
            return None
        # One warm call for each state is excluded. Compilation itself has
        # already occurred in the two full generation warmups.
        self._set_active_transfeat(False)
        self._time_direct_dit()
        self._set_active_transfeat(True)
        self._time_direct_dit()

        off: list[float] = []
        on: list[float] = []
        orders = ((False, True), (True, False), (False, True), (True, False))
        for order in orders:
            for enabled in order:
                self._set_active_transfeat(enabled)
                value = self._time_direct_dit()
                (on if enabled else off).append(value)
        self._set_active_transfeat(True)
        off_stats = timing_stats(off)
        on_stats = timing_stats(on)
        speedup = off_stats["median_ms"] / on_stats["median_ms"]
        payload = {
            "schema_version": 1,
            "transfeat": self.active_transfeat,
            "stack": list(self.stack),
            "comparison_scope": (
                "full_composed_stack_OFF_eager_unpacked_vs_ON_compiled_packed"
                if self.active_transfeat == "qkv_fusion" and self._compiled_impls
                else "active_transfeat_OFF_ON"
            ),
            "timing_scope": "one_complete_DiT_forward_on_captured_official_shape_inputs",
            "warmup_policy": "two full generation warmups, then one excluded direct-DiT warmup in each state",
            "ordering": ["OFF", "ON", "ON", "OFF", "OFF", "ON", "ON", "OFF"],
            "repeat_count_per_state": len(off),
            "off": off_stats,
            "on": on_stats,
            "median_speedup": speedup,
            "extra_diagnostic_dit_calls": 10,
            "generation_output_dependency": False,
            "output_comparison_used": False,
        }
        self._write("paired_dit_benchmark.json", payload)
        return payload

    def environment(self) -> dict[str, Any]:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return {
            "device": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "stack": list(self.stack),
            "active_transfeat": self.active_transfeat or None,
            "activation": self.activation,
            "cudagraph_mark_step_before_each_dit": self.cudagraph_mark_step,
            "invariant_cache_stats": self.invariant_cache_stats,
            "environment": self.environment(),
            "generation_passes": self.completed_passes,
            "generation_invariants": {
                "steps_per_prompt": 50,
                "model_calls_per_step": 2,
                "dit_calls_per_prompt": 100,
                "blocks_per_dit": len(self.model.blocks),
            },
        }

    def finalize(self) -> dict[str, Any]:
        payload = self.summary()
        self._write("kernel_runtime.json", payload)
        return payload

    def _write(self, name: str, payload: Any) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
