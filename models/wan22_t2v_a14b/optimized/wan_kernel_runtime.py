"""Experiment-local instrumentation for exact Wan transformer optimization.

This module is deliberately inert unless one of the WAN22_KERNEL_* guards is
enabled. It never changes scheduler state, denoising steps, model selection,
or transformer inputs. Preflight mode wraps complete DiT forwards only to
record CUDA-event latency and one operator profile per MoE expert.
"""

from __future__ import annotations

import importlib.util
import json
import os
import statistics
from pathlib import Path
from typing import Any

import torch


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


def _timing_stats(values: list[float]) -> dict[str, Any]:
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


def _shape(value: Any) -> list[int] | None:
    return list(value.shape) if isinstance(value, torch.Tensor) else None


def _event_device_us(event: Any) -> float:
    for name in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(event, name, None)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


class WanKernelRuntime:
    """Collect measured-pass DiT structure and optional preflight profiles."""

    def __init__(self, pipe: Any, output_dir: Path):
        self.pipe = pipe
        self.output_dir = Path(output_dir)
        self.profile_enabled = _enabled("WAN22_KERNEL_PROFILE")
        self.ledger_enabled = _enabled("WAN22_KERNEL_LEDGER") or self.profile_enabled
        self.stack = tuple(
            filter(
                None,
                (x.strip() for x in os.environ.get("WAN22_KERNEL_STACK", "").split(",")),
            )
        )
        supported_stack = {
            "context_parallel_ulysses4",
            "context_parallel_ring2_ulysses2",
            "packed_qkv_ulysses_a2a",
            "fused_qkv_projections",
            "compiled_block_glue",
            "compiled_ffn",
            "compiled_ffn_epilogue",
            "compiled_qk_rope",
            "compiled_qkv_norm_rope",
            "native_flash_self_attention",
            "native_cudnn_self_attention",
            "compiled_native_sdpa",
            "async_qkv_ulysses_a2a",
            "invariant_rope_cache",
            "invariant_conditioning_cache",
            "direct_ulysses_output_a2a",
            "reusable_ulysses_a2a_buffers",
            "reusable_ulysses_a2a_source_buffers",
            "pisa_attention",
        }
        unknown_stack = sorted(set(self.stack) - supported_stack)
        if unknown_stack:
            raise RuntimeError(
                "WAN22_KERNEL_STACK contains an unknown experiment-local optimization: "
                + ",".join(unknown_stack)
            )
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.world_size = (
            torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        )
        self.optimization_activations: dict[str, Any] = {}
        if "fused_qkv_projections" in self.stack:
            from wan_kernel_optimizations import install_fused_qkv_projections

            self.optimization_activations["fused_qkv_projections"] = (
                install_fused_qkv_projections(pipe)
            )
        if "compiled_block_glue" in self.stack:
            from wan_kernel_optimizations import install_compiled_block_glue

            self.optimization_activations["compiled_block_glue"] = (
                install_compiled_block_glue(pipe)
            )
        if "compiled_ffn" in self.stack:
            from wan_kernel_optimizations import install_compiled_ffn

            self.optimization_activations["compiled_ffn"] = install_compiled_ffn(pipe)
        if "compiled_ffn_epilogue" in self.stack:
            from wan_kernel_optimizations import install_compiled_ffn_epilogue

            self.optimization_activations["compiled_ffn_epilogue"] = (
                install_compiled_ffn_epilogue(pipe)
            )
        if "compiled_qk_rope" in self.stack:
            from wan_kernel_optimizations import install_compiled_qk_rope

            self.optimization_activations["compiled_qk_rope"] = (
                install_compiled_qk_rope(pipe)
            )
        if "pisa_attention" in self.stack:
            from wan_kernel_optimizations import install_pisa_attention

            self.optimization_activations["pisa_attention"] = install_pisa_attention(
                pipe, self.output_dir
            )
        if "compiled_qkv_norm_rope" in self.stack:
            from wan_kernel_optimizations import install_compiled_qkv_norm_rope

            self.optimization_activations["compiled_qkv_norm_rope"] = (
                install_compiled_qkv_norm_rope(pipe)
            )
        if "native_flash_self_attention" in self.stack:
            from wan_kernel_optimizations import install_native_flash_self_attention

            self.optimization_activations["native_flash_self_attention"] = (
                install_native_flash_self_attention(pipe)
            )
        if "native_cudnn_self_attention" in self.stack:
            from wan_kernel_optimizations import install_native_cudnn_self_attention

            self.optimization_activations["native_cudnn_self_attention"] = (
                install_native_cudnn_self_attention(pipe)
            )
        if "compiled_native_sdpa" in self.stack:
            from wan_kernel_optimizations import install_compiled_native_sdpa

            self.optimization_activations["compiled_native_sdpa"] = (
                install_compiled_native_sdpa()
            )
        if "async_qkv_ulysses_a2a" in self.stack:
            from wan_kernel_optimizations import (
                install_async_qkv_ulysses_all_to_all,
            )

            self.optimization_activations["async_qkv_ulysses_a2a"] = (
                install_async_qkv_ulysses_all_to_all()
            )
        if "direct_ulysses_output_a2a" in self.stack:
            from wan_kernel_optimizations import (
                install_direct_ulysses_output_all_to_all,
            )

            self.optimization_activations["direct_ulysses_output_a2a"] = (
                install_direct_ulysses_output_all_to_all()
            )
        if "reusable_ulysses_a2a_buffers" in self.stack:
            from wan_kernel_optimizations import install_reusable_ulysses_a2a_buffers

            self.optimization_activations["reusable_ulysses_a2a_buffers"] = (
                install_reusable_ulysses_a2a_buffers()
            )
        if "reusable_ulysses_a2a_source_buffers" in self.stack:
            from wan_kernel_optimizations import (
                install_reusable_ulysses_a2a_source_buffers,
            )

            self.optimization_activations["reusable_ulysses_a2a_source_buffers"] = (
                install_reusable_ulysses_a2a_source_buffers()
            )
        if "invariant_rope_cache" in self.stack:
            from wan_kernel_optimizations import install_invariant_rope_cache

            self.optimization_activations["invariant_rope_cache"] = (
                install_invariant_rope_cache(pipe)
            )
        if "invariant_conditioning_cache" in self.stack:
            from wan_kernel_optimizations import install_invariant_conditioning_cache

            self.optimization_activations["invariant_conditioning_cache"] = (
                install_invariant_conditioning_cache(pipe)
            )
        if "packed_qkv_ulysses_a2a" in self.stack:
            from wan_kernel_optimizations import install_packed_qkv_ulysses_all_to_all

            self.optimization_activations["packed_qkv_ulysses_a2a"] = (
                install_packed_qkv_ulysses_all_to_all()
            )
        self.current_pass: dict[str, Any] | None = None
        self.completed_passes: list[dict[str, Any]] = []
        self.profiled_labels: set[str] = set()
        self.operator_profiles: dict[str, Any] = {}
        self.input_contract: dict[str, Any] = {}
        self.original_forwards: dict[str, Any] = {}
        if self.ledger_enabled:
            self._wrap_model(getattr(pipe, "transformer", None), "transformer")
            self._wrap_model(getattr(pipe, "transformer_2", None), "transformer_2")

    @property
    def active(self) -> bool:
        return self.ledger_enabled

    def _wrap_model(self, model: Any, label: str) -> None:
        if model is None:
            return
        original = model.forward
        self.original_forwards[label] = original

        def wrapped_forward(*args, **kwargs):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            timestep = kwargs.get("timestep", args[1] if len(args) > 1 else None)
            encoder_hidden_states = kwargs.get(
                "encoder_hidden_states", args[2] if len(args) > 2 else None
            )
            if label not in self.input_contract:
                config = model.config
                patch_size = list(config.patch_size)
                latent_shape = _shape(hidden_states)
                token_count = None
                if latent_shape and len(latent_shape) == 5:
                    token_count = (
                        latent_shape[2] // patch_size[0]
                        * (latent_shape[3] // patch_size[1])
                        * (latent_shape[4] // patch_size[2])
                    )
                self.input_contract[label] = {
                    "hidden_states_shape": latent_shape,
                    "hidden_states_dtype": str(getattr(hidden_states, "dtype", "")),
                    "timestep_shape": _shape(timestep),
                    "encoder_hidden_states_shape": _shape(encoder_hidden_states),
                    "encoder_hidden_states_dtype": str(
                        getattr(encoder_hidden_states, "dtype", "")
                    ),
                    "patch_size": patch_size,
                    "token_count": token_count,
                    "num_layers": int(config.num_layers),
                    "num_attention_heads": int(config.num_attention_heads),
                    "attention_head_dim": int(config.attention_head_dim),
                    "inner_dim": int(config.num_attention_heads * config.attention_head_dim),
                    "ffn_dim": int(config.ffn_dim),
                }

            current = self.current_pass
            if current is None:
                return original(*args, **kwargs)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            should_profile = (
                self.profile_enabled
                and str(current["tag"]).startswith("p")
                and label not in self.profiled_labels
            )
            if should_profile:
                activities = [
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
                with torch.profiler.profile(
                    activities=activities,
                    record_shapes=True,
                    profile_memory=False,
                    with_stack=False,
                    with_flops=True,
                ) as profile:
                    result = original(*args, **kwargs)
                self._record_operator_profile(label, profile)
                self.profiled_labels.add(label)
            else:
                result = original(*args, **kwargs)
            end.record()
            current["calls"].append(
                {
                    "model": label,
                    "start": start,
                    "end": end,
                    "ordinal": len(current["calls"]),
                }
            )
            return result

        model.forward = wrapped_forward

    def _record_operator_profile(self, label: str, profile: Any) -> None:
        events = list(profile.key_averages(group_by_input_shape=True))
        events.sort(key=_event_device_us, reverse=True)
        top = []
        for event in events[:100]:
            top.append(
                {
                    "name": event.key,
                    "count": int(event.count),
                    "self_device_time_us": _event_device_us(event),
                    "device_time_total_us": float(
                        getattr(
                            event,
                            "device_time_total",
                            getattr(event, "cuda_time_total", 0.0),
                        )
                        or 0.0
                    ),
                    "cpu_time_total_us": float(
                        getattr(event, "cpu_time_total", 0.0) or 0.0
                    ),
                    "flops": int(getattr(event, "flops", 0) or 0),
                    "input_shapes": str(getattr(event, "input_shapes", "")),
                }
            )
        payload = {
            "model": label,
            "timing_scope": "one_complete_CP4_DiT_forward_after_two_full_warmup_generations",
            "instrumentation": "torch.profiler CPU+CUDA; shape and FLOP recording; no output comparison",
            "top_ops_by_self_device_time": top,
        }
        self.operator_profiles[label] = payload
        (self.output_dir / f"kernel_profile_{label}.txt").write_text(
            profile.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total", row_limit=100
            )
        )
        self._write_json("kernel_operator_profiles.json", self.operator_profiles)

    def start_pass(self, tag: str) -> None:
        if not self.ledger_enabled:
            return
        if self.current_pass is not None:
            raise RuntimeError(f"kernel ledger pass {self.current_pass['tag']} was not finished")
        self.current_pass = {
            "tag": tag,
            "calls": [],
            "optimization_counter_start": self._optimization_counters(),
        }

    def finish_pass(self, tag: str) -> dict[str, Any]:
        if not self.ledger_enabled:
            return {}
        if self.current_pass is None or self.current_pass["tag"] != tag:
            raise RuntimeError(f"kernel ledger pass mismatch: expected {tag!r}")
        torch.cuda.synchronize()
        calls = self.current_pass["calls"]
        call_records = []
        by_model: dict[str, list[float]] = {}
        for call in calls:
            elapsed_ms = float(call["start"].elapsed_time(call["end"]))
            by_model.setdefault(call["model"], []).append(elapsed_ms)
            call_records.append(
                {"ordinal": call["ordinal"], "model": call["model"], "elapsed_ms": elapsed_ms}
            )
        summary = {
            "tag": tag,
            "timing_scope": "complete_DiT_forward_cuda_event",
            "call_count": len(call_records),
            "calls": call_records,
            "by_model": {name: _timing_stats(values) for name, values in by_model.items()},
            "all_calls": _timing_stats(
                [record["elapsed_ms"] for record in call_records]
            ),
            "optimization_dispatch": self._counter_delta(
                self.current_pass["optimization_counter_start"],
                self._optimization_counters(),
            ),
        }
        full_block_executions = sum(
            int(self.input_contract[record["model"]]["num_layers"])
            for record in call_records
        )
        # In a composed cache run, the asynchronous Ulysses dispatch counter is
        # the authoritative count of self-attention/block executions: cache hits
        # execute block 0 and bypass blocks 1-39.  In a kernel-only run it equals
        # the full structural count.  This keeps the integration ledger factual
        # without changing either verified implementation.
        observed_async_blocks = int(
            summary["optimization_dispatch"].get("async_qkv_a2a_calls", 0)
        )
        block_executions = (
            observed_async_blocks
            if "async_qkv_ulysses_a2a" in self.stack
            else full_block_executions
        )
        summary["optimization_dispatch"].update(
            {
                "full_work_block_executions": full_block_executions,
                "observed_block_executions": block_executions,
            }
        )
        if "fused_qkv_projections" in self.stack:
            summary["optimization_dispatch"].update(
                {
                    "transformer_block_executions": block_executions,
                    "baseline_projection_linear_calls_equivalent": 6
                    * block_executions,
                    "fused_projection_linear_calls": 3 * block_executions,
                    "projection_linear_calls_eliminated": 3 * block_executions,
                    "fused_self_qkv_calls": block_executions,
                    "fused_cross_kv_calls": block_executions,
                    "unfused_cross_q_calls_unchanged": block_executions,
                }
            )
        if "compiled_block_glue" in self.stack:
            summary["optimization_dispatch"].update(
                {
                    "transformer_block_executions": block_executions,
                    "compiled_modulation_expression_calls": 2 * block_executions,
                    "compiled_gated_residual_expression_calls": 2
                    * block_executions,
                    "native_layer_norm_calls_unchanged": 3 * block_executions,
                    "attention_calls_unchanged": 2 * block_executions,
                    "ffn_calls_unchanged": block_executions,
                }
            )
        if "compiled_qk_rope" in self.stack:
            summary["optimization_dispatch"].update(
                {
                    "transformer_block_executions": block_executions,
                    "compiled_qk_rope_pair_calls": block_executions,
                    "qk_tensors_rotated": 2 * block_executions,
                    "qk_norm_calls_unchanged": 2 * block_executions,
                    "self_attention_calls_unchanged": block_executions,
                }
            )
        if "compiled_ffn" in self.stack:
            summary["optimization_dispatch"].update(
                {
                    "compiled_ffn_modules": block_executions,
                    "compiled_ffn_calls": block_executions,
                    "ffn_formula_unchanged": True,
                }
            )
        if "compiled_ffn_epilogue" in self.stack:
            summary["optimization_dispatch"].update(
                {
                    "compiled_ffn_epilogue_modules": block_executions,
                    "compiled_ffn_epilogue_calls": block_executions,
                    "ffn_formula_unchanged": True,
                    "gated_residual_fused_into_ffn_epilogue": True,
                }
            )
        if "native_flash_self_attention" in self.stack:
            summary["optimization_dispatch"].update(
                {
                    "native_flash_dense_self_attention_calls": block_executions,
                    "self_attention_calls_unchanged": block_executions,
                    "cross_attention_calls_unchanged": block_executions,
                }
            )
        if "native_cudnn_self_attention" in self.stack:
            summary["optimization_dispatch"].update(
                {
                    "native_cudnn_dense_self_attention_calls": block_executions,
                    "self_attention_calls_unchanged": block_executions,
                    "cross_attention_calls_unchanged": block_executions,
                }
            )
        if "compiled_native_sdpa" in self.stack:
            summary["optimization_dispatch"].update(
                {
                    "compiled_dense_sdpa_calls": 2 * block_executions,
                    "self_attention_calls_unchanged": block_executions,
                    "cross_attention_calls_unchanged": block_executions,
                    "dense_attention_preserved": True,
                }
            )
        self.completed_passes.append(summary)
        self.current_pass = None
        self._write_json("dit_call_ledger.json", self.summary())
        return summary

    def environment(self) -> dict[str, Any]:
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        return {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": int(properties.total_memory),
            "triton_available": importlib.util.find_spec("triton") is not None,
            "torch_inductor_available": importlib.util.find_spec("torch._inductor") is not None,
            "distributed_initialized": torch.distributed.is_initialized(),
            "rank": self.rank,
            "world_size": self.world_size,
            "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
            "mem_efficient_sdp_enabled": bool(
                torch.backends.cuda.mem_efficient_sdp_enabled()
            ),
            "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
            "cudnn_sdp_enabled": bool(torch.backends.cuda.cudnn_sdp_enabled()),
        }

    def summary(self) -> dict[str, Any]:
        measured = [
            item for item in self.completed_passes if str(item["tag"]).startswith("p")
        ]
        return {
            "enabled": self.ledger_enabled,
            "profile_enabled": self.profile_enabled,
            "activation_stack": list(self.stack),
            "optimization_activations": self.optimization_activations,
            "input_contract": self.input_contract,
            "environment": self.environment() if self.ledger_enabled else {},
            "measured_passes": measured,
            "warmup_pass_count_recorded": sum(
                1 for item in self.completed_passes if item["tag"] == "warmup"
            ),
            "operator_profile_files": sorted(
                [f"kernel_profile_{label}.txt" for label in self.operator_profiles]
            ),
        }

    def _optimization_counters(self) -> dict[str, int]:
        try:
            from wan_kernel_optimizations import get_optimization_counters
        except ImportError:
            return {}
        return get_optimization_counters()

    @staticmethod
    def _counter_delta(start: dict[str, int], end: dict[str, int]) -> dict[str, int]:
        return {key: int(end.get(key, 0) - start.get(key, 0)) for key in sorted(set(start) | set(end))}

    def _write_json(self, name: str, payload: Any) -> None:
        if self.rank != 0:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
