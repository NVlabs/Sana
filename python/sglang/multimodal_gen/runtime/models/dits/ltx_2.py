# Copied and adapted from LTX-2 and WanVideo implementations.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
import json
import os
from contextlib import nullcontext
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2ArchConfig, LTX2Config
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_parallel_rank,
    get_sp_world_size,
    get_tp_rank,
    get_tp_world_size,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention, USPAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    modelopt_fp4_apply_linear_bias_gelu,
    modelopt_fp4_apply_linear_per_col_residual_gate,
    modelopt_fp4_apply_quantized_linear,
    modelopt_fp4_quantize_activation,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import timestep_embedding
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.dit_activation_dump import (
    dump_attention_debug_from_env,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


_LTX2_PROFILE_CONTEXT: list[tuple[str, object]] = []
_LTX2_PROFILE_EVENTS: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
_LTX2_PROFILE_DUMP_REGISTERED = False
_LTX2_PERTURBATION_MASK_CACHE: dict[tuple[object, ...], torch.Tensor] = {}
_LTX2_FUSED_ADALN_KERNELS: tuple[object, object] | None = None
_LTX2_FUSED_ADALN_IMPORT_FAILED = False
_LTX2_FUSED_ADALN_RUNTIME_DISABLED = False
_LTX2_FUSED_ADALN_WARNING_EMITTED = False
_LTX2_FUSED_MODULATE_RUNTIME_DISABLED = False
_LTX2_FUSED_MODULATE_WARNING_EMITTED = False
_LTX2_FUSED_QKNORM_RUNTIME_DISABLED = False
_LTX2_FUSED_QKNORM_WARNING_EMITTED = False
_LTX2_FUSED_QKNORM_ROPE_RUNTIME_DISABLED = False
_LTX2_FUSED_QKNORM_ROPE_WARNING_EMITTED = False
_LTX2_FUSED_DUAL_MODULATE_RUNTIME_DISABLED = False
_LTX2_FUSED_DUAL_MODULATE_WARNING_EMITTED = False
_LTX2_FUSED_CA_DUAL_MODULATE_RUNTIME_DISABLED = False
_LTX2_FUSED_CA_DUAL_MODULATE_WARNING_EMITTED = False
_LTX2_FUSED_ADA_VALUES_RUNTIME_DISABLED = False
_LTX2_FUSED_ADA_VALUES_WARNING_EMITTED = False
_LTX2_FUSED_ADA_VALUES_ALL_RUNTIME_DISABLED = False
_LTX2_FUSED_ADA_VALUES_ALL_WARNING_EMITTED = False
_LTX2_FUSED_ADA_DIRECT_RUNTIME_DISABLED = False
_LTX2_FUSED_ADA_DIRECT_WARNING_EMITTED = False
_LTX2_FUSED_GELU_INPLACE_RUNTIME_DISABLED = False
_LTX2_FUSED_GELU_INPLACE_WARNING_EMITTED = False
_LTX2_FUSED_Q_GATE_RUNTIME_DISABLED = False
_LTX2_FUSED_Q_GATE_WARNING_EMITTED = False
_LTX2_COMPILED_GATE_TO_OUT = None
_LTX2_COMPILED_GATE_TO_OUT_RUNTIME_DISABLED = False
_LTX2_COMPILED_GATE_TO_OUT_WARNING_EMITTED = False
_LTX2_COMPILED_GATE_TO_OUT_RESIDUAL = None
_LTX2_COMPILED_GATE_TO_OUT_RESIDUAL_RUNTIME_DISABLED = False
_LTX2_COMPILED_GATE_TO_OUT_RESIDUAL_WARNING_EMITTED = False
_LTX2_RMS_NORM_MODULATE = None
_LTX2_RMS_NORM_MODULATE_UNAVAILABLE = False
_LTX2_SPLIT_ROPE_INPLACE = None
_LTX2_SPLIT_ROPE_QK_INPLACE = None
_LTX2_SPLIT_ROPE_INPLACE_UNAVAILABLE = False
_LTX2_SPLIT_ROPE_QK_INPLACE_UNAVAILABLE = False
_LTX2_TE_NVFP4_RECIPE = None
_LTX2_TE_NVFP4_LINEAR_CLS = None
_LTX2_TE_NVFP4_FP8_AUTOCAST = None
_LTX2_TE_NVFP4_IMPORT_FAILED = False
_LTX2_TE_NVFP4_RUNTIME_DISABLED = False
_LTX2_TE_NVFP4_WARNING_EMITTED = False
_LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU_RUNTIME_DISABLED = False
_LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU_WARNING_EMITTED = False


def _ltx2_record_functions_enabled() -> bool:
    return os.environ.get("SGLANG_DIFFUSION_RECORD_FUNCTIONS", "0") == "1"


def _ltx2_event_profile_enabled() -> bool:
    return os.environ.get("SGLANG_DIFFUSION_LTX2_EVENT_PROFILE", "0") == "1"


def _ltx2_scoped_profile_name(name: str) -> str:
    if not _LTX2_PROFILE_CONTEXT:
        return name
    phase, step_index = _LTX2_PROFILE_CONTEXT[-1]
    return f"ltx2_phase::{phase}::step_{step_index}::{name}"


def _ltx2_push_profile_context(phase: object, step_index: object) -> int:
    _LTX2_PROFILE_CONTEXT.append((str(phase), step_index))
    return len(_LTX2_PROFILE_CONTEXT)


def _ltx2_pop_profile_context(token: int) -> None:
    del _LTX2_PROFILE_CONTEXT[token - 1 :]


def _ltx2_register_event_dump() -> None:
    global _LTX2_PROFILE_DUMP_REGISTERED
    if _LTX2_PROFILE_DUMP_REGISTERED:
        return
    atexit.register(_ltx2_dump_event_profile)
    _LTX2_PROFILE_DUMP_REGISTERED = True


def _ltx2_dump_event_profile() -> None:
    if not _LTX2_PROFILE_EVENTS:
        return
    output_path = os.environ.get("SGLANG_DIFFUSION_LTX2_PROFILE_PATH")
    if not output_path:
        output_path = f"ltx2_event_profile_{os.getpid()}.json"
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stats: dict[str, dict[str, float | int]] = {}
        for name, start_event, end_event in _LTX2_PROFILE_EVENTS:
            duration_ms = float(start_event.elapsed_time(end_event))
            item = stats.setdefault(
                name,
                {
                    "count": 0,
                    "total_ms": 0.0,
                    "min_ms": duration_ms,
                    "max_ms": duration_ms,
                },
            )
            item["count"] = int(item["count"]) + 1
            item["total_ms"] = float(item["total_ms"]) + duration_ms
            item["min_ms"] = min(float(item["min_ms"]), duration_ms)
            item["max_ms"] = max(float(item["max_ms"]), duration_ms)
        rows = []
        for name, item in stats.items():
            count = int(item["count"])
            total_ms = float(item["total_ms"])
            rows.append(
                {
                    "name": name,
                    "count": count,
                    "total_ms": total_ms,
                    "avg_ms": total_ms / count if count else 0.0,
                    "min_ms": float(item["min_ms"]),
                    "max_ms": float(item["max_ms"]),
                }
            )
        rows.sort(key=lambda item: item["total_ms"], reverse=True)
        abs_output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
        tmp_output_path = f"{abs_output_path}.tmp.{os.getpid()}"
        with open(tmp_output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "event_count": len(_LTX2_PROFILE_EVENTS),
                    "stats": rows,
                },
                f,
                indent=2,
            )
        os.replace(tmp_output_path, abs_output_path)
        logger.info("Saved LTX2 event profile to: %s", abs_output_path)
    except Exception as exc:
        logger.warning("Failed to dump LTX2 event profile: %s", exc)


class _LTX2ProfileScope:
    def __init__(self, name: str):
        self.name = _ltx2_scoped_profile_name(name)
        self.record_ctx = None
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        if _ltx2_record_functions_enabled():
            self.record_ctx = torch.profiler.record_function(self.name)
            self.record_ctx.__enter__()
        if _ltx2_event_profile_enabled() and torch.cuda.is_available():
            _ltx2_register_event_dump()
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.start_event is not None and self.end_event is not None:
                self.end_event.record()
                _LTX2_PROFILE_EVENTS.append(
                    (self.name, self.start_event, self.end_event)
                )
        finally:
            if self.record_ctx is not None:
                self.record_ctx.__exit__(exc_type, exc_val, exc_tb)
                self.record_ctx = None
        return False


def _ltx2_record_function(name: str):
    if _ltx2_record_functions_enabled() or _ltx2_event_profile_enabled():
        return _LTX2ProfileScope(name)
    return nullcontext()


ADALN_NUM_BASE_PARAMS = 6
ADALN_NUM_CROSS_ATTN_PARAMS = 3


def adaln_embedding_coefficient(cross_attention_adaln: bool) -> int:
    return ADALN_NUM_BASE_PARAMS + (
        ADALN_NUM_CROSS_ATTN_PARAMS if cross_attention_adaln else 0
    )


def _ltx2_is_perturbed(
    perturbation_config: dict[str, object],
    key: str,
    block_idx: int,
) -> bool:
    value = perturbation_config.get(key)
    if value is None:
        return False
    if key.endswith("_blocks"):
        return block_idx in value
    return bool(value)


def _ltx2_build_batched_perturbation_states(
    perturbation_configs: tuple[dict[str, object], ...],
    key: str,
    block_indices: tuple[int, ...],
    values: torch.Tensor,
) -> dict[int, tuple[torch.Tensor | None, bool]]:
    mask_cache: dict[tuple[int, ...], torch.Tensor] = {}
    states: dict[int, tuple[torch.Tensor | None, bool]] = {}
    for block_idx in block_indices:
        keep_values = []
        any_perturbed = False
        all_perturbed = True
        for config in perturbation_configs:
            perturbed = _ltx2_is_perturbed(config, key, block_idx)
            any_perturbed = any_perturbed or perturbed
            all_perturbed = all_perturbed and perturbed
            keep_values.append(0 if perturbed else 1)

        if not any_perturbed:
            states[block_idx] = (None, False)
        elif all_perturbed:
            states[block_idx] = (None, True)
        else:
            cache_key = tuple(keep_values)
            mask = mask_cache.get(cache_key)
            if mask is None:
                global_cache_key = (
                    cache_key,
                    values.device.type,
                    values.device.index,
                    values.dtype,
                    values.ndim,
                )
                mask = _LTX2_PERTURBATION_MASK_CACHE.get(global_cache_key)
                if mask is None:
                    mask = torch.tensor(
                        keep_values, device=values.device, dtype=values.dtype
                    ).view(len(keep_values), *([1] * (values.ndim - 1)))
                    _LTX2_PERTURBATION_MASK_CACHE[global_cache_key] = mask
                mask_cache[cache_key] = mask
            states[block_idx] = (mask, False)
    return states


def _ltx2_child_prefix(prefix: str, child: str) -> str:
    return f"{prefix}.{child}" if prefix else child


def apply_interleaved_rotary_emb(
    x: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    return x * cos + x_rotated * sin


def apply_split_rotary_emb(
    x: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    cos, sin = freqs
    if (
        x.ndim == 3
        and cos.ndim == 4
        and sin.ndim == 4
        and x.dtype == torch.bfloat16
        and cos.dtype == torch.bfloat16
        and sin.dtype == torch.bfloat16
        and x.is_cuda
        and x.is_contiguous()
        and cos.is_cuda
        and sin.is_cuda
    ):
        from sglang.jit_kernel.diffusion.triton.ltx2_rotary import (
            apply_ltx2_split_rotary_emb,
        )

        return apply_ltx2_split_rotary_emb(x, cos, sin)

    x_dtype = x.dtype
    needs_reshape = False
    if x.ndim != 4 and cos.ndim == 4:
        b = x.shape[0]
        _, h, t, _ = cos.shape
        x = x.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    last = x.shape[-1]
    if last % 2 != 0:
        raise ValueError(
            f"Expected x.shape[-1] to be even for split rotary, got {last}."
        )
    r = last // 2

    split_x = x.reshape(*x.shape[:-1], 2, r)
    first_x = split_x[..., :1, :]
    second_x = split_x[..., 1:, :]

    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)

    out = split_x * cos_u
    first_out = out[..., :1, :]
    second_out = out[..., 1:, :]
    first_out.addcmul_(-sin_u, second_x)
    second_out.addcmul_(sin_u, first_x)

    out = out.reshape(*out.shape[:-2], last)
    if needs_reshape:
        out = out.swapaxes(1, 2).reshape(b, t, -1)
    return out.to(dtype=x_dtype)


def _get_ltx2_split_rope_inplace():
    global _LTX2_SPLIT_ROPE_INPLACE, _LTX2_SPLIT_ROPE_INPLACE_UNAVAILABLE
    if _LTX2_SPLIT_ROPE_INPLACE_UNAVAILABLE:
        return None
    if _LTX2_SPLIT_ROPE_INPLACE is None:
        try:
            from sglang.jit_kernel.diffusion.triton.ltx2_rotary import (
                apply_ltx2_split_rotary_emb_inplace,
            )
        except Exception:
            _LTX2_SPLIT_ROPE_INPLACE_UNAVAILABLE = True
            return None
        _LTX2_SPLIT_ROPE_INPLACE = apply_ltx2_split_rotary_emb_inplace
    return _LTX2_SPLIT_ROPE_INPLACE


def _get_ltx2_split_rope_qk_inplace():
    global _LTX2_SPLIT_ROPE_QK_INPLACE, _LTX2_SPLIT_ROPE_QK_INPLACE_UNAVAILABLE
    if _LTX2_SPLIT_ROPE_QK_INPLACE_UNAVAILABLE:
        return None
    if _LTX2_SPLIT_ROPE_QK_INPLACE is None:
        try:
            from sglang.jit_kernel.diffusion.triton.ltx2_rotary import (
                apply_ltx2_split_rotary_emb_qk_inplace,
            )
        except Exception:
            _LTX2_SPLIT_ROPE_QK_INPLACE_UNAVAILABLE = True
            return None
        _LTX2_SPLIT_ROPE_QK_INPLACE = apply_ltx2_split_rotary_emb_qk_inplace
    return _LTX2_SPLIT_ROPE_QK_INPLACE


def _can_use_ltx2_split_rope_inplace(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> bool:
    return (
        x.ndim == 3
        and cos.ndim == 4
        and sin.ndim == 4
        and x.dtype == torch.bfloat16
        and cos.dtype == torch.bfloat16
        and sin.dtype == torch.bfloat16
        and x.is_cuda
        and x.is_contiguous()
        and cos.is_cuda
        and sin.is_cuda
        and not x.requires_grad
        and os.environ.get("SGLANG_LTX2_FUSED_QK_ROPE", "1") != "0"
    )


def apply_split_rotary_emb_inplace(
    x: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    cos, sin = freqs
    if _can_use_ltx2_split_rope_inplace(x, cos, sin):
        ltx2_split_rope_inplace = _get_ltx2_split_rope_inplace()
        if ltx2_split_rope_inplace is not None:
            try:
                return ltx2_split_rope_inplace(x, cos, sin)
            except Exception:
                pass
    return apply_split_rotary_emb(x, freqs)


def apply_split_rotary_emb_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: Tuple[torch.Tensor, torch.Tensor],
    k_freqs: Tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = freqs
    k_cos, k_sin = k_freqs
    if (
        q.shape == k.shape
        and k_cos.shape == cos.shape
        and k_sin.shape == sin.shape
        and _can_use_ltx2_split_rope_inplace(q, cos, sin)
        and _can_use_ltx2_split_rope_inplace(k, k_cos, k_sin)
    ):
        ltx2_split_rope_qk_inplace = _get_ltx2_split_rope_qk_inplace()
        if ltx2_split_rope_qk_inplace is not None:
            try:
                return ltx2_split_rope_qk_inplace(q, k, cos, sin, k_cos, k_sin)
            except Exception:
                pass
    return apply_split_rotary_emb_inplace(q, freqs), apply_split_rotary_emb_inplace(
        k, k_freqs
    )


# ==============================================================================
# Layers and Embeddings
# ==============================================================================


class LTX2AudioVideoRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        scale_factors: Tuple[int, ...] = (8, 32, 32),
        theta: float = 10000.0,
        causal_offset: int = 1,
        modality: str = "video",
        double_precision: bool = True,
        rope_type: str = "interleaved",
        num_attention_heads: int = 32,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.patch_size = int(patch_size)
        self.patch_size_t = int(patch_size_t)

        if rope_type not in ["interleaved", "split"]:
            raise ValueError(
                f"{rope_type=} not supported. Choose between 'interleaved' and 'split'."
            )
        self.rope_type = rope_type

        self.base_num_frames = int(base_num_frames)
        self.num_attention_heads = int(num_attention_heads)

        self.base_height = int(base_height)
        self.base_width = int(base_width)

        self.sampling_rate = int(sampling_rate)
        self.hop_length = int(hop_length)
        self.audio_latents_per_second = (
            float(self.sampling_rate) / float(self.hop_length) / float(scale_factors[0])
        )

        self.scale_factors = tuple(int(x) for x in scale_factors)
        self.theta = float(theta)
        self.causal_offset = int(causal_offset)

        self.modality = modality
        if self.modality not in ["video", "audio"]:
            raise ValueError(
                f"Modality {modality} is not supported. Supported modalities are `video` and `audio`."
            )
        self.double_precision = bool(double_precision)

    def prepare_video_coords(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        fps: float = 24.0,
        *,
        start_frame: int = 0,
    ) -> torch.Tensor:
        grid_f = torch.arange(
            start=int(start_frame),
            end=int(num_frames) + int(start_frame),
            step=self.patch_size_t,
            dtype=torch.float32,
            device=device,
        )
        grid_h = torch.arange(
            start=0,
            end=height,
            step=self.patch_size,
            dtype=torch.float32,
            device=device,
        )
        grid_w = torch.arange(
            start=0,
            end=width,
            step=self.patch_size,
            dtype=torch.float32,
            device=device,
        )
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)

        patch_size = (self.patch_size_t, self.patch_size, self.patch_size)
        patch_size_delta = torch.tensor(
            patch_size, dtype=grid.dtype, device=grid.device
        )
        patch_ends = grid + patch_size_delta.view(3, 1, 1, 1)

        latent_coords = torch.stack([grid, patch_ends], dim=-1)
        latent_coords = latent_coords.flatten(1, 3)
        latent_coords = latent_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        scale_tensor = torch.tensor(self.scale_factors, device=latent_coords.device)
        broadcast_shape = [1] * latent_coords.ndim
        broadcast_shape[1] = -1
        pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)
        pixel_coords[:, 0, ...] = (
            pixel_coords[:, 0, ...] + self.causal_offset - self.scale_factors[0]
        ).clamp(min=0)
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps
        return pixel_coords

    def prepare_audio_coords(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        *,
        start_frame: int = 0,
    ) -> torch.Tensor:
        grid_f = torch.arange(
            start=int(start_frame),
            end=int(num_frames) + int(start_frame),
            step=self.patch_size_t,
            dtype=torch.float32,
            device=device,
        )

        audio_scale_factor = self.scale_factors[0]
        grid_start_mel = grid_f * audio_scale_factor
        grid_start_mel = (
            grid_start_mel + self.causal_offset - audio_scale_factor
        ).clip(min=0)
        grid_start_s = grid_start_mel * self.hop_length / self.sampling_rate

        grid_end_mel = (grid_f + self.patch_size_t) * audio_scale_factor
        grid_end_mel = (grid_end_mel + self.causal_offset - audio_scale_factor).clip(
            min=0
        )
        grid_end_s = grid_end_mel * self.hop_length / self.sampling_rate

        audio_coords = torch.stack([grid_start_s, grid_end_s], dim=-1)
        audio_coords = audio_coords.unsqueeze(0).expand(batch_size, -1, -1)
        audio_coords = audio_coords.unsqueeze(1)
        return audio_coords

    def prepare_coords(self, *args, **kwargs):
        if self.modality == "video":
            return self.prepare_video_coords(*args, **kwargs)
        return self.prepare_audio_coords(*args, **kwargs)

    def forward(
        self,
        coords: torch.Tensor,
        device: Optional[Union[str, torch.device]] = None,
        out_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or coords.device
        out_dtype = out_dtype or coords.dtype
        num_pos_dims = coords.shape[1]

        if coords.ndim == 4:
            coords_start, coords_end = coords.chunk(2, dim=-1)
            coords = (coords_start + coords_end) / 2.0
            coords = coords.squeeze(-1)

        if self.modality == "video":
            max_positions = (self.base_num_frames, self.base_height, self.base_width)
        else:
            max_positions = (self.base_num_frames,)

        grid = torch.stack(
            [coords[:, i] / max_positions[i] for i in range(num_pos_dims)], dim=-1
        ).to(device)

        num_rope_elems = num_pos_dims * 2
        # LTX-2.3 HQ is sensitive to RoPE rounding; keep frequency generation on
        # the target device instead of caching a CPU/NumPy tensor.
        freqs_dtype = torch.float64 if self.double_precision else torch.float32
        pow_indices = torch.pow(
            self.theta,
            torch.linspace(
                start=0.0,
                end=1.0,
                steps=self.dim // num_rope_elems,
                dtype=freqs_dtype,
                device=device,
            ),
        )
        freqs = (pow_indices * torch.pi / 2.0).to(dtype=torch.float32)

        freqs = (grid.unsqueeze(-1) * 2 - 1) * freqs
        freqs = freqs.transpose(-1, -2).flatten(2)

        if self.rope_type == "interleaved":
            cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
            sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

            if self.dim % num_rope_elems != 0:
                cos_padding = torch.ones_like(
                    cos_freqs[:, :, : self.dim % num_rope_elems]
                )
                sin_padding = torch.zeros_like(
                    cos_freqs[:, :, : self.dim % num_rope_elems]
                )
                cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
                sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)
        else:
            expected_freqs = self.dim // 2
            current_freqs = freqs.shape[-1]
            pad_size = expected_freqs - current_freqs
            cos_freq = freqs.cos()
            sin_freq = freqs.sin()

            if pad_size != 0:
                cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
                sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])
                cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
                sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)

            b = cos_freq.shape[0]
            t = cos_freq.shape[1]
            cos_freq = cos_freq.reshape(b, t, self.num_attention_heads, -1)
            sin_freq = sin_freq.reshape(b, t, self.num_attention_heads, -1)
            cos_freqs = torch.swapaxes(cos_freq, 1, 2)
            sin_freqs = torch.swapaxes(sin_freq, 1, 2)

        return cos_freqs.to(dtype=out_dtype), sin_freqs.to(dtype=out_dtype)


def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(x, normalized_shape=(x.shape[-1],), eps=eps)


def _ltx2_fused_adaln_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FUSED_ADALN", "0") == "1"


def _ltx2_fused_modulate_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FUSED_MODULATE", "0") == "1"


def _ltx2_fused_residual_gate_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FUSED_RESIDUAL_GATE", "0") == "1"


def _ltx2_fused_qknorm_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FUSED_QKNORM", "0") == "1"


_LTX2_OFFICIAL_FA4_ATTENTION_DISABLED_REASON: str | None = None


def _ltx2_official_fa4_attention_enabled() -> bool:
    return (
        os.environ.get("SGLANG_LTX2_OFFICIAL_FA4_ATTENTION", "0") == "1"
        and _LTX2_OFFICIAL_FA4_ATTENTION_DISABLED_REASON is None
    )


def _ltx2_fused_qknorm_rope_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FUSED_QKNORM_ROPE", "0") == "1"


def _ltx2_cache_rope_emb_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_CACHE_ROPE_EMB", "0") == "1"


def _ltx2_fused_dual_modulate_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FUSED_DUAL_MODULATE", "0") == "1"


def _ltx2_fused_ca_dual_modulate_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FUSED_CA_DUAL_MODULATE", "0") == "1"


def _ltx2_fused_ada_values_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FUSED_ADA_VALUES", "0") == "1"


def _ltx2_fused_ada_values_all_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FUSED_ADA_VALUES_ALL", "0") == "1"


def _ltx2_fused_ada_values_packed_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FUSED_ADA_VALUES_PACKED", "0") == "1"


def _ltx2_fused_ada_direct_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FUSED_ADA_DIRECT", "0") == "1"


def _ltx2_fused_q_gate_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FUSED_Q_GATE", "0") == "1"


def _ltx2_fused_qkv_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FUSED_QKV", "0") == "1"


def _ltx2_fp4_shared_qkv_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FP4_SHARED_QKV", "0") == "1"


def _ltx2_fp4_shared_q_gate_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FP4_SHARED_Q_GATE", "0") == "1"


def _ltx2_fused_audio_qkvg_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FUSED_AUDIO_QKVG", "0") == "1"


def _ltx2_fused_kv_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FUSED_KV", "0") == "1"


def _ltx2_fused_ffn_proj_in_gelu_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU", "0") == "1"


def _ltx2_fused_gelu_inplace_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FUSED_GELU_INPLACE", "0") == "1"


def _ltx2_fp4_fused_proj_in_bias_gelu_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FP4_FUSED_PROJ_IN_BIAS_GELU", "0") == "1"


def _ltx2_fp4_fused_proj_out_bias_gate_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FP4_FUSED_PROJ_OUT_BIAS_GATE", "0") == "1"


def _ltx2_fp4_fused_attn_to_out_bias_gate_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FP4_FUSED_ATTN_TO_OUT_BIAS_GATE", "0") == "1"


def _ltx2_te_nvfp4_video_ffn_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_TE_NVFP4_VIDEO_FFN", "0") == "1"


def _ltx2_te_nvfp4_fused_proj_in_gelu_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU", "0") == "1"


def _ltx2_te_nvfp4_fused_proj_out_bias_gate_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE", "0") == "1"


def _ltx2_compile_gate_to_out_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_COMPILE_GATE_TO_OUT", "0") == "1"


def _ltx2_compile_gate_to_out_residual_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_COMPILE_GATE_TO_OUT_RESIDUAL", "0") == "1"


def _ltx2_compile_a2v_gate_to_out_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_COMPILE_A2V_GATE_TO_OUT", "0") == "1"


def _ltx2_share_guidance_prefix_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_SHARE_GUIDANCE_PREFIX", "0") == "1"


def _ltx2_linear_base_for_fusion(layer: nn.Module) -> nn.Module | None:
    base_layer = getattr(layer, "base_layer", layer)
    if base_layer is not layer and not (
        getattr(layer, "merged", False) or getattr(layer, "disable_lora", False)
    ):
        return None
    return base_layer


def _ltx2_env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def _ltx2_get_te_nvfp4_context():
    global _LTX2_TE_NVFP4_RECIPE
    global _LTX2_TE_NVFP4_LINEAR_CLS
    global _LTX2_TE_NVFP4_FP8_AUTOCAST
    global _LTX2_TE_NVFP4_IMPORT_FAILED
    global _LTX2_TE_NVFP4_RUNTIME_DISABLED
    global _LTX2_TE_NVFP4_WARNING_EMITTED

    if _LTX2_TE_NVFP4_RUNTIME_DISABLED or _LTX2_TE_NVFP4_IMPORT_FAILED:
        return None

    try:
        if (
            _LTX2_TE_NVFP4_RECIPE is None
            or _LTX2_TE_NVFP4_LINEAR_CLS is None
            or _LTX2_TE_NVFP4_FP8_AUTOCAST is None
        ):
            import transformer_engine.pytorch as te
            from transformer_engine.common.recipe import NVFP4BlockScaling
            from transformer_engine.pytorch import fp8_autocast

            _LTX2_TE_NVFP4_LINEAR_CLS = te.Linear
            _LTX2_TE_NVFP4_FP8_AUTOCAST = fp8_autocast
            _LTX2_TE_NVFP4_RECIPE = NVFP4BlockScaling(
                disable_rht=_ltx2_env_flag("SGLANG_LTX2_TE_NVFP4_DISABLE_RHT", "1"),
                disable_stochastic_rounding=_ltx2_env_flag(
                    "SGLANG_LTX2_TE_NVFP4_DISABLE_STOCHASTIC_ROUNDING", "1"
                ),
                disable_2d_quantization=_ltx2_env_flag(
                    "SGLANG_LTX2_TE_NVFP4_DISABLE_2D_QUANTIZATION", "1"
                ),
            )
    except Exception as exc:
        _LTX2_TE_NVFP4_IMPORT_FAILED = True
        if not _LTX2_TE_NVFP4_WARNING_EMITTED:
            logger.warning("Disabling LTX2 TE NVFP4 video FFN fast path: %s", exc)
            _LTX2_TE_NVFP4_WARNING_EMITTED = True
        return None

    return (
        _LTX2_TE_NVFP4_LINEAR_CLS,
        _LTX2_TE_NVFP4_FP8_AUTOCAST,
        _LTX2_TE_NVFP4_RECIPE,
    )


_LTX2_GUIDANCE_PERTURBATION_KEYS = (
    "skip_video_self_attn_blocks",
    "skip_audio_self_attn_blocks",
    "skip_a2v_cross_attn",
    "skip_v2a_cross_attn",
)


def _ltx2_guidance_prefix_share_plan(
    perturbation_configs: tuple[dict[str, object], ...] | None,
    block_indices: tuple[int, ...],
    batch_size: int,
) -> tuple[int, tuple[int, ...], tuple[int, ...]] | None:
    if (
        not _ltx2_share_guidance_prefix_enabled()
        or perturbation_configs is None
        or get_tp_world_size() != 1
        or int(batch_size) != len(perturbation_configs)
        or int(batch_size) not in (3, 4)
    ):
        return None

    cond_idx = 0
    perturbed_idx = 2
    cond_config = perturbation_configs[cond_idx]
    perturbed_config = perturbation_configs[perturbed_idx]

    for block_idx in block_indices:
        if any(
            _ltx2_is_perturbed(cond_config, key, block_idx)
            for key in _LTX2_GUIDANCE_PERTURBATION_KEYS
        ):
            return None

    if _ltx2_is_perturbed(
        perturbed_config, "skip_a2v_cross_attn", -1
    ) or _ltx2_is_perturbed(perturbed_config, "skip_v2a_cross_attn", -1):
        return None

    skip_blocks = sorted(
        set(int(v) for v in (perturbed_config.get("skip_video_self_attn_blocks") or ()))
        | set(
            int(v) for v in (perturbed_config.get("skip_audio_self_attn_blocks") or ())
        )
    )
    if not skip_blocks:
        return None
    first_skip_block = int(skip_blocks[0])
    if first_skip_block not in block_indices:
        return None
    if first_skip_block <= min(block_indices):
        return None

    for block_idx in block_indices:
        if block_idx >= first_skip_block:
            break
        if any(
            _ltx2_is_perturbed(perturbed_config, key, block_idx)
            for key in _LTX2_GUIDANCE_PERTURBATION_KEYS
        ):
            return None

    keep_indices = tuple(i for i in range(int(batch_size)) if i != perturbed_idx)
    expand_indices = tuple(
        cond_idx if i == perturbed_idx else i if i < perturbed_idx else i - 1
        for i in range(int(batch_size))
    )
    return first_skip_block, keep_indices, expand_indices


def _ltx2_index_batch_dim(
    tensor: torch.Tensor | None,
    indices: tuple[int, ...],
    full_batch_size: int,
) -> torch.Tensor | None:
    if tensor is None or not torch.is_tensor(tensor):
        return tensor
    if tensor.ndim == 0 or int(tensor.shape[0]) != int(full_batch_size):
        return tensor
    index = torch.tensor(indices, device=tensor.device, dtype=torch.long)
    return tensor.index_select(0, index)


def _ltx2_index_rotary_emb(
    pe: tuple[torch.Tensor, torch.Tensor] | None,
    indices: tuple[int, ...],
    full_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if pe is None:
        return None
    return (
        _ltx2_index_batch_dim(pe[0], indices, full_batch_size),
        _ltx2_index_batch_dim(pe[1], indices, full_batch_size),
    )


def _ltx2_build_perturbation_state_maps(
    perturbation_configs: tuple[dict[str, object], ...] | None,
    block_indices: tuple[int, ...],
    hidden_states: torch.Tensor,
    audio_hidden_states: torch.Tensor,
) -> tuple[
    dict[int, tuple[torch.Tensor | None, bool]] | None,
    dict[int, tuple[torch.Tensor | None, bool]] | None,
    dict[int, tuple[torch.Tensor | None, bool]] | None,
    dict[int, tuple[torch.Tensor | None, bool]] | None,
]:
    if perturbation_configs is None:
        return None, None, None, None
    return (
        _ltx2_build_batched_perturbation_states(
            perturbation_configs,
            "skip_video_self_attn_blocks",
            block_indices,
            hidden_states,
        ),
        _ltx2_build_batched_perturbation_states(
            perturbation_configs,
            "skip_audio_self_attn_blocks",
            block_indices,
            audio_hidden_states,
        ),
        _ltx2_build_batched_perturbation_states(
            perturbation_configs,
            "skip_a2v_cross_attn",
            block_indices,
            hidden_states,
        ),
        _ltx2_build_batched_perturbation_states(
            perturbation_configs,
            "skip_v2a_cross_attn",
            block_indices,
            audio_hidden_states,
        ),
    )


def _ltx2_get_fused_adaln_kernels() -> tuple[object, object] | None:
    global _LTX2_FUSED_ADALN_KERNELS, _LTX2_FUSED_ADALN_IMPORT_FAILED
    if _LTX2_FUSED_ADALN_KERNELS is not None:
        return _LTX2_FUSED_ADALN_KERNELS
    if _LTX2_FUSED_ADALN_IMPORT_FAILED:
        return None
    try:
        from sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift import (
            fused_norm_scale_shift,
            fused_scale_residual_norm_scale_shift,
        )

        _LTX2_FUSED_ADALN_KERNELS = (
            fused_norm_scale_shift,
            fused_scale_residual_norm_scale_shift,
        )
        return _LTX2_FUSED_ADALN_KERNELS
    except Exception as exc:
        _LTX2_FUSED_ADALN_IMPORT_FAILED = True
        logger.warning("LTX2 fused AdaLN kernels are unavailable: %s", exc)
        return None


def _ltx2_disable_fused_adaln(exc: Exception) -> None:
    global _LTX2_FUSED_ADALN_RUNTIME_DISABLED, _LTX2_FUSED_ADALN_WARNING_EMITTED
    _LTX2_FUSED_ADALN_RUNTIME_DISABLED = True
    if not _LTX2_FUSED_ADALN_WARNING_EMITTED:
        logger.warning("Disabling LTX2 fused AdaLN fast path after failure: %s", exc)
        _LTX2_FUSED_ADALN_WARNING_EMITTED = True


def _ltx2_disable_fused_modulate(exc: Exception) -> None:
    global _LTX2_FUSED_MODULATE_RUNTIME_DISABLED
    global _LTX2_FUSED_MODULATE_WARNING_EMITTED
    _LTX2_FUSED_MODULATE_RUNTIME_DISABLED = True
    if not _LTX2_FUSED_MODULATE_WARNING_EMITTED:
        logger.warning(
            "Disabling LTX2 fused modulation fast path after failure: %s", exc
        )
        _LTX2_FUSED_MODULATE_WARNING_EMITTED = True


def _ltx2_disable_fused_qknorm(exc: Exception) -> None:
    global _LTX2_FUSED_QKNORM_RUNTIME_DISABLED
    global _LTX2_FUSED_QKNORM_WARNING_EMITTED
    _LTX2_FUSED_QKNORM_RUNTIME_DISABLED = True
    if not _LTX2_FUSED_QKNORM_WARNING_EMITTED:
        logger.warning("Disabling LTX2 fused q/k norm fast path after failure: %s", exc)
        _LTX2_FUSED_QKNORM_WARNING_EMITTED = True


def _ltx2_disable_fused_qknorm_rope(exc: Exception) -> None:
    global _LTX2_FUSED_QKNORM_ROPE_RUNTIME_DISABLED
    global _LTX2_FUSED_QKNORM_ROPE_WARNING_EMITTED
    _LTX2_FUSED_QKNORM_ROPE_RUNTIME_DISABLED = True
    if not _LTX2_FUSED_QKNORM_ROPE_WARNING_EMITTED:
        logger.warning(
            "Disabling LTX2 fused q/k norm + RoPE fast path after failure: %s",
            exc,
        )
        _LTX2_FUSED_QKNORM_ROPE_WARNING_EMITTED = True


def _ltx2_disable_fused_dual_modulate(exc: Exception) -> None:
    global _LTX2_FUSED_DUAL_MODULATE_RUNTIME_DISABLED
    global _LTX2_FUSED_DUAL_MODULATE_WARNING_EMITTED
    _LTX2_FUSED_DUAL_MODULATE_RUNTIME_DISABLED = True
    if not _LTX2_FUSED_DUAL_MODULATE_WARNING_EMITTED:
        logger.warning(
            "Disabling LTX2 fused dual-modulate fast path after failure: %s", exc
        )
        _LTX2_FUSED_DUAL_MODULATE_WARNING_EMITTED = True


def _ltx2_disable_fused_ca_dual_modulate(exc: Exception) -> None:
    global _LTX2_FUSED_CA_DUAL_MODULATE_RUNTIME_DISABLED
    global _LTX2_FUSED_CA_DUAL_MODULATE_WARNING_EMITTED
    _LTX2_FUSED_CA_DUAL_MODULATE_RUNTIME_DISABLED = True
    if not _LTX2_FUSED_CA_DUAL_MODULATE_WARNING_EMITTED:
        logger.warning(
            "Disabling LTX2 fused CA dual-modulate fast path after failure: %s",
            exc,
        )
        _LTX2_FUSED_CA_DUAL_MODULATE_WARNING_EMITTED = True


def _ltx2_disable_fused_ada_values(exc: Exception) -> None:
    global _LTX2_FUSED_ADA_VALUES_RUNTIME_DISABLED
    global _LTX2_FUSED_ADA_VALUES_WARNING_EMITTED
    _LTX2_FUSED_ADA_VALUES_RUNTIME_DISABLED = True
    if not _LTX2_FUSED_ADA_VALUES_WARNING_EMITTED:
        logger.warning(
            "Disabling LTX2 fused Ada values fast path after failure: %s", exc
        )
        _LTX2_FUSED_ADA_VALUES_WARNING_EMITTED = True


def _ltx2_disable_fused_ada_values_all(exc: Exception) -> None:
    global _LTX2_FUSED_ADA_VALUES_ALL_RUNTIME_DISABLED
    global _LTX2_FUSED_ADA_VALUES_ALL_WARNING_EMITTED
    _LTX2_FUSED_ADA_VALUES_ALL_RUNTIME_DISABLED = True
    if not _LTX2_FUSED_ADA_VALUES_ALL_WARNING_EMITTED:
        logger.warning(
            "Disabling LTX2 fused all-Ada-values fast path after failure: %s",
            exc,
        )
        _LTX2_FUSED_ADA_VALUES_ALL_WARNING_EMITTED = True


def _ltx2_disable_fused_ada_direct(exc: Exception) -> None:
    global _LTX2_FUSED_ADA_DIRECT_RUNTIME_DISABLED
    global _LTX2_FUSED_ADA_DIRECT_WARNING_EMITTED
    _LTX2_FUSED_ADA_DIRECT_RUNTIME_DISABLED = True
    if not _LTX2_FUSED_ADA_DIRECT_WARNING_EMITTED:
        logger.warning(
            "Disabling LTX2 fused direct-Ada fast path after failure: %s",
            exc,
        )
        _LTX2_FUSED_ADA_DIRECT_WARNING_EMITTED = True


def _ltx2_disable_fused_gelu_inplace(exc: Exception) -> None:
    global _LTX2_FUSED_GELU_INPLACE_RUNTIME_DISABLED
    global _LTX2_FUSED_GELU_INPLACE_WARNING_EMITTED
    _LTX2_FUSED_GELU_INPLACE_RUNTIME_DISABLED = True
    if not _LTX2_FUSED_GELU_INPLACE_WARNING_EMITTED:
        logger.warning(
            "Disabling LTX2 fused in-place GELU fast path after failure: %s", exc
        )
        _LTX2_FUSED_GELU_INPLACE_WARNING_EMITTED = True


def _ltx2_disable_fused_q_gate(exc: Exception) -> None:
    global _LTX2_FUSED_Q_GATE_RUNTIME_DISABLED
    global _LTX2_FUSED_Q_GATE_WARNING_EMITTED
    _LTX2_FUSED_Q_GATE_RUNTIME_DISABLED = True
    if not _LTX2_FUSED_Q_GATE_WARNING_EMITTED:
        logger.warning(
            "Disabling LTX2 fused q+gate projection fast path after failure: %s",
            exc,
        )
        _LTX2_FUSED_Q_GATE_WARNING_EMITTED = True


def _ltx2_disable_compiled_gate_to_out(exc: Exception) -> None:
    global _LTX2_COMPILED_GATE_TO_OUT_RUNTIME_DISABLED
    global _LTX2_COMPILED_GATE_TO_OUT_WARNING_EMITTED
    _LTX2_COMPILED_GATE_TO_OUT_RUNTIME_DISABLED = True
    if not _LTX2_COMPILED_GATE_TO_OUT_WARNING_EMITTED:
        logger.warning(
            "Disabling LTX2 compiled gate-to-out fast path after failure: %s",
            exc,
        )
        _LTX2_COMPILED_GATE_TO_OUT_WARNING_EMITTED = True


def _ltx2_disable_compiled_gate_to_out_residual(exc: Exception) -> None:
    global _LTX2_COMPILED_GATE_TO_OUT_RESIDUAL_RUNTIME_DISABLED
    global _LTX2_COMPILED_GATE_TO_OUT_RESIDUAL_WARNING_EMITTED
    _LTX2_COMPILED_GATE_TO_OUT_RESIDUAL_RUNTIME_DISABLED = True
    if not _LTX2_COMPILED_GATE_TO_OUT_RESIDUAL_WARNING_EMITTED:
        logger.warning(
            "Disabling LTX2 compiled gate-to-out residual fast path after failure: %s",
            exc,
        )
        _LTX2_COMPILED_GATE_TO_OUT_RESIDUAL_WARNING_EMITTED = True


def _ltx2_gate_to_out_impl(
    out: torch.Tensor,
    gate_logits: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    scaled = out * (2.0 * torch.sigmoid(gate_logits).unsqueeze(-1))
    return F.linear(scaled.reshape(*scaled.shape[:-2], -1), weight, bias)


def _ltx2_get_compiled_gate_to_out():
    global _LTX2_COMPILED_GATE_TO_OUT
    if _LTX2_COMPILED_GATE_TO_OUT is None:
        _LTX2_COMPILED_GATE_TO_OUT = torch.compile(
            _ltx2_gate_to_out_impl,
            mode="max-autotune-no-cudagraphs",
            dynamic=False,
            fullgraph=True,
        )
    return _LTX2_COMPILED_GATE_TO_OUT


def _ltx2_gate_to_out_residual_impl(
    out: torch.Tensor,
    gate_logits: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    output_gate: torch.Tensor,
) -> torch.Tensor:
    scaled = out * (2.0 * torch.sigmoid(gate_logits).unsqueeze(-1))
    projected = F.linear(scaled.reshape(*scaled.shape[:-2], -1), weight, bias)
    return torch.addcmul(residual, projected, output_gate)


def _ltx2_get_compiled_gate_to_out_residual():
    global _LTX2_COMPILED_GATE_TO_OUT_RESIDUAL
    if _LTX2_COMPILED_GATE_TO_OUT_RESIDUAL is None:
        _LTX2_COMPILED_GATE_TO_OUT_RESIDUAL = torch.compile(
            _ltx2_gate_to_out_residual_impl,
            mode="max-autotune-no-cudagraphs",
            dynamic=False,
            fullgraph=True,
        )
    return _LTX2_COMPILED_GATE_TO_OUT_RESIDUAL


def _ltx2_try_fused_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: nn.Module,
    k_norm: nn.Module,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        not _ltx2_fused_qknorm_enabled()
        or _LTX2_FUSED_QKNORM_RUNTIME_DISABLED
        or get_tp_world_size() != 1
        or not isinstance(q_norm, torch.nn.RMSNorm)
        or not isinstance(k_norm, torch.nn.RMSNorm)
        or not q.is_cuda
        or not k.is_cuda
        or q.dtype not in (torch.float16, torch.bfloat16)
        or k.dtype != q.dtype
        or q.ndim != 3
        or k.ndim != 3
        or q.shape[-1] != k.shape[-1]
        or not q.is_contiguous()
        or not k.is_contiguous()
    ):
        return None

    q_weight = q_norm.weight
    k_weight = k_norm.weight
    hidden = int(q.shape[-1])
    if (
        q_weight is None
        or k_weight is None
        or q_weight.device != q.device
        or k_weight.device != k.device
        or q_weight.dtype != q.dtype
        or k_weight.dtype != k.dtype
        or q_weight.numel() != hidden
        or k_weight.numel() != hidden
    ):
        return None

    try:
        from sglang.jit_kernel.diffusion.triton.ltx2_qknorm import (
            ltx2_qknorm_pair_inplace,
        )

        with _ltx2_record_function("ltx2_fused_qknorm::pair_inplace"):
            q_view = q.view(-1, hidden)
            k_view = k.view(-1, hidden)
            ltx2_qknorm_pair_inplace(q_view, k_view, q_weight, k_weight, eps)
        return q, k
    except Exception as exc:
        _ltx2_disable_fused_qknorm(exc)
        return None


def _ltx2_try_official_fa4_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    profile_prefix: str,
) -> torch.Tensor | None:
    if (
        not _ltx2_official_fa4_attention_enabled()
        or get_tp_world_size() != 1
        or q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
        or not q.is_cuda
        or not k.is_cuda
        or not v.is_cuda
        or q.dtype not in (torch.float16, torch.bfloat16)
        or q.dtype != k.dtype
        or q.dtype != v.dtype
    ):
        return None
    try:
        from flash_attn.cute import flash_attn_func as flash_attn_4_func

        with _ltx2_record_function(f"ltx2_official_fa4_attention::{profile_prefix}"):
            out, _ = flash_attn_4_func(q.to(v.dtype), k.to(v.dtype), v)
        return out
    except Exception as exc:
        global _LTX2_OFFICIAL_FA4_ATTENTION_DISABLED_REASON
        _LTX2_OFFICIAL_FA4_ATTENTION_DISABLED_REASON = str(exc)
        logger.warning_once(
            f"Disabling official FA4 attention compatibility path for {profile_prefix}: {exc}"
        )
        return None


def _ltx2_try_fused_qknorm_split_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: nn.Module,
    k_norm: nn.Module,
    eps: float,
    pe: tuple[torch.Tensor, torch.Tensor],
    k_pe: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        not _ltx2_fused_qknorm_rope_enabled()
        or _LTX2_FUSED_QKNORM_ROPE_RUNTIME_DISABLED
        or get_tp_world_size() != 1
        or not isinstance(q_norm, torch.nn.RMSNorm)
        or not isinstance(k_norm, torch.nn.RMSNorm)
        or not q.is_cuda
        or not k.is_cuda
        or q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or q.ndim != 3
        or k.ndim != 3
        or q.shape[-1] != k.shape[-1]
        or not q.is_contiguous()
        or not k.is_contiguous()
    ):
        return None

    q_cos, q_sin = pe
    k_cos, k_sin = pe if k_pe is None else k_pe
    if (
        q_cos.ndim != 4
        or q_sin.shape != q_cos.shape
        or k_cos.ndim != 4
        or k_sin.shape != k_cos.shape
        or q_cos.dtype != torch.bfloat16
        or q_sin.dtype != torch.bfloat16
        or k_cos.dtype != torch.bfloat16
        or k_sin.dtype != torch.bfloat16
        or not q_cos.is_cuda
        or not q_sin.is_cuda
        or not k_cos.is_cuda
        or not k_sin.is_cuda
    ):
        return None

    q_weight = q_norm.weight
    k_weight = k_norm.weight
    hidden = int(q.shape[-1])
    if (
        q_weight is None
        or k_weight is None
        or q_weight.device != q.device
        or k_weight.device != k.device
        or q_weight.dtype != q.dtype
        or k_weight.dtype != k.dtype
        or q_weight.numel() != hidden
        or k_weight.numel() != hidden
    ):
        return None

    try:
        from sglang.jit_kernel.diffusion.triton.ltx2_qknorm import (
            ltx2_qknorm_split_rope_pair,
        )

        with _ltx2_record_function("ltx2_fused_qknorm_rope::split_pair"):
            return ltx2_qknorm_split_rope_pair(
                q,
                k,
                q_weight,
                k_weight,
                q_cos,
                q_sin,
                k_cos,
                k_sin,
                eps,
            )
    except Exception as exc:
        _ltx2_disable_fused_qknorm_rope(exc)
        return None


def _ltx2_try_fused_rmsnorm_dual_modulate(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        not _ltx2_fused_dual_modulate_enabled()
        or _LTX2_FUSED_DUAL_MODULATE_RUNTIME_DISABLED
        or get_tp_world_size() != 1
        or not x.is_cuda
        or x.ndim != 3
        or not x.is_contiguous()
        or x.dtype != torch.bfloat16
    ):
        return None
    hidden = int(x.shape[-1])
    if hidden % 256 != 0 or hidden > 8192:
        return None
    for tensor in (scale0, shift0, scale1, shift1):
        if (
            not tensor.is_cuda
            or tensor.dtype != x.dtype
            or tensor.shape[-1] != hidden
            or tensor.stride(-1) != 1
        ):
            return None
    try:
        from sglang.jit_kernel.diffusion.triton.ltx2_dual_modulate import (
            ltx2_rmsnorm_dual_modulate,
        )

        with _ltx2_record_function("ltx2_fused_dual_modulate::rmsnorm_scale_shift_x2"):
            return ltx2_rmsnorm_dual_modulate(x, scale0, shift0, scale1, shift1, eps)
    except Exception as exc:
        _ltx2_disable_fused_dual_modulate(exc)
        return None


def _ltx2_try_fused_rmsnorm_ca_dual_modulate(
    x: torch.Tensor,
    temb_scale_shift: torch.Tensor,
    scale_shift_table: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        not _ltx2_fused_ca_dual_modulate_enabled()
        or _LTX2_FUSED_CA_DUAL_MODULATE_RUNTIME_DISABLED
        or get_tp_world_size() != 1
        or not x.is_cuda
        or x.ndim != 3
        or not x.is_contiguous()
        or x.dtype != torch.bfloat16
    ):
        return None
    batch, seq, hidden = x.shape
    if hidden % 256 != 0 or hidden > 8192:
        return None
    if (
        not temb_scale_shift.is_cuda
        or temb_scale_shift.dtype != x.dtype
        or temb_scale_shift.ndim != 3
        or temb_scale_shift.shape[0] != batch
        or temb_scale_shift.shape[1] != seq
        or temb_scale_shift.shape[2] != 4 * hidden
        or temb_scale_shift.stride(-1) != 1
    ):
        return None
    if (
        not scale_shift_table.is_cuda
        or scale_shift_table.dtype not in (torch.bfloat16, torch.float32)
        or scale_shift_table.ndim != 2
        or scale_shift_table.shape[0] < 4
        or scale_shift_table.shape[1] != hidden
        or scale_shift_table.stride(-1) != 1
    ):
        return None
    try:
        from sglang.jit_kernel.diffusion.triton.ltx2_dual_modulate import (
            ltx2_rmsnorm_ca_dual_modulate_from_temb,
        )

        with _ltx2_record_function(
            "ltx2_fused_ca_dual_modulate::rmsnorm_table_scale_shift_x2"
        ):
            return ltx2_rmsnorm_ca_dual_modulate_from_temb(
                x, temb_scale_shift, scale_shift_table, eps
            )
    except Exception as exc:
        _ltx2_disable_fused_ca_dual_modulate(exc)
        return None


def _ltx2_try_fused_ada_values3(
    scale_shift_table: torch.Tensor,
    batch_size: int,
    timestep: torch.Tensor,
    indices: slice,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if (
        not _ltx2_fused_ada_values_enabled()
        or _LTX2_FUSED_ADA_VALUES_RUNTIME_DISABLED
        or get_tp_world_size() != 1
        or not isinstance(indices, slice)
        or indices.step not in (None, 1)
        or indices.start is None
        or indices.stop is None
        or int(indices.stop) - int(indices.start) != 3
        or not timestep.is_cuda
        or timestep.dtype != torch.bfloat16
        or timestep.ndim != 3
        or int(timestep.shape[0]) != int(batch_size)
        or not timestep.is_contiguous()
        or not scale_shift_table.is_cuda
        or scale_shift_table.dtype not in (torch.bfloat16, torch.float32)
        or scale_shift_table.ndim != 2
        or scale_shift_table.stride(-1) != 1
    ):
        return None
    hidden = int(scale_shift_table.shape[1])
    num_ada_params = int(scale_shift_table.shape[0])
    if (
        hidden % 256 != 0
        or hidden > 8192
        or timestep.shape[-1] != num_ada_params * hidden
        or int(indices.start) < 0
        or int(indices.stop) > num_ada_params
    ):
        return None
    try:
        from sglang.jit_kernel.diffusion.triton.ltx2_ada_values import (
            ltx2_ada_values3,
        )

        with _ltx2_record_function("ltx2_fused_ada_values::triple"):
            return ltx2_ada_values3(scale_shift_table, timestep, int(indices.start))
    except Exception as exc:
        _ltx2_disable_fused_ada_values(exc)
        return None


def _ltx2_try_fused_ada_values9(
    scale_shift_table: torch.Tensor,
    batch_size: int,
    timestep: torch.Tensor,
) -> (
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None
):
    if (
        not _ltx2_fused_ada_values_all_enabled()
        or _LTX2_FUSED_ADA_VALUES_ALL_RUNTIME_DISABLED
        or get_tp_world_size() != 1
        or not timestep.is_cuda
        or timestep.dtype != torch.bfloat16
        or timestep.ndim != 3
        or int(timestep.shape[0]) != int(batch_size)
        or not timestep.is_contiguous()
        or not scale_shift_table.is_cuda
        or scale_shift_table.dtype not in (torch.bfloat16, torch.float32)
        or scale_shift_table.ndim != 2
        or int(scale_shift_table.shape[0]) != 9
        or scale_shift_table.stride(-1) != 1
    ):
        return None
    hidden = int(scale_shift_table.shape[1])
    if hidden % 256 != 0 or hidden > 8192 or timestep.shape[-1] != 9 * hidden:
        return None
    try:
        from sglang.jit_kernel.diffusion.triton.ltx2_ada_values import (
            ltx2_ada_values9,
            ltx2_ada_values9_packed,
        )

        with _ltx2_record_function("ltx2_fused_ada_values::all9"):
            if _ltx2_fused_ada_values_packed_enabled():
                return ltx2_ada_values9_packed(scale_shift_table, timestep)
            return ltx2_ada_values9(scale_shift_table, timestep)
    except Exception as exc:
        _ltx2_disable_fused_ada_values_all(exc)
        return None


def _ltx2_try_fused_ada_gates3(
    scale_shift_table: torch.Tensor,
    batch_size: int,
    timestep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if (
        not _ltx2_fused_ada_direct_enabled()
        or _LTX2_FUSED_ADA_DIRECT_RUNTIME_DISABLED
        or get_tp_world_size() != 1
        or not timestep.is_cuda
        or timestep.dtype != torch.bfloat16
        or timestep.ndim != 3
        or int(timestep.shape[0]) != int(batch_size)
        or not timestep.is_contiguous()
        or not scale_shift_table.is_cuda
        or scale_shift_table.dtype not in (torch.bfloat16, torch.float32)
        or scale_shift_table.ndim != 2
        or int(scale_shift_table.shape[0]) != 9
        or scale_shift_table.stride(-1) != 1
    ):
        return None
    hidden = int(scale_shift_table.shape[1])
    if hidden % 256 != 0 or hidden > 8192 or timestep.shape[-1] != 9 * hidden:
        return None
    try:
        from sglang.jit_kernel.diffusion.triton.ltx2_ada_values import (
            ltx2_ada_values_indices3,
        )

        with _ltx2_record_function("ltx2_fused_ada_direct::gates3"):
            return ltx2_ada_values_indices3(scale_shift_table, timestep, 2, 5, 8)
    except Exception as exc:
        _ltx2_disable_fused_ada_direct(exc)
        return None


def _ltx2_try_fused_norm_ada_scale_shift(
    x: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    shift_index: int,
    scale_index: int,
    eps: float,
) -> torch.Tensor | None:
    if (
        not _ltx2_fused_ada_direct_enabled()
        or _LTX2_FUSED_ADA_DIRECT_RUNTIME_DISABLED
        or get_tp_world_size() != 1
        or not x.is_cuda
        or not timestep.is_cuda
        or x.dtype != torch.bfloat16
        or timestep.dtype != x.dtype
        or x.ndim != 3
        or timestep.ndim != 3
        or int(timestep.shape[0]) != int(x.shape[0])
        or int(timestep.shape[1]) != int(x.shape[1])
        or x.stride(-1) != 1
        or not timestep.is_contiguous()
        or not scale_shift_table.is_cuda
        or scale_shift_table.dtype not in (torch.bfloat16, torch.float32)
        or scale_shift_table.ndim != 2
        or int(scale_shift_table.shape[0]) != 9
        or int(scale_shift_table.shape[1]) != int(x.shape[-1])
        or scale_shift_table.stride(-1) != 1
    ):
        return None
    hidden = int(x.shape[-1])
    if hidden % 256 != 0 or hidden > 8192 or timestep.shape[-1] != 9 * hidden:
        return None
    try:
        from sglang.jit_kernel.diffusion.cutedsl.ada_norm_scale_shift import (
            fused_norm_ada_scale_shift_chunked,
        )

        with _ltx2_record_function("ltx2_fused_ada_direct::norm_scale_shift"):
            return fused_norm_ada_scale_shift_chunked(
                x,
                timestep,
                scale_shift_table,
                int(shift_index),
                int(scale_index),
                "rms",
                eps,
            )
    except Exception as exc:
        _ltx2_disable_fused_ada_direct(exc)
        return None


def _ltx2_try_fused_residual_norm_ada_scale_shift(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    shift_index: int,
    scale_index: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        not _ltx2_fused_ada_direct_enabled()
        or _LTX2_FUSED_ADA_DIRECT_RUNTIME_DISABLED
        or get_tp_world_size() != 1
        or not residual.is_cuda
        or not x.is_cuda
        or not gate.is_cuda
        or not timestep.is_cuda
        or residual.dtype != torch.bfloat16
        or x.dtype != residual.dtype
        or gate.dtype != residual.dtype
        or timestep.dtype != residual.dtype
        or residual.shape != x.shape
        or gate.shape != x.shape
        or residual.ndim != 3
        or timestep.ndim != 3
        or int(timestep.shape[0]) != int(x.shape[0])
        or int(timestep.shape[1]) != int(x.shape[1])
        or residual.stride(-1) != 1
        or x.stride(-1) != 1
        or gate.stride(-1) != 1
        or not timestep.is_contiguous()
        or not scale_shift_table.is_cuda
        or scale_shift_table.dtype not in (torch.bfloat16, torch.float32)
        or scale_shift_table.ndim != 2
        or int(scale_shift_table.shape[0]) != 9
        or int(scale_shift_table.shape[1]) != int(x.shape[-1])
        or scale_shift_table.stride(-1) != 1
    ):
        return None
    hidden = int(x.shape[-1])
    if hidden % 256 != 0 or hidden > 8192 or timestep.shape[-1] != 9 * hidden:
        return None
    try:
        from sglang.jit_kernel.diffusion.cutedsl.ada_norm_scale_shift import (
            fused_scale_residual_norm_ada_scale_shift_chunked,
        )

        with _ltx2_record_function(
            "ltx2_fused_ada_direct::scale_residual_norm_scale_shift"
        ):
            return fused_scale_residual_norm_ada_scale_shift_chunked(
                residual,
                x,
                gate,
                timestep,
                scale_shift_table,
                int(shift_index),
                int(scale_index),
                "rms",
                eps,
            )
    except Exception as exc:
        _ltx2_disable_fused_ada_direct(exc)
        return None


def _ltx2_can_use_fused_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    gate: torch.Tensor | None = None,
) -> bool:
    if (
        not _ltx2_fused_adaln_enabled()
        or _LTX2_FUSED_ADALN_RUNTIME_DISABLED
        or not x.is_cuda
        or x.ndim != 3
        or not x.is_contiguous()
    ):
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    d = int(x.shape[-1])
    if d % 256 != 0 or d > 8192:
        return False
    for tensor in (scale, shift, gate):
        if tensor is None:
            continue
        if (
            tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or tensor.stride(-1) != 1
            or tensor.shape[-1] != d
        ):
            return False
    return _ltx2_get_fused_adaln_kernels() is not None


def _ltx2_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    if (
        _ltx2_fused_modulate_enabled()
        and not _LTX2_FUSED_MODULATE_RUNTIME_DISABLED
        and x.is_cuda
        and x.ndim == 3
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and scale.is_cuda
        and shift.is_cuda
        and scale.dtype == x.dtype
        and shift.dtype == x.dtype
        and scale.shape[-1] == x.shape[-1]
        and shift.shape[-1] == x.shape[-1]
    ):
        try:
            from sglang.jit_kernel.diffusion.triton.scale_shift import (
                fuse_scale_shift_kernel,
            )

            with _ltx2_record_function("ltx2_fused_modulate::scale_shift"):
                return fuse_scale_shift_kernel(x, scale, shift, scale_constant=1.0)
        except Exception as exc:
            _ltx2_disable_fused_modulate(exc)
    return x * (1 + scale) + shift


def _ltx2_residual_gate_add(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    can_use_fused = (
        _ltx2_fused_residual_gate_enabled()
        and residual.is_cuda
        and x.is_cuda
        and gate.is_cuda
        and residual.dtype == x.dtype
        and gate.dtype == x.dtype
    )
    if can_use_fused:
        with _ltx2_record_function("ltx2_fused_residual_gate::addcmul"):
            return torch.addcmul(residual, x, gate)
    return residual + x * gate


def _ltx2_try_gelu_tanh_inplace(x: torch.Tensor) -> torch.Tensor | None:
    if (
        not _ltx2_fused_gelu_inplace_enabled()
        or _LTX2_FUSED_GELU_INPLACE_RUNTIME_DISABLED
        or not x.is_cuda
        or x.dtype not in (torch.float16, torch.bfloat16)
        or not x.is_contiguous()
    ):
        return None
    try:
        from sglang.jit_kernel.diffusion.triton.ltx2_gelu import (
            ltx2_gelu_tanh_inplace,
        )

        with _ltx2_record_function("ltx2_fused_gelu_inplace::gelu_tanh"):
            return ltx2_gelu_tanh_inplace(x)
    except Exception as exc:
        _ltx2_disable_fused_gelu_inplace(exc)
        return None


def _ltx2_try_fp4_fused_proj_in_bias_gelu(
    x: torch.Tensor,
    proj_in: nn.Module,
) -> torch.Tensor | None:
    if (
        not _ltx2_fp4_fused_proj_in_bias_gelu_enabled()
        or get_tp_world_size() != 1
        or not x.is_cuda
        or x.dtype not in (torch.float16, torch.bfloat16)
        or x.ndim < 2
        or x.stride(-1) != 1
    ):
        return None
    base = _ltx2_linear_base_for_fusion(proj_in)
    if base is None:
        return None
    if getattr(base, "gather_output", False) or getattr(base, "skip_bias_add", False):
        return None
    if base.quant_method.__class__.__name__ != "ModelOptFp4LinearMethod":
        return None
    bias = getattr(base, "bias", None)
    if (
        bias is None
        or bias.device != x.device
        or bias.dtype != x.dtype
        or bias.ndim != 1
    ):
        return None

    try:
        with _ltx2_record_function(
            "ltx2_fp4_epilogue_proj_in_bias_gelu::linear_bias_gelu"
        ):
            y = modelopt_fp4_apply_linear_bias_gelu(base, x, bias)
        if y is not None:
            return y

        from sglang.jit_kernel.diffusion.triton.ltx2_gelu import (
            ltx2_bias_gelu_tanh_inplace,
        )

        with _ltx2_record_function("ltx2_fp4_fused_proj_in_bias_gelu::linear"):
            y = base.quant_method.apply(base, x, bias=None)
        if y.shape[-1] != bias.shape[0] or not y.is_contiguous():
            return None
        with _ltx2_record_function("ltx2_fp4_fused_proj_in_bias_gelu::bias_gelu"):
            return ltx2_bias_gelu_tanh_inplace(y, bias)
    except Exception as exc:
        logger.warning_once(
            f"Disabling LTX2 FP4 fused proj_in+bias+GELU fast path: {exc}"
        )
        return None


def _ltx2_try_fused_ffn_proj_in_gelu(
    x: torch.Tensor,
    proj_in: nn.Module,
) -> torch.Tensor | None:
    if (
        not _ltx2_fused_ffn_proj_in_gelu_enabled()
        or get_tp_world_size() != 1
        or not x.is_cuda
        or x.dtype not in (torch.float16, torch.bfloat16)
        or x.ndim < 2
        or x.stride(-1) != 1
    ):
        return None
    base = _ltx2_linear_base_for_fusion(proj_in)
    if base is None:
        return None
    if getattr(base, "gather_output", False) or getattr(base, "skip_bias_add", False):
        return None
    if base.quant_method.__class__.__name__ != "UnquantizedLinearMethod":
        return None

    weight = getattr(base, "weight", None)
    bias = getattr(base, "bias", None)
    if weight is None or bias is None:
        return None
    if (
        weight.device != x.device
        or bias.device != x.device
        or weight.dtype != x.dtype
        or bias.dtype != x.dtype
        or weight.ndim != 2
        or bias.ndim != 1
        or weight.shape[1] != x.shape[-1]
        or weight.shape[0] != bias.shape[0]
        or weight.stride(-1) != 1
    ):
        return None

    x_2d = x.reshape(-1, x.shape[-1])
    with _ltx2_record_function("ltx2_fused_ffn_proj_in_gelu::addmm_activation"):
        out = torch.ops.aten._addmm_activation.default(
            bias,
            x_2d,
            weight.t(),
            beta=1,
            alpha=1,
            use_gelu=True,
        )
    return out.reshape(*x.shape[:-1], weight.shape[0])


def _get_ltx2_rms_norm_modulate():
    global _LTX2_RMS_NORM_MODULATE, _LTX2_RMS_NORM_MODULATE_UNAVAILABLE
    if _LTX2_RMS_NORM_MODULATE_UNAVAILABLE:
        return None
    if _LTX2_RMS_NORM_MODULATE is None:
        try:
            from sglang.jit_kernel.diffusion.triton.ltx2_adaln import (
                ltx2_rms_norm_modulate,
            )
        except Exception:
            _LTX2_RMS_NORM_MODULATE_UNAVAILABLE = True
            return None
        _LTX2_RMS_NORM_MODULATE = ltx2_rms_norm_modulate
    return _LTX2_RMS_NORM_MODULATE


def _ltx2_try_triton_rms_norm_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    if (
        x.ndim != 3
        or scale.ndim != 3
        or shift.ndim != 3
        or x.dtype != torch.bfloat16
        or scale.dtype != torch.bfloat16
        or shift.dtype != torch.bfloat16
        or os.environ.get("SGLANG_LTX2_FUSED_RMS_ADALN", "1") == "0"
        or not x.is_cuda
        or not scale.is_cuda
        or not shift.is_cuda
    ):
        return None
    ltx2_rms_norm_modulate = _get_ltx2_rms_norm_modulate()
    if ltx2_rms_norm_modulate is None:
        return None
    try:
        with _ltx2_record_function("ltx2_fused_rms_adaln::norm_scale_shift"):
            return ltx2_rms_norm_modulate(x, scale, shift, eps)
    except Exception:
        return None


def _ltx2_norm_scale_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if _ltx2_can_use_fused_adaln(x, scale, shift):
        kernels = _ltx2_get_fused_adaln_kernels()
        assert kernels is not None
        fused_norm_scale_shift, _ = kernels
        try:
            with _ltx2_record_function("ltx2_fused_adaln::norm_scale_shift"):
                return fused_norm_scale_shift(x, None, None, scale, shift, "rms", eps)
        except Exception as exc:
            _ltx2_disable_fused_adaln(exc)
    triton_out = _ltx2_try_triton_rms_norm_modulate(x, scale, shift, eps)
    if triton_out is not None:
        return triton_out
    return _ltx2_modulate(rms_norm(x, eps), scale, shift)


def _ltx2_residual_norm_scale_shift(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _ltx2_can_use_fused_adaln(residual, scale, shift, gate) and x.is_contiguous():
        kernels = _ltx2_get_fused_adaln_kernels()
        assert kernels is not None
        _, fused_scale_residual_norm_scale_shift = kernels
        try:
            with _ltx2_record_function(
                "ltx2_fused_adaln::scale_residual_norm_scale_shift"
            ):
                return fused_scale_residual_norm_scale_shift(
                    residual, x, gate, None, None, scale, shift, "rms", eps
                )
        except Exception as exc:
            _ltx2_disable_fused_adaln(exc)
    residual_out = _ltx2_residual_gate_add(residual, x, gate)
    triton_out = _ltx2_try_triton_rms_norm_modulate(residual_out, scale, shift, eps)
    if triton_out is not None:
        return triton_out, residual_out
    return _ltx2_modulate(rms_norm(residual_out, eps), scale, shift), residual_out


class LTX2TextProjection(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int | None = None,
        act_fn: str = "gelu_tanh",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if out_features is None:
            out_features = hidden_size

        self.linear_1 = ColumnParallelLinear(
            in_features,
            hidden_size,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix=_ltx2_child_prefix(prefix, "linear_1"),
        )
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")

        self.linear_2 = ColumnParallelLinear(
            hidden_size,
            out_features,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix=_ltx2_child_prefix(prefix, "linear_2"),
        )

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class LTX2TimestepEmbedder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        in_channels: int = 256,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.linear_1 = ColumnParallelLinear(
            in_channels,
            embedding_dim,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix=_ltx2_child_prefix(prefix, "linear_1"),
        )
        self.linear_2 = ColumnParallelLinear(
            embedding_dim,
            embedding_dim,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix=_ltx2_child_prefix(prefix, "linear_2"),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        x, _ = self.linear_1(t_emb)
        x = F.silu(x)
        x, _ = self.linear_2(x)
        return x


class LTX2PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.timestep_embedder = LTX2TimestepEmbedder(
            embedding_dim,
            in_channels=256,
            quant_config=quant_config,
            prefix=_ltx2_child_prefix(prefix, "timestep_embedder"),
        )

    def forward(
        self, timestep: torch.Tensor, hidden_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        t = timestep.reshape(-1).to(dtype=torch.float32)
        t_emb = timestep_embedding(t, dim=256, max_period=10000, dtype=torch.float32)
        if hidden_dtype is not None:
            t_emb = t_emb.to(dtype=hidden_dtype)
        return self.timestep_embedder(t_emb)


class LTX2AdaLayerNormSingle(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        embedding_coefficient: int = 6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.emb = LTX2PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            quant_config=quant_config,
            prefix=_ltx2_child_prefix(prefix, "emb"),
        )
        self.silu = nn.SiLU()
        self.linear = ColumnParallelLinear(
            embedding_dim,
            embedding_coefficient * embedding_dim,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix=_ltx2_child_prefix(prefix, "linear"),
        )

    def forward(
        self, timestep: torch.Tensor, hidden_dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype).to(
            dtype=self.linear.weight.dtype
        )
        out, _ = self.linear(self.silu(embedded_timestep))
        return out, embedded_timestep


class LTX2TPRMSNormAcrossHeads(nn.Module):
    def __init__(
        self, full_hidden_size: int, local_hidden_size: int, eps: float
    ) -> None:
        super().__init__()
        self.full_hidden_size = full_hidden_size
        self.local_hidden_size = local_hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(local_hidden_size))

        tp_rank = get_tp_rank()

        def _weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            shard = loaded_weight.narrow(
                0, tp_rank * local_hidden_size, local_hidden_size
            )
            param.data.copy_(shard.to(dtype=param.dtype, device=param.device))

        setattr(self.weight, "weight_loader", _weight_loader)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep track of the original dtype. We do the statistics in fp32 for
        # numerical stability, but cast the output back to the input dtype to
        orig_dtype = x.dtype
        if get_tp_world_size() == 1:
            var = x.float().pow(2).mean(dim=-1, keepdim=True)
        else:
            local_sumsq = x.float().pow(2).sum(dim=-1, keepdim=True)
            global_sumsq = tensor_model_parallel_all_reduce(local_sumsq)
            var = global_sumsq / float(self.full_hidden_size)

        inv_rms_fp32 = torch.rsqrt(var + self.eps)
        y = (x.float() * inv_rms_fp32).to(dtype=orig_dtype)
        return y * self.weight.to(dtype=orig_dtype)


class LTX2Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        qk_norm: bool = True,
        use_local_attention: bool = False,
        apply_gated_attention: bool = False,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        profile_prefix: str | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()

        self.query_dim = int(query_dim)
        self.context_dim = int(query_dim if context_dim is None else context_dim)
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.inner_dim = self.heads * self.dim_head
        self.norm_eps = float(norm_eps)
        self.qk_norm = bool(qk_norm)
        self.use_local_attention = bool(use_local_attention)
        self.apply_gated_attention = bool(apply_gated_attention)
        self.prefix = prefix
        self.profile_prefix = profile_prefix or prefix

        tp_size = get_tp_world_size()
        if tp_size <= 0:
            raise ValueError(f"Invalid {tp_size=}. Expected tp_size >= 1.")
        if self.heads % tp_size != 0:
            raise ValueError(
                f"LTX2Attention requires heads divisible by tp_size, got "
                f"{self.heads=} {tp_size=}."
            )
        if self.inner_dim % tp_size != 0:
            # This should follow from heads % tp_size, but keep explicit for clarity.
            raise ValueError(
                f"LTX2Attention requires inner_dim divisible by tp_size, got "
                f"{self.inner_dim=} {tp_size=}."
            )
        self.local_heads = self.heads // tp_size

        self.to_q = ColumnParallelLinear(
            self.query_dim,
            self.inner_dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=_ltx2_child_prefix(prefix, "to_q"),
        )
        self.to_k = ColumnParallelLinear(
            self.context_dim,
            self.inner_dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=_ltx2_child_prefix(prefix, "to_k"),
        )
        self.to_v = ColumnParallelLinear(
            self.context_dim,
            self.inner_dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=_ltx2_child_prefix(prefix, "to_v"),
        )
        self.to_gate_logits: ColumnParallelLinear | None = None
        self._qkv_fused_cache: tuple[object, ...] | None = None
        self._qkv_fused_projection_disabled = False
        self._fp4_shared_qkv_projection_disabled = False
        self._fp4_shared_qkv_scale_checked = False
        self._fp4_shared_q_gate_projection_disabled = False
        self._fp4_shared_q_gate_scale_checked = False
        self._audio_qkvg_fused_cache: tuple[object, ...] | None = None
        self._audio_qkvg_fused_projection_disabled = False
        if self.apply_gated_attention:
            self.to_gate_logits = ColumnParallelLinear(
                self.query_dim,
                self.heads,
                bias=True,
                gather_output=False,
                quant_config=quant_config,
                prefix=_ltx2_child_prefix(prefix, "to_gate_logits"),
            )
        self._q_gate_fused_cache: tuple[object, ...] | None = None
        self._kv_fused_cache: tuple[object, ...] | None = None
        self._kv_fused_projection_disabled = False

        self.q_norm: nn.Module | None = None
        self.k_norm: nn.Module | None = None
        if self.qk_norm:
            if tp_size == 1:
                self.q_norm = torch.nn.RMSNorm(self.inner_dim, eps=self.norm_eps)
                self.k_norm = torch.nn.RMSNorm(self.inner_dim, eps=self.norm_eps)
            else:
                self.q_norm = LTX2TPRMSNormAcrossHeads(
                    full_hidden_size=self.inner_dim,
                    local_hidden_size=self.inner_dim // tp_size,
                    eps=self.norm_eps,
                )
                self.k_norm = LTX2TPRMSNormAcrossHeads(
                    full_hidden_size=self.inner_dim,
                    local_hidden_size=self.inner_dim // tp_size,
                    eps=self.norm_eps,
                )

        self.to_out = nn.Sequential(
            RowParallelLinear(
                self.inner_dim,
                self.query_dim,
                bias=True,
                input_is_parallel=True,
                quant_config=quant_config,
                prefix=_ltx2_child_prefix(prefix, "to_out.0"),
            ),
            nn.Identity(),
        )

        if self.use_local_attention:
            self.attn = LocalAttention(
                num_heads=self.local_heads,
                head_size=self.dim_head,
                num_kv_heads=self.local_heads,
                softmax_scale=None,
                causal=False,
                supported_attention_backends=supported_attention_backends,
                prefix=f"{prefix}.attn",
                # official LTX2 torch_sdpa uses cuDNN; cuda setup disables it
                allow_cudnn_sdp=True,
            )
        else:
            self.attn = USPAttention(
                num_heads=self.local_heads,
                head_size=self.dim_head,
                num_kv_heads=self.local_heads,
                dropout_rate=0,
                softmax_scale=None,
                causal=False,
                supported_attention_backends=supported_attention_backends,
                prefix=f"{prefix}.attn",
                # official LTX2 torch_sdpa uses cuDNN; cuda setup disables it
                allow_cudnn_sdp=True,
            )

    def _route_stage2_video_self_attention_to_backend(self) -> bool:
        if os.environ.get(
            "SGLANG_LTX2_STAGE2_PIECEWISE_BYPASS_OFFICIAL_FA4", "1"
        ).lower() in ("0", "false", "no"):
            return False
        if getattr(self.attn, "backend", None) != AttentionBackendEnum.PIECEWISE_ATTN:
            return False
        if not _LTX2_PROFILE_CONTEXT or _LTX2_PROFILE_CONTEXT[-1][0] != "stage2":
            return False
        if not str(self.profile_prefix).endswith(".attn1"):
            return False
        return (
            self.context_dim == self.query_dim
            and self.query_dim == 4096
            and self.inner_dim == 4096
            and self.dim_head == 128
        )

    @staticmethod
    def _q_gate_tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
        return (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
            getattr(tensor, "_version", 0),
        )

    def _try_fused_audio_qkvg_projection(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if (
            not _ltx2_fused_audio_qkvg_enabled()
            or self._audio_qkvg_fused_projection_disabled
            or get_tp_world_size() != 1
            or self.to_gate_logits is None
            or not x.is_cuda
            or x.dtype not in (torch.float16, torch.bfloat16)
            or x.shape[-1] != self.query_dim
            or self.query_dim != self.context_dim
            or self.query_dim != 2048
            or self.inner_dim != 2048
        ):
            return None
        to_q = _ltx2_linear_base_for_fusion(self.to_q)
        to_k = _ltx2_linear_base_for_fusion(self.to_k)
        to_v = _ltx2_linear_base_for_fusion(self.to_v)
        to_gate = _ltx2_linear_base_for_fusion(self.to_gate_logits)
        if to_q is None or to_k is None or to_v is None or to_gate is None:
            return None
        if any(
            getattr(layer, "gather_output", False)
            for layer in (to_q, to_k, to_v, to_gate)
        ):
            return None
        if any(
            layer.quant_method.__class__.__name__ != "UnquantizedLinearMethod"
            for layer in (to_q, to_k, to_v, to_gate)
        ):
            return None

        q_weight = getattr(to_q, "weight", None)
        k_weight = getattr(to_k, "weight", None)
        v_weight = getattr(to_v, "weight", None)
        gate_weight = getattr(to_gate, "weight", None)
        q_bias = getattr(to_q, "bias", None)
        k_bias = getattr(to_k, "bias", None)
        v_bias = getattr(to_v, "bias", None)
        gate_bias = getattr(to_gate, "bias", None)
        tensors = (
            q_weight,
            k_weight,
            v_weight,
            gate_weight,
            q_bias,
            k_bias,
            v_bias,
            gate_bias,
        )
        if any(tensor is None for tensor in tensors):
            return None
        assert q_weight is not None and k_weight is not None and v_weight is not None
        assert gate_weight is not None
        assert q_bias is not None and k_bias is not None and v_bias is not None
        assert gate_bias is not None
        if any(
            tensor.device != x.device or tensor.dtype != x.dtype for tensor in tensors
        ):
            return None
        if (
            q_weight.ndim != 2
            or k_weight.ndim != 2
            or v_weight.ndim != 2
            or gate_weight.ndim != 2
            or q_bias.ndim != 1
            or k_bias.ndim != 1
            or v_bias.ndim != 1
            or gate_bias.ndim != 1
            or q_weight.shape[1] != x.shape[-1]
            or k_weight.shape[1] != x.shape[-1]
            or v_weight.shape[1] != x.shape[-1]
            or gate_weight.shape[1] != x.shape[-1]
            or q_weight.shape[0] != q_bias.shape[0]
            or k_weight.shape[0] != k_bias.shape[0]
            or v_weight.shape[0] != v_bias.shape[0]
            or gate_weight.shape[0] != gate_bias.shape[0]
            or gate_weight.shape[0] != self.heads
            or q_weight.stride(-1) != 1
            or k_weight.stride(-1) != 1
            or v_weight.stride(-1) != 1
            or gate_weight.stride(-1) != 1
        ):
            return None

        try:
            q_out = int(q_weight.shape[0])
            k_out = int(k_weight.shape[0])
            v_out = int(v_weight.shape[0])
            gate_out = int(gate_weight.shape[0])
            signature = (
                self._q_gate_tensor_signature(q_weight),
                self._q_gate_tensor_signature(k_weight),
                self._q_gate_tensor_signature(v_weight),
                self._q_gate_tensor_signature(gate_weight),
                self._q_gate_tensor_signature(q_bias),
                self._q_gate_tensor_signature(k_bias),
                self._q_gate_tensor_signature(v_bias),
                self._q_gate_tensor_signature(gate_bias),
            )
            cache = self._audio_qkvg_fused_cache
            if cache is None or cache[0] != signature:
                with torch.no_grad():
                    fused_weight = torch.cat(
                        (
                            q_weight.detach(),
                            k_weight.detach(),
                            v_weight.detach(),
                            gate_weight.detach(),
                        ),
                        dim=0,
                    ).contiguous()
                    fused_bias = torch.cat(
                        (
                            q_bias.detach(),
                            k_bias.detach(),
                            v_bias.detach(),
                            gate_bias.detach(),
                        ),
                        dim=0,
                    ).contiguous()
                cache = (
                    signature,
                    q_out,
                    k_out,
                    v_out,
                    gate_out,
                    fused_weight,
                    fused_bias,
                )
                self._audio_qkvg_fused_cache = cache

            _, q_out, k_out, v_out, gate_out, fused_weight, fused_bias = cache
            with _ltx2_record_function("ltx2_fused_audio_qkvg::linear"):
                fused = F.linear(x, fused_weight, fused_bias)
            q = fused[..., :q_out]
            k = fused[..., q_out : q_out + k_out]
            v = fused[..., q_out + k_out : q_out + k_out + v_out]
            gate = fused[
                ...,
                q_out + k_out + v_out : q_out + k_out + v_out + gate_out,
            ]
            return q, k, v, gate
        except Exception:
            self._audio_qkvg_fused_projection_disabled = True
            self._audio_qkvg_fused_cache = None
            return None

    def _try_fused_qkv_projection(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if (
            not _ltx2_fused_qkv_enabled()
            or self._qkv_fused_projection_disabled
            or not x.is_cuda
            or x.dtype not in (torch.float16, torch.bfloat16)
            or x.shape[-1] != self.query_dim
            or self.query_dim != self.context_dim
        ):
            return None
        to_q = _ltx2_linear_base_for_fusion(self.to_q)
        to_k = _ltx2_linear_base_for_fusion(self.to_k)
        to_v = _ltx2_linear_base_for_fusion(self.to_v)
        if to_q is None or to_k is None or to_v is None:
            return None
        if (
            getattr(to_q, "gather_output", False)
            or getattr(to_k, "gather_output", False)
            or getattr(to_v, "gather_output", False)
        ):
            return None
        if (
            to_q.quant_method.__class__.__name__ != "UnquantizedLinearMethod"
            or to_k.quant_method.__class__.__name__ != "UnquantizedLinearMethod"
            or to_v.quant_method.__class__.__name__ != "UnquantizedLinearMethod"
        ):
            return None

        q_weight = getattr(to_q, "weight", None)
        k_weight = getattr(to_k, "weight", None)
        v_weight = getattr(to_v, "weight", None)
        q_bias = getattr(to_q, "bias", None)
        k_bias = getattr(to_k, "bias", None)
        v_bias = getattr(to_v, "bias", None)
        if any(
            tensor is None
            for tensor in (q_weight, k_weight, v_weight, q_bias, k_bias, v_bias)
        ):
            return None
        assert q_weight is not None and k_weight is not None and v_weight is not None
        assert q_bias is not None and k_bias is not None and v_bias is not None
        tensors = (q_weight, k_weight, v_weight, q_bias, k_bias, v_bias)
        if any(
            tensor.device != x.device or tensor.dtype != x.dtype for tensor in tensors
        ):
            return None
        if (
            q_weight.ndim != 2
            or k_weight.ndim != 2
            or v_weight.ndim != 2
            or q_bias.ndim != 1
            or k_bias.ndim != 1
            or v_bias.ndim != 1
            or q_weight.shape[1] != x.shape[-1]
            or k_weight.shape[1] != x.shape[-1]
            or v_weight.shape[1] != x.shape[-1]
            or q_weight.shape[0] != q_bias.shape[0]
            or k_weight.shape[0] != k_bias.shape[0]
            or v_weight.shape[0] != v_bias.shape[0]
            or q_weight.stride(-1) != 1
            or k_weight.stride(-1) != 1
            or v_weight.stride(-1) != 1
        ):
            return None

        try:
            q_out = int(q_weight.shape[0])
            k_out = int(k_weight.shape[0])
            v_out = int(v_weight.shape[0])
            signature = (
                self._q_gate_tensor_signature(q_weight),
                self._q_gate_tensor_signature(k_weight),
                self._q_gate_tensor_signature(v_weight),
                self._q_gate_tensor_signature(q_bias),
                self._q_gate_tensor_signature(k_bias),
                self._q_gate_tensor_signature(v_bias),
            )
            cache = self._qkv_fused_cache
            if cache is None or cache[0] != signature:
                with torch.no_grad():
                    fused_weight = torch.cat(
                        (
                            q_weight.detach(),
                            k_weight.detach(),
                            v_weight.detach(),
                        ),
                        dim=0,
                    ).contiguous()
                    fused_bias = torch.cat(
                        (q_bias.detach(), k_bias.detach(), v_bias.detach()), dim=0
                    ).contiguous()
                cache = (signature, q_out, k_out, v_out, fused_weight, fused_bias)
                self._qkv_fused_cache = cache

            _, q_out, k_out, v_out, fused_weight, fused_bias = cache
            with _ltx2_record_function("ltx2_fused_qkv::linear"):
                fused = F.linear(x, fused_weight, fused_bias)
            q = fused[..., :q_out]
            k = fused[..., q_out : q_out + k_out]
            v = fused[..., q_out + k_out : q_out + k_out + v_out]
            return q, k, v
        except Exception:
            self._qkv_fused_projection_disabled = True
            self._qkv_fused_cache = None
            return None

    def _fp4_shared_qkv_scales_match(
        self, to_q: nn.Module, to_k: nn.Module, to_v: nn.Module
    ) -> bool:
        if self._fp4_shared_qkv_scale_checked:
            return True
        scales = (
            getattr(to_q, "input_scale_inv", None),
            getattr(to_k, "input_scale_inv", None),
            getattr(to_v, "input_scale_inv", None),
        )
        if any(scale is None or scale.numel() != 1 for scale in scales):
            return False
        try:
            values = [float(scale.detach().cpu().item()) for scale in scales]
        except Exception:
            return False
        if values[0] != values[1] or values[0] != values[2]:
            return False
        self._fp4_shared_qkv_scale_checked = True
        return True

    def _try_fp4_shared_qkv_projection(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if (
            not _ltx2_fp4_shared_qkv_enabled()
            or self._fp4_shared_qkv_projection_disabled
            or not x.is_cuda
            or x.dtype not in (torch.float16, torch.bfloat16)
            or x.shape[-1] != self.query_dim
            or self.query_dim != self.context_dim
        ):
            return None
        to_q = _ltx2_linear_base_for_fusion(self.to_q)
        to_k = _ltx2_linear_base_for_fusion(self.to_k)
        to_v = _ltx2_linear_base_for_fusion(self.to_v)
        if to_q is None or to_k is None or to_v is None:
            return None
        if (
            getattr(to_q, "gather_output", False)
            or getattr(to_k, "gather_output", False)
            or getattr(to_v, "gather_output", False)
            or getattr(to_q, "skip_bias_add", False)
            or getattr(to_k, "skip_bias_add", False)
            or getattr(to_v, "skip_bias_add", False)
        ):
            return None
        if (
            to_q.quant_method.__class__.__name__ != "ModelOptFp4LinearMethod"
            or to_k.quant_method.__class__.__name__ != "ModelOptFp4LinearMethod"
            or to_v.quant_method.__class__.__name__ != "ModelOptFp4LinearMethod"
        ):
            return None
        if not self._fp4_shared_qkv_scales_match(to_q, to_k, to_v):
            self._fp4_shared_qkv_projection_disabled = True
            return None

        try:
            with _ltx2_record_function("ltx2_fp4_shared_qkv::quantize"):
                x_fp4, x_scale_interleaved, input_shape, output_dtype = (
                    modelopt_fp4_quantize_activation(x, to_q.input_scale_inv)
                )
            with _ltx2_record_function("ltx2_fp4_shared_qkv::to_q"):
                q = modelopt_fp4_apply_quantized_linear(
                    to_q,
                    x_fp4,
                    x_scale_interleaved,
                    input_shape,
                    output_dtype,
                    to_q.bias,
                )
            with _ltx2_record_function("ltx2_fp4_shared_qkv::to_k"):
                k = modelopt_fp4_apply_quantized_linear(
                    to_k,
                    x_fp4,
                    x_scale_interleaved,
                    input_shape,
                    output_dtype,
                    to_k.bias,
                )
            with _ltx2_record_function("ltx2_fp4_shared_qkv::to_v"):
                v = modelopt_fp4_apply_quantized_linear(
                    to_v,
                    x_fp4,
                    x_scale_interleaved,
                    input_shape,
                    output_dtype,
                    to_v.bias,
                )
            return q, k, v
        except Exception as exc:
            logger.warning_once(f"Disabling LTX2 FP4 shared QKV projection: {exc}")
            self._fp4_shared_qkv_projection_disabled = True
            return None

    def _fp4_shared_q_gate_scales_match(
        self, to_q: nn.Module, to_gate: nn.Module
    ) -> bool:
        if self._fp4_shared_q_gate_scale_checked:
            return True
        q_scale = getattr(to_q, "input_scale_inv", None)
        gate_scale = getattr(to_gate, "input_scale_inv", None)
        if (
            q_scale is None
            or gate_scale is None
            or q_scale.numel() != 1
            or gate_scale.numel() != 1
        ):
            return False
        try:
            q_value = float(q_scale.detach().cpu().item())
            gate_value = float(gate_scale.detach().cpu().item())
        except Exception:
            return False
        if q_value != gate_value:
            return False
        self._fp4_shared_q_gate_scale_checked = True
        return True

    def _try_fp4_shared_q_gate_projection(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            not _ltx2_fp4_shared_q_gate_enabled()
            or self._fp4_shared_q_gate_projection_disabled
            or self.to_gate_logits is None
            or get_tp_world_size() != 1
            or not x.is_cuda
            or x.dtype not in (torch.float16, torch.bfloat16)
            or x.shape[-1] != self.query_dim
            or self.query_dim != 4096
            or self.inner_dim != 4096
            or self.local_heads != 32
        ):
            return None
        to_q = _ltx2_linear_base_for_fusion(self.to_q)
        to_gate = _ltx2_linear_base_for_fusion(self.to_gate_logits)
        if to_q is None or to_gate is None:
            return None
        if (
            getattr(to_q, "gather_output", False)
            or getattr(to_gate, "gather_output", False)
            or getattr(to_q, "skip_bias_add", False)
            or getattr(to_gate, "skip_bias_add", False)
        ):
            return None
        if (
            to_q.quant_method.__class__.__name__ != "ModelOptFp4LinearMethod"
            or to_gate.quant_method.__class__.__name__ != "ModelOptFp4LinearMethod"
        ):
            return None
        if not self._fp4_shared_q_gate_scales_match(to_q, to_gate):
            self._fp4_shared_q_gate_projection_disabled = True
            return None

        try:
            with _ltx2_record_function("ltx2_fp4_shared_q_gate::quantize"):
                x_fp4, x_scale_interleaved, input_shape, output_dtype = (
                    modelopt_fp4_quantize_activation(x, to_q.input_scale_inv)
                )
            with _ltx2_record_function("ltx2_fp4_shared_q_gate::to_q"):
                q = modelopt_fp4_apply_quantized_linear(
                    to_q,
                    x_fp4,
                    x_scale_interleaved,
                    input_shape,
                    output_dtype,
                    to_q.bias,
                )
            with _ltx2_record_function("ltx2_fp4_shared_q_gate::to_gate_logits"):
                gate = modelopt_fp4_apply_quantized_linear(
                    to_gate,
                    x_fp4,
                    x_scale_interleaved,
                    input_shape,
                    output_dtype,
                    to_gate.bias,
                )
            return q, gate
        except Exception as exc:
            logger.warning_once(f"Disabling LTX2 FP4 shared q+gate projection: {exc}")
            self._fp4_shared_q_gate_projection_disabled = True
            return None

    def _try_fused_kv_projection(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            not _ltx2_fused_kv_enabled()
            or self._kv_fused_projection_disabled
            or not context.is_cuda
            or context.dtype not in (torch.float16, torch.bfloat16)
            or context.shape[-1] != self.context_dim
        ):
            return None
        to_k = _ltx2_linear_base_for_fusion(self.to_k)
        to_v = _ltx2_linear_base_for_fusion(self.to_v)
        if to_k is None or to_v is None:
            return None
        if getattr(to_k, "gather_output", False) or getattr(
            to_v, "gather_output", False
        ):
            return None
        if (
            to_k.quant_method.__class__.__name__ != "UnquantizedLinearMethod"
            or to_v.quant_method.__class__.__name__ != "UnquantizedLinearMethod"
        ):
            return None

        k_weight = getattr(to_k, "weight", None)
        v_weight = getattr(to_v, "weight", None)
        k_bias = getattr(to_k, "bias", None)
        v_bias = getattr(to_v, "bias", None)
        if any(tensor is None for tensor in (k_weight, v_weight, k_bias, v_bias)):
            return None
        assert k_weight is not None and v_weight is not None
        assert k_bias is not None and v_bias is not None
        if (
            k_weight.device != context.device
            or v_weight.device != context.device
            or k_bias.device != context.device
            or v_bias.device != context.device
            or k_weight.dtype != context.dtype
            or v_weight.dtype != context.dtype
            or k_bias.dtype != context.dtype
            or v_bias.dtype != context.dtype
            or k_weight.ndim != 2
            or v_weight.ndim != 2
            or k_bias.ndim != 1
            or v_bias.ndim != 1
            or k_weight.shape[1] != context.shape[-1]
            or v_weight.shape[1] != context.shape[-1]
            or k_weight.shape[0] != k_bias.shape[0]
            or v_weight.shape[0] != v_bias.shape[0]
            or k_weight.stride(-1) != 1
            or v_weight.stride(-1) != 1
        ):
            return None

        try:
            k_out = int(k_weight.shape[0])
            v_out = int(v_weight.shape[0])
            signature = (
                self._q_gate_tensor_signature(k_weight),
                self._q_gate_tensor_signature(v_weight),
                self._q_gate_tensor_signature(k_bias),
                self._q_gate_tensor_signature(v_bias),
            )
            cache = self._kv_fused_cache
            if cache is None or cache[0] != signature:
                with torch.no_grad():
                    fused_weight = torch.cat(
                        (k_weight.detach(), v_weight.detach()), dim=0
                    ).contiguous()
                    fused_bias = torch.cat(
                        (k_bias.detach(), v_bias.detach()), dim=0
                    ).contiguous()
                cache = (signature, k_out, v_out, fused_weight, fused_bias)
                self._kv_fused_cache = cache

            _, k_out, v_out, fused_weight, fused_bias = cache
            with _ltx2_record_function("ltx2_fused_kv::linear"):
                fused = F.linear(context, fused_weight, fused_bias)
            return fused[..., :k_out], fused[..., k_out : k_out + v_out]
        except Exception:
            self._kv_fused_projection_disabled = True
            self._kv_fused_cache = None
            return None

    def _try_fused_q_gate_projection(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            not _ltx2_fused_q_gate_enabled()
            or _LTX2_FUSED_Q_GATE_RUNTIME_DISABLED
            or self.to_gate_logits is None
            or self.to_q.gather_output
            or self.to_gate_logits.gather_output
            or not x.is_cuda
            or x.dtype not in (torch.float16, torch.bfloat16)
            or x.shape[-1] != self.query_dim
        ):
            return None
        if (
            self.to_q.quant_method.__class__.__name__ != "UnquantizedLinearMethod"
            or self.to_gate_logits.quant_method.__class__.__name__
            != "UnquantizedLinearMethod"
        ):
            return None

        q_weight = getattr(self.to_q, "weight", None)
        gate_weight = getattr(self.to_gate_logits, "weight", None)
        q_bias = getattr(self.to_q, "bias", None)
        gate_bias = getattr(self.to_gate_logits, "bias", None)
        if any(tensor is None for tensor in (q_weight, gate_weight, q_bias, gate_bias)):
            return None
        assert q_weight is not None and gate_weight is not None
        assert q_bias is not None and gate_bias is not None
        if (
            q_weight.device != x.device
            or gate_weight.device != x.device
            or q_bias.device != x.device
            or gate_bias.device != x.device
            or q_weight.dtype != x.dtype
            or gate_weight.dtype != x.dtype
            or q_bias.dtype != x.dtype
            or gate_bias.dtype != x.dtype
            or q_weight.ndim != 2
            or gate_weight.ndim != 2
            or q_bias.ndim != 1
            or gate_bias.ndim != 1
            or q_weight.shape[1] != x.shape[-1]
            or gate_weight.shape[1] != x.shape[-1]
            or q_weight.shape[0] != q_bias.shape[0]
            or gate_weight.shape[0] != gate_bias.shape[0]
            or q_weight.stride(-1) != 1
            or gate_weight.stride(-1) != 1
        ):
            return None

        try:
            q_out = int(q_weight.shape[0])
            gate_out = int(gate_weight.shape[0])
            signature = (
                self._q_gate_tensor_signature(q_weight),
                self._q_gate_tensor_signature(gate_weight),
                self._q_gate_tensor_signature(q_bias),
                self._q_gate_tensor_signature(gate_bias),
            )
            cache = self._q_gate_fused_cache
            if cache is None or cache[0] != signature:
                with torch.no_grad():
                    fused_weight = torch.cat(
                        (q_weight.detach(), gate_weight.detach()), dim=0
                    ).contiguous()
                    fused_bias = torch.cat(
                        (q_bias.detach(), gate_bias.detach()), dim=0
                    ).contiguous()
                cache = (signature, q_out, gate_out, fused_weight, fused_bias)
                self._q_gate_fused_cache = cache

            _, q_out, gate_out, fused_weight, fused_bias = cache
            with _ltx2_record_function("ltx2_fused_q_gate::linear"):
                fused = F.linear(x, fused_weight, fused_bias)
            return fused[..., :q_out], fused[..., q_out : q_out + gate_out]
        except Exception as exc:
            _ltx2_disable_fused_q_gate(exc)
            self._q_gate_fused_cache = None
            return None

    def _try_compiled_gate_to_out_residual(
        self,
        out: torch.Tensor,
        gate_logits: torch.Tensor,
        residual: torch.Tensor | None,
        output_gate: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if (
            residual is None
            or output_gate is None
            or not _ltx2_compile_gate_to_out_residual_enabled()
            or _LTX2_COMPILED_GATE_TO_OUT_RESIDUAL_RUNTIME_DISABLED
            or not _ltx2_compile_gate_to_out_enabled()
            or get_tp_world_size() != 1
            or not out.is_cuda
            or not gate_logits.is_cuda
            or not residual.is_cuda
            or not output_gate.is_cuda
            or out.dtype not in (torch.float16, torch.bfloat16)
            or gate_logits.dtype != out.dtype
            or residual.dtype != out.dtype
            or output_gate.dtype != out.dtype
            or out.ndim != 4
            or gate_logits.ndim != 3
            or out.shape[:3] != gate_logits.shape
            or residual.shape != out.shape[:2] + (self.query_dim,)
            or out.shape[2] != self.local_heads
            or out.shape[3] != self.dim_head
            or gate_logits.shape[-1] != self.local_heads
            or self.heads != 32
            or self.local_heads != 32
            or not out.is_contiguous()
            or not residual.is_contiguous()
            or gate_logits.stride(-1) != 1
        ):
            return None
        is_video_self_shape = (
            self.query_dim == self.inner_dim
            and self.query_dim == 4096
            and self.dim_head == 128
        )
        is_a2v_shape = (
            _ltx2_compile_a2v_gate_to_out_enabled()
            and self.query_dim == 4096
            and self.inner_dim == 2048
            and self.dim_head == 64
        )
        if not (is_video_self_shape or is_a2v_shape):
            return None

        to_out = _ltx2_linear_base_for_fusion(self.to_out[0])
        if to_out is None:
            return None
        if (
            not getattr(to_out, "input_is_parallel", False)
            or getattr(to_out, "skip_bias_add", False)
            or not getattr(to_out, "reduce_results", True)
            or to_out.quant_method.__class__.__name__ != "UnquantizedLinearMethod"
        ):
            return None

        weight = getattr(to_out, "weight", None)
        bias = getattr(to_out, "bias", None)
        if weight is None or bias is None:
            return None
        if (
            weight.device != out.device
            or bias.device != out.device
            or weight.dtype != out.dtype
            or bias.dtype != out.dtype
            or weight.ndim != 2
            or bias.ndim != 1
            or weight.shape != (self.query_dim, self.inner_dim)
            or bias.shape[0] != self.query_dim
            or weight.stride(-1) != 1
        ):
            return None

        try:
            compiled_gate_to_out_residual = _ltx2_get_compiled_gate_to_out_residual()
            with _ltx2_record_function(
                "ltx2_compiled_gate_to_out_residual::gate_linear_residual"
            ):
                return compiled_gate_to_out_residual(
                    out, gate_logits, weight, bias, residual, output_gate
                )
        except Exception as exc:
            _ltx2_disable_compiled_gate_to_out_residual(exc)
            return None

    def _try_compiled_gate_to_out(
        self,
        out: torch.Tensor,
        gate_logits: torch.Tensor,
    ) -> torch.Tensor | None:
        if (
            not _ltx2_compile_gate_to_out_enabled()
            or _LTX2_COMPILED_GATE_TO_OUT_RUNTIME_DISABLED
            or get_tp_world_size() != 1
            or not out.is_cuda
            or not gate_logits.is_cuda
            or out.dtype not in (torch.float16, torch.bfloat16)
            or gate_logits.dtype != out.dtype
            or out.ndim != 4
            or gate_logits.ndim != 3
            or out.shape[:3] != gate_logits.shape
            or out.shape[2] != self.local_heads
            or out.shape[3] != self.dim_head
            or gate_logits.shape[-1] != self.local_heads
            or self.heads != 32
            or self.local_heads != 32
            or not out.is_contiguous()
            or gate_logits.stride(-1) != 1
        ):
            return None
        is_video_self_shape = (
            self.query_dim == self.inner_dim
            and self.query_dim == 4096
            and self.dim_head == 128
        )
        is_a2v_shape = (
            _ltx2_compile_a2v_gate_to_out_enabled()
            and self.query_dim == 4096
            and self.inner_dim == 2048
            and self.dim_head == 64
        )
        if not (is_video_self_shape or is_a2v_shape):
            return None

        to_out = _ltx2_linear_base_for_fusion(self.to_out[0])
        if to_out is None:
            return None
        if (
            not getattr(to_out, "input_is_parallel", False)
            or getattr(to_out, "skip_bias_add", False)
            or not getattr(to_out, "reduce_results", True)
            or to_out.quant_method.__class__.__name__ != "UnquantizedLinearMethod"
        ):
            return None

        weight = getattr(to_out, "weight", None)
        bias = getattr(to_out, "bias", None)
        if weight is None or bias is None:
            return None
        if (
            weight.device != out.device
            or bias.device != out.device
            or weight.dtype != out.dtype
            or bias.dtype != out.dtype
            or weight.ndim != 2
            or bias.ndim != 1
            or weight.shape != (self.query_dim, self.inner_dim)
            or bias.shape[0] != self.query_dim
            or weight.stride(-1) != 1
        ):
            return None

        try:
            compiled_gate_to_out = _ltx2_get_compiled_gate_to_out()
            with _ltx2_record_function("ltx2_compiled_gate_to_out::gate_linear"):
                return compiled_gate_to_out(out, gate_logits, weight, bias)
        except Exception as exc:
            _ltx2_disable_compiled_gate_to_out(exc)
            return None

    def _try_fp4_to_out_with_residual_gate(
        self,
        out_flat: torch.Tensor,
        residual: torch.Tensor | None,
        gate: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if (
            residual is None
            or gate is None
            or not _ltx2_fp4_fused_attn_to_out_bias_gate_enabled()
            or get_tp_world_size() != 1
            or not out_flat.is_cuda
            or out_flat.dtype not in (torch.float16, torch.bfloat16)
            or out_flat.ndim != 3
            or residual.shape != out_flat.shape[:-1] + (self.query_dim,)
            or residual.dtype != out_flat.dtype
            or gate.dtype != out_flat.dtype
            or not residual.is_contiguous()
        ):
            return None
        to_out = _ltx2_linear_base_for_fusion(self.to_out[0])
        if to_out is None:
            return None
        if (
            not getattr(to_out, "input_is_parallel", False)
            or getattr(to_out, "skip_bias_add", False)
            or not getattr(to_out, "reduce_results", True)
            or getattr(to_out, "tp_size", 1) != 1
            or to_out.quant_method.__class__.__name__ != "ModelOptFp4LinearMethod"
        ):
            return None
        bias = getattr(to_out, "bias", None)
        if (
            bias is None
            or bias.device != out_flat.device
            or bias.dtype != out_flat.dtype
        ):
            return None

        try:
            from sglang.jit_kernel.diffusion.triton.ltx2_gelu import (
                ltx2_bias_residual_gate,
            )

            with _ltx2_record_function(
                f"ltx2_fp4_epilogue_attn_to_out_bias_gate::{self.profile_prefix}.linear_residual_gate"
            ):
                fused_residual = modelopt_fp4_apply_linear_per_col_residual_gate(
                    to_out, out_flat, residual, gate, bias
                )
            if fused_residual is not None:
                return fused_residual

            with _ltx2_record_function(
                f"ltx2_fp4_fused_attn_to_out_bias_gate::{self.profile_prefix}.linear"
            ):
                update = to_out.quant_method.apply(to_out, out_flat, bias=None)
            if not update.is_contiguous():
                return None
            with _ltx2_record_function(
                f"ltx2_fp4_fused_attn_to_out_bias_gate::{self.profile_prefix}.epilogue"
            ):
                return ltx2_bias_residual_gate(update, residual, gate, bias)
        except Exception as exc:
            logger.warning_once(
                f"Disabling LTX2 FP4 fused attn to_out+bias+gate fast path: {exc}"
            )
            return None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        with _ltx2_record_function(f"ltx2_attention::{self.profile_prefix}"):
            return self._forward_impl(*args, **kwargs)

    def _forward_impl(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
        skip_sequence_parallel_override: bool = False,
        gather_context_kv_for_sp: bool = False,
        gather_query_for_sp: bool = False,
        output_residual: torch.Tensor | None = None,
        output_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        profile_prefix = self.profile_prefix
        gate_input = x
        context_ = x if context is None else context
        use_attention = not all_perturbed
        fused_gate_logits = None
        fused_kv = None
        fused_qkv = None
        fused_audio_qkvg = None
        if use_attention and context is None:
            with _ltx2_record_function(
                f"ltx2_attention_proj::{profile_prefix}.audio_qkvg"
            ):
                fused_audio_qkvg = self._try_fused_audio_qkvg_projection(x)
        if use_attention and context is None and fused_audio_qkvg is None:
            with _ltx2_record_function(f"ltx2_attention_proj::{profile_prefix}.qkv"):
                fused_qkv = self._try_fp4_shared_qkv_projection(x)
                if fused_qkv is None:
                    fused_qkv = self._try_fused_qkv_projection(x)
        if fused_audio_qkvg is not None:
            q, k, v, fused_gate_logits = fused_audio_qkvg
        elif fused_qkv is None:
            if use_attention:
                with _ltx2_record_function(f"ltx2_attention_proj::{profile_prefix}.kv"):
                    fused_kv = self._try_fused_kv_projection(context_)
            if fused_kv is None:
                with _ltx2_record_function(
                    f"ltx2_attention_proj::{profile_prefix}.to_v"
                ):
                    v, _ = self.to_v(context_)
            else:
                k, v = fused_kv
        else:
            q, k, v = fused_qkv

        if use_attention:
            if fused_qkv is None and fused_audio_qkvg is None:
                fused_q_gate = None
                if self.to_gate_logits is not None:
                    with _ltx2_record_function(
                        f"ltx2_attention_proj::{profile_prefix}.q_gate"
                    ):
                        fused_q_gate = self._try_fp4_shared_q_gate_projection(x)
                        if fused_q_gate is None:
                            fused_q_gate = self._try_fused_q_gate_projection(x)
                if fused_q_gate is None:
                    with _ltx2_record_function(
                        f"ltx2_attention_proj::{profile_prefix}.to_q"
                    ):
                        q, _ = self.to_q(x)
                else:
                    q, fused_gate_logits = fused_q_gate
                if fused_kv is None:
                    with _ltx2_record_function(
                        f"ltx2_attention_proj::{profile_prefix}.to_k"
                    ):
                        k, _ = self.to_k(context_)

            applied_fused_qknorm_rope = False
            if self.qk_norm and pe is not None:
                assert self.q_norm is not None and self.k_norm is not None
                fused_qk_rope = _ltx2_try_fused_qknorm_split_rope(
                    q, k, self.q_norm, self.k_norm, self.norm_eps, pe, k_pe
                )
                if fused_qk_rope is not None:
                    q, k = fused_qk_rope
                    applied_fused_qknorm_rope = True

            if self.qk_norm and not applied_fused_qknorm_rope:
                assert self.q_norm is not None and self.k_norm is not None
                with _ltx2_record_function(
                    f"ltx2_attention_norm::{profile_prefix}.qk_norm"
                ):
                    fused_qk = _ltx2_try_fused_qknorm(
                        q, k, self.q_norm, self.k_norm, self.norm_eps
                    )
                    if fused_qk is None:
                        q = self.q_norm(q)
                        k = self.k_norm(k)
                    else:
                        q, k = fused_qk

            q_after_norm_for_debug = q
            k_after_norm_for_debug = k
            if pe is not None and not applied_fused_qknorm_rope:
                cos, sin = pe
                k_cos, k_sin = pe if k_pe is None else k_pe
                tp_size = get_tp_world_size()
                if tp_size > 1:
                    with _ltx2_record_function(
                        f"ltx2_attention_rotary::{profile_prefix}.tp_slice"
                    ):
                        tp_rank = get_tp_rank()
                        cos, sin = self._slice_rope_for_tp(
                            cos, sin, tp_rank=tp_rank, tp_size=tp_size
                        )
                        k_cos, k_sin = self._slice_rope_for_tp(
                            k_cos, k_sin, tp_rank=tp_rank, tp_size=tp_size
                        )
                with _ltx2_record_function(
                    f"ltx2_attention_rotary::{profile_prefix}.qk_apply"
                ):
                    if cos.dim() == 3:
                        q = apply_interleaved_rotary_emb(q, (cos, sin))
                        k = apply_interleaved_rotary_emb(k, (k_cos, k_sin))
                    else:
                        q, k = apply_split_rotary_emb_qk(
                            q, k, (cos, sin), (k_cos, k_sin)
                        )
            dump_attention_debug_from_env(
                dump_dir_env="SGLANG_LTX2_ATTENTION_DEBUG_DIR",
                env_prefix="SGLANG_LTX2_ATTENTION_DEBUG",
                name=str(profile_prefix),
                payload={
                    "q_after_norm": q_after_norm_for_debug,
                    "k_after_norm": k_after_norm_for_debug,
                    "q_after_rope": q,
                    "k_after_rope": k,
                    "v_flat": v,
                    "pe": pe,
                    "k_pe": k_pe,
                },
            )

        v = v.view(*v.shape[:-1], self.local_heads, self.dim_head)
        if use_attention:
            q = q.view(*q.shape[:-1], self.local_heads, self.dim_head)
            k = k.view(*k.shape[:-1], self.local_heads, self.dim_head)
            local_query_tokens = q.shape[1]
            query_gathered_for_sp = False

            if gather_query_for_sp:
                try:
                    sp_world_size = get_sp_world_size()
                except AssertionError:
                    sp_world_size = 1
                if sp_world_size > 1:
                    q = sequence_model_parallel_all_gather(q.contiguous(), dim=1)
                    if mask is not None:
                        query_mask_dim = None
                        for dim in range(mask.ndim):
                            if mask.shape[dim] == local_query_tokens and dim != 0:
                                query_mask_dim = dim
                                break
                        if (
                            query_mask_dim is None
                            and mask.shape[0] == local_query_tokens
                        ):
                            query_mask_dim = 0
                        if query_mask_dim is not None:
                            mask = sequence_model_parallel_all_gather(
                                mask.contiguous(), dim=query_mask_dim
                            )
                    query_gathered_for_sp = True

            if gather_context_kv_for_sp:
                with _ltx2_record_function(
                    f"ltx2_attention_sp::{profile_prefix}.all_gather"
                ):
                    k_full = sequence_model_parallel_all_gather(k.contiguous(), dim=1)
                    v_full = sequence_model_parallel_all_gather(v.contiguous(), dim=1)
                    gathered_mask = None
                    if mask is not None:
                        gathered_mask = sequence_model_parallel_all_gather(
                            mask.contiguous(), dim=1
                        )
                if self.use_local_attention:
                    with _ltx2_record_function(
                        f"ltx2_attention_core::{profile_prefix}"
                    ):
                        out = self.attn(q, k_full, v_full, attn_mask=gathered_mask)
                else:
                    with _ltx2_record_function(
                        f"ltx2_attention_core::{profile_prefix}"
                    ):
                        out = self.attn(
                            q,
                            k_full,
                            v_full,
                            attn_mask=gathered_mask,
                            skip_sequence_parallel_override=True,
                        )
            elif self.use_local_attention:
                with _ltx2_record_function(f"ltx2_attention_core::{profile_prefix}"):
                    out = self.attn(q, k, v, attn_mask=mask)
            else:
                out = None
                route_stage2_video_self_to_backend = (
                    self._route_stage2_video_self_attention_to_backend()
                )
                if (
                    mask is None
                    and not skip_sequence_parallel_override
                    and not route_stage2_video_self_to_backend
                ):
                    out = _ltx2_try_official_fa4_attention(q, k, v, profile_prefix)
                if out is None:
                    with _ltx2_record_function(
                        f"ltx2_attention_core::{profile_prefix}"
                    ):
                        out = self.attn(
                            q,
                            k,
                            v,
                            attn_mask=mask,
                            skip_sequence_parallel_override=skip_sequence_parallel_override,
                        )

            if query_gathered_for_sp:
                sp_rank = get_sp_parallel_rank()
                query_start = int(sp_rank) * int(local_query_tokens)
                out = out[
                    :, query_start : query_start + local_query_tokens
                ].contiguous()

            out_before_perturbation_for_debug = out
            if perturbation_mask is not None:
                with _ltx2_record_function(
                    f"ltx2_attention_mix::{profile_prefix}.perturbation"
                ):
                    if perturbation_mask.ndim == out.ndim - 1:
                        perturbation_mask = perturbation_mask.unsqueeze(-1)
                    out = out * perturbation_mask + v * (1 - perturbation_mask)

        if not use_attention:
            out = v
            out_before_perturbation_for_debug = out

        out_before_gate_for_debug = out

        if self.to_gate_logits is not None:
            with _ltx2_record_function(
                f"ltx2_attention_proj::{profile_prefix}.gate_logits"
            ):
                if fused_gate_logits is None:
                    gate_logits, _ = self.to_gate_logits(gate_input)
                else:
                    gate_logits = fused_gate_logits
                compiled_gate_to_out_residual = self._try_compiled_gate_to_out_residual(
                    out, gate_logits, output_residual, output_gate
                )
                if compiled_gate_to_out_residual is not None:
                    return compiled_gate_to_out_residual
                compiled_gate_to_out = self._try_compiled_gate_to_out(out, gate_logits)
                if compiled_gate_to_out is not None:
                    if output_residual is not None and output_gate is not None:
                        return _ltx2_residual_gate_add(
                            output_residual, compiled_gate_to_out, output_gate
                        )
                    return compiled_gate_to_out
                b, t = out.shape[:2]
                out = out.view(b, t, self.local_heads, self.dim_head)
                out = out * (2.0 * torch.sigmoid(gate_logits).unsqueeze(-1))
                out = out.view(b, t, self.local_heads * self.dim_head)

        dump_attention_debug_from_env(
            dump_dir_env="SGLANG_LTX2_ATTENTION_DEBUG_DIR",
            env_prefix="SGLANG_LTX2_ATTENTION_DEBUG",
            name=str(profile_prefix),
            payload={
                "attn_out_before_perturbation": out_before_perturbation_for_debug,
                "attn_out_before_gate": out_before_gate_for_debug,
                "gate_logits": gate_logits if self.to_gate_logits is not None else None,
                "out_after_gate_flat": out,
            },
        )

        out_flat = out.flatten(2)
        fused_output_residual = self._try_fp4_to_out_with_residual_gate(
            out_flat, output_residual, output_gate
        )
        if fused_output_residual is not None:
            return fused_output_residual

        with _ltx2_record_function(f"ltx2_attention_proj::{profile_prefix}.to_out"):
            out_proj, _ = self.to_out[0](out_flat)

        if output_residual is not None and output_gate is not None:
            return _ltx2_residual_gate_add(output_residual, out_proj, output_gate)
        return out_proj

    def _slice_rope_for_tp(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        tp_rank: int,
        tp_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice RoPE tensors to the local TP shard.

        - split-rope: cos/sin are shaped [B, H, T, R] (head-major), slice by heads.
        - interleaved-rope: cos/sin are shaped [B, T, D], where D matches the projected
          feature dimension and is sharded by TP.
        """
        if cos.ndim == 4:
            # [B, H, T, R]
            start = tp_rank * self.local_heads
            end = start + self.local_heads
            return cos[:, start:end, :, :], sin[:, start:end, :, :]
        elif cos.ndim == 3:
            # [B, T, D]
            d = cos.shape[-1]
            if d % tp_size != 0:
                raise ValueError(
                    f"RoPE dim must be divisible by tp_size, got {d=} {tp_size=}."
                )
            local_d = d // tp_size
            start = tp_rank * local_d
            end = start + local_d
            return cos[:, :, start:end], sin[:, :, start:end]
        raise ValueError(f"Unexpected RoPE tensor rank: {cos.ndim}. Expected 3 or 4.")


class LTX2FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        if dim_out is None:
            dim_out = dim
        inner_dim = int(dim * mult)

        self.proj_in = ColumnParallelLinear(
            dim,
            inner_dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=_ltx2_child_prefix(prefix, "proj_in"),
        )
        self.act = nn.GELU(approximate="tanh")
        self.proj_out = RowParallelLinear(
            inner_dim,
            dim_out,
            bias=True,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=_ltx2_child_prefix(prefix, "proj_out"),
        )
        self._te_nvfp4_video_ffn = (
            dim == 4096
            and dim_out == 4096
            and inner_dim == 16384
            and str(prefix).endswith(".ff")
            and not str(prefix).endswith(".audio_ff")
        )
        self._te_nvfp4_proj_in = None
        self._te_nvfp4_proj_out = None
        self._te_nvfp4_proj_out_return_bias = None

    def _get_te_nvfp4_linear_context(
        self, cache_attr: str, layer: nn.Module, *, return_bias: bool = False
    ) -> tuple[nn.Module, object, object] | None:
        if (
            not _ltx2_te_nvfp4_video_ffn_enabled()
            or not self._te_nvfp4_video_ffn
            or get_tp_world_size() != 1
        ):
            return None

        base = _ltx2_linear_base_for_fusion(layer)
        if base is None:
            return None
        if (
            getattr(base, "quant_method", None).__class__.__name__
            != "UnquantizedLinearMethod"
        ):
            return None
        weight = getattr(base, "weight", None)
        bias = getattr(base, "bias", None)
        if (
            weight is None
            or not weight.is_cuda
            or weight.dtype not in (torch.float16, torch.bfloat16)
            or getattr(base, "tp_size", 1) != 1
            or getattr(base, "skip_bias_add", False)
        ):
            return None
        if bias is not None and (
            bias.device != weight.device or bias.dtype != weight.dtype
        ):
            return None

        context = _ltx2_get_te_nvfp4_context()
        if context is None:
            return None
        te_linear_cls, fp8_autocast, recipe = context

        input_size = int(getattr(base, "input_size_per_partition", weight.shape[1]))
        output_size = int(getattr(base, "output_size_per_partition", weight.shape[0]))
        cached = getattr(self, cache_attr)
        if (
            cached is None
            or getattr(cached, "weight", None) is not weight
            or getattr(cached, "bias", None) is not bias
            or bool(getattr(cached, "return_bias", False)) != bool(return_bias)
        ):
            te_layer = te_linear_cls(
                input_size,
                output_size,
                bias=bias is not None,
                return_bias=return_bias,
                params_dtype=weight.dtype,
                device=weight.device,
            )
            te_layer.weight = weight
            if bias is not None:
                te_layer.bias = bias
            te_layer.train(self.training)
            setattr(self, cache_attr, te_layer)
            cached = te_layer
        return cached, fp8_autocast, recipe

    def _try_te_nvfp4_linear_gelu(
        self, cache_attr: str, layer: nn.Module, x: torch.Tensor
    ) -> torch.Tensor | None:
        global _LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU_RUNTIME_DISABLED
        global _LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU_WARNING_EMITTED
        if (
            not _ltx2_te_nvfp4_fused_proj_in_gelu_enabled()
            or _LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU_RUNTIME_DISABLED
            or torch.is_grad_enabled()
            or not x.is_cuda
            or x.dtype not in (torch.float16, torch.bfloat16)
        ):
            return None
        context = self._get_te_nvfp4_linear_context(cache_attr, layer)
        if context is None:
            return None
        te_layer, fp8_autocast, recipe = context
        input_shape = tuple(x.shape)
        if not input_shape or input_shape[-1] != int(te_layer.weight.shape[1]):
            return None
        x_2d = x.reshape(-1, input_shape[-1])
        original_m = int(x_2d.shape[0])
        pad_m_to = int(os.environ.get("SGLANG_LTX2_TE_NVFP4_PAD_M_TO", "16"))
        if pad_m_to > 1:
            pad_rows = (-original_m) % pad_m_to
            if pad_rows:
                x_2d = F.pad(x_2d, (0, 0, 0, pad_rows))

        try:
            from transformer_engine.pytorch.cpp_extensions import general_gemm
            from transformer_engine.pytorch.module.base import (
                _2X_ACC_FPROP,
                quantize_weight,
            )
            from transformer_engine.pytorch.quantization import FP8GlobalStateManager
            from transformer_engine.pytorch.utils import (
                assert_dim_for_fp8_exec,
                cast_if_needed,
            )

            with fp8_autocast(enabled=True, fp8_recipe=recipe):
                inp = te_layer.prepare_forward(x_2d, allow_non_contiguous=False)
                try:
                    if not getattr(te_layer, "fp8", False):
                        return None
                    weight, bias = te_layer._get_weight_and_bias_tensors()
                    assert_dim_for_fp8_exec(inp, weight)
                    (
                        input_quantizer,
                        weight_quantizer,
                        output_quantizer,
                        _grad_input_quantizer,
                        _grad_weight_quantizer,
                        _grad_output_quantizer,
                    ) = te_layer._get_quantizers(
                        fp8_output=False, fp8_grad=False, is_grad_enabled=False
                    )
                    if input_quantizer is None or weight_quantizer is None:
                        return None
                    input_quantizer.set_usage(rowwise=True, columnwise=False)
                    inputmat = input_quantizer(inp)
                    weight_quantizer.set_usage(rowwise=True, columnwise=False)
                    weightmat, _new_weight_workspace = quantize_weight(
                        tensor=weight,
                        quantizer=weight_quantizer,
                        workspace=None,
                        update_workspace=True,
                        skip_update_flag=None,
                        fsdp_group=getattr(te_layer, "fsdp_group", None),
                        workspace_dtype=getattr(te_layer, "activation_dtype", x.dtype),
                        cache=False,
                    )
                    weightmat.update_usage(rowwise_usage=True)
                    bias_dtype = getattr(te_layer, "activation_dtype", x.dtype)
                    bias = (
                        cast_if_needed(bias, bias_dtype) if bias is not None else None
                    )
                    use_split_accumulator = _2X_ACC_FPROP
                    fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
                    if hasattr(fp8_recipe, "fp8_gemm_fprop"):
                        use_split_accumulator = (
                            fp8_recipe.fp8_gemm_fprop.use_split_accumulator
                        )
                    with _ltx2_record_function(
                        f"ltx2_te_nvfp4_fused_proj_in_gelu::{self.prefix}"
                    ):
                        out, *_ = general_gemm(
                            weightmat,
                            inputmat,
                            quantization_params=output_quantizer,
                            out_dtype=getattr(te_layer, "activation_dtype", x.dtype),
                            bias=bias,
                            gelu=True,
                            use_split_accumulator=use_split_accumulator,
                        )
                finally:
                    te_layer.end_forward()
        except Exception as exc:
            _LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU_RUNTIME_DISABLED = True
            if not _LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU_WARNING_EMITTED:
                logger.warning(
                    "Disabling LTX2 TE NVFP4 fused proj_in+GELU path after failure: %s",
                    exc,
                )
                _LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU_WARNING_EMITTED = True
            return None

        if int(out.shape[0]) != original_m:
            out = out[:original_m]
        return out.reshape(*input_shape[:-1], int(out.shape[-1]))

    def _try_te_nvfp4_linear(
        self, cache_attr: str, layer: nn.Module, x: torch.Tensor
    ) -> torch.Tensor | None:
        if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
            return None
        context = self._get_te_nvfp4_linear_context(cache_attr, layer)
        if context is None:
            return None
        te_layer, fp8_autocast, recipe = context
        input_shape = tuple(x.shape)
        if not input_shape or input_shape[-1] != int(te_layer.weight.shape[1]):
            return None
        x_2d = x.reshape(-1, input_shape[-1])
        original_m = int(x_2d.shape[0])
        pad_m_to = int(os.environ.get("SGLANG_LTX2_TE_NVFP4_PAD_M_TO", "16"))
        if pad_m_to > 1:
            pad_rows = (-original_m) % pad_m_to
            if pad_rows:
                x_2d = F.pad(x_2d, (0, 0, 0, pad_rows))
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            out = te_layer(x_2d)
        if int(out.shape[0]) != original_m:
            out = out[:original_m]
        return out.reshape(*input_shape[:-1], int(out.shape[-1]))

    def _try_te_nvfp4_linear_return_bias(
        self, cache_attr: str, layer: nn.Module, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
            return None
        context = self._get_te_nvfp4_linear_context(cache_attr, layer, return_bias=True)
        if context is None:
            return None
        te_layer, fp8_autocast, recipe = context
        input_shape = tuple(x.shape)
        if not input_shape or x.shape[-1] != int(te_layer.weight.shape[1]):
            return None
        x_2d = x.reshape(-1, input_shape[-1])
        original_m = int(x_2d.shape[0])
        pad_m_to = int(os.environ.get("SGLANG_LTX2_TE_NVFP4_PAD_M_TO", "16"))
        if pad_m_to > 1:
            pad_rows = (-original_m) % pad_m_to
            if pad_rows:
                x_2d = F.pad(x_2d, (0, 0, 0, pad_rows))
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            result = te_layer(x_2d)
        if not isinstance(result, tuple) or len(result) != 2:
            return None
        out, bias = result
        if bias is None or bias.device != out.device or bias.dtype != out.dtype:
            return None
        if int(out.shape[0]) != original_m:
            out = out[:original_m]
        return out.reshape(*input_shape[:-1], int(out.shape[-1])), bias

    def _try_te_nvfp4_forward_with_residual_gate(
        self, x: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor | None:
        if (
            not _ltx2_te_nvfp4_fused_proj_out_bias_gate_enabled()
            or not _ltx2_te_nvfp4_video_ffn_enabled()
            or not self._te_nvfp4_video_ffn
            or get_tp_world_size() != 1
            or not x.is_cuda
            or x.dtype not in (torch.float16, torch.bfloat16)
            or residual.dtype != x.dtype
            or gate.dtype != x.dtype
            or residual.shape != x.shape[:-1] + (4096,)
            or not residual.is_contiguous()
        ):
            return None
        try:
            from sglang.jit_kernel.diffusion.triton.ltx2_gelu import (
                ltx2_bias_residual_gate,
            )
        except Exception as exc:
            logger.warning_once(
                f"Disabling LTX2 TE NVFP4 fused proj_out+bias+gate path: {exc}"
            )
            return None

        hidden = self._try_te_nvfp4_linear_gelu("_te_nvfp4_proj_in", self.proj_in, x)
        if hidden is None:
            hidden = self._try_te_nvfp4_linear("_te_nvfp4_proj_in", self.proj_in, x)
            if hidden is None:
                return None
            with _ltx2_record_function(f"ltx2_ffn_act::{self.prefix}.gelu"):
                fused_gelu = _ltx2_try_gelu_tanh_inplace(hidden)
                hidden = self.act(hidden) if fused_gelu is None else fused_gelu

        with _ltx2_record_function(
            f"ltx2_te_nvfp4_ffn_proj::{self.prefix}.proj_out_return_bias"
        ):
            update_bias = self._try_te_nvfp4_linear_return_bias(
                "_te_nvfp4_proj_out_return_bias", self.proj_out, hidden
            )
        if update_bias is None:
            return None
        update, bias = update_bias
        if not update.is_contiguous():
            return None
        with _ltx2_record_function(
            f"ltx2_te_nvfp4_fused_proj_out_bias_gate::{self.prefix}.epilogue"
        ):
            return ltx2_bias_residual_gate(update, residual, gate, bias)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        with _ltx2_record_function(f"ltx2_feedforward::{self.prefix}"):
            return self._forward_impl(*args, **kwargs)

    def try_forward_with_residual_gate(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor | None:
        te_residual = self._try_te_nvfp4_forward_with_residual_gate(x, residual, gate)
        if te_residual is not None:
            return te_residual

        proj_out = _ltx2_linear_base_for_fusion(self.proj_out)
        if proj_out is None:
            return None
        proj_out_size = getattr(self.proj_out, "output_size", None)
        if proj_out_size is None:
            proj_out_size = getattr(proj_out, "output_size", None)
        if (
            not _ltx2_fp4_fused_proj_out_bias_gate_enabled()
            or get_tp_world_size() != 1
            or not x.is_cuda
            or x.dtype not in (torch.float16, torch.bfloat16)
            or proj_out_size is None
            or residual.shape != x.shape[:-1] + (proj_out_size,)
            or residual.dtype != x.dtype
            or gate.dtype != x.dtype
            or not residual.is_contiguous()
        ):
            return None
        if (
            getattr(proj_out, "skip_bias_add", False)
            or getattr(proj_out, "tp_size", 1) != 1
            or proj_out.quant_method.__class__.__name__ != "ModelOptFp4LinearMethod"
        ):
            return None
        bias = getattr(proj_out, "bias", None)
        if bias is None or bias.device != x.device or bias.dtype != x.dtype:
            return None

        try:
            from sglang.jit_kernel.diffusion.triton.ltx2_gelu import (
                ltx2_bias_residual_gate,
            )

            hidden = _ltx2_try_fp4_fused_proj_in_bias_gelu(x, self.proj_in)
            if hidden is None:
                return None
            with _ltx2_record_function(
                f"ltx2_fp4_epilogue_proj_out_bias_gate::{self.prefix}.linear_residual_gate"
            ):
                fused_residual = modelopt_fp4_apply_linear_per_col_residual_gate(
                    proj_out, hidden, residual, gate, bias
                )
            if fused_residual is not None:
                return fused_residual

            with _ltx2_record_function(
                f"ltx2_fp4_fused_proj_out_bias_gate::{self.prefix}.linear"
            ):
                update = proj_out.quant_method.apply(proj_out, hidden, bias=None)
            if not update.is_contiguous():
                return None
            with _ltx2_record_function(
                f"ltx2_fp4_fused_proj_out_bias_gate::{self.prefix}.epilogue"
            ):
                return ltx2_bias_residual_gate(update, residual, gate, bias)
        except Exception as exc:
            logger.warning_once(
                f"Disabling LTX2 FP4 fused proj_out+bias+gate fast path: {exc}"
            )
            return None

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        te_proj_in_gelu = None
        with _ltx2_record_function(
            f"ltx2_te_nvfp4_ffn_proj::{self.prefix}.proj_in_gelu"
        ):
            te_proj_in_gelu = self._try_te_nvfp4_linear_gelu(
                "_te_nvfp4_proj_in", self.proj_in, x
            )
        if te_proj_in_gelu is not None:
            x = te_proj_in_gelu
        else:
            te_proj_in = None
            with _ltx2_record_function(
                f"ltx2_te_nvfp4_ffn_proj::{self.prefix}.proj_in"
            ):
                te_proj_in = self._try_te_nvfp4_linear(
                    "_te_nvfp4_proj_in", self.proj_in, x
                )
            if te_proj_in is not None:
                x = te_proj_in
                with _ltx2_record_function(f"ltx2_ffn_act::{self.prefix}.gelu"):
                    fused_gelu = _ltx2_try_gelu_tanh_inplace(x)
                    x = self.act(x) if fused_gelu is None else fused_gelu
            else:
                fused_proj_in = _ltx2_try_fp4_fused_proj_in_bias_gelu(x, self.proj_in)
                if fused_proj_in is None:
                    fused_proj_in = _ltx2_try_fused_ffn_proj_in_gelu(x, self.proj_in)
                if fused_proj_in is None:
                    with _ltx2_record_function(f"ltx2_ffn_proj::{self.prefix}.proj_in"):
                        x, _ = self.proj_in(x)
                    with _ltx2_record_function(f"ltx2_ffn_act::{self.prefix}.gelu"):
                        fused_gelu = _ltx2_try_gelu_tanh_inplace(x)
                        x = self.act(x) if fused_gelu is None else fused_gelu
                else:
                    x = fused_proj_in

        with _ltx2_record_function(f"ltx2_te_nvfp4_ffn_proj::{self.prefix}.proj_out"):
            te_proj_out = self._try_te_nvfp4_linear(
                "_te_nvfp4_proj_out", self.proj_out, x
            )
        if te_proj_out is not None:
            x = te_proj_out
        else:
            with _ltx2_record_function(f"ltx2_ffn_proj::{self.prefix}.proj_out"):
                x, _ = self.proj_out(x)
        return x


class LTX2TransformerBlock(nn.Module):
    def __init__(
        self,
        idx: int,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_dim: int,
        audio_num_attention_heads: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
        apply_gated_attention: bool = False,
        cross_attention_adaln: bool = False,
        use_local_av_cross_attention: bool = False,
        force_sdpa_v2a_cross_attention: bool = False,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.idx = idx
        self.norm_eps = norm_eps
        # LTX2.3
        self.cross_attention_adaln = cross_attention_adaln
        self.use_local_av_cross_attention = use_local_av_cross_attention

        # 1. Self-Attention (video and audio)
        self.attn1 = LTX2Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            apply_gated_attention=apply_gated_attention,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn1",
            profile_prefix=f"block_{idx}.attn1",
            quant_config=quant_config,
        )
        self.audio_attn1 = LTX2Attention(
            query_dim=audio_dim,
            heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            apply_gated_attention=apply_gated_attention,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.audio_attn1",
            profile_prefix=f"block_{idx}.audio_attn1",
            quant_config=quant_config,
        )

        # 2. Prompt Cross-Attention
        # Prompt KV is replicated across SP ranks, so prompt cross-attn should
        # stay local and preserve the explicit KV mask semantics from official.
        self.attn2 = LTX2Attention(
            query_dim=dim,
            context_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            use_local_attention=True,
            apply_gated_attention=apply_gated_attention,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn2",
            profile_prefix=f"block_{idx}.attn2",
            quant_config=quant_config,
        )
        self.audio_attn2 = LTX2Attention(
            query_dim=audio_dim,
            context_dim=audio_cross_attention_dim,
            heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            use_local_attention=True,
            apply_gated_attention=apply_gated_attention,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.audio_attn2",
            profile_prefix=f"block_{idx}.audio_attn2",
            quant_config=quant_config,
        )

        # 3. Audio-to-Video (a2v) and Video-to-Audio (v2a) Cross-Attention
        self.audio_to_video_attn = LTX2Attention(
            query_dim=dim,
            context_dim=audio_dim,
            heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            use_local_attention=use_local_av_cross_attention,
            apply_gated_attention=apply_gated_attention,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.audio_to_video_attn",
            profile_prefix=f"block_{idx}.audio_to_video_attn",
            quant_config=quant_config,
        )
        self.video_to_audio_attn = LTX2Attention(
            query_dim=audio_dim,
            context_dim=dim,
            heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            use_local_attention=use_local_av_cross_attention,
            apply_gated_attention=apply_gated_attention,
            supported_attention_backends=(
                {AttentionBackendEnum.TORCH_SDPA}
                if force_sdpa_v2a_cross_attention
                else supported_attention_backends
            ),
            prefix=f"{prefix}.video_to_audio_attn",
            profile_prefix=f"block_{idx}.video_to_audio_attn",
            quant_config=quant_config,
        )

        # 4. Feedforward layers
        self.ff = LTX2FeedForward(
            dim,
            dim_out=dim,
            prefix=_ltx2_child_prefix(prefix, "ff"),
            quant_config=quant_config,
        )
        self.audio_ff = LTX2FeedForward(
            audio_dim,
            dim_out=audio_dim,
            prefix=_ltx2_child_prefix(prefix, "audio_ff"),
            quant_config=quant_config,
        )

        # 5. Modulation Parameters
        num_ada_params = adaln_embedding_coefficient(cross_attention_adaln)
        self.scale_shift_table = nn.Parameter(
            torch.randn(num_ada_params, dim) / dim**0.5
        )
        self.audio_scale_shift_table = nn.Parameter(
            torch.randn(num_ada_params, audio_dim) / audio_dim**0.5
        )
        self.video_a2v_cross_attn_scale_shift_table = nn.Parameter(torch.randn(5, dim))
        self.audio_a2v_cross_attn_scale_shift_table = nn.Parameter(
            torch.randn(5, audio_dim)
        )
        if self.cross_attention_adaln:
            # LTX2.3
            self.prompt_scale_shift_table = nn.Parameter(torch.randn(2, dim))
            self.audio_prompt_scale_shift_table = nn.Parameter(
                torch.randn(2, audio_dim)
            )

    def get_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep: torch.Tensor,
        indices: slice,
    ) -> tuple[torch.Tensor, ...]:
        with _ltx2_record_function(f"ltx2_ada_values::block_{self.idx}"):
            fused_values = _ltx2_try_fused_ada_values3(
                scale_shift_table, batch_size, timestep, indices
            )
            if fused_values is not None:
                return fused_values

            num_ada_params = int(scale_shift_table.shape[0])
            ada_values = (
                scale_shift_table[indices]
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device=timestep.device, dtype=timestep.dtype)
                + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[
                    :, :, indices, :
                ]
            ).unbind(dim=2)
            return [t.squeeze(2) for t in ada_values]

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        temb_audio: torch.Tensor,
        temb_prompt: torch.Tensor | None,
        temb_audio_prompt: torch.Tensor | None,
        temb_ca_scale_shift: torch.Tensor,
        temb_ca_audio_scale_shift: torch.Tensor,
        temb_ca_gate: torch.Tensor,
        temb_ca_audio_gate: torch.Tensor,
        video_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        audio_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ca_video_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ca_audio_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        audio_encoder_attention_mask: Optional[torch.Tensor] = None,
        video_self_attention_mask: Optional[torch.Tensor] = None,
        audio_self_attention_mask: Optional[torch.Tensor] = None,
        a2v_cross_attention_mask: Optional[torch.Tensor] = None,
        v2a_cross_attention_mask: Optional[torch.Tensor] = None,
        skip_video_self_attn: bool = False,
        skip_audio_self_attn: bool = False,
        skip_a2v_cross_attn: bool = False,
        skip_v2a_cross_attn: bool = False,
        video_self_attn_perturbation_mask: Optional[torch.Tensor] = None,
        audio_self_attn_perturbation_mask: Optional[torch.Tensor] = None,
        a2v_cross_attn_perturbation_mask: Optional[torch.Tensor] = None,
        v2a_cross_attn_perturbation_mask: Optional[torch.Tensor] = None,
        audio_replicated_for_sp: bool = False,
        audio_latents_replicated_for_sp: bool = False,
        disable_sequence_parallel_for_replicated_sp: bool = False,
        share_block0_self_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size = hidden_states.size(0)
        audio_latents_replicated_for_sp = (
            audio_replicated_for_sp or audio_latents_replicated_for_sp
        )
        video_ada_gates = _ltx2_try_fused_ada_gates3(
            self.scale_shift_table, batch_size, temb
        )
        if video_ada_gates is None:
            video_ada_values = _ltx2_try_fused_ada_values9(
                self.scale_shift_table, batch_size, temb
            )
        else:
            video_ada_values = None
        audio_ada_values = _ltx2_try_fused_ada_values9(
            self.audio_scale_shift_table, batch_size, temb_audio
        )

        # 1. Video and Audio Self-Attention
        if video_ada_gates is not None:
            vshift_msa = vscale_msa = None
            vgate_msa = video_ada_gates[0]
        elif video_ada_values is None:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, batch_size, temb, slice(0, 3)
            )
        else:
            vshift_msa, vscale_msa, vgate_msa = video_ada_values[0:3]
        share_video_self_attn = (
            share_block0_self_attn
            and self.idx == 0
            and batch_size > 1
            and video_self_attention_mask is None
            and video_self_attn_perturbation_mask is None
            and not skip_video_self_attn
            and not audio_replicated_for_sp
            and not audio_latents_replicated_for_sp
        )
        if share_video_self_attn:
            norm_hidden_states = None
            if video_ada_gates is not None:
                norm_hidden_states = _ltx2_try_fused_norm_ada_scale_shift(
                    hidden_states[:1],
                    self.scale_shift_table,
                    temb[:1],
                    0,
                    1,
                    self.norm_eps,
                )
            if norm_hidden_states is None:
                if vscale_msa is None or vshift_msa is None:
                    vshift_msa, vscale_msa, _ = self.get_ada_values(
                        self.scale_shift_table, batch_size, temb, slice(0, 3)
                    )
                norm_hidden_states = _ltx2_norm_scale_shift(
                    hidden_states[:1], vscale_msa[:1], vshift_msa[:1], self.norm_eps
                )
            video_rotary_emb_for_attn = (
                None
                if video_rotary_emb is None
                else (video_rotary_emb[0][:1], video_rotary_emb[1][:1])
            )
            attn_hidden_states = (
                self.attn1(
                    norm_hidden_states,
                    pe=video_rotary_emb_for_attn,
                )
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        else:
            norm_hidden_states = None
            if video_ada_gates is not None:
                norm_hidden_states = _ltx2_try_fused_norm_ada_scale_shift(
                    hidden_states,
                    self.scale_shift_table,
                    temb,
                    0,
                    1,
                    self.norm_eps,
                )
            if norm_hidden_states is None:
                if vscale_msa is None or vshift_msa is None:
                    vshift_msa, vscale_msa, _ = self.get_ada_values(
                        self.scale_shift_table, batch_size, temb, slice(0, 3)
                    )
                norm_hidden_states = _ltx2_norm_scale_shift(
                    hidden_states, vscale_msa, vshift_msa, self.norm_eps
                )
            attn_hidden_states = self.attn1(
                norm_hidden_states,
                mask=video_self_attention_mask,
                pe=video_rotary_emb,
                perturbation_mask=video_self_attn_perturbation_mask,
                all_perturbed=skip_video_self_attn,
                gather_context_kv_for_sp=(
                    audio_replicated_for_sp
                    and not disable_sequence_parallel_for_replicated_sp
                ),
                skip_sequence_parallel_override=disable_sequence_parallel_for_replicated_sp,
            )
        if not self.cross_attention_adaln:
            hidden_states = _ltx2_residual_gate_add(
                hidden_states, attn_hidden_states, vgate_msa
            )

        if audio_ada_values is None:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, batch_size, temb_audio, slice(0, 3)
            )
        else:
            ashift_msa, ascale_msa, agate_msa = audio_ada_values[0:3]
        share_audio_self_attn = (
            share_block0_self_attn
            and self.idx == 0
            and batch_size > 1
            and audio_self_attention_mask is None
            and audio_self_attn_perturbation_mask is None
            and not skip_audio_self_attn
            and not audio_replicated_for_sp
            and not audio_latents_replicated_for_sp
        )
        if share_audio_self_attn:
            norm_audio_hidden_states = _ltx2_norm_scale_shift(
                audio_hidden_states[:1], ascale_msa[:1], ashift_msa[:1], self.norm_eps
            )
            audio_rotary_emb_for_attn = (
                None
                if audio_rotary_emb is None
                else (audio_rotary_emb[0][:1], audio_rotary_emb[1][:1])
            )
            attn_audio_hidden_states = (
                self.audio_attn1(
                    norm_audio_hidden_states,
                    pe=audio_rotary_emb_for_attn,
                )
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        else:
            norm_audio_hidden_states = _ltx2_norm_scale_shift(
                audio_hidden_states, ascale_msa, ashift_msa, self.norm_eps
            )
            attn_audio_hidden_states = self.audio_attn1(
                norm_audio_hidden_states,
                mask=audio_self_attention_mask,
                pe=audio_rotary_emb,
                perturbation_mask=audio_self_attn_perturbation_mask,
                all_perturbed=skip_audio_self_attn,
                skip_sequence_parallel_override=(
                    disable_sequence_parallel_for_replicated_sp
                    or audio_latents_replicated_for_sp
                ),
            )
        if not self.cross_attention_adaln:
            audio_hidden_states = _ltx2_residual_gate_add(
                audio_hidden_states, attn_audio_hidden_states, agate_msa
            )
        # 2. Prompt Cross-Attention
        if self.cross_attention_adaln:
            # LTX2.3
            if temb_prompt is None or temb_audio_prompt is None:
                raise ValueError(
                    "cross_attention_adaln requires prompt modulation tensors."
                )
            if video_ada_gates is not None:
                vshift_q = vscale_q = None
                vgate_q = video_ada_gates[2]
            elif video_ada_values is None:
                vshift_q, vscale_q, vgate_q = self.get_ada_values(
                    self.scale_shift_table, batch_size, temb, slice(6, 9)
                )
            else:
                vshift_q, vscale_q, vgate_q = video_ada_values[6:9]
            v_prompt_shift, v_prompt_scale = self.get_ada_values(
                self.prompt_scale_shift_table, batch_size, temb_prompt, slice(None)
            )
            video_q_residual_norm = None
            if video_ada_gates is not None:
                video_q_residual_norm = _ltx2_try_fused_residual_norm_ada_scale_shift(
                    hidden_states,
                    attn_hidden_states,
                    vgate_msa,
                    self.scale_shift_table,
                    temb,
                    6,
                    7,
                    self.norm_eps,
                )
            if video_q_residual_norm is None:
                if vscale_q is None or vshift_q is None:
                    vshift_q, vscale_q, _ = self.get_ada_values(
                        self.scale_shift_table, batch_size, temb, slice(6, 9)
                    )
                norm_hidden_states, hidden_states = _ltx2_residual_norm_scale_shift(
                    hidden_states,
                    attn_hidden_states,
                    vgate_msa,
                    vscale_q,
                    vshift_q,
                    self.norm_eps,
                )
            else:
                norm_hidden_states, hidden_states = video_q_residual_norm
            mod_encoder_hidden_states = _ltx2_modulate(
                encoder_hidden_states, v_prompt_scale, v_prompt_shift
            )
            if _ltx2_fp4_fused_attn_to_out_bias_gate_enabled():
                hidden_states = self.attn2(
                    norm_hidden_states,
                    context=mod_encoder_hidden_states,
                    mask=encoder_attention_mask,
                    skip_sequence_parallel_override=disable_sequence_parallel_for_replicated_sp,
                    output_residual=hidden_states,
                    output_gate=vgate_q,
                )
            else:
                attn_hidden_states = self.attn2(
                    norm_hidden_states,
                    context=mod_encoder_hidden_states,
                    mask=encoder_attention_mask,
                    skip_sequence_parallel_override=disable_sequence_parallel_for_replicated_sp,
                )
                hidden_states = _ltx2_residual_gate_add(
                    hidden_states, attn_hidden_states, vgate_q
                )

            if audio_ada_values is None:
                ashift_q, ascale_q, agate_q = self.get_ada_values(
                    self.audio_scale_shift_table, batch_size, temb_audio, slice(6, 9)
                )
            else:
                ashift_q, ascale_q, agate_q = audio_ada_values[6:9]
            a_prompt_shift, a_prompt_scale = self.get_ada_values(
                self.audio_prompt_scale_shift_table,
                batch_size,
                temb_audio_prompt,
                slice(None),
            )
            norm_audio_hidden_states, audio_hidden_states = (
                _ltx2_residual_norm_scale_shift(
                    audio_hidden_states,
                    attn_audio_hidden_states,
                    agate_msa,
                    ascale_q,
                    ashift_q,
                    self.norm_eps,
                )
            )
            mod_audio_encoder_hidden_states = _ltx2_modulate(
                audio_encoder_hidden_states, a_prompt_scale, a_prompt_shift
            )
            if _ltx2_fp4_fused_attn_to_out_bias_gate_enabled():
                audio_hidden_states = self.audio_attn2(
                    norm_audio_hidden_states,
                    context=mod_audio_encoder_hidden_states,
                    mask=audio_encoder_attention_mask,
                    skip_sequence_parallel_override=disable_sequence_parallel_for_replicated_sp,
                    output_residual=audio_hidden_states,
                    output_gate=agate_q,
                )
            else:
                attn_audio_hidden_states = self.audio_attn2(
                    norm_audio_hidden_states,
                    context=mod_audio_encoder_hidden_states,
                    mask=audio_encoder_attention_mask,
                    skip_sequence_parallel_override=disable_sequence_parallel_for_replicated_sp,
                )
                audio_hidden_states = _ltx2_residual_gate_add(
                    audio_hidden_states, attn_audio_hidden_states, agate_q
                )
        else:
            norm_hidden_states = rms_norm(hidden_states, self.norm_eps)
            attn_hidden_states = self.attn2(
                norm_hidden_states,
                context=encoder_hidden_states,
                mask=encoder_attention_mask,
                skip_sequence_parallel_override=disable_sequence_parallel_for_replicated_sp,
            )
            hidden_states = hidden_states + attn_hidden_states

            norm_audio_hidden_states = rms_norm(audio_hidden_states, self.norm_eps)
            attn_audio_hidden_states = self.audio_attn2(
                norm_audio_hidden_states,
                context=audio_encoder_hidden_states,
                mask=audio_encoder_attention_mask,
                skip_sequence_parallel_override=disable_sequence_parallel_for_replicated_sp,
            )
            audio_hidden_states = audio_hidden_states + attn_audio_hidden_states
        # 3. Audio-to-Video and Video-to-Audio Cross-Attention
        video_dual_mod = _ltx2_try_fused_rmsnorm_ca_dual_modulate(
            hidden_states,
            temb_ca_scale_shift,
            self.video_a2v_cross_attn_scale_shift_table[:4, :],
            self.norm_eps,
        )
        audio_dual_mod = _ltx2_try_fused_rmsnorm_ca_dual_modulate(
            audio_hidden_states,
            temb_ca_audio_scale_shift,
            self.audio_a2v_cross_attn_scale_shift_table[:4, :],
            self.norm_eps,
        )

        # Compute combined ada params
        with _ltx2_record_function(f"ltx2_cross_ada_values::block_{self.idx}"):
            video_per_layer_ca_gate = self.video_a2v_cross_attn_scale_shift_table[4:, :]
            a2v_gate = (
                video_per_layer_ca_gate[None, None, :, :].to(
                    dtype=temb_ca_gate.dtype, device=temb_ca_gate.device
                )
                + temb_ca_gate.reshape(batch_size, temb_ca_gate.shape[1], 1, -1)
            ).squeeze(2)

            audio_per_layer_ca_gate = self.audio_a2v_cross_attn_scale_shift_table[4:, :]
            v2a_gate = (
                audio_per_layer_ca_gate[None, None, :, :].to(
                    dtype=temb_ca_audio_gate.dtype,
                    device=temb_ca_audio_gate.device,
                )
                + temb_ca_audio_gate.reshape(
                    batch_size, temb_ca_audio_gate.shape[1], 1, -1
                )
            ).squeeze(2)

            if video_dual_mod is None:
                video_per_layer_ca_scale_shift = (
                    self.video_a2v_cross_attn_scale_shift_table[:4, :]
                )
                video_ca_scale_shift_table = (
                    video_per_layer_ca_scale_shift[None, None, :, :].to(
                        dtype=temb_ca_scale_shift.dtype,
                        device=temb_ca_scale_shift.device,
                    )
                    + temb_ca_scale_shift.reshape(
                        batch_size, temb_ca_scale_shift.shape[1], 4, -1
                    )
                ).unbind(dim=2)

                (
                    video_a2v_ca_scale,
                    video_a2v_ca_shift,
                    video_v2a_ca_scale,
                    video_v2a_ca_shift,
                ) = [t.squeeze(2) for t in video_ca_scale_shift_table]

            if audio_dual_mod is None:
                audio_per_layer_ca_scale_shift = (
                    self.audio_a2v_cross_attn_scale_shift_table[:4, :]
                )
                audio_ca_scale_shift_table = (
                    audio_per_layer_ca_scale_shift[None, None, :, :].to(
                        dtype=temb_ca_audio_scale_shift.dtype,
                        device=temb_ca_audio_scale_shift.device,
                    )
                    + temb_ca_audio_scale_shift.reshape(
                        batch_size, temb_ca_audio_scale_shift.shape[1], 4, -1
                    )
                ).unbind(dim=2)

                (
                    audio_a2v_ca_scale,
                    audio_a2v_ca_shift,
                    audio_v2a_ca_scale,
                    audio_v2a_ca_shift,
                ) = [t.squeeze(2) for t in audio_ca_scale_shift_table]

        if video_dual_mod is None:
            video_dual_mod = _ltx2_try_fused_rmsnorm_dual_modulate(
                hidden_states,
                video_a2v_ca_scale,
                video_a2v_ca_shift,
                video_v2a_ca_scale,
                video_v2a_ca_shift,
                self.norm_eps,
            )
        if video_dual_mod is None:
            norm_hidden_states = rms_norm(hidden_states, self.norm_eps)
            mod_norm_hidden_states_a2v = _ltx2_modulate(
                norm_hidden_states, video_a2v_ca_scale, video_a2v_ca_shift
            )
            mod_norm_hidden_states_v2a = _ltx2_modulate(
                norm_hidden_states, video_v2a_ca_scale, video_v2a_ca_shift
            )
        else:
            mod_norm_hidden_states_a2v, mod_norm_hidden_states_v2a = video_dual_mod

        if audio_dual_mod is None:
            audio_dual_mod = _ltx2_try_fused_rmsnorm_dual_modulate(
                audio_hidden_states,
                audio_a2v_ca_scale,
                audio_a2v_ca_shift,
                audio_v2a_ca_scale,
                audio_v2a_ca_shift,
                self.norm_eps,
            )
        if audio_dual_mod is None:
            norm_audio_hidden_states = rms_norm(audio_hidden_states, self.norm_eps)
            mod_norm_audio_hidden_states_a2v = _ltx2_modulate(
                norm_audio_hidden_states, audio_a2v_ca_scale, audio_a2v_ca_shift
            )
            mod_norm_audio_hidden_states_v2a = _ltx2_modulate(
                norm_audio_hidden_states, audio_v2a_ca_scale, audio_v2a_ca_shift
            )
        else:
            (
                mod_norm_audio_hidden_states_a2v,
                mod_norm_audio_hidden_states_v2a,
            ) = audio_dual_mod

        # A2V
        if not skip_a2v_cross_attn:
            a2v_attn_hidden_states = self.audio_to_video_attn(
                mod_norm_hidden_states_a2v,
                context=mod_norm_audio_hidden_states_a2v,
                pe=ca_video_rotary_emb,
                k_pe=ca_audio_rotary_emb,
                mask=a2v_cross_attention_mask,
                skip_sequence_parallel_override=(
                    disable_sequence_parallel_for_replicated_sp
                    or audio_latents_replicated_for_sp
                ),
                gather_query_for_sp=(
                    audio_latents_replicated_for_sp
                    and not disable_sequence_parallel_for_replicated_sp
                ),
            )
            if a2v_cross_attn_perturbation_mask is not None:
                a2v_attn_hidden_states = (
                    a2v_attn_hidden_states * a2v_cross_attn_perturbation_mask
                )
        else:
            a2v_attn_hidden_states = None

        # V2A
        if not skip_v2a_cross_attn:
            v2a_attn_hidden_states = self.video_to_audio_attn(
                mod_norm_audio_hidden_states_v2a,
                context=mod_norm_hidden_states_v2a,
                pe=ca_audio_rotary_emb,
                k_pe=ca_video_rotary_emb,
                mask=v2a_cross_attention_mask,
                skip_sequence_parallel_override=disable_sequence_parallel_for_replicated_sp,
                gather_context_kv_for_sp=(
                    audio_latents_replicated_for_sp
                    and not disable_sequence_parallel_for_replicated_sp
                ),
            )
            if v2a_cross_attn_perturbation_mask is not None:
                v2a_attn_hidden_states = (
                    v2a_attn_hidden_states * v2a_cross_attn_perturbation_mask
                )
        else:
            v2a_attn_hidden_states = None
        # 4. Feedforward
        if video_ada_gates is not None:
            vshift_mlp = vscale_mlp = None
            vgate_mlp = video_ada_gates[1]
        elif video_ada_values is None:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, batch_size, temb, slice(3, 6)
            )
        else:
            vshift_mlp, vscale_mlp, vgate_mlp = video_ada_values[3:6]
        if a2v_attn_hidden_states is None:
            norm_hidden_states = None
            if video_ada_gates is not None:
                norm_hidden_states = _ltx2_try_fused_norm_ada_scale_shift(
                    hidden_states,
                    self.scale_shift_table,
                    temb,
                    3,
                    4,
                    self.norm_eps,
                )
            if norm_hidden_states is None:
                if vscale_mlp is None or vshift_mlp is None:
                    vshift_mlp, vscale_mlp, _ = self.get_ada_values(
                        self.scale_shift_table, batch_size, temb, slice(3, 6)
                    )
                norm_hidden_states = _ltx2_norm_scale_shift(
                    hidden_states, vscale_mlp, vshift_mlp, self.norm_eps
                )
        else:
            video_mlp_residual_norm = None
            if video_ada_gates is not None:
                video_mlp_residual_norm = _ltx2_try_fused_residual_norm_ada_scale_shift(
                    hidden_states,
                    a2v_attn_hidden_states,
                    a2v_gate,
                    self.scale_shift_table,
                    temb,
                    3,
                    4,
                    self.norm_eps,
                )
            if video_mlp_residual_norm is None:
                if vscale_mlp is None or vshift_mlp is None:
                    vshift_mlp, vscale_mlp, _ = self.get_ada_values(
                        self.scale_shift_table, batch_size, temb, slice(3, 6)
                    )
                norm_hidden_states, hidden_states = _ltx2_residual_norm_scale_shift(
                    hidden_states,
                    a2v_attn_hidden_states,
                    a2v_gate,
                    vscale_mlp,
                    vshift_mlp,
                    self.norm_eps,
                )
            else:
                norm_hidden_states, hidden_states = video_mlp_residual_norm
        fused_ff_residual = self.ff.try_forward_with_residual_gate(
            norm_hidden_states, hidden_states, vgate_mlp
        )
        if fused_ff_residual is None:
            ff_output = self.ff(norm_hidden_states)
            hidden_states = _ltx2_residual_gate_add(hidden_states, ff_output, vgate_mlp)
        else:
            hidden_states = fused_ff_residual

        if audio_ada_values is None:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, batch_size, temb_audio, slice(3, 6)
            )
        else:
            ashift_mlp, ascale_mlp, agate_mlp = audio_ada_values[3:6]
        if v2a_attn_hidden_states is None:
            norm_audio_hidden_states = _ltx2_norm_scale_shift(
                audio_hidden_states, ascale_mlp, ashift_mlp, self.norm_eps
            )
        else:
            norm_audio_hidden_states, audio_hidden_states = (
                _ltx2_residual_norm_scale_shift(
                    audio_hidden_states,
                    v2a_attn_hidden_states,
                    v2a_gate,
                    ascale_mlp,
                    ashift_mlp,
                    self.norm_eps,
                )
            )
        audio_ff_output = self.audio_ff(norm_audio_hidden_states)
        audio_hidden_states = _ltx2_residual_gate_add(
            audio_hidden_states, audio_ff_output, agate_mlp
        )
        return hidden_states, audio_hidden_states


class LTX2VideoTransformer3DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    _fsdp_shard_conditions = LTX2ArchConfig()._fsdp_shard_conditions
    _compile_conditions = LTX2ArchConfig()._compile_conditions
    _supported_attention_backends = LTX2ArchConfig()._supported_attention_backends
    param_names_mapping = LTX2ArchConfig().param_names_mapping
    reverse_param_names_mapping = LTX2ArchConfig().reverse_param_names_mapping
    lora_param_names_mapping = LTX2ArchConfig().lora_param_names_mapping

    @staticmethod
    def _collapse_prompt_timestep(timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim <= 1:
            return timestep
        return timestep.amax(dim=tuple(range(1, timestep.ndim)))

    def _scale_timestep_for_adaln(self, timestep: torch.Tensor) -> torch.Tensor:
        ltx_variant = str(getattr(self.config.arch_config, "ltx_variant", "ltx_2"))
        if ltx_variant == "ltx_2_3" and bool(
            getattr(self, "_sglang_use_ltx23_hq_timestep_semantics", False)
        ):
            return timestep * float(self.timestep_scale_multiplier)
        return timestep

    @staticmethod
    def _ltx2_rope_cache_tensor_signature(
        tensor: torch.Tensor | None,
    ) -> tuple[object, ...] | None:
        if tensor is None:
            return None
        # tensor._version raises RuntimeError (not AttributeError, so getattr's
        # default won't catch it) on inference-mode tensors -- NVFP4/TE outputs
        # or token-pruned/sliced coords from the efficiency-framework midpoint
        # prune. Fall back to 0: under inference mode the tensor is not mutated
        # in place, so a stale cache key is safe.
        try:
            version = tensor._version
        except RuntimeError:
            version = 0
        return (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
            version,
        )

    def _ltx2_rope_cache_key(
        self,
        *,
        video_coords: torch.Tensor,
        audio_coords: torch.Tensor,
        generated_video_coords: bool,
        generated_audio_coords: bool,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        fps: float,
        audio_num_frames: int,
        hidden_device: torch.device,
        hidden_dtype: torch.dtype,
        audio_device: torch.device,
        audio_dtype: torch.dtype,
    ) -> tuple[object, ...]:
        video_sig = (
            None
            if generated_video_coords
            else self._ltx2_rope_cache_tensor_signature(video_coords)
        )
        audio_sig = (
            None
            if generated_audio_coords
            else self._ltx2_rope_cache_tensor_signature(audio_coords)
        )
        return (
            id(self.rope),
            id(self.audio_rope),
            id(self.cross_attn_rope),
            id(self.cross_attn_audio_rope),
            bool(generated_video_coords),
            bool(generated_audio_coords),
            int(batch_size),
            int(num_frames),
            int(height),
            int(width),
            float(fps),
            int(audio_num_frames),
            str(hidden_device),
            hidden_dtype,
            str(audio_device),
            audio_dtype,
            video_sig,
            audio_sig,
        )

    def _ltx2_get_rope_embeddings(
        self,
        *,
        video_coords: torch.Tensor,
        audio_coords: torch.Tensor,
        generated_video_coords: bool,
        generated_audio_coords: bool,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        fps: float,
        audio_num_frames: int,
        hidden_device: torch.device,
        hidden_dtype: torch.dtype,
        audio_device: torch.device,
        audio_dtype: torch.dtype,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        if _ltx2_cache_rope_emb_enabled():
            key = self._ltx2_rope_cache_key(
                video_coords=video_coords,
                audio_coords=audio_coords,
                generated_video_coords=generated_video_coords,
                generated_audio_coords=generated_audio_coords,
                batch_size=batch_size,
                num_frames=num_frames,
                height=height,
                width=width,
                fps=fps,
                audio_num_frames=audio_num_frames,
                hidden_device=hidden_device,
                hidden_dtype=hidden_dtype,
                audio_device=audio_device,
                audio_dtype=audio_dtype,
            )
            cached = self._ltx2_rope_emb_cache.get(key)
            if cached is not None:
                return cached

        video_rotary_emb = self.rope(
            video_coords,
            device=hidden_device,
            out_dtype=hidden_dtype,
        )
        audio_rotary_emb = self.audio_rope(
            audio_coords,
            device=audio_device,
            out_dtype=audio_dtype,
        )
        ca_video_rotary_emb = self.cross_attn_rope(
            video_coords[:, 0:1, :],
            device=hidden_device,
            out_dtype=hidden_dtype,
        )
        ca_audio_rotary_emb = self.cross_attn_audio_rope(
            audio_coords[:, 0:1, :],
            device=audio_device,
            out_dtype=audio_dtype,
        )
        value = (
            video_rotary_emb,
            audio_rotary_emb,
            ca_video_rotary_emb,
            ca_audio_rotary_emb,
        )
        if _ltx2_cache_rope_emb_enabled():
            if len(self._ltx2_rope_emb_cache) >= 4:
                self._ltx2_rope_emb_cache.clear()
            self._ltx2_rope_emb_cache[key] = value
        return value

    def _validate_tp_config(self, *, arch: LTX2ArchConfig, tp_size: int) -> None:
        """Validate TP-related dimension constraints (fail-fast)."""
        if tp_size < 1:
            raise ValueError(f"Invalid tp_size={tp_size}. Expected tp_size >= 1.")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "video hidden_size must be divisible by num_attention_heads, got "
                f"{self.hidden_size=} {self.num_attention_heads=}."
            )
        if self.audio_hidden_size % self.audio_num_attention_heads != 0:
            raise ValueError(
                "audio_hidden_size must be divisible by audio_num_attention_heads, got "
                f"{self.audio_hidden_size=} {self.audio_num_attention_heads=}."
            )

        if tp_size == 1:
            return

        if self.num_attention_heads % tp_size != 0:
            raise ValueError(
                "num_attention_heads must be divisible by tp_size, got "
                f"{self.num_attention_heads=} {tp_size=}."
            )
        if self.audio_num_attention_heads % tp_size != 0:
            raise ValueError(
                "audio_num_attention_heads must be divisible by tp_size, got "
                f"{self.audio_num_attention_heads=} {tp_size=}."
            )
        if self.hidden_size % tp_size != 0:
            raise ValueError(
                "hidden_size must be divisible by tp_size for TP-sharded projections, got "
                f"{self.hidden_size=} {tp_size=}."
            )
        if self.audio_hidden_size % tp_size != 0:
            raise ValueError(
                "audio_hidden_size must be divisible by tp_size for TP-sharded projections, got "
                f"{self.audio_hidden_size=} {tp_size=}."
            )
        if int(arch.out_channels) % tp_size != 0:
            raise ValueError(
                "out_channels must be divisible by tp_size for TP-sharded output projection, got "
                f"{arch.out_channels=} {tp_size=}."
            )
        if int(arch.audio_out_channels) % tp_size != 0:
            raise ValueError(
                "audio_out_channels must be divisible by tp_size for TP-sharded output projection, got "
                f"{arch.audio_out_channels=} {tp_size=}."
            )

    def __init__(
        self,
        config: LTX2Config,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)

        arch = config.arch_config
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.audio_hidden_size = arch.audio_hidden_size
        self.audio_num_attention_heads = arch.audio_num_attention_heads
        self.norm_eps = arch.norm_eps

        tp_size = get_tp_world_size()
        self._validate_tp_config(arch=arch, tp_size=tp_size)

        # 1. Patchification input projections
        # Matches LTX2Config().param_names_mapping
        self.patchify_proj = ColumnParallelLinear(
            arch.in_channels,
            self.hidden_size,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix="patchify_proj",
        )
        self.audio_patchify_proj = ColumnParallelLinear(
            arch.audio_in_channels,
            self.audio_hidden_size,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix="audio_patchify_proj",
        )

        # 2. Prompt embeddings
        self.caption_projection: LTX2TextProjection | None = None
        self.audio_caption_projection: LTX2TextProjection | None = None
        if not arch.caption_proj_before_connector:
            self.caption_projection = LTX2TextProjection(
                in_features=arch.caption_channels,
                hidden_size=self.hidden_size,
                quant_config=quant_config,
                prefix="caption_projection",
            )
            self.audio_caption_projection = LTX2TextProjection(
                in_features=arch.caption_channels,
                hidden_size=self.audio_hidden_size,
                quant_config=quant_config,
                prefix="audio_caption_projection",
            )

        # 3. Timestep Modulation Params and Embedding
        self.adaln_single = LTX2AdaLayerNormSingle(
            self.hidden_size,
            embedding_coefficient=adaln_embedding_coefficient(
                arch.cross_attention_adaln
            ),
            quant_config=quant_config,
            prefix="adaln_single",
        )
        self.audio_adaln_single = LTX2AdaLayerNormSingle(
            self.audio_hidden_size,
            embedding_coefficient=adaln_embedding_coefficient(
                arch.cross_attention_adaln
            ),
            quant_config=quant_config,
            prefix="audio_adaln_single",
        )
        self.prompt_adaln_single: LTX2AdaLayerNormSingle | None = None
        self.audio_prompt_adaln_single: LTX2AdaLayerNormSingle | None = None
        if arch.cross_attention_adaln:
            self.prompt_adaln_single = LTX2AdaLayerNormSingle(
                self.hidden_size,
                embedding_coefficient=2,
                quant_config=quant_config,
                prefix="prompt_adaln_single",
            )
            self.audio_prompt_adaln_single = LTX2AdaLayerNormSingle(
                self.audio_hidden_size,
                embedding_coefficient=2,
                quant_config=quant_config,
                prefix="audio_prompt_adaln_single",
            )

        # Global Cross Attention Modulation Parameters
        self.av_ca_video_scale_shift_adaln_single = LTX2AdaLayerNormSingle(
            self.hidden_size,
            embedding_coefficient=4,
            quant_config=quant_config,
            prefix="av_ca_video_scale_shift_adaln_single",
        )
        self.av_ca_a2v_gate_adaln_single = LTX2AdaLayerNormSingle(
            self.hidden_size,
            embedding_coefficient=1,
            quant_config=quant_config,
            prefix="av_ca_a2v_gate_adaln_single",
        )
        self.av_ca_audio_scale_shift_adaln_single = LTX2AdaLayerNormSingle(
            self.audio_hidden_size,
            embedding_coefficient=4,
            quant_config=quant_config,
            prefix="av_ca_audio_scale_shift_adaln_single",
        )
        self.av_ca_v2a_gate_adaln_single = LTX2AdaLayerNormSingle(
            self.audio_hidden_size,
            embedding_coefficient=1,
            quant_config=quant_config,
            prefix="av_ca_v2a_gate_adaln_single",
        )

        # Output Layer Scale/Shift Modulation parameters
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, self.hidden_size) / self.hidden_size**0.5
        )
        self.audio_scale_shift_table = nn.Parameter(
            torch.randn(2, self.audio_hidden_size) / self.audio_hidden_size**0.5
        )

        hf_patch_size = int(hf_config.get("patch_size", 1))
        hf_patch_size_t = int(hf_config.get("patch_size_t", 1))
        self.patch_size = (hf_patch_size_t, hf_patch_size, hf_patch_size)

        hf_audio_patch_size = int(hf_config.get("audio_patch_size", 1))
        hf_audio_patch_size_t = int(hf_config.get("audio_patch_size_t", 1))

        rope_type = (
            arch.rope_type.value
            if hasattr(arch.rope_type, "value")
            else str(arch.rope_type)
        )
        frequencies_precision = hf_config.get("frequencies_precision")
        if frequencies_precision is None:
            frequencies_precision = getattr(arch, "frequencies_precision", None)

        # diffusers/LTX configs use `frequencies_precision` for this RoPE switch
        rope_double_precision = (
            str(frequencies_precision) == "float64"
            if frequencies_precision is not None
            else bool(
                hf_config.get("rope_double_precision", arch.double_precision_rope)
            )
        )
        self.quantize_video_rope_coords_to_hidden_dtype = bool(
            hf_config.get("quantize_video_rope_coords_to_hidden_dtype", False)
        )
        causal_offset = int(hf_config.get("causal_offset", 1))

        pos_embed_max_pos = int(arch.positional_embedding_max_pos[0])
        base_height = int(arch.positional_embedding_max_pos[1])
        base_width = int(arch.positional_embedding_max_pos[2])

        audio_pos_embed_max_pos = int(arch.audio_positional_embedding_max_pos[0])

        self.video_scale_factors = (8, 32, 32)
        self.audio_scale_factors = (4,)

        self.rope = LTX2AudioVideoRotaryPosEmbed(
            dim=self.hidden_size,
            patch_size=hf_patch_size,
            patch_size_t=hf_patch_size_t,
            base_num_frames=pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            scale_factors=self.video_scale_factors,
            theta=float(arch.positional_embedding_theta),
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=self.num_attention_heads,
        )
        self.audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=self.audio_hidden_size,
            patch_size=hf_audio_patch_size,
            patch_size_t=hf_audio_patch_size_t,
            base_num_frames=audio_pos_embed_max_pos,
            sampling_rate=16000,
            hop_length=160,
            scale_factors=self.audio_scale_factors,
            theta=float(arch.positional_embedding_theta),
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=self.audio_num_attention_heads,
        )

        cross_attn_pos_embed_max_pos = max(pos_embed_max_pos, audio_pos_embed_max_pos)
        self.cross_attn_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=int(arch.audio_cross_attention_dim),
            patch_size=hf_patch_size,
            patch_size_t=hf_patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            theta=float(arch.positional_embedding_theta),
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=self.num_attention_heads,
        )
        self.cross_attn_audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=int(arch.audio_cross_attention_dim),
            patch_size=hf_audio_patch_size,
            patch_size_t=hf_audio_patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            sampling_rate=16000,
            hop_length=160,
            scale_factors=self.audio_scale_factors,
            theta=float(arch.positional_embedding_theta),
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=self.audio_num_attention_heads,
        )

        self.cross_pe_max_pos = cross_attn_pos_embed_max_pos
        self._ltx2_rope_emb_cache: dict[
            tuple[object, ...],
            tuple[
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
            ],
        ] = {}

        # 5. Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                LTX2TransformerBlock(
                    idx=idx,
                    dim=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.hidden_size // self.num_attention_heads,
                    cross_attention_dim=arch.cross_attention_dim,
                    audio_dim=self.audio_hidden_size,
                    audio_num_attention_heads=self.audio_num_attention_heads,
                    audio_attention_head_dim=self.audio_hidden_size
                    // self.audio_num_attention_heads,
                    audio_cross_attention_dim=arch.audio_cross_attention_dim,
                    norm_eps=self.norm_eps,
                    qk_norm=True,  # Always True in LTX2
                    apply_gated_attention=arch.apply_gated_attention,
                    cross_attention_adaln=arch.cross_attention_adaln,
                    use_local_av_cross_attention=bool(
                        getattr(arch, "use_local_av_cross_attention", False)
                    ),
                    force_sdpa_v2a_cross_attention=bool(
                        getattr(arch, "force_sdpa_v2a_cross_attention", False)
                    ),
                    supported_attention_backends=self._supported_attention_backends,
                    prefix=f"transformer_blocks.{idx}",
                    quant_config=quant_config,
                )
                for idx in range(arch.num_layers)
            ]
        )

        # 6. Output layers
        self.norm_out = nn.LayerNorm(
            self.hidden_size, eps=self.norm_eps, elementwise_affine=False
        )
        self.proj_out = ColumnParallelLinear(
            self.hidden_size,
            arch.out_channels,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix="proj_out",
        )

        self.audio_norm_out = nn.LayerNorm(
            self.audio_hidden_size, eps=self.norm_eps, elementwise_affine=False
        )
        self.audio_proj_out = ColumnParallelLinear(
            self.audio_hidden_size,
            arch.audio_out_channels,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix="audio_proj_out",
        )

        self.out_channels_raw = arch.out_channels // (
            self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        )
        self.audio_out_channels = arch.audio_out_channels
        self.timestep_scale_multiplier = arch.timestep_scale_multiplier
        self.av_ca_timestep_scale_multiplier = arch.av_ca_timestep_scale_multiplier

        self.layer_names = ["transformer_blocks"]

    def _maybe_quantize_video_rope_coords(
        self,
        video_coords: torch.Tensor,
        hidden_device: torch.device,
        hidden_dtype: torch.dtype,
    ) -> torch.Tensor:
        ltx_variant = str(getattr(self.config.arch_config, "ltx_variant", "ltx_2"))
        if self.quantize_video_rope_coords_to_hidden_dtype and not (
            ltx_variant == "ltx_2_3"
            and bool(getattr(self, "_sglang_use_ltx23_hq_timestep_semantics", False))
        ):
            return video_coords.to(device=hidden_device, dtype=hidden_dtype)
        return video_coords.to(device=hidden_device)

    def _get_av_ca_gate_timestep_factor(self) -> float:
        ltx_variant = str(getattr(self.config.arch_config, "ltx_variant", "ltx_2"))
        if ltx_variant == "ltx_2_3":
            return self.av_ca_timestep_scale_multiplier / self.timestep_scale_multiplier
        return float(self.av_ca_timestep_scale_multiplier)

    def _get_av_ca_timesteps(
        self,
        timestep: torch.Tensor,
        audio_timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        audio_prompt_timestep: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ltx_variant = str(getattr(self.config.arch_config, "ltx_variant", "ltx_2"))
        if ltx_variant != "ltx_2_3":
            return timestep, audio_timestep, timestep, audio_timestep

        video_scale_shift_timestep = timestep
        audio_scale_shift_timestep = audio_timestep
        a2v_gate_timestep = (
            self._collapse_prompt_timestep(audio_timestep)
            if audio_prompt_timestep is None
            else audio_prompt_timestep
        )
        v2a_gate_timestep = (
            self._collapse_prompt_timestep(timestep)
            if prompt_timestep is None
            else prompt_timestep
        )
        return (
            video_scale_shift_timestep,
            audio_scale_shift_timestep,
            a2v_gate_timestep,
            v2a_gate_timestep,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        audio_timestep: Optional[torch.LongTensor] = None,
        prompt_timestep: Optional[torch.Tensor] = None,
        audio_prompt_timestep: Optional[torch.Tensor] = None,
        encoder_hidden_states_projected: bool = False,
        audio_encoder_hidden_states_projected: bool = False,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        audio_encoder_attention_mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: float = 24.0,
        audio_num_frames: Optional[int] = None,
        video_coords: Optional[torch.Tensor] = None,
        audio_coords: Optional[torch.Tensor] = None,
        video_self_attention_mask: Optional[torch.Tensor] = None,
        audio_self_attention_mask: Optional[torch.Tensor] = None,
        a2v_cross_attention_mask: Optional[torch.Tensor] = None,
        v2a_cross_attention_mask: Optional[torch.Tensor] = None,
        skip_video_self_attn_blocks: Optional[tuple[int, ...]] = None,
        skip_audio_self_attn_blocks: Optional[tuple[int, ...]] = None,
        disable_a2v_cross_attn: bool = False,
        disable_v2a_cross_attn: bool = False,
        audio_replicated_for_sp: bool = False,
        audio_latents_replicated_for_sp: bool = False,
        disable_sequence_parallel_for_replicated_sp: bool = False,
        share_block0_self_attn: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:

        batch_size = hidden_states.size(0)
        audio_latents_replicated_for_sp = (
            audio_replicated_for_sp or audio_latents_replicated_for_sp
        )
        audio_timestep = audio_timestep if audio_timestep is not None else timestep

        if num_frames is None or height is None or width is None:
            raise ValueError(
                "num_frames/height/width must be provided for RoPE coordinate generation."
            )
        if audio_num_frames is None:
            raise ValueError(
                "audio_num_frames must be provided for RoPE coordinate generation."
            )
        perturbation_configs = kwargs.get("perturbation_configs")
        if perturbation_configs is not None and len(perturbation_configs) != batch_size:
            raise ValueError(
                "perturbation_configs length must match batch size, got "
                f"{len(perturbation_configs)=} {batch_size=}."
            )

        generated_video_coords = video_coords is None
        generated_audio_coords = audio_coords is None
        if video_coords is None:
            # Wan-style SP-RoPE: when SP is enabled, each rank runs on its local
            # time shard but RoPE positions must be offset to global time.
            #
            # We assume equal time sharding across SP ranks.
            if model_parallel_is_initialized():
                sp_world_size = get_sp_world_size()
                sp_rank = get_sp_parallel_rank()
            else:
                sp_world_size = 1
                sp_rank = 0

            video_shift = (
                int(sp_rank) * int(num_frames)
                if sp_world_size > 1 and not disable_sequence_parallel_for_replicated_sp
                else 0
            )
            video_coords = self.rope.prepare_video_coords(
                batch_size=batch_size,
                num_frames=num_frames,
                height=height,
                width=width,
                device=hidden_states.device,
                fps=fps,
                start_frame=video_shift,
            )
        if audio_coords is None:
            audio_coords = self.audio_rope.prepare_audio_coords(
                batch_size=batch_size,
                num_frames=audio_num_frames,
                device=audio_hidden_states.device,
            )

        video_coords = self._maybe_quantize_video_rope_coords(
            video_coords, hidden_states.device, hidden_states.dtype
        )
        audio_coords = audio_coords.to(device=audio_hidden_states.device)
        (
            video_rotary_emb,
            audio_rotary_emb,
            ca_video_rotary_emb,
            ca_audio_rotary_emb,
        ) = self._ltx2_get_rope_embeddings(
            video_coords=video_coords,
            audio_coords=audio_coords,
            generated_video_coords=generated_video_coords,
            generated_audio_coords=generated_audio_coords,
            batch_size=batch_size,
            num_frames=int(num_frames),
            height=int(height),
            width=int(width),
            fps=float(fps),
            audio_num_frames=int(audio_num_frames),
            hidden_device=hidden_states.device,
            hidden_dtype=hidden_states.dtype,
            audio_device=audio_hidden_states.device,
            audio_dtype=audio_hidden_states.dtype,
        )

        # 2. Patchify input projections
        hidden_states, _ = self.patchify_proj(hidden_states)
        audio_hidden_states, _ = self.audio_patchify_proj(audio_hidden_states)
        # 3. Prepare timestep embeddings
        # 3.1. Prepare global modality (video and audio) timestep embedding and modulation parameters
        timestep_for_adaln = self._scale_timestep_for_adaln(timestep)
        audio_timestep_for_adaln = self._scale_timestep_for_adaln(audio_timestep)
        temb, embedded_timestep = self.adaln_single(
            timestep_for_adaln.flatten(),
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.size(-1)
        )

        temb_audio, audio_embedded_timestep = self.audio_adaln_single(
            audio_timestep_for_adaln.flatten(),
            hidden_dtype=audio_hidden_states.dtype,
        )
        temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))
        audio_embedded_timestep = audio_embedded_timestep.view(
            batch_size, -1, audio_embedded_timestep.size(-1)
        )
        temb_prompt = None
        temb_audio_prompt = None
        if self.prompt_adaln_single is not None:
            prompt_timestep = (
                self._collapse_prompt_timestep(timestep)
                if prompt_timestep is None
                else prompt_timestep
            )
            prompt_timestep_for_adaln = self._scale_timestep_for_adaln(prompt_timestep)
            temb_prompt, _ = self.prompt_adaln_single(
                prompt_timestep_for_adaln.flatten(), hidden_dtype=hidden_states.dtype
            )
            temb_prompt = temb_prompt.view(batch_size, -1, temb_prompt.size(-1))
        if self.audio_prompt_adaln_single is not None:
            audio_prompt_timestep = (
                self._collapse_prompt_timestep(audio_timestep)
                if audio_prompt_timestep is None
                else audio_prompt_timestep
            )
            audio_prompt_timestep_for_adaln = self._scale_timestep_for_adaln(
                audio_prompt_timestep
            )
            temb_audio_prompt, _ = self.audio_prompt_adaln_single(
                audio_prompt_timestep_for_adaln.flatten(),
                hidden_dtype=audio_hidden_states.dtype,
            )
            temb_audio_prompt = temb_audio_prompt.view(
                batch_size, -1, temb_audio_prompt.size(-1)
            )

        # 3.2. Prepare global modality cross attention modulation parameters
        hidden_dtype = hidden_states.dtype
        (
            av_ca_video_scale_shift_timestep,
            av_ca_audio_scale_shift_timestep,
            av_ca_a2v_gate_timestep,
            av_ca_v2a_gate_timestep,
        ) = self._get_av_ca_timesteps(
            timestep,
            audio_timestep,
            prompt_timestep,
            audio_prompt_timestep,
        )
        av_ca_video_timestep_for_adaln = self._scale_timestep_for_adaln(
            av_ca_video_scale_shift_timestep
        )
        av_ca_audio_timestep_for_adaln = self._scale_timestep_for_adaln(
            av_ca_audio_scale_shift_timestep
        )
        temb_ca_scale_shift, _ = self.av_ca_video_scale_shift_adaln_single(
            av_ca_video_timestep_for_adaln.flatten(), hidden_dtype=hidden_dtype
        )
        temb_ca_scale_shift = temb_ca_scale_shift.view(
            batch_size, -1, temb_ca_scale_shift.shape[-1]
        )

        av_ca_gate_factor = self._get_av_ca_gate_timestep_factor()
        av_ca_a2v_gate_timestep_for_adaln = self._scale_timestep_for_adaln(
            av_ca_a2v_gate_timestep
        )
        temb_ca_gate, _ = self.av_ca_a2v_gate_adaln_single(
            av_ca_a2v_gate_timestep_for_adaln.flatten() * av_ca_gate_factor,
            hidden_dtype=hidden_dtype,
        )
        temb_ca_gate = temb_ca_gate.view(batch_size, -1, temb_ca_gate.shape[-1])

        temb_ca_audio_scale_shift, _ = self.av_ca_audio_scale_shift_adaln_single(
            av_ca_audio_timestep_for_adaln.flatten(),
            hidden_dtype=audio_hidden_states.dtype,
        )
        temb_ca_audio_scale_shift = temb_ca_audio_scale_shift.view(
            batch_size, -1, temb_ca_audio_scale_shift.shape[-1]
        )

        av_ca_v2a_gate_timestep_for_adaln = self._scale_timestep_for_adaln(
            av_ca_v2a_gate_timestep
        )
        temb_ca_audio_gate, _ = self.av_ca_v2a_gate_adaln_single(
            av_ca_v2a_gate_timestep_for_adaln.flatten() * av_ca_gate_factor,
            hidden_dtype=audio_hidden_states.dtype,
        )
        temb_ca_audio_gate = temb_ca_audio_gate.view(
            batch_size, -1, temb_ca_audio_gate.shape[-1]
        )

        # 4. Prepare prompt embeddings
        if self.caption_projection is not None and not encoder_hidden_states_projected:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        if (
            self.audio_caption_projection is not None
            and not audio_encoder_hidden_states_projected
        ):
            audio_encoder_hidden_states = self.audio_caption_projection(
                audio_encoder_hidden_states
            )
        # 5. Run blocks
        skip_video_self_attn_blocks = set(skip_video_self_attn_blocks or ())
        skip_audio_self_attn_blocks = set(skip_audio_self_attn_blocks or ())
        block_indices = tuple(
            getattr(block, "idx", -1) for block in self.transformer_blocks
        )
        full_perturbation_states = _ltx2_build_perturbation_state_maps(
            perturbation_configs, block_indices, hidden_states, audio_hidden_states
        )
        (
            video_self_attn_perturbation_states,
            audio_self_attn_perturbation_states,
            a2v_cross_attn_perturbation_states,
            v2a_cross_attn_perturbation_states,
        ) = full_perturbation_states

        prefix_share_plan = _ltx2_guidance_prefix_share_plan(
            perturbation_configs, block_indices, batch_size
        )
        prefix_share_expanded = False
        if prefix_share_plan is not None:
            (
                prefix_first_skip_block,
                prefix_keep_indices,
                prefix_expand_indices,
            ) = prefix_share_plan
            full_block_inputs = (
                encoder_hidden_states,
                audio_encoder_hidden_states,
                temb,
                temb_audio,
                temb_prompt,
                temb_audio_prompt,
                temb_ca_scale_shift,
                temb_ca_audio_scale_shift,
                temb_ca_gate,
                temb_ca_audio_gate,
                video_rotary_emb,
                audio_rotary_emb,
                ca_video_rotary_emb,
                ca_audio_rotary_emb,
                encoder_attention_mask,
                audio_encoder_attention_mask,
                video_self_attention_mask,
                audio_self_attention_mask,
                a2v_cross_attention_mask,
                v2a_cross_attention_mask,
            )
            assert perturbation_configs is not None
            prefix_perturbation_configs = tuple(
                perturbation_configs[index] for index in prefix_keep_indices
            )
            prefix_perturbation_states = _ltx2_build_perturbation_state_maps(
                prefix_perturbation_configs,
                block_indices,
                hidden_states,
                audio_hidden_states,
            )
            hidden_states = _ltx2_index_batch_dim(
                hidden_states, prefix_keep_indices, batch_size
            )
            audio_hidden_states = _ltx2_index_batch_dim(
                audio_hidden_states, prefix_keep_indices, batch_size
            )
            encoder_hidden_states = _ltx2_index_batch_dim(
                encoder_hidden_states, prefix_keep_indices, batch_size
            )
            audio_encoder_hidden_states = _ltx2_index_batch_dim(
                audio_encoder_hidden_states, prefix_keep_indices, batch_size
            )
            temb = _ltx2_index_batch_dim(temb, prefix_keep_indices, batch_size)
            temb_audio = _ltx2_index_batch_dim(
                temb_audio, prefix_keep_indices, batch_size
            )
            temb_prompt = _ltx2_index_batch_dim(
                temb_prompt, prefix_keep_indices, batch_size
            )
            temb_audio_prompt = _ltx2_index_batch_dim(
                temb_audio_prompt, prefix_keep_indices, batch_size
            )
            temb_ca_scale_shift = _ltx2_index_batch_dim(
                temb_ca_scale_shift, prefix_keep_indices, batch_size
            )
            temb_ca_audio_scale_shift = _ltx2_index_batch_dim(
                temb_ca_audio_scale_shift, prefix_keep_indices, batch_size
            )
            temb_ca_gate = _ltx2_index_batch_dim(
                temb_ca_gate, prefix_keep_indices, batch_size
            )
            temb_ca_audio_gate = _ltx2_index_batch_dim(
                temb_ca_audio_gate, prefix_keep_indices, batch_size
            )
            video_rotary_emb = _ltx2_index_rotary_emb(
                video_rotary_emb, prefix_keep_indices, batch_size
            )
            audio_rotary_emb = _ltx2_index_rotary_emb(
                audio_rotary_emb, prefix_keep_indices, batch_size
            )
            ca_video_rotary_emb = _ltx2_index_rotary_emb(
                ca_video_rotary_emb, prefix_keep_indices, batch_size
            )
            ca_audio_rotary_emb = _ltx2_index_rotary_emb(
                ca_audio_rotary_emb, prefix_keep_indices, batch_size
            )
            encoder_attention_mask = _ltx2_index_batch_dim(
                encoder_attention_mask, prefix_keep_indices, batch_size
            )
            audio_encoder_attention_mask = _ltx2_index_batch_dim(
                audio_encoder_attention_mask, prefix_keep_indices, batch_size
            )
            video_self_attention_mask = _ltx2_index_batch_dim(
                video_self_attention_mask, prefix_keep_indices, batch_size
            )
            audio_self_attention_mask = _ltx2_index_batch_dim(
                audio_self_attention_mask, prefix_keep_indices, batch_size
            )
            a2v_cross_attention_mask = _ltx2_index_batch_dim(
                a2v_cross_attention_mask, prefix_keep_indices, batch_size
            )
            v2a_cross_attention_mask = _ltx2_index_batch_dim(
                v2a_cross_attention_mask, prefix_keep_indices, batch_size
            )
            (
                video_self_attn_perturbation_states,
                audio_self_attn_perturbation_states,
                a2v_cross_attn_perturbation_states,
                v2a_cross_attn_perturbation_states,
            ) = prefix_perturbation_states
        else:
            prefix_first_skip_block = -1
            prefix_keep_indices = ()
            prefix_expand_indices = ()
            full_block_inputs = None

        teacache_coordinator = None
        teacache_decision = None
        original_hidden_states_for_teacache = None
        original_audio_hidden_states_for_teacache = None
        try:
            from sglang.multimodal_gen.runtime.cache.ltx2_teacache import (
                get_ltx2_teacache_coordinator,
            )

            teacache_coordinator = get_ltx2_teacache_coordinator(self)
        except Exception as exc:
            logger.warning("Failed to initialize LTX2 TeaCache hook: %s", exc)
            teacache_coordinator = None
        if teacache_coordinator is not None:
            teacache_decision = teacache_coordinator.lookup(
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                temb=temb,
                temb_audio=temb_audio,
                skip_video_self_attn_blocks=skip_video_self_attn_blocks,
                skip_audio_self_attn_blocks=skip_audio_self_attn_blocks,
                disable_a2v_cross_attn=disable_a2v_cross_attn,
                disable_v2a_cross_attn=disable_v2a_cross_attn,
                perturbation_configs=perturbation_configs,
                pass_id=str(getattr(self, "_sglang_ltx2_pass_id", "default")),
            )
            if teacache_decision.should_skip:
                assert teacache_decision.hidden_states is not None
                assert teacache_decision.audio_hidden_states is not None
                hidden_states = teacache_decision.hidden_states
                audio_hidden_states = teacache_decision.audio_hidden_states
            else:
                original_hidden_states_for_teacache = hidden_states
                original_audio_hidden_states_for_teacache = audio_hidden_states

        profile_token = _ltx2_push_profile_context(
            getattr(self, "_sgl_ltx2_profile_phase", "unknown"),
            getattr(self, "_sgl_ltx2_profile_step", "unknown"),
        )
        try:
            if teacache_decision is not None and teacache_decision.should_skip:
                pass
            else:
                for block in self.transformer_blocks:
                    block_idx = getattr(block, "idx", -1)
                    if (
                        prefix_share_plan is not None
                        and not prefix_share_expanded
                        and block_idx >= prefix_first_skip_block
                    ):
                        hidden_states = _ltx2_index_batch_dim(
                            hidden_states,
                            prefix_expand_indices,
                            len(prefix_keep_indices),
                        )
                        audio_hidden_states = _ltx2_index_batch_dim(
                            audio_hidden_states,
                            prefix_expand_indices,
                            len(prefix_keep_indices),
                        )
                        assert full_block_inputs is not None
                        (
                            encoder_hidden_states,
                            audio_encoder_hidden_states,
                            temb,
                            temb_audio,
                            temb_prompt,
                            temb_audio_prompt,
                            temb_ca_scale_shift,
                            temb_ca_audio_scale_shift,
                            temb_ca_gate,
                            temb_ca_audio_gate,
                            video_rotary_emb,
                            audio_rotary_emb,
                            ca_video_rotary_emb,
                            ca_audio_rotary_emb,
                            encoder_attention_mask,
                            audio_encoder_attention_mask,
                            video_self_attention_mask,
                            audio_self_attention_mask,
                            a2v_cross_attention_mask,
                            v2a_cross_attention_mask,
                        ) = full_block_inputs
                        (
                            video_self_attn_perturbation_states,
                            audio_self_attn_perturbation_states,
                            a2v_cross_attn_perturbation_states,
                            v2a_cross_attn_perturbation_states,
                        ) = full_perturbation_states
                        prefix_share_expanded = True

                    video_self_attn_perturbation_mask = None
                    audio_self_attn_perturbation_mask = None
                    a2v_cross_attn_perturbation_mask = None
                    v2a_cross_attn_perturbation_mask = None
                    skip_video_self_attn = block_idx in skip_video_self_attn_blocks
                    skip_audio_self_attn = block_idx in skip_audio_self_attn_blocks
                    skip_a2v_cross_attn = disable_a2v_cross_attn
                    skip_v2a_cross_attn = disable_v2a_cross_attn
                    if perturbation_configs is not None:
                        if not skip_video_self_attn:
                            assert video_self_attn_perturbation_states is not None
                            state = video_self_attn_perturbation_states[block_idx]
                            video_self_attn_perturbation_mask, skip_video_self_attn = (
                                state
                            )
                        if not skip_audio_self_attn:
                            assert audio_self_attn_perturbation_states is not None
                            state = audio_self_attn_perturbation_states[block_idx]
                            audio_self_attn_perturbation_mask, skip_audio_self_attn = (
                                state
                            )
                        if not skip_a2v_cross_attn:
                            assert a2v_cross_attn_perturbation_states is not None
                            state = a2v_cross_attn_perturbation_states[block_idx]
                            a2v_cross_attn_perturbation_mask, skip_a2v_cross_attn = (
                                state
                            )
                        if not skip_v2a_cross_attn:
                            assert v2a_cross_attn_perturbation_states is not None
                            state = v2a_cross_attn_perturbation_states[block_idx]
                            v2a_cross_attn_perturbation_mask, skip_v2a_cross_attn = (
                                state
                            )
                    with _ltx2_record_function(f"ltx2_dit_block::{block_idx}"):
                        hidden_states, audio_hidden_states = block(
                            hidden_states,
                            audio_hidden_states,
                            encoder_hidden_states,
                            audio_encoder_hidden_states,
                            # Keep the first 4 args positional to stay compatible with cache-dit's
                            # LTX2 adapter, which treats `audio_hidden_states` as `encoder_hidden_states`
                            # under ForwardPattern.Pattern_0.
                            temb=temb,
                            temb_audio=temb_audio,
                            temb_prompt=temb_prompt,
                            temb_audio_prompt=temb_audio_prompt,
                            temb_ca_scale_shift=temb_ca_scale_shift,
                            temb_ca_audio_scale_shift=temb_ca_audio_scale_shift,
                            temb_ca_gate=temb_ca_gate,
                            temb_ca_audio_gate=temb_ca_audio_gate,
                            video_rotary_emb=video_rotary_emb,
                            audio_rotary_emb=audio_rotary_emb,
                            ca_video_rotary_emb=ca_video_rotary_emb,
                            ca_audio_rotary_emb=ca_audio_rotary_emb,
                            encoder_attention_mask=encoder_attention_mask,
                            audio_encoder_attention_mask=audio_encoder_attention_mask,
                            video_self_attention_mask=video_self_attention_mask,
                            audio_self_attention_mask=audio_self_attention_mask,
                            a2v_cross_attention_mask=a2v_cross_attention_mask,
                            v2a_cross_attention_mask=v2a_cross_attention_mask,
                            skip_video_self_attn=skip_video_self_attn,
                            skip_audio_self_attn=skip_audio_self_attn,
                            skip_a2v_cross_attn=skip_a2v_cross_attn,
                            skip_v2a_cross_attn=skip_v2a_cross_attn,
                            video_self_attn_perturbation_mask=video_self_attn_perturbation_mask,
                            audio_self_attn_perturbation_mask=audio_self_attn_perturbation_mask,
                            a2v_cross_attn_perturbation_mask=a2v_cross_attn_perturbation_mask,
                            v2a_cross_attn_perturbation_mask=v2a_cross_attn_perturbation_mask,
                            audio_replicated_for_sp=audio_replicated_for_sp,
                            audio_latents_replicated_for_sp=audio_latents_replicated_for_sp,
                            disable_sequence_parallel_for_replicated_sp=disable_sequence_parallel_for_replicated_sp,
                            share_block0_self_attn=share_block0_self_attn,
                        )
        finally:
            _ltx2_pop_profile_context(profile_token)
        if (
            teacache_coordinator is not None
            and teacache_decision is not None
            and not teacache_decision.should_skip
            and original_hidden_states_for_teacache is not None
            and original_audio_hidden_states_for_teacache is not None
        ):
            teacache_coordinator.store(
                teacache_decision,
                original_hidden_states=original_hidden_states_for_teacache,
                original_audio_hidden_states=original_audio_hidden_states_for_teacache,
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                temb=temb,
                temb_audio=temb_audio,
            )

        # 6. Output layers
        # Video
        scale_shift_values = self.scale_shift_table[None, None].to(
            device=hidden_states.device, dtype=hidden_states.dtype
        ) + embedded_timestep[:, :, None].to(dtype=hidden_states.dtype)
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states, _ = self.proj_out(hidden_states)

        # Audio
        audio_scale_shift_values = self.audio_scale_shift_table[None, None].to(
            device=audio_hidden_states.device, dtype=audio_hidden_states.dtype
        ) + audio_embedded_timestep[:, :, None].to(dtype=audio_hidden_states.dtype)
        audio_shift, audio_scale = (
            audio_scale_shift_values[:, :, 0],
            audio_scale_shift_values[:, :, 1],
        )
        with torch.autocast(device_type=audio_hidden_states.device.type, enabled=False):
            audio_hidden_states = self.audio_norm_out(audio_hidden_states)
        audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
        audio_hidden_states, _ = self.audio_proj_out(audio_hidden_states)
        # Unpatchify if requested (default True for pipeline compatibility)
        return_latents = kwargs.get("return_latents", True)

        if return_latents:
            # Unpatchify Video
            # [B, N, C_out_raw*patch_vol] -> [B, C_out_raw, T, H, W]
            # Requires num_frames, height, width to be known
            if num_frames is not None and height is not None and width is not None:
                p_t, p_h, p_w = self.patch_size
                post_t, post_h, post_w = num_frames // p_t, height // p_h, width // p_w
                b = batch_size
                hidden_states = hidden_states.reshape(
                    b, post_t, post_h, post_w, self.out_channels_raw, p_t, p_h, p_w
                )
                hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
                    b, self.out_channels_raw, num_frames, height, width
                )

            # Unpatchify Audio
            # [B, N, C_out] -> [B, C_out, T] (or 4D/5D)
            if audio_num_frames is not None:
                b = batch_size
                # simple reshape for 1D patch
                audio_hidden_states = audio_hidden_states.permute(0, 2, 1)  # [B, C, T]

        return hidden_states, audio_hidden_states


# Backward-compatible alias (older internal name).
LTXModel = LTX2VideoTransformer3DModel
EntryClass = LTX2VideoTransformer3DModel
