from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import warnings
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _configure_concise_import_logs() -> None:
    if _env_flag("LINGBOT_VERBOSE_LOGS"):
        return
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    warnings.filterwarnings(
        "ignore",
        message=r"Unable to import `torchao` Tensor objects.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`enable_parallelism` is an experimental feature.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"barrier\(\): using the device under current context.*",
    )
    logging.getLogger(
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config"
    ).setLevel(logging.ERROR)


_configure_concise_import_logs()

from lingbot_video.inference_backend import (  # noqa: E402
    resolve_backend_engine,
    resolve_negative_prompt_arg,
)

try:
    import transformers as _transformers
    import transformers.cache_utils as _transformers_cache_utils
except Exception as exc:  # pragma: no cover - deployment dependency guard
    _transformers = None
    _transformers_cache_utils = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

try:
    import diffusers.utils.import_utils as _diffusers_import_utils
except Exception as exc:  # pragma: no cover - deployment dependency guard
    _diffusers_import_utils = None
    _DIFFUSERS_IMPORT_UTILS_ERROR = exc
else:
    _DIFFUSERS_IMPORT_UTILS_ERROR = None

try:
    from diffusers.utils import logging as _diffusers_logging
except Exception:
    _diffusers_logging = None

try:
    from transformers.utils import logging as _transformers_logging
except Exception:
    _transformers_logging = None

try:
    import huggingface_hub.utils as _hf_hub_utils
except Exception:
    _hf_hub_utils = None


def _install_sglang_import_shims() -> None:
    if _transformers is not None and _transformers_cache_utils is not None:
        hybrid_cache = getattr(_transformers_cache_utils, "HybridCache", None)
        if hybrid_cache is None:
            hybrid_cache = getattr(_transformers_cache_utils, "DynamicCache", object)
            _transformers_cache_utils.HybridCache = hybrid_cache
        import_structure = getattr(_transformers, "_import_structure", {})
        cache_exports = import_structure.setdefault("cache_utils", [])
        if "HybridCache" not in cache_exports:
            cache_exports.append("HybridCache")
        _transformers.HybridCache = hybrid_cache
    if _diffusers_import_utils is not None:
        _diffusers_import_utils._peft_available = False


_install_sglang_import_shims()

try:
    from lingbot_video.fsdp_inference import (
        apply_fsdp_inference,
        init_fsdp_inference_mesh,
    )
    from lingbot_video.model_paths import (
        effective_refiner_model_dir,
        model_component_dir,
    )
    from lingbot_video.pipeline_lingbot_video import (
        DEFAULT_NEGATIVE_PROMPT,
        DEFAULT_NEGATIVE_PROMPT_IMAGE,
        LingBotVideoPipeline,
    )
    from lingbot_video.transformer_lingbot_video import set_lingbot_pisa_phase
    from lingbot_video.pipeline_lingbot_video_i2v import (
        LingBotVideoImageToVideoPipeline,
    )
    from lingbot_video.utils import (
        caption_from_sample,
        load_first_frame_condition_tensor,
        load_refiner_video_tensor,
        num_frames_from_duration,
        prepare_refiner_latent,
    )
except Exception as exc:  # pragma: no cover - reported when generation is attempted
    DEFAULT_NEGATIVE_PROMPT = ""
    DEFAULT_NEGATIVE_PROMPT_IMAGE = ""
    LingBotVideoPipeline = None
    LingBotVideoImageToVideoPipeline = None
    apply_fsdp_inference = None
    init_fsdp_inference_mesh = None
    effective_refiner_model_dir = None
    model_component_dir = None
    caption_from_sample = None
    load_first_frame_condition_tensor = None
    load_refiner_video_tensor = None
    num_frames_from_duration = None
    prepare_refiner_latent = None
    _LINGBOT_PIPELINE_IMPORT_ERROR = exc
else:
    _LINGBOT_PIPELINE_IMPORT_ERROR = None

try:
    from lingbot_video.transformer_lingbot_video import (
        LingBotVideoTransformer3DModel,
    )
except Exception as exc:  # pragma: no cover - reported when generation is attempted
    LingBotVideoTransformer3DModel = None
    _TRANSFORMER_IMPORT_ERROR = exc
else:
    _TRANSFORMER_IMPORT_ERROR = None

try:
    from diffusers import ContextParallelConfig
    from diffusers.hooks.context_parallel import apply_context_parallel
    from diffusers.models._modeling_parallel import ParallelConfig
    from diffusers.models.attention import AttentionModuleMixin
    from diffusers.models.attention_processor import Attention, MochiAttention
    from diffusers.utils import export_to_video
except Exception as exc:  # pragma: no cover - reported by the selected engine
    ContextParallelConfig = None
    ParallelConfig = None
    Attention = None
    AttentionModuleMixin = None
    MochiAttention = None
    apply_context_parallel = None
    export_to_video = None
    _DIFFUSERS_IMPORT_ERROR = exc
else:
    _DIFFUSERS_IMPORT_ERROR = None


def _cuda_sdp_state() -> dict[str, bool | None]:
    state: dict[str, bool | None] = {}
    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is None:
        return state
    getters = {
        "flash": "flash_sdp_enabled",
        "mem_efficient": "mem_efficient_sdp_enabled",
        "math": "math_sdp_enabled",
        "cudnn": "cudnn_sdp_enabled",
    }
    for key, getter in getters.items():
        fn = getattr(cuda_backends, getter, None)
        if fn is None:
            state[key] = None
            continue
        try:
            state[key] = bool(fn())
        except Exception:
            state[key] = None
    return state


def _restore_cuda_sdp_state(state: dict[str, bool | None]) -> None:
    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is None:
        return
    setters = {
        "flash": "enable_flash_sdp",
        "mem_efficient": "enable_mem_efficient_sdp",
        "math": "enable_math_sdp",
        "cudnn": "enable_cudnn_sdp",
    }
    for key, setter in setters.items():
        value = state.get(key)
        if value is None:
            continue
        fn = getattr(cuda_backends, setter, None)
        if fn is None:
            continue
        try:
            fn(value)
        except Exception:
            pass


_BASELINE_CUDA_SDP_STATE = _cuda_sdp_state()


RESOLUTION_BUCKETS: dict[str, dict[str, tuple[int, int]]] = {
    "192p": {
        "1:1": (192, 192),
        "9:16": (192, 320),
        "16:9": (320, 192),
        "3:4": (192, 256),
        "4:3": (256, 192),
    },
    "480p": {
        "1:1": (480, 480),
        "9:16": (480, 832),
        "16:9": (832, 480),
        "3:4": (480, 640),
        "4:3": (640, 480),
    },
    "720p": {
        "1:1": (736, 736),
        "9:16": (736, 1280),
        "16:9": (1280, 736),
        "3:4": (736, 960),
        "4:3": (960, 736),
    },
    "1080p": {
        "1:1": (1088, 1088),
        "9:16": (1088, 1920),
        "16:9": (1920, 1088),
        "3:4": (1088, 1440),
        "4:3": (1440, 1088),
    },
    "2k": {
        "1:1": (1440, 1440),
        "9:16": (1440, 2560),
        "16:9": (2560, 1440),
        "3:4": (1440, 1920),
        "4:3": (1920, 1440),
    },
    "4k": {
        "1:1": (2176, 2176),
        "9:16": (2176, 3840),
        "16:9": (3840, 2176),
        "3:4": (2176, 2880),
        "4:3": (2880, 2176),
    },
}


def _distributed_env() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def _default_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda", torch.cuda.current_device())


def _init_parallel(
    cfg_degree: int,
    context_degree: int,
    enable_fsdp_inference: bool,
) -> tuple[int, int, int, Any | None, DeviceMesh | None, int, int]:
    rank, local_rank, world_size = _distributed_env()
    if cfg_degree < 1 or context_degree < 1:
        raise ValueError("Parallel degrees must be >= 1.")
    expected_world_size = cfg_degree * context_degree
    if expected_world_size <= 1:
        if enable_fsdp_inference and world_size > 1:
            if not torch.cuda.is_available():
                raise RuntimeError("FSDP inference requires CUDA devices.")
            torch.cuda.set_device(local_rank)
            if not dist.is_initialized():
                dist.init_process_group("nccl", timeout=timedelta(minutes=30))
            return rank, local_rank, world_size, None, None, 0, 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return rank, local_rank, world_size, None, None, 0, 0
    if world_size != expected_world_size:
        raise ValueError(
            f"Parallel topology cfg={cfg_degree}, context={context_degree} requires "
            f"WORLD_SIZE={expected_world_size}, got {world_size}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed parallel inference requires CUDA devices.")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))

    cfg_branch_rank = rank // context_degree
    context_rank = rank % context_degree
    cfg_parallel_group = None
    if cfg_degree > 1:
        lane_groups = [
            dist.new_group(
                ranks=[cfg_rank * context_degree + context_rank for cfg_rank in range(cfg_degree)]
            )
            for context_rank in range(context_degree)
        ]
        cfg_parallel_group = lane_groups[context_rank]

    context_mesh = None
    if context_degree > 1 and cfg_degree > 1:
        mesh_ranks = [
            [
                cfg_rank * context_degree + context_rank
                for context_rank in range(context_degree)
            ]
            for cfg_rank in range(cfg_degree)
        ]
        context_mesh = DeviceMesh(
            "cuda",
            mesh_ranks,
            mesh_dim_names=("ring", "ulysses"),
        )

    return rank, local_rank, world_size, cfg_parallel_group, context_mesh, cfg_branch_rank, context_rank

try:
    from lingbot_video.native_backend import (
        LingBotVideoNativePipeline,
        register_lingbot_native_pipeline,
    )
except Exception as exc:  # pragma: no cover - only needed for --engine sglang-native
    LingBotVideoNativePipeline = None
    register_lingbot_native_pipeline = None
    _NATIVE_BACKEND_IMPORT_ERROR = exc
else:
    _NATIVE_BACKEND_IMPORT_ERROR = None
finally:
    _restore_cuda_sdp_state(_BASELINE_CUDA_SDP_STATE)


def _parse_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _make_dtype_map(args: argparse.Namespace) -> dict[str, torch.dtype]:
    default_dtype = _parse_dtype(args.default_dtype)
    return {
        "default": default_dtype,
        "transformer": _parse_dtype(args.transformer_dtype),
        "text_encoder": _parse_dtype(args.text_encoder_dtype),
        "vae": _parse_dtype(args.vae_dtype),
    }


def _make_default_image(height: int, width: int) -> Image.Image:
    yy, xx = np.mgrid[0:height, 0:width]
    red = (xx / max(width - 1, 1) * 255).astype(np.uint8)
    green = (yy / max(height - 1, 1) * 255).astype(np.uint8)
    blue = (((xx // 24 + yy // 24) % 2) * 180 + 40).astype(np.uint8)
    return Image.fromarray(np.stack([red, green, blue], axis=-1), mode="RGB")


def _load_prompt_sample(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if not data:
            raise ValueError(f"`--prompt_json` is empty: {path}")
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError(f"`--prompt_json` must contain a dict or a non-empty list of dicts: {path}")
    return data


def _caption_from_sample(sample: dict[str, Any]) -> str:
    if caption_from_sample is not None:
        return caption_from_sample(sample)
    if "caption" in sample:
        caption = sample["caption"]
    else:
        runtime_keys = {
            "duration",
            "fps",
            "height",
            "width",
            "num_frames",
            "resolution",
            "ratio",
        }
        caption = {key: value for key, value in sample.items() if key not in runtime_keys}
    if isinstance(caption, dict):
        return str(caption.get("qwen_long_caption") or caption.get("comprehensive_description") or caption)
    return str(caption)


def _height_width_from_bucket(resolution: str, ratio: str) -> tuple[int, int]:
    if resolution not in RESOLUTION_BUCKETS:
        choices = ", ".join(sorted(RESOLUTION_BUCKETS))
        raise ValueError(f"unsupported resolution {resolution!r}; choices: {choices}")
    ratios = RESOLUTION_BUCKETS[resolution]
    if ratio not in ratios:
        choices = ", ".join(sorted(ratios))
        raise ValueError(f"unsupported ratio {ratio!r} for {resolution}; choices: {choices}")
    return ratios[ratio]


def _module_dtype(module: Any) -> str | None:
    if module is None:
        return None
    if hasattr(module, "dtype"):
        dtype = getattr(module, "dtype")
        if isinstance(dtype, torch.dtype):
            return _dtype_name(dtype)
    if isinstance(module, torch.nn.Module):
        try:
            return _dtype_name(next(module.parameters()).dtype)
        except StopIteration:
            return None
    return None


def _component_dtypes(pipe: Any) -> dict[str, str | None]:
    return {
        "transformer": _module_dtype(getattr(pipe, "transformer", None)),
        "text_encoder": _module_dtype(getattr(pipe, "text_encoder", None)),
        "vae": _module_dtype(getattr(pipe, "vae", None)),
    }


def _disable_external_progress_bars() -> None:
    for logging_module in (_diffusers_logging, _transformers_logging):
        fn = getattr(logging_module, "disable_progress_bar", None)
        if callable(fn):
            fn()
    hf_fn = getattr(_hf_hub_utils, "disable_progress_bars", None)
    if callable(hf_fn):
        hf_fn()


def _configure_pipeline_logs(pipe: Any) -> None:
    inner_pipe = _inner_diffusers_pipe(pipe)
    set_progress_bar_config = getattr(inner_pipe, "set_progress_bar_config", None)
    if callable(set_progress_bar_config):
        set_progress_bar_config(
            disable=not _show_progress(),
            desc="denoising",
            dynamic_ncols=True,
        )


@contextmanager
def _patch_qwen3vl_from_pretrained():
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except Exception:
        yield
        return

    original_from_pretrained = Qwen3VLForConditionalGeneration.from_pretrained
    attn_implementation = os.environ.get("LINGBOT_QWEN_ATTN_IMPLEMENTATION", "flash_attention_3")

    @classmethod
    def patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if attn_implementation:
            kwargs.setdefault("attn_implementation", attn_implementation)
        if "torch_dtype" in kwargs and "dtype" not in kwargs:
            kwargs["dtype"] = kwargs.pop("torch_dtype")
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    Qwen3VLForConditionalGeneration.from_pretrained = patched_from_pretrained
    try:
        yield
    finally:
        Qwen3VLForConditionalGeneration.from_pretrained = original_from_pretrained


def _destroy_parallel_if_needed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _sync_parallel_if_needed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _current_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _is_main_process() -> bool:
    return _current_rank() == 0


def _show_progress() -> bool:
    return _is_main_process() and not _env_flag("LINGBOT_QUIET_PROGRESS")


def _log_progress(message: str) -> None:
    if _show_progress():
        print(message, flush=True)


# --- Lightweight phase timing (pure logging; does not change any computation) ---
_PHASE_T0 = None
_PHASE_PREV = None


def _phase(label: str) -> None:
    """Log wall-clock elapsed since start and since the previous phase (rank 0 only).

    Instrumentation only — emits `PHASE <label> dt=<s> total=<s>` lines the agent greps.
    Guarded by LINGBOT_PHASE_TIMING=1 so it is a no-op OFF path otherwise.
    """
    global _PHASE_T0, _PHASE_PREV
    if not _env_flag("LINGBOT_PHASE_TIMING"):
        return
    if not _is_main_process():
        return
    import time as _t

    now = _t.time()
    if _PHASE_T0 is None:
        _PHASE_T0 = now
        _PHASE_PREV = now
    dt = now - _PHASE_PREV
    total = now - _PHASE_T0
    _PHASE_PREV = now
    print(f"PHASE {label} dt={dt:.2f} total={total:.2f}", flush=True)


def _apply_fsdp_inference_if_requested(
    pipe: Any,
    enabled: bool,
    mesh: DeviceMesh | None,
) -> Any | None:
    if not enabled:
        return None
    if apply_fsdp_inference is None:
        raise RuntimeError("FSDP inference helpers are not importable.") from _LINGBOT_PIPELINE_IMPORT_ERROR
    inner_pipe = _inner_diffusers_pipe(pipe)
    transformer = getattr(inner_pipe, "transformer", None)
    if not isinstance(transformer, torch.nn.Module):
        raise ValueError(
            "FSDP inference requires a pipeline with a torch.nn.Module transformer."
        )
    _log_progress("applying FSDP inference sharding")
    info = apply_fsdp_inference(transformer, mesh)
    _log_progress(f"applied FSDP inference sharding: {info}")
    return info


def _cache_prompt_conditions(
    pipe: Any,
    prompt: str,
    negative_prompt: str,
    *,
    device: torch.device,
    null_cond_clone_zero: bool,
    images: list[Any] | None = None,
) -> dict[str, torch.Tensor]:
    encode_kwargs: dict[str, Any] = {"device": device}
    if images is not None:
        encode_kwargs["images"] = images
    prompt_embeds, prompt_mask = pipe.encode_prompt(prompt, **encode_kwargs)
    if null_cond_clone_zero:
        negative_embeds = torch.zeros_like(prompt_embeds)
        negative_mask = prompt_mask.clone()
    else:
        negative_embeds, negative_mask = pipe.encode_prompt(negative_prompt, **encode_kwargs)
    return {
        "prompt_embeds": prompt_embeds.detach().cpu(),
        "prompt_mask": prompt_mask.detach().cpu(),
        "negative_prompt_embeds": negative_embeds.detach().cpu(),
        "negative_prompt_mask": negative_mask.detach().cpu(),
    }


def _inner_diffusers_pipe(pipe: Any) -> Any:
    return getattr(pipe, "diffusers_pipe", pipe)


def _cache_ti2v_prompt_conditions(
    pipe: Any,
    prompt: str,
    negative_prompt: str,
    image: Image.Image,
    *,
    height: int,
    width: int,
    device: torch.device,
    null_cond_clone_zero: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    condition_pipe = _inner_diffusers_pipe(pipe)
    if not hasattr(condition_pipe, "preprocess_image") or not hasattr(condition_pipe, "_vlm_image"):
        raise ValueError("TI2V condition reuse requires a LingBotVideo image-to-video pipeline.")
    pixel = condition_pipe.preprocess_image(image, height, width).to(
        device=device,
        dtype=torch.float32,
    )
    vlm_image = condition_pipe._vlm_image(pixel)
    return (
        _cache_prompt_conditions(
            condition_pipe,
            prompt,
            negative_prompt,
            device=device,
            null_cond_clone_zero=null_cond_clone_zero,
            images=[vlm_image],
        ),
        pixel.detach().cpu(),
    )


def _condition_call_kwargs(
    cache: dict[str, torch.Tensor] | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not cache:
        return {}
    return {key: value.to(device=device) for key, value in cache.items()}


def _pipeline_class_for_mode(mode: str) -> Any:
    if mode == "ti2v":
        if LingBotVideoImageToVideoPipeline is None:
            raise RuntimeError(
                "LingBotVideoImageToVideoPipeline is not importable."
            ) from _LINGBOT_PIPELINE_IMPORT_ERROR
        return LingBotVideoImageToVideoPipeline
    if LingBotVideoPipeline is None:
        raise RuntimeError("LingBotVideoPipeline is not importable.") from _LINGBOT_PIPELINE_IMPORT_ERROR
    return LingBotVideoPipeline


def _bcast_weights_enabled() -> bool:
    return _env_flag("LINGBOT_BCAST_WEIGHTS")


def _build_empty_transformer(
    model_dir: Path,
    transformer_subfolder: str,
    transformer_dtype: torch.dtype,
) -> Any:
    """Construct the transformer architecture WITHOUT reading weight shards from disk.

    Used on non-zero ranks under LINGBOT_BCAST_WEIGHTS: only rank 0 parses the 57GB
    checkpoint; every other rank builds an empty (garbage) module of the identical
    architecture and receives the real weights over NVLink via _broadcast_transformer_weights.
    """
    cfg = LingBotVideoTransformer3DModel.load_config(
        str(model_dir), subfolder=transformer_subfolder
    )
    with torch.device("meta"):
        model = LingBotVideoTransformer3DModel.from_config(cfg)
    model = model.to(transformer_dtype)  # cheap on meta; broadcast overwrites per-param dtype
    model = model.to_empty(device="cpu")
    return model


def _broadcast_transformer_weights(transformer: Any, src: int = 0, device: Any = None) -> None:
    """Broadcast every parameter/buffer of `transformer` from `src` to all ranks over NCCL.

    Name-keyed and shape/dtype-synced (rank 0 announces the ordered spec), so every rank
    ends with rank 0's exact weights → mathematically identical to loading from disk on
    each rank, but the parse happens once. Leaves tensors on the current CUDA device;
    FSDP fully_shard reshards them afterwards.
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    world = dist.get_world_size()
    if world <= 1:
        return
    rank = dist.get_rank()
    dev = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    params = {n: p for n, p in transformer.named_parameters()}
    buffers = {n: b for n, b in transformer.named_buffers() if b is not None}
    if rank == src:
        spec = [(n, tuple(p.shape), p.dtype, "p") for n in sorted(params)
                for p in (params[n],)]
        spec += [(n, tuple(b.shape), b.dtype, "b") for n in sorted(buffers)
                 for b in (buffers[n],)]
    else:
        spec = None
    box = [spec]
    dist.broadcast_object_list(box, src=src)
    spec = box[0]
    _log_progress(f"broadcasting {len(spec)} transformer tensors from rank {src} over NVLink")
    with torch.no_grad():
        for name, shape, dtype, kind in spec:
            store = params if kind == "p" else buffers
            if rank == src:
                g = store[name].data.to(device=dev, dtype=dtype)
            else:
                g = torch.empty(shape, dtype=dtype, device=dev)
            dist.broadcast(g, src=src)
            if name in store:
                store[name].data = g
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _sharded_load_enabled() -> bool:
    return _env_flag("LINGBOT_SHARDED_LOAD")


_ST_DTYPE = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}


def _sharded_load_and_broadcast(
    transformer: Any,
    model_dir: Path,
    transformer_subfolder: str,
    device: Any = None,
) -> None:
    """Distribute the checkpoint PARSE across ranks, then assemble via NVLink broadcast.

    Each rank safetensors-loads a disjoint ~1/4 of the shard files (its owned params get
    real weights; the rest stay empty), then every param is broadcast from its owning rank.
    Cuts the single-rank parse (~69s for 57GB) to ~1/world. Result on every rank is the
    exact checkpoint weights → mathematically identical to a full per-rank load.
    """
    import json as _json
    from safetensors import safe_open

    world = dist.get_world_size()
    rank = dist.get_rank()
    dev = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    sub = Path(model_dir) / transformer_subfolder
    index_path = sub / "diffusion_pytorch_model.safetensors.index.json"
    weight_map = _json.load(open(index_path))["weight_map"]  # name -> shard file
    shard_files = sorted(set(weight_map.values()))
    owner = {name: shard_files.index(shard) % world for name, shard in weight_map.items()}

    # Every rank cheaply reads all shard HEADERS (no tensor data) to learn shape/dtype.
    shape_of: dict[str, tuple] = {}
    dtype_of: dict[str, torch.dtype] = {}
    for shard in shard_files:
        with safe_open(str(sub / shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                sl = f.get_slice(name)
                shape_of[name] = tuple(sl.get_shape())
                dtype_of[name] = _ST_DTYPE[sl.get_dtype()]

    named = {n: p for n, p in transformer.named_parameters()}
    named.update({n: b for n, b in transformer.named_buffers() if b is not None})

    # Load only this rank's owned shards' tensors into the empty model (CPU).
    my_shards = [s for i, s in enumerate(shard_files) if i % world == rank]
    with torch.no_grad():
        for shard in my_shards:
            with safe_open(str(sub / shard), framework="pt", device="cpu") as f:
                for name in f.keys():
                    if name in named:
                        named[name].data = f.get_tensor(name)
    _log_progress(
        f"sharded-load rank {rank}: parsed {len(my_shards)}/{len(shard_files)} shards; "
        f"broadcasting {len(weight_map)} tensors over NVLink"
    )
    # Per-param broadcast from its owner rank.
    with torch.no_grad():
        for name in sorted(weight_map):
            src = owner[name]
            if rank == src:
                g = named[name].data.to(device=dev, dtype=dtype_of[name])
            else:
                g = torch.empty(shape_of[name], dtype=dtype_of[name], device=dev)
            dist.broadcast(g, src=src)
            if name in named:
                named[name].data = g
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _distribute_transformer_weights(
    transformer: Any,
    model_dir: Any,
    transformer_subfolder: str,
    device: Any = None,
) -> None:
    """Dispatch: sharded parallel-load+broadcast, or rank-0-load+broadcast, or nothing."""
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return
    if _sharded_load_enabled():
        _sharded_load_and_broadcast(transformer, Path(model_dir), transformer_subfolder, device)
    elif _bcast_weights_enabled():
        _broadcast_transformer_weights(transformer, device=device)


def _load_transformer_component(
    model_dir: Path,
    transformer_subfolder: str,
    dtype_map: dict[str, torch.dtype],
) -> Any:
    if LingBotVideoTransformer3DModel is None:
        raise RuntimeError(
            "LingBotVideoTransformer3DModel is not importable."
        ) from _TRANSFORMER_IMPORT_ERROR
    if model_component_dir is None:
        raise RuntimeError("model path helpers are not importable.") from _LINGBOT_PIPELINE_IMPORT_ERROR
    model_component_dir(model_dir, transformer_subfolder)
    transformer_dtype = dtype_map.get(
        "transformer",
        dtype_map.get("default", torch.float32),
    )
    _multi = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    # Sharded mode: EVERY rank builds empty; each parses a disjoint 1/4 of shards later.
    if _sharded_load_enabled() and _multi:
        _log_progress(
            f"building EMPTY transformer subfolder={transformer_subfolder} "
            f"dtype={_dtype_name(transformer_dtype)} (sharded parallel load + broadcast)"
        )
        return _build_empty_transformer(model_dir, transformer_subfolder, transformer_dtype)
    # Broadcast mode: only rank 0 reads/parses the checkpoint; other ranks build empty.
    if _bcast_weights_enabled() and _multi and dist.get_rank() != 0:
        _log_progress(
            f"building EMPTY transformer subfolder={transformer_subfolder} "
            f"dtype={_dtype_name(transformer_dtype)} (weights via NVLink broadcast)"
        )
        return _build_empty_transformer(model_dir, transformer_subfolder, transformer_dtype)
    _disable_external_progress_bars()
    _log_progress(
        f"loading transformer subfolder={transformer_subfolder} "
        f"dtype={_dtype_name(transformer_dtype)} model_dir={model_dir}"
    )
    transformer = LingBotVideoTransformer3DModel.from_pretrained(
        str(model_dir),
        subfolder=transformer_subfolder,
        torch_dtype=transformer_dtype,
    )
    _log_progress(f"loaded transformer subfolder={transformer_subfolder}")
    _maybe_compile_moe(transformer)
    return transformer


def _maybe_compile_moe(transformer: Any) -> None:
    """Approach 1 (megafusion): torch.compile each MoE block's ffn so inductor fuses
    the pointwise glue (router sigmoid/topk, restore scatter+weighted-sum, silu-gate,
    shared MLP) into fused Triton kernels — the 'GEMM + epilogue' fusion. The restore
    scatter alone is ~46% of the MoE block; fusion ~halves the block (measured 2.13x).
    Gated by LINGBOT_COMPILE_MOE (OFF path unchanged)."""
    if not _env_flag("LINGBOT_COMPILE_MOE"):
        return
    blocks = getattr(transformer, "blocks", None)
    if not blocks:
        return
    mode = os.environ.get("LINGBOT_COMPILE_MODE", "max-autotune-no-cudagraphs")
    n = 0
    for blk in blocks:
        ffn = getattr(blk, "ffn", None)
        if ffn is not None and ffn.__class__.__name__ == "LingBotVideoSparseMoeBlock":
            blk.ffn = torch.compile(ffn, mode=mode)
            n += 1
    _log_progress(f"compiled {n} MoE ffn blocks (megafusion, mode={mode})")


def _move_pipeline_aux_modules_to_device(pipe: Any, device: torch.device) -> Any:
    inner_pipe = _inner_diffusers_pipe(pipe)
    for name in ("text_encoder", "vae"):
        module = getattr(inner_pipe, name, None)
        if isinstance(module, torch.nn.Module):
            _log_progress(f"moving {name} to {device}")
            module.to(device)
    return pipe


def _load_diffusers_pipe(
    model_dir: Path,
    dtype_map: dict[str, torch.dtype],
    mode: str,
    transformer_subfolder: str,
    defer_transformer_to_device: bool = False,
    defer_aux_to_device: bool = False,
) -> Any:
    pipeline_class = _pipeline_class_for_mode(mode)
    transformer = _load_transformer_component(model_dir, transformer_subfolder, dtype_map)

    _disable_external_progress_bars()
    _log_progress(f"loading pipeline mode={mode} model_dir={model_dir}")
    with _patch_qwen3vl_from_pretrained():
        pipe = pipeline_class.from_pretrained(
            str(model_dir),
            transformer=transformer,
            trust_remote_code=True,
            torch_dtype=dtype_map,
        )
    _log_progress(f"loaded pipeline mode={mode}")
    device = _default_device()
    if defer_aux_to_device:
        # CPU-only load: keep BOTH transformer and aux modules (text_encoder, vae)
        # on CPU so this can run in a background thread with zero CUDA calls
        # (overlap the refiner load with base denoise). Caller finalizes on-device later.
        return pipe
    if defer_transformer_to_device:
        return _move_pipeline_aux_modules_to_device(pipe, device)
    _log_progress(f"moving pipeline to {device}")
    return pipe.to(device)


def _load_sglang_native_pipe(
    model_dir: Path,
    dtype_map: dict[str, torch.dtype],
    mode: str,
    transformer_subfolder: str,
    defer_transformer_to_device: bool = False,
) -> Any:
    if LingBotVideoNativePipeline is None or register_lingbot_native_pipeline is None:
        raise RuntimeError(
            "LingBotVideo SGLang native backend is not importable."
        ) from _NATIVE_BACKEND_IMPORT_ERROR
    _restore_cuda_sdp_state(_BASELINE_CUDA_SDP_STATE)
    try:
        register_lingbot_native_pipeline()
        diffusers_pipe = _load_diffusers_pipe(
            model_dir,
            dtype_map,
            mode=mode,
            transformer_subfolder=transformer_subfolder,
            defer_transformer_to_device=defer_transformer_to_device,
        )
        return LingBotVideoNativePipeline.from_diffusers_pipe(
            diffusers_pipe,
            model_path=model_dir,
        )
    finally:
        _restore_cuda_sdp_state(_BASELINE_CUDA_SDP_STATE)


def _load_pipe(
    args: argparse.Namespace,
    dtype_map: dict[str, torch.dtype],
    *,
    defer_transformer_to_device: bool = False,
    defer_aux_to_device: bool = False,
):
    model_dir = Path(args.model_dir).resolve()
    if args.engine == "diffusers":
        return (
            _load_diffusers_pipe(
                model_dir,
                dtype_map,
                mode=args.mode,
                transformer_subfolder=args.transformer_subfolder,
                defer_transformer_to_device=defer_transformer_to_device,
                defer_aux_to_device=defer_aux_to_device,
            ),
            "diffusers-reference",
        )
    if args.engine == "sglang-native":
        return (
            _load_sglang_native_pipe(
                model_dir,
                dtype_map,
                mode=args.mode,
                transformer_subfolder=args.transformer_subfolder,
                defer_transformer_to_device=defer_transformer_to_device,
            ),
            "sglang-native",
        )
    raise ValueError(f"unsupported engine: {args.engine}")


def _refiner_model_available(
    model_dir: str | None,
    transformer_subfolder: str = "refiner",
) -> bool:
    if not model_dir:
        return False
    root = Path(model_dir)
    if not root.is_dir():
        return False
    required = ("model_index.json", "scheduler", "text_encoder", "vae", transformer_subfolder)
    return all((root / name).exists() for name in required)


def _refiner_skip_reason(
    requested: bool,
    model_dir: str | None,
    transformer_subfolder: str = "refiner",
) -> str | None:
    if not requested:
        return None
    if not model_dir:
        return "missing_refiner_model_dir"
    if not _refiner_model_available(model_dir, transformer_subfolder):
        return f"missing_refiner_component:{transformer_subfolder}"
    return None


def _maybe_preload_refiner(
    args: argparse.Namespace,
    dtype_map: dict[str, torch.dtype],
    device: torch.device,
    context_parallel_rank: int,
    context_parallel_mesh: DeviceMesh | None,
    defer_transformer_to_device: bool,
    cpu_only: bool = False,
) -> dict[str, Any]:
    requested = bool(args.run_refiner)
    available = (
        _refiner_model_available(args.refiner_model_dir, args.refiner_transformer_subfolder)
        if requested
        else False
    )
    state: dict[str, Any] = {
        "refiner_requested": requested,
        "refiner_available": available,
        "refiner_skipped_reason": _refiner_skip_reason(
            requested,
            args.refiner_model_dir,
            args.refiner_transformer_subfolder,
        ),
        "pipe": None,
        "engine_name": None,
        "component_dtypes": None,
    }
    if not requested or not available:
        return state

    refiner_args = argparse.Namespace(**vars(args))
    refiner_args.model_dir = args.refiner_model_dir
    refiner_args.transformer_subfolder = args.refiner_transformer_subfolder
    refiner_args.vae_dtype = args.refiner_vae_dtype
    # The refiner always samples through the text-only t2v pipeline; ti2v
    # first-frame conditioning is injected as a clean frame-0 latent
    # (`cond_latent`), not through the image-to-video pipeline.
    refiner_args.mode = "t2v"
    refiner_dtype_map = dict(dtype_map)
    refiner_dtype_map["vae"] = _parse_dtype(args.refiner_vae_dtype)
    # cpu_only: load transformer + aux modules to CPU with ZERO CUDA calls so this
    # can run in a background thread overlapping base denoise. Device move, CP-enable,
    # and FSDP are deferred to _finalize_preloaded_refiner() on the main thread.
    refiner_pipe, refiner_engine_name = _load_pipe(
        refiner_args,
        refiner_dtype_map,
        defer_transformer_to_device=defer_transformer_to_device,
        defer_aux_to_device=cpu_only,
    )
    refiner_component_dtypes = _component_dtypes(refiner_pipe)
    expected_refiner_vae_dtype = _dtype_name(refiner_dtype_map["vae"])
    if refiner_component_dtypes.get("vae") != expected_refiner_vae_dtype:
        raise AssertionError(
            f"Refiner VAE dtype mismatch: requested {expected_refiner_vae_dtype}, "
            f"got {refiner_component_dtypes}"
        )
    if not cpu_only and (_sharded_load_enabled() or _bcast_weights_enabled()):
        # Non-overlap path: all ranks are here together → safe collective distribute.
        _distribute_transformer_weights(
            refiner_pipe.transformer, refiner_args.model_dir, refiner_args.transformer_subfolder
        )
    if not cpu_only and args.context_parallel_degree > 1:
        _enable_context_parallel(
            refiner_pipe.transformer,
            args.context_parallel_degree,
            args.context_parallel_ulysses_anything,
            context_parallel_rank,
            device,
            context_parallel_mesh,
            args.cfg_parallel_degree == 1,
        )
    state.update(
        {
            "pipe": refiner_pipe,
            "engine_name": refiner_engine_name,
            "component_dtypes": refiner_component_dtypes,
            "finalized": not cpu_only,
        }
    )
    return state


def _finalize_preloaded_refiner(
    state: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    context_parallel_rank: int,
    context_parallel_mesh: DeviceMesh | None,
) -> None:
    """Main-thread GPU setup for a CPU-only preloaded refiner: move aux modules to
    device and enable context parallel. FSDP is applied by the caller afterwards."""
    if state.get("finalized"):
        return
    refiner_pipe = state["pipe"]
    if refiner_pipe is None:
        return
    if _sharded_load_enabled() or _bcast_weights_enabled():
        # Overlap: rank 0 loaded the full refiner in the bg thread during base denoise
        # (bcast mode), or all ranks built empty (sharded mode). All ranks on main thread
        # here → safe collective. Sharded mode parses 1/4 shards/rank now, then broadcasts.
        _distribute_transformer_weights(
            refiner_pipe.transformer, args.refiner_model_dir, args.refiner_transformer_subfolder
        )
    _move_pipeline_aux_modules_to_device(refiner_pipe, device)
    if args.context_parallel_degree > 1:
        _enable_context_parallel(
            refiner_pipe.transformer,
            args.context_parallel_degree,
            args.context_parallel_ulysses_anything,
            context_parallel_rank,
            device,
            context_parallel_mesh,
            args.cfg_parallel_degree == 1,
        )
    state["finalized"] = True


def _extract_frames(result: Any) -> np.ndarray:
    frames = result.frames if hasattr(result, "frames") else result[0]
    if isinstance(frames, torch.Tensor):
        tensor = frames.detach().cpu().float()
        if tensor.ndim == 5:
            # Accept B,T,C,H,W or B,C,T,H,W.
            if tensor.shape[2] in (1, 3, 4):
                tensor = tensor[0].permute(0, 2, 3, 1)
            else:
                tensor = tensor[0].permute(1, 2, 3, 0)
        elif tensor.ndim == 4:
            if tensor.shape[1] in (1, 3, 4):
                tensor = tensor.permute(0, 2, 3, 1)
        return tensor.numpy()
    if isinstance(frames, list):
        return np.asarray(frames[0])
    return np.asarray(frames)


def _save_frames(
    frames: np.ndarray,
    mode: str,
    output: Path,
    fps: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    if mode == "t2i":
        Image.fromarray((frames[0] * 255).clip(0, 255).astype(np.uint8)).save(output)
    else:
        if export_to_video is None:
            raise RuntimeError("diffusers.utils.export_to_video is not importable.") from _DIFFUSERS_IMPORT_ERROR
        export_to_video(frames, str(output), fps=fps)

    print(f"saved {output} shape={tuple(frames.shape)}", flush=True)


def _enable_context_parallel(
    transformer: torch.nn.Module,
    degree: int,
    ulysses_anything: bool,
    rank: int,
    device: torch.device,
    mesh: DeviceMesh | None,
    use_native_mesh: bool,
) -> None:
    if degree <= 1:
        return
    if (
        ContextParallelConfig is None
        or ParallelConfig is None
        or apply_context_parallel is None
        or Attention is None
        or MochiAttention is None
        or AttentionModuleMixin is None
    ):
        raise RuntimeError("diffusers context-parallel helpers are unavailable.") from _DIFFUSERS_IMPORT_ERROR
    cp_config = ContextParallelConfig(
        ulysses_degree=degree,
        ulysses_anything=ulysses_anything,
    )
    if use_native_mesh:
        transformer.enable_parallelism(config=cp_config)
        return

    if mesh is None:
        raise ValueError("A context-parallel mesh is required when context_parallel_degree > 1.")

    config = ParallelConfig(context_parallel_config=cp_config)
    config.setup(rank, degree, device, mesh=mesh)
    if cp_config.ring_degree == 1 and int(mesh.mesh.numel()) != degree:
        cp_config._flattened_mesh = cp_config._ulysses_mesh._flatten()
    transformer._parallel_config = config

    attention_classes = (Attention, MochiAttention, AttentionModuleMixin)
    for module in transformer.modules():
        if not isinstance(module, attention_classes):
            continue
        processor = module.processor
        if processor is not None and hasattr(processor, "_parallel_config"):
            processor._parallel_config = config

    cp_plan = getattr(transformer, "_cp_plan", None)
    if cp_plan is None:
        raise ValueError("Transformer does not define a context-parallel plan.")
    apply_context_parallel(transformer, cp_config, cp_plan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="LingBot-Video model root directory.")
    parser.add_argument(
        "--backend",
        choices=["diffusers", "sglang"],
        default=None,
        help=(
            "public backend selector; `sglang` uses SGLang Diffusion when "
            "available and falls back to diffusers otherwise"
        ),
    )
    parser.add_argument(
        "--engine",
        choices=["sglang-native", "diffusers"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--mode", choices=["t2i", "t2v", "ti2v"], required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt_json", default=None)
    parser.add_argument("--negative_prompt", default=None,
                        help="negative prompt; if unset, uses the mode's default "
                             "(image default for t2i, video default for t2v/ti2v)")
    parser.add_argument(
        "--negative_prompt_json",
        default=None,
        help="path to the JSON file produced by rewriter/auto_negative.py",
    )
    parser.add_argument("--image", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--ratio", default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num_frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--default_dtype", default="bf16")
    parser.add_argument("--transformer_dtype", default="bf16")
    parser.add_argument("--transformer_subfolder", default="transformer")
    parser.add_argument("--text_encoder_dtype", default="bf16")
    parser.add_argument("--vae_dtype", default="fp32")
    parser.add_argument("--diffusers_attn_backend", default=os.environ.get("DIFFUSERS_ATTN_BACKEND", ""))
    parser.add_argument("--allow_tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--quiet_progress",
        action="store_true",
        help="Disable model-loading logs and denoising progress bars.",
    )
    parser.add_argument("--cfg_parallel_degree", type=int, default=1)
    parser.add_argument("--context_parallel_degree", type=int, default=1)
    parser.add_argument("--context_parallel_ulysses_anything", action="store_true")
    parser.add_argument(
        "--enable_fsdp_inference",
        action="store_true",
        help="Shard the base/refiner DiT transformers with PyTorch composable FSDP2.",
    )
    parser.add_argument("--batch_cfg", action="store_true")
    parser.add_argument("--null_cond_clone_zero", action="store_true")
    parser.add_argument("--reuse_condition_features", action="store_true")
    parser.add_argument("--run_refiner", action="store_true")
    parser.add_argument("--refiner_model_dir", default=None)
    parser.add_argument("--refiner_transformer_subfolder", default="refiner")
    parser.add_argument("--refiner_output", default=None)
    parser.add_argument("--refiner_height", type=int, default=1088)
    parser.add_argument("--refiner_width", type=int, default=1920)
    parser.add_argument("--refiner_steps", type=int, default=8)
    parser.add_argument("--refiner_guidance_scale", type=float, default=3.0)
    parser.add_argument("--refiner_shift", type=float, default=3.0)
    parser.add_argument("--refiner_t_thresh", type=float, default=0.85)
    parser.add_argument("--refiner_sigma_tail_steps", type=int, default=2)
    parser.add_argument("--refiner_fps", type=int, default=24)
    parser.add_argument("--refiner_sample_fps", type=int, default=24)
    parser.add_argument("--refiner_max_video_frames", type=int, default=None)
    parser.add_argument("--refiner_vae_dtype", default="fp32")
    parser.add_argument("--refiner_batch_cfg", action="store_true")
    parser.add_argument("--refiner_no_null_cond_clone_zero", action="store_true")
    parser.add_argument("--refiner_offload_vae_during_denoise", action="store_true")
    args = parser.parse_args()
    if args.quiet_progress:
        os.environ["LINGBOT_QUIET_PROGRESS"] = "1"
    args.engine = resolve_backend_engine(
        engine=args.engine,
        backend=args.backend,
        stderr=sys.stderr,
    )
    args.negative_prompt = resolve_negative_prompt_arg(
        args.negative_prompt,
        args.negative_prompt_json,
    )

    prompt_sample = None
    if args.prompt_json:
        prompt_sample = _load_prompt_sample(Path(args.prompt_json))
        args.prompt = _caption_from_sample(prompt_sample)
        if args.duration is None and "duration" in prompt_sample:
            args.duration = float(prompt_sample["duration"])
    if args.prompt is None:
        raise ValueError(
            "Provide `--prompt_json <path_to_structured_prompt.json>` or an explicit `--prompt`."
        )
    if args.resolution or args.ratio:
        if not args.resolution or not args.ratio:
            raise ValueError("`--resolution` and `--ratio` must be provided together.")
        args.height, args.width = _height_width_from_bucket(args.resolution, args.ratio)
    if args.negative_prompt is None:
        args.negative_prompt = (
            DEFAULT_NEGATIVE_PROMPT_IMAGE if args.mode == "t2i" else DEFAULT_NEGATIVE_PROMPT
        )
    if args.mode != "t2i" and args.duration is not None:
        if num_frames_from_duration is None:
            raise RuntimeError("num_frames_from_duration is not importable.") from _LINGBOT_PIPELINE_IMPORT_ERROR
        args.num_frames = num_frames_from_duration(args.duration, args.fps)
    args.run_refiner = bool(args.run_refiner or args.refiner_model_dir)
    if args.run_refiner:
        if effective_refiner_model_dir is None:
            raise RuntimeError("model path helpers are not importable.") from _LINGBOT_PIPELINE_IMPORT_ERROR
        args.refiner_model_dir = str(effective_refiner_model_dir(args))

    (
        rank,
        local_rank,
        world_size,
        cfg_parallel_group,
        context_parallel_mesh,
        cfg_branch_rank,
        context_parallel_rank,
    ) = _init_parallel(
        args.cfg_parallel_degree,
        args.context_parallel_degree,
        args.enable_fsdp_inference,
    )
    if args.cfg_parallel_degree > 1 and args.batch_cfg:
        raise ValueError("`--cfg_parallel_degree > 1` and `--batch_cfg` are mutually exclusive.")
    if args.refiner_model_dir and args.cfg_parallel_degree > 1 and args.refiner_batch_cfg:
        raise ValueError(
            "`--cfg_parallel_degree > 1` and `--refiner_batch_cfg` are mutually exclusive."
        )
    if args.run_refiner and args.mode == "t2i":
        raise ValueError("Refiner is only supported for video modes.")
    if args.run_refiner and args.mode == "ti2v" and not args.image:
        raise ValueError(
            "The ti2v refiner conditions on the clean first frame; `--image` is required."
        )
    if args.diffusers_attn_backend:
        os.environ["DIFFUSERS_ATTN_BACKEND"] = args.diffusers_attn_backend
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if args.mode == "t2i":
        args.num_frames = 1

    fsdp_mesh = None
    if args.enable_fsdp_inference:
        if init_fsdp_inference_mesh is None:
            raise RuntimeError("FSDP inference helpers are not importable.") from _LINGBOT_PIPELINE_IMPORT_ERROR
        fsdp_mesh = init_fsdp_inference_mesh()

    _phase("init_parallel_done")
    dtype_map = _make_dtype_map(args)
    defer_transformer_to_device = fsdp_mesh is not None
    pipe, engine_name = _load_pipe(
        args,
        dtype_map,
        defer_transformer_to_device=defer_transformer_to_device,
    )
    _configure_pipeline_logs(pipe)
    component_dtypes = _component_dtypes(pipe)
    expected_vae_dtype = _dtype_name(dtype_map["vae"])
    if component_dtypes.get("vae") != expected_vae_dtype:
        raise AssertionError(
            f"VAE dtype mismatch: requested {expected_vae_dtype}, got {component_dtypes}"
        )
    device = _default_device()
    if _sharded_load_enabled() or _bcast_weights_enabled():
        # Distribute base transformer weights (sharded parse or rank-0 load) over NVLink.
        _distribute_transformer_weights(pipe.transformer, args.model_dir, args.transformer_subfolder)
    if args.context_parallel_degree > 1:
        _enable_context_parallel(
            pipe.transformer,
            args.context_parallel_degree,
            args.context_parallel_ulysses_anything,
            context_parallel_rank,
            device,
            context_parallel_mesh,
            args.cfg_parallel_degree == 1,
        )
    base_fsdp_info = _apply_fsdp_inference_if_requested(pipe, args.enable_fsdp_inference, fsdp_mesh)
    _phase("base_pipe_loaded")

    # Stage-transition scheduling: the refiner is only needed AFTER base denoise, but
    # loading it (~108s of lustre IO + CPU) currently runs serially BEFORE base denoise
    # (~89s of pure-GPU work). With LINGBOT_OVERLAP_REFINER_LOAD=1 we load the refiner
    # to CPU in a background thread (zero CUDA calls) that overlaps base denoise, then
    # finalize (device move + CP + FSDP) on the main thread once base denoise is done.
    overlap_refiner_load = (
        _env_flag("LINGBOT_OVERLAP_REFINER_LOAD")
        and bool(args.run_refiner)
        and _refiner_model_available(args.refiner_model_dir, args.refiner_transformer_subfolder)
    )
    refiner_state = None
    refiner_load_thread = None
    refiner_fsdp_info = None
    if overlap_refiner_load:
        import threading

        _refiner_holder: dict[str, Any] = {}

        def _bg_load_refiner() -> None:
            _refiner_holder["state"] = _maybe_preload_refiner(
                args,
                dtype_map,
                device,
                context_parallel_rank,
                context_parallel_mesh,
                defer_transformer_to_device,
                cpu_only=True,
            )

        refiner_load_thread = threading.Thread(
            target=_bg_load_refiner, name="refiner-cpu-load", daemon=True
        )
        refiner_load_thread.start()
        # Placeholder so `should_cache_conditions` below sees the refiner as available.
        refiner_state = {
            "refiner_requested": True,
            "refiner_available": True,
            "refiner_skipped_reason": None,
            "pipe": None,
            "engine_name": None,
            "component_dtypes": None,
            "_holder": _refiner_holder,
        }
    else:
        refiner_state = _maybe_preload_refiner(
            args,
            dtype_map,
            device,
            context_parallel_rank,
            context_parallel_mesh,
            defer_transformer_to_device,
        )
        if refiner_state["pipe"] is not None:
            _configure_pipeline_logs(refiner_state["pipe"])
        if refiner_state["pipe"] is not None:
            refiner_fsdp_info = _apply_fsdp_inference_if_requested(
                refiner_state["pipe"],
                args.enable_fsdp_inference,
                fsdp_mesh,
            )
    if rank == 0 and refiner_state["refiner_requested"] and not refiner_state["refiner_available"]:
        print(
            "WARNING: refiner requested but unavailable; "
            f"reason={refiner_state['refiner_skipped_reason']} "
            f"refiner_model_dir={args.refiner_model_dir}",
            flush=True,
        )

    _phase("refiner_preloaded")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    condition_cache = None
    input_image = None
    ti2v_image_tensor_cpu = None
    should_cache_conditions = (
        args.reuse_condition_features
        or bool(refiner_state["refiner_available"])
        or args.batch_cfg
        or args.null_cond_clone_zero
    )
    if args.mode == "ti2v":
        input_image = (
            Image.open(args.image).convert("RGB")
            if args.image
            else _make_default_image(args.height, args.width)
        )
        if should_cache_conditions:
            condition_cache, ti2v_image_tensor_cpu = _cache_ti2v_prompt_conditions(
                pipe,
                args.prompt,
                args.negative_prompt,
                input_image,
                height=args.height,
                width=args.width,
                device=device,
                null_cond_clone_zero=args.null_cond_clone_zero,
            )
    elif should_cache_conditions:
        condition_cache = _cache_prompt_conditions(
            pipe,
            args.prompt,
            args.negative_prompt,
            device=device,
            null_cond_clone_zero=args.null_cond_clone_zero,
        )
    call_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        shift=args.shift,
        generator=generator,
        output_type="np",
        **_condition_call_kwargs(condition_cache, device),
    )
    call_kwargs["batch_cfg"] = args.batch_cfg
    if args.mode != "ti2v":
        call_kwargs["null_cond_clone_zero"] = args.null_cond_clone_zero
    if args.mode == "ti2v":
        call_kwargs["image"] = input_image
        if ti2v_image_tensor_cpu is not None:
            call_kwargs["image_tensor"] = ti2v_image_tensor_cpu.to(device=device)
    if args.cfg_parallel_degree > 1:
        call_kwargs["cfg_parallel_group"] = cfg_parallel_group

    if rank == 0:
        print(
            "runtime "
            f"engine={engine_name} model_dir={args.model_dir} mode={args.mode} device={device} "
            f"rank={rank}/{world_size} local_rank={local_rank} "
            f"cfg_branch_rank={cfg_branch_rank} context_parallel_rank={context_parallel_rank} "
            f"cfg_parallel_degree={args.cfg_parallel_degree} "
            f"context_parallel_degree={args.context_parallel_degree} "
            f"height={args.height} width={args.width} frames={args.num_frames} steps={args.steps} "
            f"guidance={args.guidance_scale} shift={args.shift} seed={args.seed} "
            f"attn_backend={os.environ.get('DIFFUSERS_ATTN_BACKEND')} "
            f"allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
            f"fsdp_inference={base_fsdp_info} "
            f"component_dtypes={component_dtypes}",
            flush=True,
        )

    _phase("base_conditions_cached")
    set_lingbot_pisa_phase("base")
    with torch.no_grad():
        result = pipe(**call_kwargs)
    _phase("base_denoise_done")
    # Overlap path: the refiner finished loading to CPU on the background thread during
    # base denoise. Join it and finalize GPU setup (device move + CP + FSDP) now.
    if refiner_load_thread is not None:
        refiner_load_thread.join()
        loaded_state = refiner_state["_holder"].get("state")
        if loaded_state is None:
            raise RuntimeError("Background refiner load produced no state.")
        refiner_state = loaded_state
        if refiner_state["pipe"] is not None:
            _configure_pipeline_logs(refiner_state["pipe"])
            _finalize_preloaded_refiner(
                refiner_state,
                args,
                device,
                context_parallel_rank,
                context_parallel_mesh,
            )
            refiner_fsdp_info = _apply_fsdp_inference_if_requested(
                refiner_state["pipe"],
                args.enable_fsdp_inference,
                fsdp_mesh,
            )
        _phase("refiner_finalized")
    frames = _extract_frames(result) if rank == 0 else None
    if rank == 0:
        _save_frames(
            frames,
            args.mode,
            Path(args.output),
            args.fps,
        )
    _phase("base_vae_saved")

    if not refiner_state["refiner_requested"] or not refiner_state["refiner_available"]:
        _sync_parallel_if_needed()
        _destroy_parallel_if_needed()
        return

    _sync_parallel_if_needed()
    del result, frames, call_kwargs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Single-GPU two-stage otherwise OOMs at the 1080p refiner because the runner
    # keeps BOTH 30B DiTs resident (~120GB). base and refiner run sequentially, so
    # evict the base pipeline to CPU here — only the 60GB refiner stays on GPU.
    # Opt-in (LINGBOT_OFFLOAD_BASE_BEFORE_REFINER=1); skipped under FSDP (already sharded).
    if _env_flag("LINGBOT_OFFLOAD_BASE_BEFORE_REFINER") and not args.enable_fsdp_inference:
        _log_progress("offloading base pipeline to CPU before refiner")
        try:
            _inner_diffusers_pipe(pipe).to("cpu")
        except Exception as exc:  # pragma: no cover - best-effort memory relief
            _log_progress(f"base offload skipped: {exc}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if load_refiner_video_tensor is None or prepare_refiner_latent is None:
        raise RuntimeError("Refiner helpers are not importable.") from _LINGBOT_PIPELINE_IMPORT_ERROR
    refiner_pipe = refiner_state["pipe"]
    if refiner_pipe is None:
        raise RuntimeError("Refiner was marked available but was not preloaded.")
    refiner_engine_name = str(refiner_state["engine_name"])
    refiner_component_dtypes = refiner_state["component_dtypes"]

    refiner_generator = torch.Generator(device=device).manual_seed(args.seed)
    lowres_video, lowres_meta = load_refiner_video_tensor(
        args.output,
        args.refiner_height,
        args.refiner_width,
        sample_fps=args.refiner_sample_fps,
        vae_tc=getattr(refiner_pipe, "vae_scale_factor_temporal", 4),
        max_frames=args.refiner_max_video_frames,
    )

    # TI2V always refines with the clean input first frame as a fixed frame-0
    # latent: injected into the initial latent here and re-clamped after every
    # scheduler step via `cond_latent`.
    first_frame_condition_enabled = args.mode == "ti2v"
    refiner_cond_latent = None
    with torch.no_grad():
        x_up = refiner_pipe.encode_video_latent(lowres_video, generator=refiner_generator)
        if first_frame_condition_enabled:
            condition_pixels = load_first_frame_condition_tensor(
                args.image,
                args.refiner_height,
                args.refiner_width,
                geometry_height=args.height,
                geometry_width=args.width,
            )
            clean_x0 = refiner_pipe.encode_video_latent(
                condition_pixels,
                generator=refiner_generator,
            )
            refiner_cond_latent = clean_x0[:, :, 0:1].contiguous()
            x_up[:, :, 0:1] = refiner_cond_latent.to(dtype=x_up.dtype)
            del condition_pixels, clean_x0
        noise = torch.randn(
            x_up.shape,
            device=device,
            dtype=x_up.dtype,
            generator=refiner_generator,
        )
        initial_latent = prepare_refiner_latent(x_up, noise, args.refiner_t_thresh)
    _phase("refiner_vae_encode_done")
    del lowres_video, x_up, noise
    refiner_null_clone_zero = not args.refiner_no_null_cond_clone_zero
    # The ti2v condition cache carries image tokens; the refiner conditions on
    # text only, so it rebuilds t2v-style embeddings instead of reusing it.
    refiner_condition_cache = None if args.mode == "ti2v" else condition_cache
    should_cache_refiner_conditions = not (
        args.cfg_parallel_degree > 1 and cfg_branch_rank == 1 and not refiner_null_clone_zero
    )
    if refiner_condition_cache is None and should_cache_refiner_conditions:
        refiner_condition_cache = _cache_prompt_conditions(
            refiner_pipe,
            args.prompt,
            args.negative_prompt,
            device=device,
            null_cond_clone_zero=refiner_null_clone_zero,
        )
    elif refiner_null_clone_zero:
        refiner_condition_cache = dict(refiner_condition_cache)
        refiner_condition_cache["negative_prompt_embeds"] = torch.zeros_like(
            refiner_condition_cache["prompt_embeds"]
        )
        refiner_condition_cache["negative_prompt_mask"] = refiner_condition_cache["prompt_mask"].clone()

    if rank == 0:
        print(
            "runtime "
            f"engine={refiner_engine_name} model_dir={args.refiner_model_dir} mode=refiner "
            f"height={args.refiner_height} width={args.refiner_width} "
            f"frames={lowres_meta['sample_frame']} steps={args.refiner_steps} "
            f"guidance={args.refiner_guidance_scale} shift={args.refiner_shift} "
            f"t_thresh={args.refiner_t_thresh} tail_steps={args.refiner_sigma_tail_steps} "
            f"seed={args.seed} "
            f"batch_cfg={args.refiner_batch_cfg} "
            f"first_frame_condition={first_frame_condition_enabled} "
            f"fsdp_inference={refiner_fsdp_info} "
            f"component_dtypes={refiner_component_dtypes}",
            flush=True,
        )

    refiner_call_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.refiner_height,
        width=args.refiner_width,
        num_frames=int(lowres_meta["sample_frame"]),
        num_inference_steps=args.refiner_steps,
        guidance_scale=args.refiner_guidance_scale,
        shift=args.refiner_shift,
        t_thresh=args.refiner_t_thresh,
        refiner_sigma_tail_steps=args.refiner_sigma_tail_steps,
        generator=refiner_generator,
        latents=initial_latent,
        output_type="np",
        batch_cfg=args.refiner_batch_cfg,
        null_cond_clone_zero=refiner_null_clone_zero,
        offload_vae_during_denoise=args.refiner_offload_vae_during_denoise,
        **_condition_call_kwargs(refiner_condition_cache, device),
    )
    if refiner_cond_latent is not None:
        refiner_call_kwargs["cond_latent"] = refiner_cond_latent
    if args.cfg_parallel_degree > 1:
        refiner_call_kwargs["cfg_parallel_group"] = cfg_parallel_group

    _phase("refiner_conditions_cached")
    set_lingbot_pisa_phase("refiner")
    with torch.no_grad():
        refiner_result = refiner_pipe(**refiner_call_kwargs)
    _phase("refiner_denoise_done")
    if rank == 0:
        refiner_frames = _extract_frames(refiner_result)
        _save_frames(
            refiner_frames,
            args.mode,
            Path(args.refiner_output or str(Path(args.output).with_name(Path(args.output).stem + "_refined.mp4"))),
            args.refiner_fps,
        )
    _phase("refiner_vae_saved")

    _sync_parallel_if_needed()
    _destroy_parallel_if_needed()


if __name__ == "__main__":
    main()
