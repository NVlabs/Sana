from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers import DiffusionPipeline

from lingbot_video.model_paths import model_component_dir
from lingbot_video.pipeline_lingbot_video import (
    DEFAULT_NEGATIVE_PROMPT,
    LingBotVideoPipeline,
    LingBotVideoPipelineOutput,
)
from lingbot_video.pipeline_lingbot_video_i2v import LingBotVideoImageToVideoPipeline
from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel

try:
    from sglang.multimodal_gen import registry as sglang_registry
except Exception:  # pragma: no cover - SGLang is an optional deployment dep
    sglang_registry = None

try:
    from sglang.multimodal_gen.runtime.server_args import (
        Backend,
        ServerArgs,
        set_global_server_args,
    )
except Exception:  # pragma: no cover - SGLang is an optional deployment dep
    Backend = None
    ServerArgs = None
    set_global_server_args = None


@dataclass
class LingBotVideoNativePipelineConfig:
    """Minimal config object for the LingBotVideo native adapter."""

    flow_shift: float = 3.0


@dataclass
class LingBotVideoNativeSamplingParams:
    """Default sampling parameters for the native pipeline."""

    prompt: str | None = None
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 24
    num_inference_steps: int = 40
    guidance_scale: float = 3.0
    seed: int = 42
    shift: float = 3.0


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


def _module_dtype_name(module: Any) -> str | None:
    if module is None:
        return None
    dtype = getattr(module, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return str(dtype).replace("torch.", "")
    if isinstance(module, torch.nn.Module):
        try:
            return str(next(module.parameters()).dtype).replace("torch.", "")
        except StopIteration:
            return None
    return None


def _default_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda", torch.cuda.current_device())


def _pipeline_class_for_mode(mode: str) -> type[DiffusionPipeline]:
    if mode == "ti2v":
        return LingBotVideoImageToVideoPipeline
    return LingBotVideoPipeline


def _load_lingbot_diffusers_pipe(
    model_path: str | Path,
    dtype_map: dict[str, torch.dtype] | torch.dtype | None,
    mode: str,
    transformer_subfolder: str,
) -> DiffusionPipeline:
    model_root = Path(model_path)
    model_component_dir(model_root, transformer_subfolder)
    transformer_dtype = (
        dtype_map.get("transformer", dtype_map.get("default", torch.float32))
        if isinstance(dtype_map, dict)
        else dtype_map
    )
    transformer = LingBotVideoTransformer3DModel.from_pretrained(
        str(model_root),
        subfolder=transformer_subfolder,
        torch_dtype=transformer_dtype,
    )
    load_kwargs: dict[str, Any] = {
        "transformer": transformer,
        "trust_remote_code": True,
    }
    if dtype_map is not None:
        load_kwargs["torch_dtype"] = dtype_map
    with _patch_qwen3vl_from_pretrained():
        pipe = _pipeline_class_for_mode(mode).from_pretrained(
            str(model_root),
            **load_kwargs,
        )
    return pipe.to(_default_device())


class LingBotVideoNativeExecutionStage:
    """Single-stage LingBotVideo execution."""

    def __init__(self, diffusers_pipe: DiffusionPipeline):
        self.diffusers_pipe = diffusers_pipe

    def __call__(self, **kwargs: Any) -> LingBotVideoPipelineOutput:
        return self.forward(**kwargs)

    @torch.no_grad()
    def forward(self, **kwargs: Any) -> LingBotVideoPipelineOutput:
        output = self.diffusers_pipe(**kwargs)
        frames = output.frames if hasattr(output, "frames") else output[0]
        return LingBotVideoPipelineOutput(frames=frames)


class LingBotVideoNativePipeline:
    """SGLang-native LingBotVideo pipeline adapter."""

    pipeline_name = "LingBotVideoNativePipeline"
    pipeline_config_cls = LingBotVideoNativePipelineConfig
    sampling_params_cls = LingBotVideoNativeSamplingParams
    is_video_pipeline = True

    def __init__(
        self,
        diffusers_pipe: DiffusionPipeline | str | Path,
        *,
        model_path: str | Path | None = None,
        server_args: Any | None = None,
        dtype_map: dict[str, torch.dtype] | torch.dtype | None = None,
        mode: str = "t2v",
        transformer_subfolder: str = "transformer",
        **_: Any,
    ):
        if isinstance(diffusers_pipe, (str, Path)):
            model_path = diffusers_pipe if model_path is None else model_path
            diffusers_pipe = _load_lingbot_diffusers_pipe(
                model_path,
                dtype_map,
                mode=mode,
                transformer_subfolder=transformer_subfolder,
            )
        if model_path is None:
            raise ValueError("model_path is required for LingBotVideoNativePipeline")
        self.diffusers_pipe = diffusers_pipe
        self.model_path = str(model_path)
        self.server_args = server_args
        self.execution_stage = LingBotVideoNativeExecutionStage(diffusers_pipe)
        self.modules = {"lingbot_video_pipeline": diffusers_pipe}
        self.memory_usages: dict[str, float] = {}

    @property
    def transformer(self) -> Any:
        return getattr(self.diffusers_pipe, "transformer", None)

    @property
    def text_encoder(self) -> Any:
        return getattr(self.diffusers_pipe, "text_encoder", None)

    @property
    def vae(self) -> Any:
        return getattr(self.diffusers_pipe, "vae", None)

    @property
    def scheduler(self) -> Any:
        return getattr(self.diffusers_pipe, "scheduler", None)

    def encode_prompt(self, *args: Any, **kwargs: Any) -> Any:
        return self.diffusers_pipe.encode_prompt(*args, **kwargs)

    def encode_video_latent(self, *args: Any, **kwargs: Any) -> Any:
        return self.diffusers_pipe.encode_video_latent(*args, **kwargs)

    @classmethod
    def from_diffusers_pipe(
        cls,
        diffusers_pipe: DiffusionPipeline,
        *,
        model_path: str | Path,
        server_args: Any | None = None,
    ) -> "LingBotVideoNativePipeline":
        return cls(diffusers_pipe, model_path=model_path, server_args=server_args)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        torch_dtype: dict[str, torch.dtype] | torch.dtype | None = None,
        server_args: Any | None = None,
        mode: str = "t2v",
        transformer_subfolder: str = "transformer",
        **kwargs: Any,
    ) -> "LingBotVideoNativePipeline":
        return cls(
            model_path,
            model_path=model_path,
            server_args=server_args,
            dtype_map=torch_dtype,
            mode=mode,
            transformer_subfolder=transformer_subfolder,
            **kwargs,
        )

    @torch.no_grad()
    def __call__(
        self,
        *,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 40,
        guidance_scale: float = 3.0,
        shift: float = 3.0,
        generator: torch.Generator | None = None,
        image: Any | None = None,
        output_type: str = "np",
        **kwargs: Any,
    ) -> LingBotVideoPipelineOutput:
        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            generator=generator,
            output_type=output_type,
        )
        if image is not None:
            call_kwargs["image"] = image
        call_kwargs.update(kwargs)
        return self.execution_stage(**call_kwargs)

    def component_dtypes(self) -> dict[str, str | None]:
        return {
            "transformer": _module_dtype_name(self.transformer),
            "text_encoder": _module_dtype_name(self.text_encoder),
            "vae": _module_dtype_name(self.vae),
        }


def register_lingbot_native_pipeline() -> bool:
    """Register the adapter in SGLang's native registry when SGLang is present."""

    if sglang_registry is None:
        return False

    try:
        sglang_registry._PIPELINE_REGISTRY[LingBotVideoNativePipeline.pipeline_name] = (  # type: ignore[attr-defined]
            LingBotVideoNativePipeline
        )
        sglang_registry._PIPELINE_CONFIG_REGISTRY[LingBotVideoNativePipeline.pipeline_name] = (  # type: ignore[attr-defined]
            LingBotVideoNativePipelineConfig,
            LingBotVideoNativeSamplingParams,
        )
    except Exception:
        return False
    return True


def load_lingbot_native_pipeline(
    model_dir: Path,
    dtype_map: dict[str, torch.dtype],
    mode: str = "t2v",
    transformer_subfolder: str = "transformer",
) -> LingBotVideoNativePipeline:
    """Load a LingBotVideo native adapter from a diffusers-format model dir."""

    server_args = None
    if Backend is not None and ServerArgs is not None and set_global_server_args is not None:
        try:
            server_args = ServerArgs(
                model_path=str(model_dir),
                backend=Backend.SGLANG,
                trust_remote_code=True,
                pipeline_class_name=LingBotVideoNativePipeline.pipeline_name,
                pipeline_config=LingBotVideoNativePipelineConfig(),
                dit_cpu_offload=False,
                dit_layerwise_offload=False,
                layerwise_offload_components=[],
                text_encoder_cpu_offload=False,
                image_encoder_cpu_offload=False,
                vae_cpu_offload=False,
            )
            set_global_server_args(server_args)
        except Exception:
            server_args = None

    register_lingbot_native_pipeline()

    return LingBotVideoNativePipeline.from_pretrained(
        model_path=model_dir,
        torch_dtype=dtype_map,
        server_args=server_args,
        mode=mode,
        transformer_subfolder=transformer_subfolder,
    )
