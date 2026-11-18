#!/usr/bin/env python3
"""Run the distilled LTX-2.5 pipeline with Stage-2 Sol-Attn on RTX 5090."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from .attention import LTX25Stage2SolAttention
from .exact_adaln import LTX25ExactAdaLN
from .memory import FeedForwardChunking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=("bf16", "nvfp4"), required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--nvfp4-embedding-source", type=Path)
    parser.add_argument("--exact-adaln-table", type=Path)
    parser.add_argument("official_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.official_args[:1] == ["--"]:
        args.official_args = args.official_args[1:]
    if not args.official_args:
        parser.error("official LTX-2.5 arguments are required after --")
    if args.pipeline == "nvfp4" and (
        args.nvfp4_embedding_source is None or args.exact_adaln_table is None
    ):
        parser.error("NVFP4 requires --nvfp4-embedding-source and --exact-adaln-table")
    return args


def find_keyframe_embedding_key(checkpoint) -> str:
    keys = [
        key
        for key in checkpoint.keys()
        if key.endswith("keyframes_abs_pos_embedding")
    ]
    if len(keys) != 1:
        raise RuntimeError("expected one keyframe embedding")
    return keys[0]


def install_nvfp4_embedding(source: Path) -> str:
    """Inject the BF16 keyframe embedding omitted from the NVFP4 checkpoint."""

    import torch
    from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
    from ltx_pipelines.utils.quantization_factory import QuantizationKind
    from safetensors import safe_open

    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        source_key = find_keyframe_embedding_key(checkpoint)
        embedding = checkpoint.get_tensor(source_key).clone()

    original_to_policy = QuantizationKind.to_policy

    def repaired_to_policy(self: QuantizationKind, checkpoint_path: str | None = None):
        policy = original_to_policy(self, checkpoint_path)
        if self is not QuantizationKind.NVFP4_PREQUANT:
            return policy

        def inject(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
            results = [KeyValueOperationResult(key, value)]
            if key == "patchify_proj.bias":
                results.append(
                    KeyValueOperationResult(
                        "keyframes_abs_pos_embedding",
                        embedding.to(device=value.device, non_blocking=True),
                    )
                )
            return results

        injection = SDOps("LTX25_NVFP4_KEYFRAME_EMBEDDING").with_kv_operation(
            key_suffix="patchify_proj.bias",
            operation=inject,
        )
        sd_ops = SDOps(
            name=f"{policy.sd_ops.name}+{injection.name}",
            mapping=(*policy.sd_ops.mapping, *injection.mapping),
            allowed_keys=policy.sd_ops.allowed_keys,
        )
        return replace(policy, sd_ops=sd_ops)

    QuantizationKind.to_policy = repaired_to_policy
    return source_key


def sync() -> None:
    import torch

    torch.cuda.synchronize()


class Timings:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = defaultdict(list)

    def measure(self, name: str, function, *args, **kwargs):
        sync()
        start = time.perf_counter()
        result = function(*args, **kwargs)
        sync()
        self.values[name].append(time.perf_counter() - start)
        return result

    def total(self, name: str) -> float:
        return sum(self.values.get(name, ()))


class TimedCallable:
    def __init__(self, name: str, wrapped: Any, timings: Timings) -> None:
        self.name = name
        self.wrapped = wrapped
        self.timings = timings

    def __call__(self, *args, **kwargs):
        return self.timings.measure(self.name, self.wrapped, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)


class TimedVideoDecoder(TimedCallable):
    def __call__(self, *args, **kwargs) -> Iterator[Any]:
        iterator = self.timings.measure("video_decoder_build", self.wrapped, *args, **kwargs)

        def measured() -> Iterator[Any]:
            sync()
            start = time.perf_counter()
            try:
                yield from iterator
            finally:
                sync()
                self.timings.values[self.name].append(time.perf_counter() - start)

        return measured()


class TimedStage:
    def __init__(
        self,
        name: str,
        stage: Any,
        timings: Timings,
        attention: LTX25Stage2SolAttention,
        exact_adaln: LTX25ExactAdaLN | None = None,
    ) -> None:
        self.name = name
        self.stage = stage
        self.timings = timings
        self.attention = attention
        self.exact_adaln = exact_adaln
        self.call_index = 0

    def __call__(self, *args, **kwargs):
        from ltx_pipelines.utils.samplers import euler_denoising_loop

        if self.name == "stage":
            self.call_index += 1
            stage_index = self.call_index
            stage_name = f"stage_{stage_index}"
        else:
            stage_index = int(self.name.rsplit("_", 1)[1])
            stage_name = self.name
        original_loop = kwargs.get("loop") or euler_denoising_loop

        def timed_loop(*loop_args, **loop_kwargs):
            transformer = loop_kwargs["transformer"]
            if self.exact_adaln is not None and stage_index == 2:
                self.exact_adaln.install(transformer)
            try:
                return self.timings.measure(
                    f"{stage_name}_denoise",
                    original_loop,
                    *loop_args,
                    **loop_kwargs,
                )
            finally:
                if self.exact_adaln is not None and stage_index == 2:
                    self.exact_adaln.uninstall(transformer)

        kwargs["loop"] = timed_loop
        with self.attention.stage2(stage_index == 2):
            return self.timings.measure(stage_name, self.stage, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stage, name)


def parse_official_args(argv: list[str]) -> argparse.Namespace:
    from ltx_pipelines.utils.args import (
        add_generated_keyframes_arg,
        default_2_stage_distilled_arg_parser,
        resolve_cli_params,
    )

    old_argv = sys.argv
    try:
        sys.argv = ["ltx25-distilled", *argv]
        params = resolve_cli_params(distilled=True)
        parser = default_2_stage_distilled_arg_parser(params=params, supports_auto_duration=True)
        return add_generated_keyframes_arg(parser).parse_args(argv)
    finally:
        sys.argv = old_argv


def build_pipeline(args: argparse.Namespace):
    from ltx_pipelines.distilled import DistilledPipeline

    return DistilledPipeline(
        model_paths=args.model_paths,
        spatial_upsampler_path=args.spatial_upsampler_path,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
        diffvae_optimization=args.diffvae_optimization,
    )


def instrument_pipeline(
    pipeline: Any,
    attention: LTX25Stage2SolAttention,
    exact_adaln: LTX25ExactAdaLN | None,
    timings: Timings,
) -> None:
    pipeline.stage = TimedStage(
        "stage",
        pipeline.stage.with_attention(attention),
        timings,
        attention,
        exact_adaln,
    )
    pipeline.prompt_encoder = TimedCallable("prompt_encode", pipeline.prompt_encoder, timings)
    pipeline.upsampler = TimedCallable("video_upsample", pipeline.upsampler, timings)
    pipeline.video_decoder = TimedVideoDecoder("video_decode", pipeline.video_decoder, timings)
    pipeline.audio_decoder = TimedCallable("audio_decode", pipeline.audio_decoder, timings)


def run_pipeline(pipeline: Any, args: argparse.Namespace) -> None:
    import torch
    from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
    from ltx_pipelines.utils.media_io import (
        encode_video,
        resolve_hdr_color_space,
        vae_dtype_for_hdr,
    )

    hdr = resolve_hdr_color_space(images=args.images, hdr=args.hdr)
    video, audio, num_frames, tiling_config = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        vae_dtype=vae_dtype_for_hdr(hdr, torch.bfloat16),
        color_space=hdr,
        enhance_prompt=args.enhance_prompt,
        enhance_static_cache=args.enhance_static_cache,
        tiling_config=AUTO_TILING,
        generated_keyframes=args.num_generated_keyframes,
    )
    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=get_video_chunks_number(num_frames, tiling_config),
        color_space=hdr,
    )


def main() -> None:
    args = parse_args()
    os.environ["SOL_ATTN_STRICT"] = "1"

    import torch

    embedding_key = None
    if args.pipeline == "nvfp4":
        embedding_key = install_nvfp4_embedding(args.nvfp4_embedding_source)
    official = parse_official_args(args.official_args)
    exact_adaln = (
        LTX25ExactAdaLN(args.exact_adaln_table, Path(official.transformer_path))
        if args.pipeline == "nvfp4"
        else None
    )
    ffn_chunking = FeedForwardChunking() if args.pipeline == "bf16" else None
    if ffn_chunking is not None:
        ffn_chunking.install()

    timings = Timings()
    attention = LTX25Stage2SolAttention()
    try:
        pipeline = build_pipeline(official)
        instrument_pipeline(pipeline, attention, exact_adaln, timings)
        torch.cuda.reset_peak_memory_stats()
        sync()
        start = time.perf_counter()
        with torch.inference_mode():
            run_pipeline(pipeline, official)
        sync()
        e2e_seconds = time.perf_counter() - start
    finally:
        if ffn_chunking is not None:
            ffn_chunking.uninstall()

    metrics = {
        "pipeline": args.pipeline,
        "stage_1_seconds": timings.total("stage_1"),
        "stage_2_seconds": timings.total("stage_2"),
        "video_vae_seconds": timings.total("video_decoder_build") + timings.total("video_decode"),
        "e2e_seconds": e2e_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "attention": attention.stats(),
        "exact_adaln": exact_adaln.stats() if exact_adaln is not None else None,
        "nvfp4_embedding_key": embedding_key,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
