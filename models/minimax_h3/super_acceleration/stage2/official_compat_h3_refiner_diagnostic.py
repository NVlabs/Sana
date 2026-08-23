#!/usr/bin/env python3
"""Two-case H3 -> LTX-2.5 Stage-2 compatibility diagnostic.

This intentionally does not replace the formal A/B/C benchmark.  It keeps the
current H3 pixel round-trip (TAEHV encode followed by the learned LTX spatial
upsampler), but restores the parts of the official LTX-2.5 two-stage Stage 2
that the formal ablation omitted:

* a full-resolution first-frame conditioning encoded by the LTX-2.5 Video VAE;
* an H3 audio latent encoded by the LTX-2.5 Audio VAE and jointly denoised;
* final decoding by the official LTX-2.5 Video VAE decoder.

The jointly denoised audio output is deliberately discarded.  The original H3
audio is muxed into the result, matching the use of the Stage-1 audio output in
the official two-stage pipeline without adding an avoidable audio decode loss.
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from dataclasses import asdict, is_dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.loader.registry import ModelRegistry
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_core.tiling import DimensionSizeConfig, TileSizeConfig
from ltx_core.tools import AudioLatentTools
from ltx_core.types import AudioLatentShape, VideoPixelShape
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.blocks import AudioConditioner, ImageConditioner, VideoDecoder
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    audio_latent_from_file,
    combined_image_conditionings,
    create_noised_state,
    ensure_tiling_config,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video
from ltx_pipelines.utils.samplers import euler_denoising_loop

import refiner_encoder_ablation_single_gpu as base


STAGE2_SCHEDULES = {
    "default": base.STAGE2_SIGMAS,
    "skip_0p9": (0.725, 0.421875, 0.0),
}
OUTPUT_DECODERS = ("official_vae", "taehv")


class OfficialCompatRefiner(base.ResidentRefiner):
    """Add official image/audio conditioning and full decoding to the H3 arm."""

    @torch.inference_mode()
    def __init__(self, args: argparse.Namespace) -> None:
        # The formal resident runner installs post-load guards at the end of its
        # constructor.  Defer those hooks while this diagnostic loads the three
        # official components that are absent from the formal A arm.
        self._defer_weight_guards = True
        self.stage2_audio_enabled = bool(getattr(args, "stage2_audio", True))
        self.retain_allocator_cache = bool(
            getattr(args, "retain_allocator_cache", False)
        )
        self.output_decoder_backend = str(getattr(args, "output_decoder", "official_vae"))
        self.stage2_schedule_name = str(getattr(args, "stage2_schedule", "default"))
        if self.output_decoder_backend not in OUTPUT_DECODERS:
            raise ValueError(f"unsupported output decoder: {self.output_decoder_backend}")
        if self.stage2_schedule_name not in STAGE2_SCHEDULES:
            raise ValueError(f"unsupported Stage-2 schedule: {self.stage2_schedule_name}")
        super().__init__(args)
        self.stage2_sigmas = STAGE2_SCHEDULES[self.stage2_schedule_name]
        self.sigmas = torch.tensor(
            self.stage2_sigmas, dtype=torch.float32, device=self.device
        )

        registry = ModelRegistry()
        image_conditioner = ImageConditioner(
            str(args.video_vae),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=AllocatorTrimStrategy.DEFER,
        )
        # Resolve before measured requests so model-version CRF detection is not
        # hidden inside a sample.
        self.image_conditioning_crf = image_conditioner.default_image_crf
        self.image_encoder = (
            image_conditioner._encoder_builder.build(device=self.device, dtype=self.dtype)
            .eval()
            .requires_grad_(False)
        )

        self.audio_encoder = None
        if self.stage2_audio_enabled:
            audio_conditioner = AudioConditioner(
                str(args.audio_vae),
                self.dtype,
                self.device,
                registry=registry,
                alloc_trim_strategy=AllocatorTrimStrategy.DEFER,
            )
            self.audio_encoder = (
                audio_conditioner._encoder_builder.build(device=self.device, dtype=self.dtype)
                .eval()
                .requires_grad_(False)
            )

        pixel_shape = VideoPixelShape(
            batch=1,
            frames=base.FRAME_COUNT,
            height=base.HEIGHT,
            width=base.WIDTH,
            fps=base.FPS,
        )
        self.video_decoder = None
        self.tiling_config = None
        self.last_decoder_stream_info: dict[str, int] | None = None
        if self.output_decoder_backend == "official_vae":
            decoder_block = VideoDecoder(
                str(args.video_vae),
                self.dtype,
                self.device,
                registry=registry,
                alloc_trim_strategy=AllocatorTrimStrategy.DEFER,
            )
            # Match the native pipeline's ordering exactly: AUTO_TILING is
            # resolved before the lazy VideoDecoder builds its weights.  The
            # recommendation already subtracts the checkpoint's estimated
            # decoder weight bytes, so resolving it after a resident decoder is
            # built would double-count those weights and choose unnecessarily
            # aggressive tiles.
            auto_tiling_config = ensure_tiling_config(
                AUTO_TILING,
                scale_factors=tiling_scale_factors_for_vae(str(args.video_vae)),
                vae_checkpoint_path=str(args.video_vae),
                video_shape=pixel_shape,
                diffvae_optimization=decoder_block.diffvae_optimization,
                device=self.device,
            )
            forced_spatial_tile = int(
                getattr(args, "diffvae_spatial_tile_size", 0)
            )
            if forced_spatial_tile:
                if not isinstance(auto_tiling_config, TileSizeConfig):
                    raise RuntimeError("DiffVAE AUTO_TILING did not return TileSizeConfig")
                self.tiling_config = TileSizeConfig(
                    frames=auto_tiling_config.frames,
                    height=DimensionSizeConfig(
                        tile_size=forced_spatial_tile,
                        overlap=auto_tiling_config.height.overlap,
                    ),
                    width=DimensionSizeConfig(
                        tile_size=forced_spatial_tile,
                        overlap=auto_tiling_config.width.overlap,
                    ),
                )
                self.tiling_config.validate(
                    tiling_scale_factors_for_vae(str(args.video_vae)), pixel_shape
                )
                self.tiling_source = (
                    "official_TileSizeConfig_checkpoint_overlap_spatial_override"
                )
            else:
                self.tiling_config = auto_tiling_config
                self.tiling_source = "official_AUTO_TILING"
            self.video_decoder = (
                decoder_block._decoder_builder.build(device=self.device, dtype=self.dtype)
                .eval()
                .requires_grad_(False)
            )
        self.audio_tools = None
        if self.stage2_audio_enabled:
            audio_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
            self.audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), audio_shape)
        self.first_frame_root = args.first_frame_root.resolve()
        self.video_vae_path = args.video_vae.resolve()
        self.audio_vae_path = args.audio_vae.resolve()

        official_residency = {
            "official_ltx25_image_encoder": base._module_residency(
                "official_ltx25_image_encoder", self.image_encoder
            ),
        }
        if self.video_decoder is not None:
            official_residency["official_ltx25_video_decoder"] = base._module_residency(
                "official_ltx25_video_decoder", self.video_decoder
            )
        if self.audio_encoder is not None:
            official_residency["official_ltx25_audio_encoder"] = base._module_residency(
                "official_ltx25_audio_encoder", self.audio_encoder
            )
        self.residency.update(official_residency)
        self._defer_weight_guards = False
        # The base constructor deliberately deferred both guards while the
        # official image/audio encoders and full decoder were still loading.
        # Install them only after the complete official-compatible fleet is
        # resident so measured requests fail closed on accidental reloads.
        self._install_safetensors_load_guard()
        self._install_torch_load_guard()
        gc.collect()
        torch.cuda.empty_cache()
        base._cuda_sync()

    def _install_safetensors_load_guard(self) -> None:
        if not getattr(self, "_defer_weight_guards", False):
            super()._install_safetensors_load_guard()

    def _install_torch_load_guard(self) -> None:
        if not getattr(self, "_defer_weight_guards", False):
            super()._install_torch_load_guard()

    def first_frame_path(self, record: dict[str, Any]) -> Path:
        relative = record.get("first_frame")
        if relative is not None:
            relative_path = Path(str(relative))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"unsafe first-frame path: {relative!r}")
            candidate = (self.first_frame_root / relative_path).resolve()
            if not candidate.is_relative_to(self.first_frame_root):
                raise ValueError(f"first-frame path escapes its root: {relative!r}")
            matches = [candidate]
        else:
            source_index = int(record.get("source_index", record["index"]))
            prompt_id = str(record["prompt_id"])
            matches = [
                self.first_frame_root / f"{source_index:02d}-{prompt_id}{suffix}"
                for suffix in (".png", ".jpg", ".jpeg")
            ]
        present = [path for path in matches if path.is_file() and path.stat().st_size > 0]
        if len(present) != 1:
            raise FileNotFoundError(
                "expected exactly one first frame; "
                f"candidates={[str(path) for path in matches]} "
                f"found={[str(path) for path in present]}"
            )
        expected_sha256 = record.get("first_frame_sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError("refiner manifest lacks first_frame_sha256")
        actual_sha256 = base._sha256(present[0])
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "first-frame SHA-256 mismatch: "
                f"expected={expected_sha256} actual={actual_sha256}"
            )
        return present[0]

    def encode_prompt_multimodal(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        raw_outputs = self.gemma.encode([prompt])
        if len(raw_outputs) != 1:
            raise RuntimeError(f"Gemma returned {len(raw_outputs)} outputs")
        hidden_states, attention_mask = raw_outputs[0]
        processed = self.embeddings_processor.process_hidden_states(hidden_states, attention_mask)
        if processed.video_encoding is None or processed.audio_encoding is None:
            raise RuntimeError("LTX-2.5 prompt encoder did not return both video and audio contexts")
        return processed.video_encoding.detach(), processed.audio_encoding.detach()

    def encode_highres_first_frame(self, image_path: Path) -> list[Any]:
        images = [
            ImageConditioningInput(
                path=str(image_path),
                frame_idx=0,
                strength=1.0,
                crf=self.image_conditioning_crf,
            )
        ]
        return combined_image_conditionings(
            images=images,
            height=base.HEIGHT,
            width=base.WIDTH,
            video_encoder=self.image_encoder,
            dtype=self.dtype,
            device=self.device,
        )

    def encode_source_audio(self, source_path: Path) -> torch.Tensor:
        if self.audio_encoder is None or self.audio_tools is None:
            raise RuntimeError("Audio Stage 2 is disabled for this refiner")
        pixel_shape = VideoPixelShape(
            batch=1,
            frames=base.FRAME_COUNT,
            height=base.HEIGHT,
            width=base.WIDTH,
            fps=base.FPS,
        )
        latent = audio_latent_from_file(
            audio_encoder=self.audio_encoder,
            file_path=str(source_path),
            output_shape=pixel_shape,
            device=self.device,
            dtype=self.dtype,
            max_duration=base.FRAME_COUNT / base.FPS,
        )
        if latent is None:
            raise RuntimeError(f"H3 source has no decodable audio stream: {source_path}")
        expected = tuple(self.audio_tools.target_shape)
        actual = tuple(latent.shape)
        if actual != expected:
            raise RuntimeError(f"unexpected audio latent shape {actual}, expected {expected}")
        return latent

    def prepare_multimodal_states(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        conditionings: list[Any],
        seed: int,
    ) -> tuple[Any, Any, torch.Generator]:
        if self.audio_tools is None:
            raise RuntimeError("Audio Stage 2 is disabled for this refiner")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        video_state = create_noised_state(
            tools=self.video_tools,
            conditionings=conditionings,
            noiser=noiser,
            dtype=self.dtype,
            device=self.device,
            noise_scale=self.stage2_sigmas[0],
            initial_latent=video_latent,
        )
        audio_state = create_noised_state(
            tools=self.audio_tools,
            conditionings=[],
            noiser=noiser,
            dtype=self.dtype,
            device=self.device,
            noise_scale=self.stage2_sigmas[0],
            initial_latent=audio_latent,
        )
        return video_state, audio_state, generator

    def prepare_video_only_state(
        self,
        video_latent: torch.Tensor,
        conditionings: list[Any],
        seed: int,
    ) -> tuple[Any, torch.Generator]:
        """Create the same conditioned video state without any audio branch."""
        generator = torch.Generator(device=self.device).manual_seed(seed)
        video_state = create_noised_state(
            tools=self.video_tools,
            conditionings=conditionings,
            noiser=GaussianNoiser(generator=generator),
            dtype=self.dtype,
            device=self.device,
            noise_scale=self.stage2_sigmas[0],
            initial_latent=video_latent,
        )
        return video_state, generator

    def denoise_multimodal(
        self,
        video_state: Any,
        audio_state: Any,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
    ) -> tuple[Any, Any]:
        self.sol_attention.begin_denoise()
        video_state, audio_state = euler_denoising_loop(
            sigmas=self.sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=base.EulerDiffusionStep(),
            transformer=self.wrapped_transformer,
            denoiser=SimpleDenoiser(v_context=video_context, a_context=audio_context),
        )
        if video_state is None or audio_state is None:
            raise RuntimeError("official-compatible Stage 2 did not return both modalities")
        self._checked_attention_stats()
        return video_state, audio_state

    def stream_full_vae(
        self,
        latent: torch.Tensor,
        generator: torch.Generator,
        phases: dict[str, float],
    ) -> Iterable[torch.Tensor]:
        """Stream the official decoder exactly as the native pipeline does.

        ``encode_video`` is the consumer of this iterator in the official
        LTX-2.5 pipeline.  Do not materialize it with ``list()``: DiffVAE owns
        temporal overlap state until the iterator is exhausted.  Time only the
        synchronous ``next()`` calls so CPU colour conversion/H.264 work between
        chunks is not charged to the Video VAE decoder.
        """
        if self.video_decoder is None or self.tiling_config is None:
            raise RuntimeError("official Video VAE decoder is not loaded")
        decoder_iterator = iter(
            self.video_decoder.decode_video(
                latent.to(self.dtype), self.tiling_config, generator=generator
            )
        )

        def measured_chunks() -> Iterable[torch.Tensor]:
            frame_count = 0
            chunk_count = 0
            decode_s = 0.0
            while True:
                base._cuda_sync()
                started = time.perf_counter()
                try:
                    chunk = next(decoder_iterator)
                except StopIteration:
                    base._cuda_sync()
                    decode_s += time.perf_counter() - started
                    break
                base._cuda_sync()
                decode_s += time.perf_counter() - started
                if chunk.ndim != 4 or tuple(chunk.shape[1:]) != (
                    base.HEIGHT,
                    base.WIDTH,
                    3,
                ):
                    raise RuntimeError(
                        f"unexpected official decoder chunk shape {tuple(chunk.shape)}"
                    )
                if not bool(torch.isfinite(chunk).all().item()):
                    raise RuntimeError("official Video VAE decoder returned NaN or Inf")
                frame_count += int(chunk.shape[0])
                chunk_count += 1
                yield chunk
            phases["official_video_vae_decode_s"] = decode_s
            if frame_count != base.FRAME_COUNT:
                raise RuntimeError(
                    f"official decoder returned {frame_count} frames, "
                    f"expected {base.FRAME_COUNT}"
                )
            # The native pipeline uses ``get_video_chunks_number`` only as the
            # tqdm total.  DiffVAE can merge temporal tile groups into fewer
            # emitted chunks, so preserve both values without requiring them to
            # match.
            self.last_decoder_stream_info = {
                "actual_chunks": chunk_count,
                "progress_expected_chunks": get_video_chunks_number(
                    base.FRAME_COUNT, self.tiling_config
                ),
                "frames": frame_count,
            }

        return measured_chunks()

    def tiling_metadata(self) -> dict[str, Any] | None:
        if self.tiling_config is None:
            return None
        config = (
            asdict(self.tiling_config)
            if is_dataclass(self.tiling_config)
            else {"repr": repr(self.tiling_config)}
        )
        return {
            "source": self.tiling_source,
            "config": config,
            "progress_expected_chunks": get_video_chunks_number(
                base.FRAME_COUNT, self.tiling_config
            ),
            "last_stream": self.last_decoder_stream_info,
        }

    def write_full_vae_video(
        self,
        chunks: Iterable[torch.Tensor],
        chunk_count: int,
        source_path: Path,
        output_path: Path,
    ) -> None:
        audio = base._normalize_audio(
            decode_audio_from_file(
                str(source_path),
                device=torch.device("cpu"),
                max_duration=base.FRAME_COUNT / base.FPS,
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.stem}.partial-{os.getpid()}.mp4")
        temporary.unlink(missing_ok=True)
        try:
            encode_video(
                video=iter(chunks),
                fps=int(base.FPS),
                audio=audio,
                output_path=str(temporary),
                video_chunks_number=chunk_count,
                crf=19,
                preset="veryfast",
                thread_count=8,
            )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError(f"video encoder produced no output: {temporary}")
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)

    @torch.inference_mode()
    def run_diagnostic(self, record: dict[str, Any], output_path: Path) -> dict[str, Any]:
        source_path = Path(record["_source_path"])
        first_frame = self.first_frame_path(record)
        prompt_stats = self.prompt_stats(record["prompt"])
        started = time.perf_counter()
        phases: dict[str, float] = {}

        (pixels, input_info), phases["input_decode_resize_s"] = base._timed_cuda(
            lambda: self.prepare_input(source_path)
        )
        if self.stage2_audio_enabled:
            (video_context, audio_context), phases["gemma_multimodal_embedding_s"] = base._timed_cuda(
                lambda: self.encode_prompt_multimodal(record["prompt"])
            )
        else:
            video_context, phases["gemma_video_embedding_s"] = base._timed_cuda(
                lambda: self.encode_prompt(record["prompt"])
            )
            audio_context = None
        input_encode_phase = (
            "taehv_video_encode_s"
            if self.variant_config["encoder"] == "taehv"
            else "official_input_video_vae_encode_s"
        )
        normalized, phases[input_encode_phase] = base._timed_cuda(
            lambda: self.video_encode(pixels)
        )
        del pixels
        if bool(self.variant_config["latent_upsampler"]):
            video_latent, phases["learned_latent_upsample_s"] = base._timed_cuda(
                lambda: self.upsample(normalized)
            )
        else:
            video_latent = normalized
            phases["learned_latent_upsample_s"] = 0.0
        del normalized
        conditionings, phases["highres_first_frame_condition_s"] = base._timed_cuda(
            lambda: self.encode_highres_first_frame(first_frame)
        )
        if self.stage2_audio_enabled:
            audio_latent, phases["audio_vae_encode_s"] = base._timed_cuda(
                lambda: self.encode_source_audio(source_path)
            )
            (video_state, audio_state, generator), phases["multimodal_state_prepare_s"] = base._timed_cuda(
                lambda: self.prepare_multimodal_states(
                    video_latent,
                    audio_latent,
                    conditionings,
                    int(record["seed"]),
                )
            )
            del video_latent, audio_latent, conditionings
            (video_state, audio_state), phases["joint_video_audio_stage2_s"] = base._timed_cuda(
                lambda: self.denoise_multimodal(
                    video_state, audio_state, video_context, audio_context
                )
            )
            del video_context, audio_context, audio_state
        else:
            (video_state, generator), phases["video_state_prepare_s"] = base._timed_cuda(
                lambda: self.prepare_video_only_state(
                    video_latent,
                    conditionings,
                    int(record["seed"]),
                )
            )
            del video_latent, conditionings
            video_state, phases["video_only_stage2_s"] = base._timed_cuda(
                lambda: self.denoise(video_state, video_context)
            )
            self._checked_attention_stats()
            del video_context
        video_state = self.video_tools.clear_conditioning(video_state)
        video_state = self.video_tools.unpatchify(video_state)
        latent = video_state.latent
        del video_state
        if self.output_decoder_backend == "official_vae":
            chunks = self.stream_full_vae(latent, generator, phases)
            _, phases["decode_stream_and_h264_mux_s"] = base._timed_cuda(
                lambda: self.write_full_vae_video(
                    chunks,
                    get_video_chunks_number(base.FRAME_COUNT, self.tiling_config),
                    source_path,
                    output_path,
                )
            )
            del latent, generator
            phases["h264_encode_mux_s"] = max(
                0.0,
                phases["decode_stream_and_h264_mux_s"]
                - phases["official_video_vae_decode_s"],
            )
            del chunks
        else:
            decoded, phases["taehv_video_decode_s"] = base._timed_cuda(
                lambda: self.tae_decode(latent, validate=True)
            )
            del latent, generator
            _, phases["h264_encode_mux_s"] = base._timed_cuda(
                lambda: self.write_video(decoded, source_path, output_path)
            )
            del decoded
        output_info, phases["output_verify_s"] = base._timed_cuda(
            lambda: base._verify_video(output_path)
        )
        attention = self._checked_attention_stats()
        cleanup_started = time.perf_counter()
        if not self.retain_allocator_cache:
            gc.collect()
            torch.cuda.empty_cache()
        base._cuda_sync()
        phases["per_request_gc_allocator_cleanup_s"] = (
            time.perf_counter() - cleanup_started
        )
        return {
            "status": "succeeded",
            "index": int(record["index"]),
            "source_index": int(record.get("source_index", record["index"])),
            "prompt_id": record["prompt_id"],
            "seed": int(record["seed"]),
            "prompt_stats": prompt_stats,
            "source_video": str(source_path),
            "highres_first_frame": str(first_frame),
            "output": str(output_path),
            "input_info": input_info,
            "output_info": output_info,
            "attention": attention,
            "official_vae_tiling": self.tiling_metadata(),
            "phases_s": {key: round(value, 6) for key, value in phases.items()},
            "wall_s": round(time.perf_counter() - started, 6),
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--first-frame-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--video-vae", type=Path, required=True)
    parser.add_argument("--audio-vae", type=Path, required=True)
    parser.add_argument("--upsampler", type=Path, required=True)
    parser.add_argument("--refiner-lora", type=Path, required=True)
    parser.add_argument("--taehv-source", type=Path, required=True)
    parser.add_argument("--taehv-checkpoint", type=Path, required=True)
    parser.add_argument("--sample-indices", type=int, nargs="+", default=[15, 28])
    parser.add_argument("--output-decoder", choices=OUTPUT_DECODERS, default="official_vae")
    parser.add_argument("--stage2-schedule", choices=tuple(STAGE2_SCHEDULES), default="default")
    return parser


def _base_args(args: argparse.Namespace) -> argparse.Namespace:
    input_encoder = str(getattr(args, "input_encoder", "taehv_upsampler"))
    variant_by_encoder = {
        "taehv_upsampler": "latent_upsampler_taehv",
        "official_vae": "pixel_resize_vae",
        "official_vae_upsampler": "latent_upsampler_vae",
    }
    if input_encoder not in variant_by_encoder:
        raise ValueError(f"unsupported official-compatible input encoder: {input_encoder}")
    return argparse.Namespace(
        variant=variant_by_encoder[input_encoder],
        source_width=int(getattr(args, "source_width", base.SOURCE_WIDTH)),
        source_height=int(getattr(args, "source_height", base.SOURCE_HEIGHT)),
        source_frames=int(getattr(args, "source_frames", base.SOURCE_FRAME_COUNT)),
        compile=bool(getattr(args, "compile", False)),
        compile_mode=getattr(args, "compile_mode", None),
        compile_cache_root=getattr(args, "compile_cache_root", None),
        retain_allocator_cache=bool(
            getattr(args, "retain_allocator_cache", False)
        ),
        transformer=args.transformer,
        text_encoder=args.text_encoder,
        video_vae=args.video_vae,
        audio_vae=args.audio_vae,
        first_frame_root=args.first_frame_root,
        upsampler=args.upsampler,
        refiner_lora=args.refiner_lora,
        taehv_source=args.taehv_source,
        taehv_checkpoint=args.taehv_checkpoint,
        stage2_audio=not bool(getattr(args, "disable_stage2_audio", False)),
        output_decoder=str(getattr(args, "output_decoder", "official_vae")),
        stage2_schedule=str(getattr(args, "stage2_schedule", "default")),
        attention_backend=str(getattr(args, "attention_backend", "sol")),
        diffvae_spatial_tile_size=int(
            getattr(args, "diffvae_spatial_tile_size", 0)
        ),
    )


def main() -> int:
    args = _build_parser().parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        raise RuntimeError("launch with torch.distributed.run/torchrun")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    rank = dist.get_rank()
    try:
        if dist.get_world_size() != 1 or torch.cuda.device_count() != 1:
            raise RuntimeError("official compatibility diagnostic requires exactly one visible GPU")
        required = (
            args.input_root,
            args.manifest,
            args.first_frame_root,
            args.transformer,
            args.text_encoder,
            args.video_vae,
            args.audio_vae,
            args.upsampler,
            args.refiner_lora,
            args.taehv_source,
            args.taehv_checkpoint,
        )
        for path in required:
            if not path.exists() or (path.is_file() and path.stat().st_size == 0):
                raise FileNotFoundError(path)
        if len(args.sample_indices) not in (1, 2) or len(set(args.sample_indices)) != len(args.sample_indices):
            raise ValueError("choose one or two unique sample indices")
        if base._sha256(args.taehv_checkpoint) != base.TAEHV_WEIGHT_SHA256:
            raise RuntimeError("TAEHV diagnostic checkpoint SHA-256 mismatch")

        records = base._load_records(args.manifest, args.input_root, expected_count=29)
        by_index = {int(record["index"]): record for record in records}
        selected = []
        for index in args.sample_indices:
            if index not in by_index:
                raise KeyError(f"sample index {index} is absent from the manifest")
            selected.append(by_index[index])

        args.output_dir.mkdir(parents=True, exist_ok=True)
        outputs = [
            args.output_dir
            / (
                f"{int(record['index']):02d}-{record['prompt_id']}-"
                f"official_compat-taehv-upsample-firstframe-av-fullvae-"
                f"{base.WIDTH}x{base.HEIGHT}-{base.FRAME_COUNT}f.mp4"
            )
            for record in selected
        ]
        existing = [str(path) for path in outputs if path.exists()]
        if existing or args.metadata_path.exists():
            raise FileExistsError(
                "refusing to overwrite diagnostic artifacts: "
                + ", ".join(existing + ([str(args.metadata_path)] if args.metadata_path.exists() else []))
            )

        load_started = time.perf_counter()
        models = OfficialCompatRefiner(_base_args(args))
        load_s = time.perf_counter() - load_started
        results = []
        for record, output in zip(selected, outputs, strict=True):
            print(
                f"[official-compat] start index={record['index']} id={record['prompt_id']}",
                flush=True,
            )
            result = models.run_diagnostic(record, output)
            results.append(result)
            print(
                f"[official-compat] done index={record['index']} wall_s={result['wall_s']} "
                f"output={output}",
                flush=True,
            )

        if rank == 0:
            base._write_json(
                args.metadata_path,
                {
                    "schema_version": 1,
                    "status": "succeeded",
                    "purpose": "quality-only two-case official LTX-2.5 Stage-2 compatibility diagnostic",
                    "timing_is_benchmark": False,
                    "model_load_s": round(load_s, 6),
                    "hardware": "1xGB200",
                    "source_stage1": "MiniMax-H3 MP4; no native LTX Stage-1 latent is available",
                    "video_initial_latent": "H3 pixels -> TAEHV -> learned LTX x2 spatial upsampler",
                    "stage2_schedule": models.stage2_schedule_name,
                    "stage2_sigmas": list(models.stage2_sigmas),
                    "stage2_updates": len(models.stage2_sigmas) - 1,
                    "distilled_lora_strength": base.LORA_STRENGTH,
                    "highres_first_frame_conditioning": {
                        "enabled": True,
                        "width": base.WIDTH,
                        "height": base.HEIGHT,
                        "frame_idx": 0,
                        "strength": 1.0,
                        "crf": models.image_conditioning_crf,
                        "encoder": str(args.video_vae.resolve()),
                    },
                    "stage2_audio": {
                        "enabled": True,
                        "encoder": str(args.audio_vae.resolve()),
                        "noise_scale": models.stage2_sigmas[0],
                        "jointly_denoised_with_video": True,
                        "denoised_audio_output_used": False,
                        "final_mux_audio": "original MiniMax-H3 source audio",
                    },
                    "video_decoder": (
                        str(args.video_vae.resolve())
                        if models.output_decoder_backend == "official_vae"
                        else str(args.taehv_checkpoint.resolve())
                    ),
                    "output_decoder_backend": models.output_decoder_backend,
                    "transformer": str(args.transformer.resolve()),
                    "refiner_lora": str(args.refiner_lora.resolve()),
                    "taehv_checkpoint": str(args.taehv_checkpoint.resolve()),
                    "results": results,
                },
            )
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
