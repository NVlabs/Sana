#!/usr/bin/env python3
"""Resident exact-handoff LTX-2.5 Stage-2 service for one pipeline pair."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from handoff_protocol import AUDIO_SPEC, VIDEO_SPEC, JsonServer, TensorServer

import official_compat_h3_refiner_diagnostic as compat
import refiner_encoder_ablation_single_gpu as base
from ltx_core.model.audio_vae import encode_audio
from ltx_core.types import Audio


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class TemporalTileRefiner(compat.OfficialCompatRefiner):
    """Keep the official encoder, changing only its temporal tile policy."""

    @torch.inference_mode()
    def __init__(self, args: argparse.Namespace) -> None:
        self.input_vae_temporal_tile_mode = args.input_vae_temporal_tile_mode
        self._direct_pixels: torch.Tensor | None = None
        self._direct_audio_gpu: torch.Tensor | None = None
        self._direct_audio_cpu: torch.Tensor | None = None
        self._direct_input_info: dict[str, Any] | None = None
        super().__init__(args)

    @torch.inference_mode()
    def prepare_input(self, source_path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
        if self._direct_pixels is None:
            return super().prepare_input(source_path)
        pixels = self._direct_pixels
        expected = VIDEO_SPEC.shape
        if (
            tuple(pixels.shape) != expected
            or pixels.device != self.device
            or pixels.dtype != self.dtype
            or not pixels.is_contiguous()
        ):
            raise RuntimeError(
                "direct LTX pixels violate the fixed contract: "
                f"shape={tuple(pixels.shape)} dtype={pixels.dtype} device={pixels.device}"
            )
        if not bool(torch.isfinite(pixels).all().item()):
            raise RuntimeError("direct LTX pixels contain NaN or Inf")
        low = float(pixels.amin().item())
        high = float(pixels.amax().item())
        if low < -1.001 or high > 1.001:
            raise RuntimeError(f"direct LTX pixels are outside [-1,1]: {low}, {high}")
        info = dict(self._direct_input_info or {})
        info.update(
            {
                "transport": "direct_tensor_binary_tcp",
                "source_width": 896,
                "source_height": 512,
                "source_frames": 124,
                "source_fps": 24.0,
                "consumed_frames": 121,
                "dropped_tail_frames": 3,
                "preprocess": "stage1_cuda_bilinear_resize_then_vae_range",
                "encoder_backend": self.variant_config["encoder"],
                "encoder_input_width": 672,
                "encoder_input_height": 384,
                "encoder_pixels_shape": list(pixels.shape),
                "latent_upsampler": bool(self.variant_config["latent_upsampler"]),
                "observed_range": [low, high],
            }
        )
        return pixels, info

    @torch.inference_mode()
    def encode_source_audio(self, source_path: Path) -> torch.Tensor:
        if self._direct_audio_gpu is None:
            return super().encode_source_audio(source_path)
        if self.audio_encoder is None or self.audio_tools is None:
            raise RuntimeError("Audio Stage 2 is disabled for this refiner")
        waveform = self._direct_audio_gpu
        if (
            tuple(waveform.shape) != AUDIO_SPEC.shape
            or waveform.device != self.device
            or waveform.dtype != torch.float32
            or not waveform.is_contiguous()
        ):
            raise RuntimeError(
                "direct PCM violates the fixed contract: "
                f"shape={tuple(waveform.shape)} dtype={waveform.dtype} device={waveform.device}"
            )
        if not bool(torch.isfinite(waveform).all().item()):
            raise RuntimeError("direct PCM contains NaN or Inf")
        audio = Audio(waveform=waveform, sampling_rate=32_000)
        latent = encode_audio(audio, self.audio_encoder, None).to(
            device=self.device, dtype=self.dtype
        )
        expected = tuple(int(value) for value in self.audio_tools.target_shape)
        if latent.ndim != len(expected):
            raise RuntimeError(
                f"direct AudioVAE latent rank {latent.ndim} != expected {len(expected)}"
            )
        for dimension, (actual, wanted) in enumerate(zip(latent.shape, expected)):
            if dimension != 2 and int(actual) != wanted:
                raise RuntimeError(
                    f"direct AudioVAE latent dim {dimension}={actual} != {wanted}"
                )
        conformed = torch.zeros(expected, device=self.device, dtype=self.dtype)
        frames = min(int(latent.shape[2]), expected[2])
        destination = [slice(None)] * latent.ndim
        destination[2] = slice(0, frames)
        conformed[tuple(destination)] = latent[tuple(destination)]
        if tuple(conformed.shape) != expected:
            raise RuntimeError(
                f"conformed direct audio latent {tuple(conformed.shape)} != {expected}"
            )
        return conformed.contiguous()

    def write_video(
        self, decoded: torch.Tensor, source_path: Path, output_path: Path
    ) -> None:
        if self._direct_audio_cpu is None:
            return super().write_video(decoded, source_path, output_path)
        audio = base._normalize_audio(
            Audio(waveform=self._direct_audio_cpu, sampling_rate=32_000)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(
            f".{output_path.stem}.partial-{os.getpid()}.mp4"
        )
        temporary.unlink(missing_ok=True)

        def chunks():
            for start in range(0, base.FRAME_COUNT, 16):
                chunk = decoded[0, start : start + 16]
                yield chunk.permute(0, 2, 3, 1).float().contiguous()

        try:
            base.encode_video(
                video=chunks(),
                fps=int(base.FPS),
                audio=audio,
                output_path=str(temporary),
                video_chunks_number=math.ceil(base.FRAME_COUNT / 16),
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
    def run_diagnostic_direct(
        self,
        record: dict[str, Any],
        output_path: Path,
        *,
        pixels: torch.Tensor,
        audio_gpu: torch.Tensor,
        audio_cpu: torch.Tensor,
        input_info: dict[str, Any],
    ) -> dict[str, Any]:
        if any(
            value is not None
            for value in (
                self._direct_pixels,
                self._direct_audio_gpu,
                self._direct_audio_cpu,
            )
        ):
            raise RuntimeError("a direct Stage-2 request is already active")
        self._direct_pixels = pixels
        self._direct_audio_gpu = audio_gpu
        self._direct_audio_cpu = audio_cpu
        self._direct_input_info = dict(input_info)
        try:
            result = super().run_diagnostic(record, output_path)
            result["handoff_mode"] = "direct_tensor"
            result["input"] = str(record["_tensor_token"])
            return result
        finally:
            self._direct_pixels = None
            self._direct_audio_gpu = None
            self._direct_audio_cpu = None
            self._direct_input_info = None

    @torch.inference_mode()
    def video_encode(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.input_vae_temporal_tile_mode == "default":
            return super().video_encode(pixels)
        if self.variant_config["encoder"] == "taehv":
            raise RuntimeError("full temporal tiles require the official input Video VAE")
        if self.original_vae_encoder is None:
            raise RuntimeError("official LTX-2.5 input Video VAE is not resident")
        tile = compat.TileSizeConfig(
            # 121 frames fit in one 128-frame tile.  Keep the official overlap
            # for an explicit, valid TileSizeConfig while avoiding a temporal
            # split for this fixed profile.
            frames=compat.DimensionSizeConfig(tile_size=128, overlap=24),
            height=compat.DimensionSizeConfig(tile_size=768, overlap=64),
            width=compat.DimensionSizeConfig(tile_size=768, overlap=64),
        )
        latent = self.original_vae_encoder.tiled_encode(pixels, tile)
        expected = (
            1,
            128,
            base.LATENT_FRAME_COUNT,
            int(self.variant_config["pixel_height"]) // 32,
            int(self.variant_config["pixel_width"]) // 32,
        )
        if tuple(latent.shape) != expected:
            raise RuntimeError(
                f"full-temporal input VAE latent {tuple(latent.shape)} != {expected}"
            )
        return latent.contiguous()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--auth-token", required=True)
    parser.add_argument("--pair-id", type=int, required=True)
    parser.add_argument(
        "--handoff-mode", choices=("direct_tensor", "mp4"), required=True
    )
    parser.add_argument("--hot-repeats", type=int, choices=(1, 10), required=True)
    parser.add_argument("--template-manifest", type=Path, required=True)
    parser.add_argument("--first-frame-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--compile-cache-root", type=Path, required=True)
    parser.add_argument(
        "--input-vae-temporal-tile-mode", choices=("default", "full"), default="full"
    )
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--video-vae", type=Path, required=True)
    parser.add_argument("--audio-vae", type=Path, required=True)
    parser.add_argument("--upsampler", type=Path, required=True)
    parser.add_argument("--refiner-lora", type=Path, required=True)
    parser.add_argument("--taehv-source", type=Path, required=True)
    parser.add_argument("--taehv-checkpoint", type=Path, required=True)
    return parser


def _template(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ValueError("template manifest must contain exactly one record")
    row = dict(payload[0])
    for key in ("prompt_id", "prompt", "seed"):
        if key not in row:
            raise ValueError(f"template manifest record lacks {key}")
    return row


def _record(
    request: dict[str, Any],
    template: dict[str, Any],
    pair_id: int,
    handoff_mode: str,
) -> dict[str, Any]:
    if int(request.get("pair_id", -1)) != pair_id:
        raise ValueError("request was routed to the wrong pair")
    for key in (
        "prompt_id",
        "prompt",
        "seed",
        "source_index",
        "first_frame_sha256",
    ):
        if request.get(key) != template.get(key):
            raise ValueError(
                f"request {key}={request.get(key)!r} != pinned template {template.get(key)!r}"
            )
    row = dict(template)
    if request.get("handoff_mode") != handoff_mode:
        raise ValueError("control request handoff mode mismatch")
    if handoff_mode == "direct_tensor":
        tensor_token = request.get("tensor_token")
        if not isinstance(tensor_token, str) or not tensor_token.startswith(
            "h3tensor://"
        ):
            raise ValueError("direct control request lacks an h3tensor token")
        row["_source_path"] = "/dev/null"
        row["_tensor_token"] = tensor_token
    else:
        source = Path(str(request["source_mp4"]))
        if (
            source.suffix.lower() != ".mp4"
            or not source.is_file()
            or source.stat().st_size <= 0
        ):
            raise FileNotFoundError(source)
        row["_source_path"] = str(source.resolve())
    return row


def main() -> int:
    args = _parser().parse_args()
    if not args.compile_cache_root.is_absolute():
        raise ValueError("compile cache must be a persistent absolute path")
    for path in (
        args.template_manifest,
        args.first_frame_root,
        args.transformer,
        args.text_encoder,
        args.video_vae,
        args.audio_vae,
        args.upsampler,
        args.refiner_lora,
        args.taehv_source,
        args.taehv_checkpoint,
    ):
        if not path.exists() or (path.is_file() and path.stat().st_size <= 0):
            raise FileNotFoundError(path)
    if base._sha256(args.taehv_checkpoint) != base.TAEHV_WEIGHT_SHA256:
        raise RuntimeError("TAEHV checkpoint SHA-256 mismatch")
    template = _template(args.template_manifest)
    if (
        template.get("prompt_id") != "bamboo-forest-wuxia-pair-en"
        or int(template.get("seed", -1)) != 50803
        or int(template.get("source_index", -1)) != 3
    ):
        raise RuntimeError("template is not the pinned bamboo/seed50803 integration sample")

    server: JsonServer = (
        TensorServer(args.endpoint, timeout_s=1800.0)
        if args.handoff_mode == "direct_tensor"
        else JsonServer(args.endpoint, timeout_s=1800.0)
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        raise RuntimeError("launch Stage 2 with torchrun")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    models: TemporalTileRefiner | None = None
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "pair_id": args.pair_id,
        "endpoint": args.endpoint,
        "handoff_mode": args.handoff_mode,
        "hot_repeats_expected": args.hot_repeats,
        "configuration": {
            "input_encoder": "official_vae_upsampler",
            "handoff_mode": args.handoff_mode,
            "direct_tensor_contract": (
                {
                    "video": VIDEO_SPEC.descriptor(),
                    "audio": AUDIO_SPEC.descriptor(),
                    "transport": "binary_tcp_loopback_pinned_cpu",
                }
                if args.handoff_mode == "direct_tensor"
                else None
            ),
            "input_vae_temporal_tile_mode": args.input_vae_temporal_tile_mode,
            "input_vae_full_temporal_tile": {
                "frames": [128, 24],
                "height": [768, 64],
                "width": [768, 64],
            },
            "highres_first_frame_condition": True,
            "stage2_audio": True,
            "output_decoder": "taehv",
            "stage2_schedule": "default",
            "sigmas": [0.909375, 0.725, 0.421875, 0.0],
            "updates": 3,
            "attention": "layer0_dense_layers1_47_sol_strict",
            "compile": "max-autotune-no-cudagraphs",
            "retain_allocator_cache": True,
            "compile_cache_root": str(args.compile_cache_root),
        },
        "excluded": {},
        "hot": [],
    }
    try:
        if dist.get_world_size() != 1 or torch.cuda.device_count() != 1:
            raise RuntimeError("Stage 2 requires exactly one visible GB200")
        model_args = argparse.Namespace(
            input_encoder="official_vae_upsampler",
            source_width=896,
            source_height=512,
            source_frames=124,
            compile=True,
            compile_mode=base.COMPILE_MODE,
            compile_cache_root=args.compile_cache_root,
            retain_allocator_cache=True,
            transformer=args.transformer,
            text_encoder=args.text_encoder,
            video_vae=args.video_vae,
            audio_vae=args.audio_vae,
            first_frame_root=args.first_frame_root,
            upsampler=args.upsampler,
            refiner_lora=args.refiner_lora,
            taehv_source=args.taehv_source,
            taehv_checkpoint=args.taehv_checkpoint,
            disable_stage2_audio=False,
            output_decoder="taehv",
            stage2_schedule="default",
            attention_backend="sol",
            diffvae_spatial_tile_size=0,
            input_vae_temporal_tile_mode=args.input_vae_temporal_tile_mode,
        )
        loaded = time.perf_counter()
        resident_args = compat._base_args(model_args)
        # _base_args intentionally copies only the original benchmark fields.
        resident_args.input_vae_temporal_tile_mode = args.input_vae_temporal_tile_mode
        models = TemporalTileRefiner(resident_args)
        report["excluded"]["model_load_s"] = time.perf_counter() - loaded
        args.output_dir.mkdir(parents=True, exist_ok=True)

        warmup_done = False

        def run_control_request(
            request_value: dict[str, Any],
            direct_payload: dict[str, Any] | None,
        ) -> dict[str, Any]:
            nonlocal warmup_done
            if request_value.get("token") != args.auth_token:
                raise PermissionError("handoff token mismatch")
            operation = request_value.get("op")
            record = _record(
                request_value, template, args.pair_id, args.handoff_mode
            )
            if args.handoff_mode == "direct_tensor":
                if direct_payload is None:
                    raise RuntimeError("direct control request has no staged tensors")
                if request_value.get("tensor_token") != direct_payload["tensor_token"]:
                    raise RuntimeError("control tensor token does not match staged tensors")
            elif direct_payload is not None:
                raise RuntimeError("MP4 request unexpectedly supplied staged tensors")

            if operation == "warmup":
                if warmup_done:
                    raise RuntimeError("Stage-2 warmup was already completed")
                output = args.output_dir / ".warmup_excluded.mp4"
                output.unlink(missing_ok=True)
            elif operation == "refine":
                if not warmup_done:
                    raise RuntimeError("refusing a hot request before full warmup")
                repeat = int(request_value["repeat"])
                if repeat != len(report["hot"]):
                    raise RuntimeError(
                        f"expected hot repeat {len(report['hot'])}, got {repeat}"
                    )
                output = (
                    args.output_dir
                    / f"hot_{repeat:02d}_refined_1344x768_121f.mp4"
                )
                if output.exists():
                    raise FileExistsError(output)
            else:
                raise ValueError(f"unsupported handoff operation {operation!r}")

            if direct_payload is None:
                result = models.run_diagnostic(record, output)
            else:
                result = models.run_diagnostic_direct(
                    record,
                    output,
                    pixels=direct_payload["pixels"],
                    audio_gpu=direct_payload["audio_gpu"],
                    audio_cpu=direct_payload["audio_cpu"],
                    input_info=direct_payload["input_info"],
                )
                result["tensor_stage"] = direct_payload["timing"]

            if operation == "warmup":
                output.unlink(missing_ok=True)
                models.prepare_steady_state()
                warmup_done = True
                report["excluded"]["full_warmup"] = result
            else:
                result["repeat"] = repeat
                report["hot"].append(result)
            return result

        if args.handoff_mode == "direct_tensor":
            if not isinstance(server, TensorServer):
                raise RuntimeError("direct mode did not construct TensorServer")
            video_cpu = torch.empty(
                VIDEO_SPEC.shape,
                dtype=torch.bfloat16,
                device="cpu",
                pin_memory=True,
            )
            audio_cpu = torch.empty(
                AUDIO_SPEC.shape,
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            while len(report["hot"]) < args.hot_repeats:
                expected_seq = 2 if not warmup_done else 3 + len(report["hot"])
                expected_phase = "warmup" if not warmup_done else "hot"
                header, tensor_handle = server.receive_into(
                    video_cpu,
                    audio_cpu,
                    expected_token=args.auth_token,
                    expected_pair_id=args.pair_id,
                    expected_seq=expected_seq,
                )
                receive_timing = server.last_tensor_receive_timing
                if receive_timing is None:
                    raise RuntimeError("tensor receiver did not publish transport timing")
                accept_wait_s = float(receive_timing["accept_wait_s"])
                payload_receive_s = float(receive_timing["payload_receive_s"])
                try:
                    metadata = header.get("metadata") or {}
                    if (
                        header.get("op") != "stage_tensor"
                        or metadata.get("phase") != expected_phase
                        or float(metadata.get("fps", -1.0)) != 24.0
                        or int(metadata.get("audio_sample_rate", -1)) != 32_000
                    ):
                        raise RuntimeError(f"invalid direct staging header: {header}")
                    h2d_started = time.perf_counter()
                    pixels = video_cpu.to(
                        device=models.device, dtype=models.dtype, non_blocking=True
                    )
                    audio_gpu = audio_cpu.to(
                        device=models.device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                    torch.cuda.synchronize(models.device)
                    h2d_s = time.perf_counter() - h2d_started
                    tensor_token = (
                        f"h3tensor://pair-{args.pair_id}/seq-{expected_seq}"
                    )
                    stage_timing = {
                        # `accept_wait_s` is pipeline overlap/idle time while
                        # Stage 2 is ready before H3.  It is not transport
                        # latency.  Keep `receive_s` as a compatibility alias
                        # for the measured header+payload receive interval.
                        "accept_wait_s": accept_wait_s,
                        "payload_receive_s": payload_receive_s,
                        "receive_s": payload_receive_s,
                        "h2d_s": h2d_s,
                        "receive_and_h2d_s": payload_receive_s + h2d_s,
                        "video_nbytes": VIDEO_SPEC.nbytes,
                        "audio_nbytes": AUDIO_SPEC.nbytes,
                    }
                    server.ack_staged(
                        tensor_handle,
                        tensor_token=tensor_token,
                        copied_to_cuda=True,
                        timing=stage_timing,
                    )
                except Exception:
                    stream, connection, _ = tensor_handle
                    stream.close()
                    connection.close()
                    raise

                direct_payload = {
                    "tensor_token": tensor_token,
                    "pixels": pixels,
                    "audio_gpu": audio_gpu,
                    # The next staging connection cannot arrive until the
                    # control response below returns, so this pinned buffer is
                    # also the zero-copy source for the one final audio mux.
                    "audio_cpu": audio_cpu,
                    "input_info": {
                        "tensor_token": tensor_token,
                        "wire_metadata": metadata,
                        "tensor_stage": stage_timing,
                    },
                    "timing": stage_timing,
                }
                request_value, control_handle = server.receive()
                try:
                    result = run_control_request(request_value, direct_payload)
                    JsonServer.respond(
                        control_handle, {"status": "succeeded", "result": result}
                    )
                except Exception as exc:
                    JsonServer.respond(
                        control_handle,
                        {
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        },
                    )
                    raise
                del direct_payload, pixels, audio_gpu
        else:
            while len(report["hot"]) < args.hot_repeats:
                request_value, handle = server.receive()
                try:
                    result = run_control_request(request_value, None)
                    JsonServer.respond(
                        handle, {"status": "succeeded", "result": result}
                    )
                except Exception as exc:
                    JsonServer.respond(
                        handle,
                        {
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        },
                    )
                    raise
        report["status"] = "complete"
        _atomic_json(args.metadata_path, report)
        return 0
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        _atomic_json(args.metadata_path, report)
        raise
    finally:
        server.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
