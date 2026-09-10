"""Resident Spark H3 ×2, adapter, three-update joint AV refiner session.

The module imports GPU frameworks only when Session is constructed. A request
consumes one same-request normalized H3/PCM bundle and the exact offline generic
post-connector AV cache. Original PCM is muxed into the final MP4.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import gc
import os
from pathlib import Path
import time

from .latent_transfer import (WIRE_METADATA, load_latent_capture, load_payload,
                              sha256, validate_request_id, write_new)
from .prompt_cache import FIXED_PROMPT, load_cache
from .stage2_ops.h3_upscale import (ADAPTER_OUTPUT, H3_INPUT, H3_UPSCALED,
                                   REFINER_INPUT, upscale_comfy_normalized_h3)
from .stage2_ops.models import SIGMAS, TILES, construct, public_refiner_modules
from .stage2_ops.conv_direct_nhwc import installed_direct_upsample_nhwc
from .stage2_ops.writer_chunking import tracked_writer_chunks

REQUIRED_PATHS = ("transformer", "refiner_lora", "output_video_vae", "audio_vae",
                  "adapter_dir", "h3_upscaler_source", "h3_upscaler_checkpoint", "prompt_cache")


def _refiner_class(base, compat, session):
    torch = session.torch

    class SparkRefiner(compat.OfficialCompatRefiner):
        def __init__(self):
            construct(self, session.paths, torch=torch, dist=session.dist,
                      compile_enabled=session.compile_enabled, base=base, compat=compat)

        def first_frame_path(self, record):
            return None

        def encode_highres_first_frame(self, image_path):
            if image_path is not None:
                raise ValueError("latent transfer has no RGB first-frame conditioning")
            return []

        def prompt_stats(self, prompt):
            return session.prompt_cache.stats(prompt)

        def encode_prompt_multimodal(self, prompt):
            return session.prompt_cache.contexts(prompt, self.device)

        def prepare_input(self, source_path):
            return session.payload["h3_normalized"].to(self.device), {
                "wire_metadata": dict(WIRE_METADATA), "transport": "same_request_H3_and_PCM",
                "preprocess": "normalized_h3_x2_then_adapter_crop_t16",
                "latent_upsampler": False}

        def video_encode(self, normalized):
            def checked(value, shape):
                if (tuple(value.shape) != shape or value.dtype != self.dtype
                        or value.device != self.device or not value.is_contiguous()
                        or not bool(torch.isfinite(value).all())):
                    raise RuntimeError(f"H3 bridge shape/dtype/layout contract failed: {shape}")
                return value

            checked(normalized, H3_INPUT)
            self.h3_upscaler_calls += 1
            highres, self._h3_upscale_s = base._timed_cuda(
                lambda: upscale_comfy_normalized_h3(self.h3_upscaler, normalized))
            checked(highres, H3_UPSCALED)
            self.adapter_convert_calls += 1
            latent, self._h3_adapter_s = base._timed_cuda(lambda: self.h3_ltx_adapter.convert(
                highres, pixel_frames=124, pixel_height=768, pixel_width=1344,
                input_normalization="normalized"))
            checked(latent, ADAPTER_OUTPUT)
            return checked(latent[:, :, :16].contiguous(), REFINER_INPUT)

        def encode_source_audio(self, source_path):
            from ltx_core.model.audio_vae import encode_audio
            from ltx_core.types import Audio

            audio = Audio(waveform=session.payload["audio"].to(self.device), sampling_rate=32000)
            latent = encode_audio(audio, self.audio_encoder, None).to(device=self.device, dtype=self.dtype)
            expected = tuple(int(value) for value in self.audio_tools.target_shape)
            if (latent.ndim != len(expected) or any(int(actual) != wanted
                    for dim, (actual, wanted) in enumerate(zip(latent.shape, expected)) if dim != 2)):
                raise RuntimeError("direct AudioVAE latent shape mismatch")
            conformed = torch.zeros(expected, device=self.device, dtype=self.dtype)
            frames = min(int(latent.shape[2]), expected[2])
            conformed[:, :, :frames] = latent[:, :, :frames]
            return conformed.contiguous()

        def _checked_attention_stats(self):
            stats = self.sol_attention.stats()
            stats.update(selected_backend=self.sol_backend, architecture="sm121-triton")
            expected = {"completed_steps": 3, "dense_calls": 3, "sol_calls": 141}
            kernel = stats.get("kernel", {})
            if (any(stats.get(key) != value for key, value in expected.items())
                    or kernel.get("kernel_calls") != 141 or kernel.get("dense_guard_calls") != 0
                    or kernel.get("hunyuan_calls") != 0 or self.sol_backend != "triton"):
                raise RuntimeError(f"three-update Triton Sol contract failed: {stats}")
            return stats

        def write_full_vae_video(self, chunks, chunk_count, source_path, output_path):
            from ltx_core.types import Audio

            session.writer_chunk_receipt = {"chunk_frames": 16, "source_chunks": 0,
                "emitted_chunks": 0, "frames": 0, "source_layouts": [], "complete": False}
            chunks = tracked_writer_chunks(chunks, session.writer_chunk_receipt,
                                           expected_dtype=torch.bfloat16)
            audio = base._normalize_audio(Audio(waveform=session.payload["audio"], sampling_rate=32000))
            temporary = output_path.with_name(f".{output_path.stem}.partial-{os.getpid()}.mp4")
            if temporary.exists() or output_path.exists():
                raise FileExistsError(output_path)
            try:
                base.encode_video(video=iter(chunks), fps=24, audio=audio,
                    output_path=str(temporary), video_chunks_number=8,
                    crf=19, preset="veryfast", thread_count=8)
                if not temporary.is_file() or temporary.stat().st_size == 0:
                    raise RuntimeError("video encoder produced no output")
                os.replace(temporary, output_path)
            finally:
                temporary.unlink(missing_ok=True)
            self.final_mp4_complete_monotonic_ns = time.monotonic_ns()

        def prepare_steady_state(self):
            session.dist.barrier()
            if self.compile_enabled:
                self.attention_manager.all2all_timeout_seconds = self.steady_all2all_timeout_s
            torch.cuda.synchronize()
            session.dist.barrier()

    return SparkRefiner


class Session:
    """Construct once, prepare/run sequential requests, then close the worker."""

    def __init__(self, paths: dict, work_dir: str, config: dict):
        self.started_ns = time.monotonic_ns()
        self.stack, self.models, self.dist = ExitStack(), None, None
        self._owns_dist = False
        self.closed = self.failed = False
        self.prepared = self.current_capture = self.payload = None
        self.completed_requests = 0
        self.final_mp4_complete_monotonic_ns = None
        self.writer_chunk_receipt = None
        self.timings = {}
        config = config.get("stage2", config)
        self.compile_enabled = config.get("compile", True)
        self.fixed_prompt = config.get("fixed_prompt", FIXED_PROMPT)
        if type(self.compile_enabled) is not bool:
            raise ValueError("compile must be a boolean")
        self.paths = {key: Path(paths[key]).expanduser().resolve(strict=True) for key in REQUIRED_PATHS}
        self.ltx_root = Path(paths["ltx_root"]).expanduser().resolve(strict=True)
        files = [path for key, path in self.paths.items() if key != "adapter_dir"]
        files += [self.paths["adapter_dir"] / name for name in ("config.json", "model.safetensors")]
        for path in files:
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(path)
        self.work_dir = Path(work_dir).resolve()
        cache_root = self.work_dir / "compile"
        for env, part in (("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton")):
            directory = cache_root / part
            directory.mkdir(parents=True, exist_ok=True)
            os.environ[env] = str(directory)
        try:
            self._construct()
        except BaseException:
            self.close()
            raise

    def _construct(self):
        import torch
        import torch.distributed as dist

        self.torch, self.dist = torch, dist
        if int(os.environ.get("LOCAL_RANK", "-1")) != 0 or torch.cuda.device_count() != 1:
            raise RuntimeError("launch Stage2 with torchrun --nproc-per-node=1 and one visible GPU")
        torch.cuda.set_device(0)
        if tuple(torch.cuda.get_device_capability(0)) != (12, 1):
            raise RuntimeError("this release profile requires Spark SM121")
        if not dist.is_initialized():
            dist.init_process_group("nccl")
            self._owns_dist = True
        if dist.get_world_size() != 1:
            raise RuntimeError("Spark requires one distributed rank")
        self.prompt_cache = load_cache(self.paths["prompt_cache"], prompt=self.fixed_prompt, torch_module=torch)
        base, compat = public_refiner_modules(self.ltx_root)
        with torch.inference_mode():
            self.models = _refiner_class(base, compat, self)()
        self.conv_layout_receipt = self.stack.enter_context(installed_direct_upsample_nhwc(torch_module=torch))
        self.ready_ns = time.monotonic_ns()
        self.session_receipt = {"schema_version": 1, "status": "READY", "model_load_count": 1,
            "session_started_monotonic_ns": self.started_ns, "session_ready_monotonic_ns": self.ready_ns,
            "session_initialization_s": (self.ready_ns - self.started_ns) / 1e9,
            "refiner_recipe": {"sigmas": list(SIGMAS), "updates": 3, "lora_strength": 0.8,
                               "joint_audio_video": True, "expected_sol_backend": "triton"},
            "spatial_route": self.models.h3_spatial_route, "conv_actual_tiling": asdict(self.models.tiling_config),
            "actual_parameter_dtypes": self.models.parameter_dtypes,
            "stage2_gemma_loaded": False, "stage2_connector_loaded": False,
            "fixed_prompt_cache": self.prompt_cache.receipt(), "residency": self.models.residency,
            "all2all_copy_elision": self.models.all2all_receipt,
            "denoiser_dummy_warmup": False, "latent_only_transfer": True}
        self.timings["startup_s"] = self.session_receipt["session_initialization_s"]

    def prepare(self, request_id, output_root):
        started_ns = time.monotonic_ns()
        if self.closed or self.failed or self.prepared is not None or self.current_capture is not None:
            raise RuntimeError("prepare requires an idle session")
        validate_request_id(request_id)
        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=False)
        ready_ns = time.monotonic_ns()
        self.prepared = {"status": "READY", "request_id": request_id, "output_root": str(root),
            "stage2_prepare_started_monotonic_ns": started_ns, "conditioning_ready_monotonic_ns": ready_ns,
            "stage2_prepare_s": (ready_ns - started_ns) / 1e9,
            "stage2_gemma_loaded": False, "stage2_connector_loaded": False}
        write_new(root / "stage2_prepare.json", self.prepared)
        return dict(self.prepared)

    def run(self, capture_dir, output_root, request_id):
        started_ns = time.monotonic_ns()
        root = Path(output_root).resolve()
        if (self.closed or self.failed or self.current_capture is not None or self.prepared is None
                or self.prepared["request_id"] != request_id or self.prepared["output_root"] != str(root)):
            raise RuntimeError("run requires the matching completed prepare identity and output root")
        receipt = {"status": "RUNNING", "request_id": request_id, "stage2_started_monotonic_ns": started_ns,
            "request_index": self.completed_requests, "preparation": dict(self.prepared),
            "timing_boundary": "run_entry_after_prepare_to_atomic_final_muxed_MP4"}
        try:
            row = load_latent_capture(capture_dir, request_id=request_id)
            self.current_capture = row
            self.payload = load_payload(row, torch_module=self.torch)
            receipt.update(case_id=row["case_id"], capture=row)
            models = self.models
            before = models.h3_upscaler_calls, models.adapter_convert_calls, self.prompt_cache.context_calls
            layout_before = self.conv_layout_receipt["calls"]
            output = root / "refined_1344x768_121f.mp4"
            record = {"prompt_id": row["case_id"], "prompt": self.fixed_prompt, "seed": row["seed"],
                      "source_index": row["source_index"], "index": 0, "_source_path": "/dev/null"}
            with self.torch.inference_mode():
                result = models.run_diagnostic(record, output)
            calls = tuple(a - b for a, b in zip((models.h3_upscaler_calls,
                models.adapter_convert_calls, self.prompt_cache.context_calls), before))
            if calls != (1, 1, 1) or not self.writer_chunk_receipt["complete"]:
                raise RuntimeError("expected one H3 upscale, adapter and cached AV context consumption")
            phases = result["phases_s"]
            phases["fixed_prompt_context_copy_s"] = phases.pop("gemma_multimodal_embedding_s")
            phases["h3_upscale_and_adapter_s"] = phases.pop("official_input_video_vae_encode_s")
            phases.update(h3_latent_upscale_s=models._h3_upscale_s, h3_ltx_adapter_s=models._h3_adapter_s)
            result["highres_first_frame"] = None
            result["spatial_route"] = {**models.h3_spatial_route,
                "actual_calls": {"h3_upscaler": 1, "adapter": 1, "ltx_learned_x2": 0, "input_video_vae": 0}}
            finished_ns = models.final_mp4_complete_monotonic_ns
            self.final_mp4_complete_monotonic_ns = finished_ns
            receipt.update(status="PASS", result=result, output=str(output), output_sha256=sha256(output),
                output_bytes=output.stat().st_size, final_mp4_complete_monotonic_ns=finished_ns,
                stage2_request_s=(finished_ns - started_ns) / 1e9,
                stage2_prepare_to_mp4_s=(finished_ns - self.prepared["stage2_prepare_started_monotonic_ns"]) / 1e9,
                stage2_gemma_loaded=False, stage2_connector_loaded=False,
                fixed_prompt_cache=self.prompt_cache.receipt(), writer_chunking=deepcopy(self.writer_chunk_receipt),
                conv_output_layout_request_calls=self.conv_layout_receipt["calls"] - layout_before)
            self.timings.update(stage2_request_s=receipt["stage2_request_s"], phases_s=dict(phases))
            write_new(root / "final_file_manifest.json", {key: receipt[key] for key in (
                "status", "request_id", "case_id", "output", "output_sha256", "output_bytes",
                "stage2_started_monotonic_ns", "final_mp4_complete_monotonic_ns", "stage2_request_s")})
            if self.completed_requests == 0:
                models.prepare_steady_state()
            self.completed_requests += 1
            return receipt
        except BaseException as error:
            self.failed = True
            receipt.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            self.current_capture = self.payload = self.prepared = None
            write_new(root / "stage2_result.json", receipt)

    def release_idle_cache(self, *, qwen_resident=True):
        if self.closed or self.failed or self.current_capture is not None or self.prepared is not None:
            raise RuntimeError("cache release requires an idle session")
        started_ns = time.monotonic_ns()
        gc.collect()
        self.torch.cuda.empty_cache()
        self.torch.cuda.synchronize()
        return {"status": "PASS", "elapsed_s": (time.monotonic_ns() - started_ns) / 1e9}

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.models = None
        self.stack.close()
        gc.collect()
        if self._owns_dist and self.dist is not None and self.dist.is_initialized():
            self.dist.destroy_process_group()
