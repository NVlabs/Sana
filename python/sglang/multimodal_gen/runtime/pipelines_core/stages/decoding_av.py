import os
import time
from contextlib import contextmanager

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    get_is_main_process,
    init_logger,
)
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes", "on")


def _decode_profile_enabled() -> bool:
    return _env_flag_enabled("SGLANG_DIFFUSION_DECODE_PROFILE")


def _fast_video_postprocess_enabled() -> bool:
    return _env_flag_enabled("SGLANG_LTX2_FAST_VIDEO_POSTPROCESS")


def _compile_video_vae_decoder_enabled() -> bool:
    return _env_flag_enabled("SGLANG_LTX2_COMPILE_VAE_DECODER")


def _compile_tiled_video_vae_decoder_enabled() -> bool:
    return _env_flag_enabled("SGLANG_LTX2_COMPILE_TILED_VAE_DECODER")


def _video_vae_compile_mode() -> str:
    return os.environ.get("SGLANG_LTX2_VAE_COMPILE_MODE", "max-autotune-no-cudagraphs")


def _ltx2_stage1_export_path(batch: Req) -> str | None:
    if _env_flag_enabled("SGLANG_LTX2_STAGE1_ONLY_OUTPUT"):
        return None
    explicit_path = os.environ.get("SGLANG_LTX2_STAGE1_OUTPUT_PATH")
    if explicit_path:
        return explicit_path
    if not _env_flag_enabled("SGLANG_LTX2_SAVE_STAGE1_OUTPUT"):
        return None
    output_file_path = batch.output_file_path()
    if not output_file_path:
        return None
    stem, ext = os.path.splitext(output_file_path)
    return f"{stem}.stage1{ext or '.mp4'}"


def _indexed_output_path(output_path: str, idx: int, count: int) -> str:
    if count <= 1:
        return output_path
    stem, ext = os.path.splitext(output_path)
    return f"{stem}_{idx}{ext}"


def _save_ltx2_output_batch(
    *,
    output_batch: OutputBatch,
    batch: Req,
    output_path: str,
) -> list[str]:
    from sglang.multimodal_gen.runtime.entrypoints.utils import save_outputs

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    count = len(output_batch.output) if output_batch.output is not None else 0
    return save_outputs(
        output_batch.output,
        batch.data_type,
        batch.fps,
        True,
        lambda idx: _indexed_output_path(output_path, idx, count),
        audio=output_batch.audio,
        audio_sample_rate=output_batch.audio_sample_rate,
        output_compression=batch.output_compression,
        enable_frame_interpolation=batch.enable_frame_interpolation,
        frame_interpolation_exp=batch.frame_interpolation_exp,
        frame_interpolation_scale=batch.frame_interpolation_scale,
        frame_interpolation_model_path=batch.frame_interpolation_model_path,
        enable_upscaling=batch.enable_upscaling,
        upscaling_model_path=batch.upscaling_model_path,
        upscaling_scale=batch.upscaling_scale,
    )


def _postprocess_video_to_uint8_hwc_tensor(video_processor, video: torch.Tensor):
    if video_processor.config.do_normalize:
        video = (video * 0.5 + 0.5).clamp(0, 1)
    else:
        video = video.clamp(0, 1)

    video = video.mul(255.0).clamp_(0, 255).to(torch.uint8)
    return video.permute(0, 2, 3, 4, 1).contiguous()


@contextmanager
def _decode_profile_scope(batch: Req, name: str):
    metrics = getattr(batch, "metrics", None)
    if not _decode_profile_enabled() or metrics is None:
        yield
        return

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metrics.record_stage(f"LTX2AVDecodingStage.{name}", time.perf_counter() - start)


class LTX2AVDecodingStage(DecodingStage):
    """
    LTX-2 specific decoding stage that handles both video and audio decoding.
    """

    def __init__(self, vae, audio_vae, vocoder, pipeline=None):
        super().__init__(vae, pipeline)
        self.audio_vae = audio_vae
        self.vocoder = vocoder
        self._compiled_video_vae_decoder = None
        self._compiled_video_vae_decoder_id = None
        self._compile_video_vae_decoder_failed = False
        self._compiled_tiled_video_vae_decoder = None
        self._compiled_tiled_video_vae_decoder_id = None
        self._compiled_tiled_video_vae_decoder_base = None
        self._compile_tiled_video_vae_decoder_failed = False
        # Add video processor for postprocessing
        from diffusers.video_processor import VideoProcessor

        self.video_processor = VideoProcessor(vae_scale_factor=32)

    def _can_use_direct_video_vae_decoder(self) -> bool:
        return not (
            getattr(self.vae, "use_slicing", False)
            or getattr(self.vae, "use_tiling", False)
            or getattr(self.vae, "use_framewise_decoding", False)
        )

    def _get_direct_video_vae_decoder(self):
        if (
            not _compile_video_vae_decoder_enabled()
            or self._compile_video_vae_decoder_failed
            or not self._can_use_direct_video_vae_decoder()
        ):
            return None

        decoder = getattr(self.vae, "decoder", None)
        if not isinstance(decoder, torch.nn.Module):
            return None

        decoder_id = id(decoder)
        if (
            self._compiled_video_vae_decoder is not None
            and self._compiled_video_vae_decoder_id == decoder_id
        ):
            return self._compiled_video_vae_decoder

        mode = _video_vae_compile_mode()
        try:
            logger.info("Compiling LTX2 video VAE decoder with mode: %s", mode)
            self._compiled_video_vae_decoder = torch.compile(
                decoder, fullgraph=False, dynamic=False, mode=mode
            )
            self._compiled_video_vae_decoder_id = decoder_id
            return self._compiled_video_vae_decoder
        except Exception:
            logger.warning(
                "Failed to compile LTX2 video VAE decoder; falling back to eager",
                exc_info=True,
            )
            self._compile_video_vae_decoder_failed = True
            return decoder

    def _maybe_compile_tiled_video_vae_decoder(self) -> None:
        if (
            not _compile_tiled_video_vae_decoder_enabled()
            or self._compile_tiled_video_vae_decoder_failed
            or not getattr(self.vae, "use_tiling", False)
            or getattr(self.vae, "use_slicing", False)
            or getattr(self.vae, "use_framewise_decoding", False)
        ):
            return

        decoder = getattr(self.vae, "decoder", None)
        if not isinstance(decoder, torch.nn.Module):
            return
        if decoder is self._compiled_tiled_video_vae_decoder:
            return

        decoder_id = id(decoder)
        if (
            self._compiled_tiled_video_vae_decoder is not None
            and self._compiled_tiled_video_vae_decoder_id == decoder_id
        ):
            self.vae.decoder = self._compiled_tiled_video_vae_decoder
            return

        mode = _video_vae_compile_mode()
        try:
            logger.info("Compiling tiled LTX2 video VAE decoder with mode: %s", mode)
            self._compiled_tiled_video_vae_decoder_base = decoder
            self._compiled_tiled_video_vae_decoder = torch.compile(
                decoder, fullgraph=False, dynamic=False, mode=mode
            )
            self._compiled_tiled_video_vae_decoder_id = decoder_id
            self.vae.decoder = self._compiled_tiled_video_vae_decoder
        except Exception:
            logger.warning(
                "Failed to compile tiled LTX2 video VAE decoder; falling back to eager",
                exc_info=True,
            )
            self._compile_tiled_video_vae_decoder_failed = True

    def _decode_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        direct_decoder = self._get_direct_video_vae_decoder()
        if direct_decoder is not None:
            try:
                return direct_decoder(latents, None, causal=None)
            except Exception:
                if direct_decoder is self._compiled_video_vae_decoder:
                    logger.warning(
                        "Compiled LTX2 video VAE decoder failed; falling back to eager",
                        exc_info=True,
                    )
                    self._compile_video_vae_decoder_failed = True
                else:
                    raise

        self._maybe_compile_tiled_video_vae_decoder()
        try:
            decode_output = self.vae.decode(latents)
        except Exception:
            if self.vae.decoder is self._compiled_tiled_video_vae_decoder:
                logger.warning(
                    "Compiled tiled LTX2 video VAE decoder failed; falling back to eager",
                    exc_info=True,
                )
                self._compile_tiled_video_vae_decoder_failed = True
                eager_decoder = self._compiled_tiled_video_vae_decoder_base
                if isinstance(eager_decoder, torch.nn.Module):
                    self.vae.decoder = eager_decoder
                    decode_output = self.vae.decode(latents)
                else:
                    raise
            else:
                raise
        if isinstance(decode_output, tuple):
            return decode_output[0]
        if hasattr(decode_output, "sample"):
            return decode_output.sample
        return decode_output

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(stage_name, "vae", target_dtype=torch.bfloat16),
            ComponentUse(stage_name, "audio_vae"),
            ComponentUse(stage_name, "vocoder"),
        ]

    @staticmethod
    def _ltx2_should_externally_denorm_video_latents(server_args: ServerArgs) -> bool:
        arch_config = server_args.pipeline_config.vae_config.arch_config
        return str(getattr(arch_config, "video_decoder_variant", "ltx_2")) != "ltx_2_3"

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        self.load_model()

        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        original_dtype = vae_dtype
        with self.use_declared_component(component_name="vae", module=self.vae) as vae:
            assert vae is not None
            self.vae = vae
            self.vae.eval()
            with _decode_profile_scope(batch, "video_latents_to_device"):
                latents = batch.latents.to(
                    get_local_torch_device(), dtype=torch.bfloat16
                )
                if self._ltx2_should_externally_denorm_video_latents(server_args):
                    std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(latents)
                    mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(latents)
                    latents = latents * std + mean
            with _decode_profile_scope(batch, "video_preprocess"):
                latents = server_args.pipeline_config.preprocess_decoding(
                    latents, server_args, vae=self.vae
                )

            with _decode_profile_scope(batch, "video_vae_decode"):
                with torch.autocast(
                    device_type=current_platform.device_type,
                    dtype=vae_dtype,
                    enabled=vae_autocast_enabled,
                ):
                    try:
                        if server_args.pipeline_config.vae_tiling:
                            self.vae.enable_tiling()
                    except Exception:
                        pass
                    video = self._decode_video_latents(latents)

            with _decode_profile_scope(batch, "video_vae_to_original_dtype"):
                self.vae.to(original_dtype)
        with _decode_profile_scope(batch, "video_postprocess"):
            if _fast_video_postprocess_enabled():
                video = _postprocess_video_to_uint8_hwc_tensor(
                    self.video_processor, video
                )
            else:
                video = self.video_processor.postprocess_video(video, output_type="np")

        output_batch = OutputBatch(
            output=video,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            trajectory_decoded=None,
            metrics=batch.metrics,
        )

        # 2. Decode Audio
        try:
            audio_latents = batch.audio_latents
        except AttributeError:
            audio_latents = None
        if audio_latents is not None:
            # Ensure device/dtype
            device = get_local_torch_device()
            with self.use_declared_component(
                component_name="audio_vae",
                module=self.audio_vae,
            ) as audio_vae:
                assert audio_vae is not None
                self.audio_vae = audio_vae
                self.audio_vae.eval()
                try:
                    dtype = self.audio_vae.dtype
                except AttributeError:
                    dtype = None
                if dtype is None:
                    try:
                        dtype = next(self.audio_vae.parameters()).dtype
                    except StopIteration:
                        dtype = torch.float32
                with _decode_profile_scope(batch, "audio_latents_prepare"):
                    audio_latents = audio_latents.to(device, dtype=dtype)
                    try:
                        latents_std = self.audio_vae.latents_std
                    except AttributeError:
                        latents_std = None
                    if isinstance(latents_std, torch.Tensor) and torch.all(
                        latents_std == 0
                    ):
                        logger.warning(
                            "audio_vae.latents_std is all zeros; audio denorm may be incorrect."
                        )
                    try:
                        latents_mean = self.audio_vae.latents_mean
                    except AttributeError:
                        latents_mean = None
                    if isinstance(latents_mean, torch.Tensor) and isinstance(
                        latents_std, torch.Tensor
                    ):
                        latents_mean = latents_mean.to(device=device, dtype=dtype)
                        latents_std = latents_std.to(device=device, dtype=dtype)
                        if audio_latents.ndim == 4:
                            latents_mean = latents_mean.view(
                                1, audio_latents.shape[1], 1, audio_latents.shape[3]
                            )
                            latents_std = latents_std.view(
                                1, audio_latents.shape[1], 1, audio_latents.shape[3]
                            )
                        audio_latents = audio_latents * latents_std + latents_mean

                with _decode_profile_scope(batch, "audio_vae_decode"):
                    with torch.no_grad():
                        # Decode latents to spectrogram
                        spectrogram = self.audio_vae.decode(
                            audio_latents, return_dict=False
                        )[0]

            with self.use_declared_component(
                component_name="vocoder",
                module=self.vocoder,
            ) as vocoder:
                assert vocoder is not None
                self.vocoder = vocoder
                self.vocoder.eval()
                if hasattr(self.vocoder, "conv_in") and hasattr(
                    self.vocoder.conv_in, "in_channels"
                ):
                    expected_in = int(self.vocoder.conv_in.in_channels)
                    actual_in = int(spectrogram.shape[1]) * int(spectrogram.shape[3])
                    if actual_in != expected_in:
                        raise ValueError(
                            f"Vocoder expects channels*mel_bins={expected_in}, got {actual_in} from spectrogram shape {tuple(spectrogram.shape)}"
                        )
                with _decode_profile_scope(batch, "vocoder"):
                    # Decode spectrogram to waveform
                    with torch.no_grad():
                        waveform = self.vocoder(spectrogram)
            with _decode_profile_scope(batch, "audio_cpu_copy"):
                output_batch.audio = waveform.cpu().float()
            try:
                pipeline_audio_cfg = server_args.pipeline_config.audio_vae_config
            except AttributeError:
                pipeline_audio_cfg = None
            try:
                pipeline_audio_arch = pipeline_audio_cfg.arch_config  # type: ignore[union-attr]
            except AttributeError:
                pipeline_audio_arch = None
            try:
                pipeline_audio_sr = pipeline_audio_arch.sample_rate  # type: ignore[union-attr]
            except AttributeError:
                pipeline_audio_sr = None

            try:
                vocoder_sr = self.vocoder.sample_rate
            except AttributeError:
                vocoder_sr = None
            try:
                audio_vae_sr = self.audio_vae.sample_rate
            except AttributeError:
                audio_vae_sr = None
            output_batch.audio_sample_rate = (
                vocoder_sr or audio_vae_sr or pipeline_audio_sr
            )

        self._save_stage1_export_if_requested(batch, server_args)
        return output_batch

    def _save_stage1_export_if_requested(
        self, batch: Req, server_args: ServerArgs
    ) -> None:
        if batch.extra.get("_ltx2_stage1_export_decode_active"):
            return
        output_path = _ltx2_stage1_export_path(batch)
        if not output_path or not get_is_main_process():
            return
        stage1_latents = batch.extra.get("ltx2_stage1_export_latents")
        if not isinstance(stage1_latents, torch.Tensor):
            return

        original_latents = batch.latents
        original_audio_latents = batch.audio_latents
        original_flag = batch.extra.get("_ltx2_stage1_export_decode_active")
        batch.latents = stage1_latents
        batch.audio_latents = batch.extra.get("ltx2_stage1_export_audio_latents")
        batch.extra["_ltx2_stage1_export_decode_active"] = True
        try:
            stage1_output_batch = self.forward(batch, server_args)
            if stage1_output_batch.output is not None:
                saved_paths = _save_ltx2_output_batch(
                    output_batch=stage1_output_batch,
                    batch=batch,
                    output_path=output_path,
                )
                logger.info("Saved LTX2 stage1 pre-refine output: %s", saved_paths)
        except Exception:
            logger.warning(
                "Failed to save LTX2 stage1 pre-refine output", exc_info=True
            )
        finally:
            batch.latents = original_latents
            batch.audio_latents = original_audio_latents
            if original_flag is None:
                batch.extra.pop("_ltx2_stage1_export_decode_active", None)
            else:
                batch.extra["_ltx2_stage1_export_decode_active"] = original_flag


class LTX2Stage1ExportStage(PipelineStage):
    """Capture pre-refinement two-stage latents for deferred Stage1 export."""

    def __init__(self, vae, audio_vae, vocoder, pipeline=None):
        super().__init__()

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return []

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        output_path = _ltx2_stage1_export_path(batch)
        if not output_path:
            return batch

        if "ltx2_stage1_export_latents" not in batch.extra and isinstance(
            batch.latents, torch.Tensor
        ):
            batch.extra["ltx2_stage1_export_latents"] = batch.latents.detach().cpu()
        if "ltx2_stage1_export_audio_latents" not in batch.extra and isinstance(
            batch.audio_latents, torch.Tensor
        ):
            batch.extra["ltx2_stage1_export_audio_latents"] = (
                batch.audio_latents.detach().cpu()
            )

        return batch
