import os

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    is_ltx23_native_variant,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    is_ltx2_two_stage_pipeline_name,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _load_saved_latents(path: str, tensor_name: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        tensor = payload
    elif isinstance(payload, dict):
        tensor = None
        for key in ("latents", tensor_name, f"{tensor_name}_latents"):
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                tensor = value
                break
        if tensor is None:
            raise ValueError(
                f"{path} does not contain a tensor under latents/{tensor_name}/{tensor_name}_latents"
            )
    else:
        raise TypeError(f"{path} contains unsupported payload type {type(payload)!r}")

    if tensor.ndim != 3:
        raise ValueError(
            f"{path} contains {tensor_name} latents with shape {tuple(tensor.shape)}, expected packed [B, S, D]"
        )
    return tensor


def _dump_stage1_initial_latents(tensor_name: str, tensor: torch.Tensor) -> None:
    dump_dir = os.environ.get("SGLANG_LTX2_DUMP_STAGE1_INITIAL_LATENTS_DIR")
    if not dump_dir:
        return
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"sglang_stage1_{tensor_name}_initial.pt")
    torch.save(
        {
            "latents": tensor.detach().cpu(),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        },
        path,
    )


class LTX2AVLatentPreparationStage(LatentPreparationStage):
    """
    LTX-2 specific latent preparation stage that handles both video and audio latents.
    """

    def __init__(self, scheduler, transformer=None, audio_vae=None):
        super().__init__(scheduler, transformer)
        self.audio_vae = audio_vae

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify latent preparation stage inputs."""
        result = VerificationResult()
        result.add_check(
            "prompt_or_embeds",
            None,
            lambda _: V.string_or_list_strings(batch.prompt)
            or V.list_not_empty(batch.prompt_embeds)
            or V.is_tensor(batch.prompt_embeds),
        )

        if isinstance(batch.prompt_embeds, list):
            result.add_check("prompt_embeds", batch.prompt_embeds, V.list_of_tensors)
        else:
            result.add_check("prompt_embeds", batch.prompt_embeds, V.is_tensor)

        result.add_check(
            "num_videos_per_prompt", batch.num_outputs_per_prompt, V.positive_int
        )
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        return result

    def _get_latent_dtype(
        self,
        batch: Req,
        server_args: ServerArgs,
    ):
        if is_ltx23_native_variant(server_args.pipeline_config.vae_config.arch_config):
            if is_ltx2_two_stage_pipeline_name(server_args.pipeline_class_name):
                return server_args.pipeline_config.get_latent_dtype(
                    batch.prompt_embeds[0].dtype
                )
            return torch.float32
        return torch.float32

    @staticmethod
    def _packed_video_latent_shape(
        latent_shape: tuple[int, int, int, int, int],
        pipeline_config,
    ) -> tuple[int, int, int]:
        batch_size, channels, num_frames, height, width = latent_shape
        patch_size_t = int(pipeline_config.patch_size_t)
        patch_size = int(pipeline_config.patch_size)
        return (
            batch_size,
            (num_frames // patch_size_t)
            * (height // patch_size)
            * (width // patch_size),
            channels * patch_size_t * patch_size * patch_size,
        )

    @staticmethod
    def _packed_audio_latent_shape(
        latent_shape: tuple[int, int, int, int],
    ) -> tuple[int, int, int]:
        batch_size, channels, latent_length, mel_bins = latent_shape
        return (batch_size, latent_length, channels * mel_bins)

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if not is_ltx23_native_variant(
            server_args.pipeline_config.vae_config.arch_config
        ):
            batch = super().forward(batch, server_args)

            try:
                generate_audio = batch.generate_audio
            except AttributeError:
                generate_audio = True
            if not generate_audio:
                batch.audio_latents = None
                batch.raw_audio_latent_shape = None
                return batch

            device = get_local_torch_device()
            dtype = self._get_latent_dtype(batch, server_args)
            generator = batch.generator

            audio_latents = batch.audio_latents
            batch_size = batch.batch_size
            num_frames = batch.num_frames

            if audio_latents is None:
                shape = server_args.pipeline_config.prepare_audio_latent_shape(
                    batch, batch_size, num_frames
                )

                audio_latents = randn_tensor(
                    shape, generator=generator, device=device, dtype=dtype
                )
            else:
                audio_latents = audio_latents.to(device)

            audio_latents = server_args.pipeline_config.maybe_pack_audio_latents(
                audio_latents, batch_size, batch
            )

            batch.audio_latents = audio_latents
            batch.raw_audio_latent_shape = audio_latents.shape
            return batch

        # 1. Prepare video latents directly in packed token space.
        # Official LTX-2.3 pipelines sample noise after patchify; generating unpacked
        # [B, C, F, H, W] noise and packing afterwards changes token ordering.
        latent_num_frames = self.adjust_video_length(batch, server_args)
        batch_size = batch.batch_size
        dtype = self._get_latent_dtype(batch, server_args)
        device = get_local_torch_device()
        generator = batch.generator

        latents = batch.latents
        num_frames = (
            latent_num_frames if latent_num_frames is not None else batch.num_frames
        )

        video_latents_path = os.environ.get("SGLANG_LTX2_STAGE1_VIDEO_LATENTS_PATH")
        if latents is None and video_latents_path:
            latents = _load_saved_latents(video_latents_path, "video").to(
                device=device, dtype=dtype
            )
            batch.extra["ltx2_stage1_packed_video_shape"] = tuple(latents.shape)

            latent_ids = server_args.pipeline_config.maybe_prepare_latent_ids(latents)
            if latent_ids is not None:
                batch.latent_ids = latent_ids.to(device=device)
        elif latents is None:
            latent_shape = server_args.pipeline_config.prepare_latent_shape(
                batch, batch_size, num_frames
            )
            packed_video_shape = self._packed_video_latent_shape(
                latent_shape, server_args.pipeline_config
            )
            latents = randn_tensor(
                packed_video_shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            batch.extra["ltx2_stage1_packed_video_shape"] = tuple(packed_video_shape)

            latent_ids = server_args.pipeline_config.maybe_prepare_latent_ids(latents)
            if latent_ids is not None:
                batch.latent_ids = latent_ids.to(device=device)
        else:
            latents = latents.to(device)
            latents = server_args.pipeline_config.maybe_pack_latents(
                latents, batch_size, batch
            )
            batch.extra["ltx2_stage1_packed_video_shape"] = tuple(latents.shape)

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma

        batch.latents = latents
        batch.raw_latent_shape = latents.shape
        _dump_stage1_initial_latents("video", latents)

        # 2. Prepare Audio Latents (optional)
        # Default to True if not specified
        try:
            generate_audio = batch.generate_audio
        except AttributeError:
            generate_audio = True
        if not generate_audio:
            batch.audio_latents = None
            batch.raw_audio_latent_shape = None
            return batch

        audio_latents = batch.audio_latents

        audio_latents_path = os.environ.get("SGLANG_LTX2_STAGE1_AUDIO_LATENTS_PATH")
        if audio_latents is None and audio_latents_path:
            audio_latents = _load_saved_latents(audio_latents_path, "audio").to(
                device=device, dtype=dtype
            )
            batch.extra["ltx2_stage1_packed_audio_shape"] = tuple(audio_latents.shape)
        elif audio_latents is None:
            latent_shape = server_args.pipeline_config.prepare_audio_latent_shape(
                batch, batch_size, batch.num_frames
            )
            packed_audio_shape = self._packed_audio_latent_shape(latent_shape)
            audio_latents = randn_tensor(
                packed_audio_shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            batch.extra["ltx2_stage1_packed_audio_shape"] = tuple(packed_audio_shape)
        else:
            audio_latents = audio_latents.to(device)
            audio_latents = server_args.pipeline_config.maybe_pack_audio_latents(
                audio_latents, batch_size, batch
            )
            batch.extra["ltx2_stage1_packed_audio_shape"] = tuple(audio_latents.shape)

        # Store in batch
        batch.audio_latents = audio_latents
        batch.raw_audio_latent_shape = audio_latents.shape
        _dump_stage1_initial_latents("audio", audio_latents)

        return batch
