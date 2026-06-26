# SPDX-License-Identifier: Apache-2.0
#
# Pipeline configuration for SANA-Video text-to-video (T2V).
#
# Modeled on the Wan T2V config (same Wan VAE for the 480p checkpoint, 5D
# latents, standard pipeline stages). Differences from Wan: Gemma2 text encoder
# (caption_channels 2304), SANA-Video DiT, and the DPMSolverMultistep scheduler
# loaded from the checkpoint. 3D RoPE is computed inside the DiT, so the cond
# kwargs only need encoder_attention_mask (like image SANA).
#
# The 720p checkpoint uses the LTX-2 video VAE instead of Wan; use
# SanaVideo720PPipelineConfig for it.

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.sana_video import SanaVideoConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.gemma2 import Gemma2Config
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


def sana_video_postprocess_text(
    outputs: BaseEncoderOutput, _text_inputs
) -> torch.Tensor:
    # SANA-Video uses the final hidden state of Gemma2 directly as text conditioning.
    return outputs.last_hidden_state


@dataclass
class SanaVideoPipelineConfig(PipelineConfig):
    """SANA-Video 480p (Wan VAE) text-to-video pipeline config."""

    task_type: ModelTaskType = ModelTaskType.T2V

    # Standard classifier-free guidance via guidance_scale; no embedded guidance token.
    should_use_guidance: bool = False
    enable_autocast: bool = False

    vae_tiling: bool = False
    vae_sp: bool = False
    vae_precision: str = "fp32"
    precision: str = "bf16"

    dit_config: DiTConfig = field(default_factory=SanaVideoConfig)
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Gemma2Config(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [{"padding": True, "return_attention_mask": True}]
    )
    preprocess_text_funcs: tuple[Callable[[str], str] | None, ...] = field(
        default_factory=lambda: (None,)
    )
    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (sana_video_postprocess_text,)
    )

    def __post_init__(self):
        # T2V: only the VAE decoder is needed.
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        out = {}
        m = batch.prompt_attention_mask
        if isinstance(m, (list, tuple)):
            out["encoder_attention_mask"] = m[0] if m else None
        elif m is not None:
            out["encoder_attention_mask"] = m
        return out

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        out = {}
        m = batch.negative_attention_mask
        if isinstance(m, (list, tuple)):
            out["encoder_attention_mask"] = m[0] if m else None
        elif m is not None:
            out["encoder_attention_mask"] = m
        return out


@dataclass
class SanaVideo720PPipelineConfig(SanaVideoPipelineConfig):
    """SANA-Video 720p variant: LTX-2 video VAE + sample_size=22 DiT.

    The DiT in_channels/out_channels and VAE are overridden in __post_init__ to
    match the 720p checkpoint (LTX-2 VAE, 128 latent channels).
    """

    def __post_init__(self):
        # Lazy import to avoid a hard dependency when only 480p is used.
        from sglang.multimodal_gen.configs.models.vaes.ltx_video import (
            LTXVideoVAEConfig,
        )

        self.dit_config.arch_config.sample_size = 22
        self.dit_config.arch_config.in_channels = 128
        self.dit_config.arch_config.out_channels = 128
        self.dit_config.arch_config.num_channels_latents = 128
        self.vae_config = LTXVideoVAEConfig()
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True

    def preprocess_decoding(self, latents, server_args=None, vae=None):
        # SANA-Video 720p uses the LTX-2 (AutoencoderKLLTX2Video) VAE, whose
        # per-channel latent denormalization is applied EXTERNALLY by the
        # reference SanaVideoPipeline (latents = latents * latents_std +
        # latents_mean, using the 128-ch stats stored in the checkpoint). The
        # generic decode stage only applies scaling_factor (=1.0, a no-op), so
        # without this the decoder receives un-denormalized latents -> visually
        # wrong output. Mirrors DecodingAVStage._ltx2_should_externally_denorm.
        std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents)
        mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents)
        return latents * std + mean
