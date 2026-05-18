from diffusion.refiner.vendor.ltx_core.loader.sd_ops import SDOps
from diffusion.refiner.vendor.ltx_core.model.model_protocol import ModelConfigurator
from diffusion.refiner.vendor.ltx_core.model.video_vae.enums import (
    LogVarianceType,
    NormLayerType,
    PaddingModeType,
)
from diffusion.refiner.vendor.ltx_core.model.video_vae.video_vae import VideoDecoder, VideoEncoder


class VideoEncoderConfigurator(ModelConfigurator[VideoEncoder]):
    """Configurator for creating a video VAE Encoder from a configuration dictionary."""

    @classmethod
    def from_config(cls: type[VideoEncoder], config: dict) -> VideoEncoder:
        config = config.get("vae", {})
        convolution_dimensions = config.get("dims", 3)
        in_channels = config.get("in_channels", 3)
        latent_channels = config.get("latent_channels", 128)
        spatial_padding_mode = PaddingModeType(config.get("spatial_padding_mode", "zeros"))
        encoder_blocks = config.get("encoder_blocks", [])
        patch_size = config.get("patch_size", 4)
        norm_layer_str = config.get("norm_layer", "pixel_norm")
        latent_log_var_str = config.get("latent_log_var", "uniform")

        return VideoEncoder(
            convolution_dimensions=convolution_dimensions,
            in_channels=in_channels,
            out_channels=latent_channels,
            encoder_blocks=encoder_blocks,
            patch_size=patch_size,
            norm_layer=NormLayerType(norm_layer_str),
            latent_log_var=LogVarianceType(latent_log_var_str),
            encoder_spatial_padding_mode=spatial_padding_mode,
        )


class VideoDecoderConfigurator(ModelConfigurator[VideoDecoder]):
    """Configurator for creating a video VAE Decoder from a configuration dictionary."""

    @classmethod
    def from_config(cls: type[VideoDecoder], config: dict) -> VideoDecoder:
        config = config.get("vae", {})
        convolution_dimensions = config.get("dims", 3)
        latent_channels = config.get("latent_channels", 128)
        spatial_padding_mode = PaddingModeType(config.get("spatial_padding_mode", "reflect"))
        out_channels = config.get("out_channels", 3)
        decoder_blocks = config.get("decoder_blocks", [])
        patch_size = config.get("patch_size", 4)
        norm_layer_str = config.get("norm_layer", "pixel_norm")
        causal = config.get("causal_decoder", False)
        timestep_conditioning = config.get("timestep_conditioning", True)
        base_channels = config.get("decoder_base_channels", 128)

        return VideoDecoder(
            convolution_dimensions=convolution_dimensions,
            in_channels=latent_channels,
            out_channels=out_channels,
            decoder_blocks=decoder_blocks,
            patch_size=patch_size,
            norm_layer=NormLayerType(norm_layer_str),
            causal=causal,
            timestep_conditioning=timestep_conditioning,
            decoder_spatial_padding_mode=spatial_padding_mode,
            base_channels=base_channels,
        )


VAE_DECODER_COMFY_KEYS_FILTER = (
    SDOps("VAE_DECODER_COMFY_KEYS_FILTER")
    .with_matching(prefix="vae.decoder.")
    .with_matching(prefix="vae.per_channel_statistics.")
    .with_replacement("vae.decoder.", "")
    .with_replacement("vae.per_channel_statistics.", "per_channel_statistics.")
)

VAE_ENCODER_COMFY_KEYS_FILTER = (
    SDOps("VAE_ENCODER_COMFY_KEYS_FILTER")
    .with_matching(prefix="vae.encoder.")
    .with_matching(prefix="vae.per_channel_statistics.")
    .with_replacement("vae.encoder.", "")
    .with_replacement("vae.per_channel_statistics.", "per_channel_statistics.")
)
