import dataclasses
from pathlib import Path

import safetensors
import torch

from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.model.video_vae.enums import LogVarianceType, NormLayerType, PaddingModeType
from ltx_core.model.video_vae.transformer.dsl_kernels import (
    FUSED_CTX_BIAS_BUFFER,
    FUSED_CTX_WEIGHT_BUFFER,
    NA_SOFTMAX_BOUND_BUFFER,
)
from ltx_core.model.video_vae.video_vae import ConvVideoDecoder, DiffusionVideoDecoder, VideoDecoder, VideoEncoder

_CONV_VAE_CLASS_NAME = "CausalVideoAutoencoder"


def _vae_class_name_from_metadata(metadata: dict) -> str:
    """Return the top-level ``vae._class_name`` from checkpoint metadata."""
    return metadata.get("config", {}).get("vae", {}).get("_class_name", _CONV_VAE_CLASS_NAME)


def is_diffusion_video_vae(checkpoint_path: str) -> bool:
    """Whether ``checkpoint_path`` metadata describes a diffusion video VAE.
    Decision is based on ``config.vae._class_name`` (same field
    ``VideoDecoderConfigurator`` uses), not on how the path was supplied.
    A standalone conv-VAE extract must still get conv SDOps / memory-efficient
    decode.
    """
    metadata = SafetensorsModelStateDictLoader().metadata(checkpoint_path)
    return _vae_class_name_from_metadata(metadata) != _CONV_VAE_CLASS_NAME


def _prepare_video_encoder_kwargs(config: dict) -> dict:
    """Extract ``VideoEncoder`` init kwargs from a ``vae`` configuration dictionary.
    Two checkpoint layouts:
    - Flat ``CausalVideoAutoencoder`` (LTX-2): fields live on ``vae`` itself.
      Latent width is ``latent_channels``; top-level ``out_channels`` is decoder
      RGB and must not be used as the encoder's latent size.
    - Nested ``CausalDiffusionVAE``: fields live under ``vae.encoder`` with
      ``blocks`` / ``out_channels`` naming (``out_channels`` is the latent width).
    """
    if "encoder" in config:
        encoder_config = config["encoder"]
        out_channels = encoder_config.get("out_channels", config.get("latent_channels", 128))
        encoder_blocks = encoder_config.get("blocks", encoder_config.get("encoder_blocks", []))
    else:
        encoder_config = config
        out_channels = config.get("latent_channels", 128)
        encoder_blocks = config.get("encoder_blocks", [])

    return {
        "convolution_dimensions": encoder_config.get("dims", config.get("dims", 3)),
        "in_channels": encoder_config.get("in_channels", 3),
        "out_channels": out_channels,
        "encoder_blocks": encoder_blocks,
        "patch_size": encoder_config.get("patch_size", 4),
        "norm_layer": NormLayerType(encoder_config.get("norm_layer", "pixel_norm")),
        "latent_log_var": LogVarianceType(encoder_config.get("latent_log_var", "uniform")),
        "encoder_spatial_padding_mode": PaddingModeType(
            encoder_config.get(
                "spatial_padding_mode",
                config.get("encoder_spatial_padding_mode", "zeros"),
            )
        ),
    }


class VideoEncoderConfigurator(ModelConfigurator[VideoEncoder]):
    """Configurator for creating a video VAE Encoder from a configuration dictionary."""

    @classmethod
    def from_metadata(cls, metadata: dict) -> VideoEncoder:
        config = metadata.get("config", {}).get("vae", {})
        return VideoEncoder(**_prepare_video_encoder_kwargs(config))


def _build_conv_video_decoder(config: dict) -> ConvVideoDecoder:
    """Build a ``ConvVideoDecoder`` from a ``vae`` configuration dictionary."""
    return ConvVideoDecoder(
        convolution_dimensions=config.get("dims", 3),
        in_channels=config.get("latent_channels", 128),
        out_channels=config.get("out_channels", 3),
        decoder_blocks=config.get("decoder_blocks", []),
        patch_size=config.get("patch_size", 4),
        norm_layer=NormLayerType(config.get("norm_layer", "pixel_norm")),
        causal=config.get("causal_decoder", False),
        timestep_conditioning=config.get("timestep_conditioning", True),
        decoder_spatial_padding_mode=PaddingModeType(config.get("spatial_padding_mode", "reflect")),
        base_channels=config.get("decoder_base_channels", 128),
    )


def _build_diffusion_video_decoder(config: dict) -> DiffusionVideoDecoder:
    """Build a ``DiffusionVideoDecoder`` from a ``vae`` configuration dictionary.
    Real checkpoints (``_class_name: "CausalDiffusionVAE"``) nest decoder
    hyperparameters under ``vae.decoder``; fall back to the flat ``vae`` dict
    itself for configs that put them there directly. Architecture fields
    absent from config (e.g. ``t_emb_dim``, stage ladders) are left at the
    class defaults by omitting them from the constructor call.
    """
    decoder_config = config.get("decoder", config)
    # Optional architecture overrides - only pass when present so class defaults apply.
    architecture: dict = {}
    if "stage_channels" in decoder_config:
        architecture["stage_channels"] = tuple(decoder_config["stage_channels"])
    if "stage_depths" in decoder_config:
        architecture["stage_depths"] = tuple(decoder_config["stage_depths"])
    if "stage_kernels" in decoder_config:
        architecture["stage_kernels"] = tuple(tuple(kernel) for kernel in decoder_config["stage_kernels"])
    if "upsamples" in decoder_config:
        architecture["upsamples"] = tuple(
            (tuple(stride), reduction) for stride, reduction in decoder_config["upsamples"]
        )
    if "stage5_kernel" in decoder_config:
        architecture["stage5_kernel"] = tuple(decoder_config["stage5_kernel"])
    if "stage5_channels" in decoder_config:
        architecture["stage5_channels"] = decoder_config["stage5_channels"]

    return DiffusionVideoDecoder(
        in_channels=decoder_config.get("in_channels", config.get("latent_channels", 128)),
        out_channels=decoder_config.get("out_channels", 3),
        patch_size=decoder_config.get("patch_size", 4),
        head_dim=decoder_config.get("head_dim", decoder_config.get("na_head_dim", 64)),
        t_emb_dim=decoder_config.get("t_emb_dim", 384),
        default_num_inference_steps=decoder_config.get("default_num_inference_steps", 2),
        timestep_scale_multiplier=decoder_config.get("timestep_scale_multiplier", 1.0),
        # Sibling of "decoder" at the top vae level (CausalDiffusionVAE.from_config
        # reads it there too), not nested under decoder_config.
        model_output_type=config.get("model_output_type", "v"),
        **architecture,
    )


def stage4_channels_from_decoder_config(decoder_config: dict) -> int:
    """Channels at stage-4 input (= output of deterministic stages 1-3).
    ``stage_channels[3]`` after the first three upsample hops.
    """
    if "stage_channels" not in decoder_config:
        raise ValueError("VAE decoder config missing stage_channels")
    channels = decoder_config["stage_channels"]
    if len(channels) < 4:
        raise ValueError(f"stage_channels must have length >= 4, got {len(channels)}")
    dim = int(channels[3])
    if dim < 1:
        raise ValueError(f"stage4_channels must be >= 1, got {dim}")
    return dim


def stage5_channels_from_decoder_config(decoder_config: dict) -> int:
    """Stage-5 feature width from VAE config: ``num_heads * head_dim``.
    Prefers explicit ``stage5_channels``, else ``stage_channels[-1]``. Both must be
    divisible by ``head_dim`` (required in config as ``head_dim`` / ``na_head_dim``).
    """
    if "head_dim" in decoder_config:
        head_dim = int(decoder_config["head_dim"])
    elif "na_head_dim" in decoder_config:
        head_dim = int(decoder_config["na_head_dim"])
    else:
        raise ValueError("VAE decoder config missing head_dim / na_head_dim")
    if head_dim < 1:
        raise ValueError(f"head_dim must be >= 1, got {head_dim}")
    if "stage5_channels" in decoder_config:
        dim = int(decoder_config["stage5_channels"])
    elif "stage_channels" in decoder_config:
        dim = int(decoder_config["stage_channels"][-1])
    else:
        raise ValueError("VAE decoder config missing stage5_channels / stage_channels")
    if dim % head_dim != 0:
        raise ValueError(f"stage5 dim {dim} is not divisible by head_dim={head_dim}")
    return (dim // head_dim) * head_dim


def diffvae_tiling_geometry_from_vae_config(config: dict) -> dict:
    """Resolve DiffVAE tiling geometry from a ``vae`` config dict (no module build).
    Returns kwargs for :func:`ltx_core.model.video_vae.diffusion_tiling.recommended_decode_tiling_config`
    except resolution / memory fields: ``tile_halos``, ``pixel_scale``, ``min_tile_size_s4``,
    ``patch_size``, ``stage5_channels``, ``stage4_channels``, ``upsample_strides``,
    ``natten_trailing_pad_latent_frames``, ``out_channels``.
    """
    # Local import: avoid a hard cycle at module import time.
    from ltx_core.model.video_vae import diffusion_tiling as dt  # noqa: PLC0415
    from ltx_core.model.video_vae import diffusion_video_decoder as dvd  # noqa: PLC0415

    decoder_config = config.get("decoder", config)
    stage_kernels = (
        tuple(tuple(k) for k in decoder_config["stage_kernels"])
        if "stage_kernels" in decoder_config
        else dvd._L_STAGE_KERNELS
    )
    stage_depths = (
        tuple(decoder_config["stage_depths"]) if "stage_depths" in decoder_config else dvd._DIFF_STAGE_DEPTHS_DEFAULT
    )
    upsamples = (
        tuple((tuple(stride), reduction) for stride, reduction in decoder_config["upsamples"])
        if "upsamples" in decoder_config
        else dvd._L_UPSAMPLES
    )
    stage5_kernel = (
        tuple(decoder_config["stage5_kernel"]) if "stage5_kernel" in decoder_config else dvd._DIFF_STAGE5_KERNEL_DEFAULT
    )
    patch_size = int(decoder_config.get("patch_size", 4))
    up3_stride = upsamples[3][0]
    return {
        "tile_halos": dt.compute_tile_halos(
            stage_kernels[3],
            stage_depths[3],
            stage5_kernel,
            stage_depths[-1],
            up3_stride,
        ),
        "pixel_scale": dt.stage4_to_pixel_scale_factors(up3_stride, patch_size),
        "min_tile_size_s4": dt.compute_tile_min_size(stage_kernels[3], stage5_kernel, up3_stride),
        "patch_size": patch_size,
        "stage5_channels": stage5_channels_from_decoder_config(decoder_config),
        "stage4_channels": stage4_channels_from_decoder_config(decoder_config),
        "upsample_strides": tuple(stride for stride, _reduction in upsamples),
        "natten_trailing_pad_latent_frames": (stage_kernels[0][0] // 2) * 2,
        "out_channels": int(decoder_config.get("out_channels", 3)),
    }


def estimate_diffusion_decoder_weight_bytes(checkpoint_path: str | Path) -> int:
    """Sum on-disk bf16-sized nbytes for ``vae.decoder`` / ``decoder`` tensors (header only)."""
    path = Path(checkpoint_path)
    total = 0
    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118  # safetensors SafeOpen is not a Mapping
            if not (key.startswith(("vae.decoder.", "decoder.")) or "per_channel_statistics" in key):
                continue
            shape = handle.get_slice(key).get_shape()
            n = 1
            for dim in shape:
                n *= int(dim)
            total += n * 2  # bf16
    return total


class VideoDecoderConfigurator(ModelConfigurator[VideoDecoder]):
    """Configurator for creating a video VAE Decoder from a configuration dictionary."""

    @classmethod
    def from_metadata(cls, metadata: dict) -> VideoDecoder:
        config = metadata.get("config", {}).get("vae", {})
        if _vae_class_name_from_metadata(metadata) == _CONV_VAE_CLASS_NAME:
            return _build_conv_video_decoder(config)
        return _build_diffusion_video_decoder(config)


# Accept both monolithic ``vae.*`` prefixes and Comfy-split bare ``encoder.*`` /
# ``decoder.*`` keys (standalone video VAE files under ``vae/``).
VAE_DECODER_COMFY_KEYS_FILTER = (
    SDOps("VAE_DECODER_COMFY_KEYS_FILTER")
    .with_matching(prefix="vae.decoder.")
    .with_matching(prefix="vae.per_channel_statistics.")
    .with_matching(prefix="decoder.")
    .with_matching(prefix="per_channel_statistics.")
    .with_replacement("vae.decoder.", "")
    .with_replacement("vae.per_channel_statistics.", "per_channel_statistics.")
    # Comfy-split bare keys: strip the leading ``decoder.`` after the vae. rule above.
    .with_replacement("decoder.", "")
)

VAE_ENCODER_COMFY_KEYS_FILTER = (
    SDOps("VAE_ENCODER_COMFY_KEYS_FILTER")
    .with_matching(prefix="vae.encoder.")
    .with_matching(prefix="vae.per_channel_statistics.")
    .with_matching(prefix="encoder.")
    .with_matching(prefix="per_channel_statistics.")
    .with_replacement("vae.encoder.", "")
    .with_replacement("vae.per_channel_statistics.", "per_channel_statistics.")
    .with_replacement("encoder.", "")
)


def _split_fused_qkv_param(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
    """Split fused ``...qkv.weight`` / ``...qkv.bias`` into ``to_q`` / ``to_k`` / ``to_v``.
    Checkpoints store a single Linear(dim, 3*dim) under ``qkv.{weight,bias}``;
    ``QKVProjections`` owns three separate linears, so the loader rewrites keys here.
    """
    if value.shape[0] % 3 != 0:
        msg = f"fused QKV param {key!r} leading dim {value.shape[0]} is not divisible by 3"
        raise ValueError(msg)
    d = value.shape[0] // 3
    # key is "...qkv.weight" or "...qkv.bias"
    leaf = "weight" if key.endswith(".weight") else "bias"
    prefix = key[: -len(leaf)]  # "...qkv."
    return [
        KeyValueOperationResult(f"{prefix}to_q.{leaf}", value[:d].detach().clone()),
        KeyValueOperationResult(f"{prefix}to_k.{leaf}", value[d : 2 * d].detach().clone()),
        KeyValueOperationResult(f"{prefix}to_v.{leaf}", value[2 * d :].detach().clone()),
    ]


_RAW_DIFFUSION_DECODER_PREFIX = "vae.decoder."
_BARE_DIFFUSION_DECODER_PREFIX = "decoder."
_K_NORM_SUFFIX = ".attn.k_norm.weight"
_CONTEXT_PROJ_WEIGHT_SUFFIX = ".context_proj.weight"
_CONTEXT_PROJ_BIAS_SUFFIX = ".context_proj.bias"
_UPSAMPLE3_WEIGHT_KEY = "upsamples.3.proj.weight"
_UPSAMPLE3_BIAS_KEY = "upsamples.3.proj.bias"
_GATE_PARAM_SUFFIXES = (".gate_msa", ".gate_mlp", ".gate_ctx")
# Post-rename Linear leaf → sibling gate key suffix (folded W←g·W, b←g·b).
_GATE_FOLD_TARGETS: tuple[tuple[str, str], ...] = (
    (".attn.proj.weight", ".gate_msa"),
    (".attn.proj.bias", ".gate_msa"),
    (".mlp.w_down.weight", ".gate_mlp"),
    (".mlp.w_down.bias", ".gate_mlp"),
    (".context_proj.weight", ".gate_ctx"),
    (".context_proj.bias", ".gate_ctx"),
)


def _strip_diffusion_decoder_prefix(key: str) -> str | None:
    """Strip ``vae.decoder.`` or bare ``decoder.``; return None if neither matches."""
    if key.startswith(_RAW_DIFFUSION_DECODER_PREFIX):
        return key.removeprefix(_RAW_DIFFUSION_DECODER_PREFIX)
    if key.startswith(_BARE_DIFFUSION_DECODER_PREFIX):
        return key.removeprefix(_BARE_DIFFUSION_DECODER_PREFIX)
    return None


def _read_diff_vae_gates(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Return ``{post_rename_gate_key: tensor}`` for every DiffVAE ``gate_*``.
    Keys are stripped of ``vae.decoder.`` / bare ``decoder.`` so they match the
    names the loader passes to kv-ops after SDOps rename. Streaming load is
    one-key-at-a-time, so siblings must be pre-read before fold ops run (same
    pattern as ``_read_scales`` / ``_build_prequant_fold_sd_ops`` in fp8_cast).
    """
    out: dict[str, torch.Tensor] = {}
    with safetensors.safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118
            stripped = _strip_diffusion_decoder_prefix(key)
            if stripped is None or not key.endswith(_GATE_PARAM_SUFFIXES):
                continue
            out[stripped] = handle.get_tensor(key)
    return out


def _drop_coarse_param(_key: str, _value: torch.Tensor) -> list[KeyValueOperationResult]:
    """Drop bundled DiffVAE preview/coarse head weights (``coarse_*``)."""
    return []


def _gate_key_for_fold_target(param_key: str) -> str | None:
    for leaf, gate_suffix in _GATE_FOLD_TARGETS:
        if param_key.endswith(leaf):
            return param_key[: -len(leaf)] + gate_suffix
    return None


def _fold_gate_into_linear(
    param_key: str, value: torch.Tensor, gates: dict[str, torch.Tensor]
) -> list[KeyValueOperationResult]:
    """Fold a static gate into a Linear weight/bias when present; else pass through."""
    gate_key = _gate_key_for_fold_target(param_key)
    if gate_key is None:
        return [KeyValueOperationResult(param_key, value)]
    gate = gates.get(gate_key)
    if gate is None:
        return [KeyValueOperationResult(param_key, value)]
    gate_f = gate.to(device=value.device, dtype=torch.float32)
    value_f = value.to(dtype=torch.float32)
    if value.ndim == 2:
        folded = gate_f.unsqueeze(1) * value_f
    elif value.ndim == 1:
        folded = gate_f * value_f
    else:
        raise ValueError(f"Unsupported param rank {value.ndim} for gate fold on {param_key}")
    return [KeyValueOperationResult(param_key, folded.to(dtype=value.dtype))]


def _build_diffusion_vae_decoder_sd_ops(
    gates: dict[str, torch.Tensor],
    *,
    stage4_hop: "_Stage4Hop | None" = None,
) -> SDOps:
    """DiffVAE decoder SDOps: rename, QKV split, coarse drop, gate fold/drop.
    *gates* is keyed by post-rename gate param names (see ``_read_diff_vae_gates``).
    Empty map (bundled n1 / QKV unit tests): fold is a no-op; drop never fires.
    *stage4_hop* additionally packs the stage-4 upsample into every block's
    ``context_proj``, which is what lets the fused CuTe DSL block read stage-4 directly.
    """

    def _drop_gate(gate_key: str, _value: torch.Tensor) -> list[KeyValueOperationResult]:
        if gate_key not in gates:
            raise ValueError(
                f"Gate key {gate_key!r} has no matching entry in the pre-read gates dict; "
                f"_read_diff_vae_gates and the loader's rename map have drifted"
            )
        return []

    def _on_attn_proj(param_key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        return _fold_gate_into_linear(param_key, value, gates)

    def _on_mlp_w_down(param_key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        return _fold_gate_into_linear(param_key, value, gates)

    def _on_context_proj(param_key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        return _fold_gate_into_linear(param_key, value, gates)

    def _on_context_proj_weight(param_key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        folded = _fold_gate_into_linear(param_key, value, gates)
        if stage4_hop is None:
            return folded
        return folded + stage4_hop.fused_results(param_key, folded[0].new_value, gates)

    # Drop/fold ops before QKV split so first-match kv-ops never mis-route.
    # Match both monolith ``vae.decoder.*`` and Comfy-split bare ``decoder.*``.
    return (
        SDOps("DIFFUSION_VAE_DECODER_COMFY_KEYS_FILTER")
        .with_matching(prefix="vae.decoder.")
        .with_matching(prefix="decoder.")
        .with_replacement("vae.decoder.", "")
        .with_replacement("decoder.", "")
        .with_replacement("t_embedder.mlp.0.", "t_embedder.timestep_embedder.linear_1.")
        .with_replacement("t_embedder.mlp.2.", "t_embedder.timestep_embedder.linear_2.")
        .with_matching(prefix="vae.per_channel_statistics.")
        .with_matching(prefix="per_channel_statistics.")
        .with_replacement("vae.per_channel_statistics.", "per_channel_statistics.")
        .with_kv_operation(operation=_drop_coarse_param, key_prefix="coarse_")
        .with_kv_operation(operation=_drop_gate, key_suffix=".gate_msa")
        .with_kv_operation(operation=_drop_gate, key_suffix=".gate_mlp")
        .with_kv_operation(operation=_drop_gate, key_suffix=".gate_ctx")
        .with_kv_operation(operation=_on_attn_proj, key_suffix=".attn.proj.weight")
        .with_kv_operation(operation=_on_attn_proj, key_suffix=".attn.proj.bias")
        .with_kv_operation(operation=_on_mlp_w_down, key_suffix=".mlp.w_down.weight")
        .with_kv_operation(operation=_on_mlp_w_down, key_suffix=".mlp.w_down.bias")
        .with_kv_operation(operation=_on_context_proj_weight, key_suffix=".context_proj.weight")
        .with_kv_operation(operation=_on_context_proj, key_suffix=".context_proj.bias")
        .with_kv_operation(operation=_split_fused_qkv_param, key_suffix=".qkv.weight")
        .with_kv_operation(operation=_split_fused_qkv_param, key_suffix=".qkv.bias")
    )


# Empty-gate build: QKV/t_embedder/coarse SDOps for unit tests. Weight loading of
# gated (standalone distilled) checkpoints must use ``video_decoder_sd_ops_for_checkpoint``.
DIFFUSION_VAE_DECODER_COMFY_KEYS_FILTER = _build_diffusion_vae_decoder_sd_ops({})


def _emit_na_softmax_bound(param_key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
    """Pass ``attn.k_norm.weight`` through and emit the DSL kernel's softmax bound beside it.
    The CuTe DSL neighborhood-attention kernel replaces the online row max with a fixed
    per-row offset built from ``sqrt(head_dim) * max|k_norm.weight|``. Deriving it here
    means it is computed once, during load, into the buffer
    ``DiffVAEMode.BLACKWELL_DSL`` registered -- rather than per forward, where reading it
    off the weight would be a device-to-host sync inside a compiled region.
    The extra key is harmless when the kernel is not in use: loading is not strict, so
    a model without the buffer ignores it.
    """
    from ltx_kernels.vae import softmax_row_bound  # noqa: PLC0415

    bound_key = param_key.removesuffix(_K_NORM_SUFFIX) + f".attn.{NA_SOFTMAX_BOUND_BUFFER}"
    return [
        KeyValueOperationResult(param_key, value),
        KeyValueOperationResult(bound_key, softmax_row_bound(value)),
    ]


def pack_fused_upsample_ctx_weights(
    w_upsample: torch.Tensor,
    b_upsample: torch.Tensor,
    w_ctx: torch.Tensor,
    b_ctx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``context_proj(pixel_shuffle(upsample(.)))`` into packed Linear weights.
    Thin wrapper over :func:`ltx_kernels.vae.pack_fused_ctx_weights` so SDOps tests
    can import from ``model_configurator`` without a hard kernels dependency at
    module import time.
    """
    from ltx_kernels.vae import pack_fused_ctx_weights  # noqa: PLC0415

    return pack_fused_ctx_weights(w_upsample, b_upsample, w_ctx, b_ctx)


@dataclasses.dataclass(frozen=True)
class _Stage4Hop:
    """``upsamples[3].proj`` and every ``context_proj.bias``, pre-read from a checkpoint.
    The fused weight a block needs is built from three tensors that arrive at three
    different keys, and a streaming load sees one key at a time -- so the two the
    ``context_proj.weight`` op cannot wait for are read up front. The upsample is one
    tensor for the whole decoder; the biases are one small vector per block. (Same
    pattern as :func:`_read_diff_vae_gates`, which pre-reads the gates for the fold.)
    """

    upsample_weight: torch.Tensor
    upsample_bias: torch.Tensor
    context_biases: dict[str, torch.Tensor]  # post-rename ``....context_proj.bias``

    @classmethod
    def read(cls, checkpoint_path: str | Path) -> "_Stage4Hop":
        upsample_weight = upsample_bias = None
        context_biases: dict[str, torch.Tensor] = {}
        with safetensors.safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118
                if not key.startswith(_RAW_DIFFUSION_DECODER_PREFIX):
                    continue
                short = key.removeprefix(_RAW_DIFFUSION_DECODER_PREFIX)
                if short == _UPSAMPLE3_WEIGHT_KEY:
                    upsample_weight = handle.get_tensor(key)
                elif short == _UPSAMPLE3_BIAS_KEY:
                    upsample_bias = handle.get_tensor(key)
                elif short.endswith(_CONTEXT_PROJ_BIAS_SUFFIX):
                    context_biases[short] = handle.get_tensor(key)
        if upsample_weight is None or upsample_bias is None:
            raise ValueError(
                f"DiffVAE checkpoint {checkpoint_path!r} has no {_UPSAMPLE3_WEIGHT_KEY}, which "
                "na_dsl_kernel needs to fuse the stage-4 hop into context_proj"
            )
        return cls(upsample_weight, upsample_bias, context_biases)

    def fused_results(
        self, weight_key: str, folded_weight: torch.Tensor, gates: dict[str, torch.Tensor]
    ) -> list[KeyValueOperationResult]:
        """The two fused buffers for the block owning ``weight_key``.
        ``folded_weight`` is that block's already gate-folded ``context_proj.weight``; the
        bias gets the same fold here, so the packed pair carries ``gate_ctx`` exactly once
        and matches what the unfused ``context_proj`` would have computed.
        """
        prefix = weight_key.removesuffix(_CONTEXT_PROJ_WEIGHT_SUFFIX)
        bias_key = prefix + _CONTEXT_PROJ_BIAS_SUFFIX
        raw_bias = self.context_biases.get(bias_key)
        if raw_bias is None:
            raise ValueError(f"No pre-read {bias_key!r} to fuse alongside {weight_key!r}")
        folded_bias = _fold_gate_into_linear(bias_key, raw_bias, gates)[0].new_value
        weight, bias = pack_fused_upsample_ctx_weights(
            self.upsample_weight, self.upsample_bias, folded_weight, folded_bias
        )
        return [
            KeyValueOperationResult(f"{prefix}.{FUSED_CTX_WEIGHT_BUFFER}", weight),
            KeyValueOperationResult(f"{prefix}.{FUSED_CTX_BIAS_BUFFER}", bias),
        ]


def video_decoder_sd_ops_for_checkpoint(
    checkpoint_path: str,
    *,
    diffusion_vae: bool | None = None,
    na_dsl_kernel: bool = False,
) -> SDOps:
    """Pick decoder SDOps from checkpoint metadata (conv vs diffusion).
    For diffusion VAEs, pre-reads ``gate_*`` and builds fold/drop ops so standalone
    distilled checkpoints load into the ungated ``DiffusionNABlock``.
    ``na_dsl_kernel`` additionally derives the CuTe DSL kernel's softmax bound from each
    ``k_norm`` weight and fuses ``upsamples[3]`` into every block's ``context_proj``, so
    the fused block reads stage-4 directly instead of a full-resolution context. Must be
    paired with ``DiffVAEMode.BLACKWELL_DSL`` (registers the receiving buffers); apply
    alone leaves NaN / meta / empty buffers that fail at first forward. ``VideoDecoder``
    ties both to one flag -- custom builders must do the same.
    Pass ``diffusion_vae`` when the caller already ran ``is_diffusion_video_vae`` to
    skip a second metadata open.
    """
    if diffusion_vae is None:
        diffusion_vae = is_diffusion_video_vae(checkpoint_path)
    if not diffusion_vae:
        return VAE_DECODER_COMFY_KEYS_FILTER
    gates = _read_diff_vae_gates(checkpoint_path)
    if not na_dsl_kernel:
        return _build_diffusion_vae_decoder_sd_ops(gates)
    return _build_diffusion_vae_decoder_sd_ops(gates, stage4_hop=_Stage4Hop.read(checkpoint_path)).with_kv_operation(
        operation=_emit_na_softmax_bound, key_suffix=_K_NORM_SUFFIX
    )
