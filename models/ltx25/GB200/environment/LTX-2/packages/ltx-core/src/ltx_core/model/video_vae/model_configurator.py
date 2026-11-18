from pathlib import Path

import safetensors
import torch

from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.model.video_vae.enums import LogVarianceType, NormLayerType, PaddingModeType
from ltx_core.model.video_vae.video_vae import ConvVideoDecoder, DiffusionVideoDecoder, VideoDecoder, VideoEncoder

_CONV_VAE_CLASS_NAME = "CausalVideoAutoencoder"


def _vae_class_name_from_metadata(metadata: dict) -> str:
    """Return the top-level ``vae._class_name`` from checkpoint metadata."""
    return metadata.get("config", {}).get("vae", {}).get("_class_name", _CONV_VAE_CLASS_NAME)


def is_diffusion_video_vae(checkpoint_path: str) -> bool:
    """Whether ``checkpoint_path`` metadata describes a diffusion video VAE.
    Decision is based on ``config.vae._class_name`` (same field
    ``VideoDecoderConfigurator`` uses), not on whether the path was passed via
    ``--vae-checkpoint-path``. A standalone conv-VAE extract must still get
    conv SDOps / memory-efficient decode.
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
    # Optional architecture overrides — only pass when present so class defaults apply.
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


class VideoDecoderConfigurator(ModelConfigurator[VideoDecoder]):
    """Configurator for creating a video VAE Decoder from a configuration dictionary."""

    @classmethod
    def from_metadata(cls, metadata: dict) -> VideoDecoder:
        config = metadata.get("config", {}).get("vae", {})
        if _vae_class_name_from_metadata(metadata) == _CONV_VAE_CLASS_NAME:
            return _build_conv_video_decoder(config)
        return _build_diffusion_video_decoder(config)


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


def _read_diff_vae_gates(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Return ``{post_rename_gate_key: tensor}`` for every DiffVAE ``gate_*``.
    Keys are stripped of ``vae.decoder.`` so they match the names the loader
    passes to kv-ops after SDOps rename. Streaming load is one-key-at-a-time,
    so siblings must be pre-read before fold ops run (same pattern as
    ``_read_scales`` / ``_build_prequant_fold_sd_ops`` in fp8_cast).
    """
    out: dict[str, torch.Tensor] = {}
    with safetensors.safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118
            if not key.startswith(_RAW_DIFFUSION_DECODER_PREFIX):
                continue
            if not key.endswith(_GATE_PARAM_SUFFIXES):
                continue
            out[key.removeprefix(_RAW_DIFFUSION_DECODER_PREFIX)] = handle.get_tensor(key)
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


def _build_diffusion_vae_decoder_sd_ops(gates: dict[str, torch.Tensor]) -> SDOps:
    """DiffVAE decoder SDOps: rename, QKV split, coarse drop, gate fold/drop.
    *gates* is keyed by post-rename gate param names (see ``_read_diff_vae_gates``).
    Empty map (bundled n1 / QKV unit tests): fold is a no-op; drop never fires.
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

    # Drop/fold ops before QKV split so first-match kv-ops never mis-route.
    return (
        SDOps("DIFFUSION_VAE_DECODER_COMFY_KEYS_FILTER")
        .with_matching(prefix="vae.decoder.")
        .with_replacement("vae.decoder.", "")
        .with_replacement("t_embedder.mlp.0.", "t_embedder.timestep_embedder.linear_1.")
        .with_replacement("t_embedder.mlp.2.", "t_embedder.timestep_embedder.linear_2.")
        .with_matching(prefix="vae.per_channel_statistics.")
        .with_replacement("vae.per_channel_statistics.", "per_channel_statistics.")
        .with_kv_operation(operation=_drop_coarse_param, key_prefix="coarse_")
        .with_kv_operation(operation=_drop_gate, key_suffix=".gate_msa")
        .with_kv_operation(operation=_drop_gate, key_suffix=".gate_mlp")
        .with_kv_operation(operation=_drop_gate, key_suffix=".gate_ctx")
        .with_kv_operation(operation=_on_attn_proj, key_suffix=".attn.proj.weight")
        .with_kv_operation(operation=_on_attn_proj, key_suffix=".attn.proj.bias")
        .with_kv_operation(operation=_on_mlp_w_down, key_suffix=".mlp.w_down.weight")
        .with_kv_operation(operation=_on_mlp_w_down, key_suffix=".mlp.w_down.bias")
        .with_kv_operation(operation=_on_context_proj, key_suffix=".context_proj.weight")
        .with_kv_operation(operation=_on_context_proj, key_suffix=".context_proj.bias")
        .with_kv_operation(operation=_split_fused_qkv_param, key_suffix=".qkv.weight")
        .with_kv_operation(operation=_split_fused_qkv_param, key_suffix=".qkv.bias")
    )


# Empty-gate build: QKV/t_embedder/coarse SDOps for unit tests. Weight loading of
# gated (standalone distilled) checkpoints must use ``video_decoder_sd_ops_for_checkpoint``.
DIFFUSION_VAE_DECODER_COMFY_KEYS_FILTER = _build_diffusion_vae_decoder_sd_ops({})


def video_decoder_sd_ops_for_checkpoint(
    checkpoint_path: str,
    *,
    diffusion_vae: bool | None = None,
) -> SDOps:
    """Pick decoder SDOps from checkpoint metadata (conv vs diffusion).
    For diffusion VAEs, pre-reads ``gate_*`` and builds fold/drop ops so standalone
    distilled checkpoints load into the ungated ``DiffusionNABlock``.
    Pass ``diffusion_vae`` when the caller already ran ``is_diffusion_video_vae`` to
    skip a second metadata open.
    """
    if diffusion_vae is None:
        diffusion_vae = is_diffusion_video_vae(checkpoint_path)
    if diffusion_vae:
        return _build_diffusion_vae_decoder_sd_ops(_read_diff_vae_gates(checkpoint_path))
    return VAE_DECODER_COMFY_KEYS_FILTER
