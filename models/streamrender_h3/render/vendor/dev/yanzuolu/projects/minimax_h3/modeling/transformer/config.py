# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

# ===== JARVIS PATCH BEGIN(sglang runtime coupling: DiTArchConfig/DiTConfig bases, FSDP is_block, AttentionBackendEnum are sglang serving config plumbing; the dataclasses become standalone) =====
# from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
# from sglang.multimodal_gen.configs.models.fsdp import is_block
# from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
# ===== JARVIS PATCH END =====

MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT = 64
MINIMAX_H3_ADALN_MODALITY_NUM = 3


@dataclass
# ===== JARVIS PATCH BEGIN(sglang runtime coupling: DiTArchConfig is sglang's FSDP/loader base; its fields are inlined here with neutral defaults so model.py stays byte-identical upstream, and the FSDP predicate + attention-backend enum drop out) =====
# class MiniMaxH3DiTArchConfig(DiTArchConfig):
#     _fsdp_shard_conditions: list = field(default_factory=lambda: [is_block])
#
#     lora_param_names_mapping: dict = field(default_factory=dict)
#
#     _supported_attention_backends: set[AttentionBackendEnum] = field(
#         default_factory=lambda: {
#             AttentionBackendEnum.FA,
#             AttentionBackendEnum.AITER,
#             AttentionBackendEnum.TORCH_SDPA,
#         }
#     )
class MiniMaxH3DiTArchConfig:
    _fsdp_shard_conditions: list = field(default_factory=list)
    _compile_conditions: list = field(default_factory=list)
    param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)
    _supported_attention_backends: set = field(default_factory=set)
    # ===== JARVIS PATCH END =====

    num_layers: int = 50
    token_refiner_num_layers: int = 2
    hidden_size: int = 5376
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    # Clean latent channels concatenated after the noisy video latent on each
    # patchified row; they widen video_patch_proj's input only.
    video_condition_channels: int = 0
    # Per-token AdaLN condition. A positive dim adds an encoder from
    # adaln_condition_channels-channel per-latent maps to one dim-wide feature
    # per video token, and to every block a zero-initialized head turning it
    # into shift and scale deltas on the rows the caller lists.
    adaln_condition_dim: int = 0
    adaln_condition_channels: int = 0
    # Per-token input condition. Positive channels add an encoder from
    # input_condition_channels-channel per-latent maps, at
    # 2 ** (len(input_condition_widths) - 1) times the token grid, to one
    # hidden-wide delta per video token. Its zero-initialized output projection
    # adds the delta to the input embedding of the rows the caller lists.
    input_condition_channels: int = 0
    input_condition_widths: tuple[int, ...] = (256, 512, 768)
    input_condition_blocks: int = 2
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    adaln_out_features: int = 18 * 5376
    final_adaln_out_features: int = 2 * 5376
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        # ===== JARVIS PATCH BEGIN(sglang runtime coupling: base __post_init__ only derives _compile_conditions from _fsdp_shard_conditions) =====
        # super().__post_init__()
        # ===== JARVIS PATCH END =====
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)
        if len(self.patch_size) != 3:
            raise ValueError(f"patch_size must have 3 values, got {self.patch_size}.")
        if self.video_condition_channels < 0:
            raise ValueError(
                "video_condition_channels must be non-negative, got "
                f"{self.video_condition_channels}."
            )
        if self.adaln_condition_dim < 0 or self.adaln_condition_channels < 0 or (
            (self.adaln_condition_dim > 0) != (self.adaln_condition_channels > 0)
        ):
            raise ValueError(
                "adaln_condition_dim and adaln_condition_channels must both be zero "
                f"or both positive, got {self.adaln_condition_dim} and "
                f"{self.adaln_condition_channels}."
            )
        if isinstance(self.input_condition_widths, list):
            self.input_condition_widths = tuple(self.input_condition_widths)
        if self.input_condition_channels < 0:
            raise ValueError(
                "input_condition_channels must be non-negative, got "
                f"{self.input_condition_channels}."
            )
        if self.input_condition_channels and (
            not self.input_condition_widths
            or any(width <= 0 or width % 32 for width in self.input_condition_widths)
            or self.input_condition_blocks < 0
        ):
            raise ValueError(
                "input_condition_widths must be positive multiples of 32 and "
                "input_condition_blocks non-negative, got "
                f"{self.input_condition_widths} and {self.input_condition_blocks}."
            )
        self.num_channels_latents = self.latents_dim


@dataclass
# ===== JARVIS PATCH BEGIN(sglang runtime coupling: drop DiTConfig base, which carries prefix/quant_config/torch_compile_mode CLI plumbing) =====
# class MiniMaxH3DiTConfig(DiTConfig):
class MiniMaxH3DiTConfig:
    # ===== JARVIS PATCH END =====
    arch_config: MiniMaxH3DiTArchConfig = field(default_factory=MiniMaxH3DiTArchConfig)


__all__ = [
    "MINIMAX_H3_ADALN_MODALITY_NUM",
    "MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT",
    "MiniMaxH3DiTArchConfig",
    "MiniMaxH3DiTConfig",
]
