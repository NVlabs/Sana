# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from diffusion.utils.config import (
    ModelVideoConfig,
    SanaVideoConfig,
    TrainVideoConfig,
    VideoDataConfig,
    model_video_init_config,
)


@dataclass
class VideoDataCamCtrlConfig(VideoDataConfig):
    caption_proportion: Dict[str, Any] = field(default_factory=lambda: {"prompt": 1})
    # Training-mixture fields; harmless at inference time but required so
    # pyrallis can deserialise the YAML.
    data_repeat: Optional[Dict[str, int]] = None
    shuffle_mode: str = "zip_group"
    external_caption_suffixes: List[str] = field(default_factory=list)
    return_raymap: bool = True
    use_plucker: bool = True
    vae_ratio: Tuple[int, int] = (4, 8)  # (time_downsample, spatial_downsample)
    cam_sample_strategy: str = "last"  # first, last
    use_relative_pose: bool = False
    normalize_poses: bool = False
    precompute_pos_embed: bool = False
    pos_embed_type: str = "wan_rope"
    attention_head_dim: int = 256
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    max_seq_len: int = 1024
    pack_latents: bool = False
    precompute_cam_pos_embed: bool = False
    camctrl_type: str = "PRoPE"
    return_delta_actions: bool = False
    return_chunk_plucker: bool = False
    s3_config: Optional[Dict[str, Any]] = None
    s3_path_map: Optional[Dict[str, str]] = None


@dataclass
class ModelVideoCamCtrlConfig(ModelVideoConfig):
    camctrl_type: Optional[str] = None
    init_cam_from_base: bool = False
    cam_attn_compress: int = 1
    use_delta_actions: bool = False
    delta_action_dim: int = 64
    use_delta_translation: bool = False
    use_delta_pose_additive: bool = False
    delta_pose_additive_dim: int = 64
    use_chunk_plucker_input: bool = False
    use_chunk_plucker_post_attn: bool = False
    chunk_plucker_channels: int = 48
    chunk_plucker_post_attn_blocks: int = -1  # -1 = all blocks, N = first N blocks only
    fp32_norm: bool = False
    chunk_size: Optional[int] = None
    chunk_split_strategy: str = "uniform"
    conv_kernel_size: int = 4  # Temporal kernel size for ShortConvolution in GDN
    k_conv_only: bool = True  # Only apply ShortConvolution on K (skip Q and V)
    softmax_every_n: int = 4  # Hybrid GDN-Softmax: replace every N-th block with softmax (0=disabled)
    use_autograd_kernel: bool = False  # Switch Triton GDN blocks to autograd-enabled fused kernels (training mode)


@dataclass
class TrainVideoCamCtrlConfig(TrainVideoConfig):
    only_train_self_attn: bool = False
    only_train_cam_attn: bool = False
    # Per-sample mixture for chunk timestep sampling (incremental strategy only).
    # Keys must sum to 1.0. When None, the legacy `same_timestep_prob` +
    # `last_chunk_anchor_prob` paths run instead. See respace.py docs for semantics.
    chunk_mixture_probs: Optional[Dict[str, float]] = None
    # Probability of forcing the LAST chunk as the anchor in the legacy
    # incremental sampler (applies only when `chunk_mixture_probs` is None).
    last_chunk_anchor_prob: float = 0.0
    # Print per-step mixture-mode counts for the first N training steps for debugging.
    chunk_mixture_debug_steps: int = 0
    mixed_finetune: bool = False
    main_lora_target_modules: Optional[List[str]] = None
    main_lora_include: List[str] = field(default_factory=lambda: [".attn.", ".mlp.", ".ffn."])
    main_lora_exclude: List[str] = field(default_factory=list)
    cam_branch_keywords: List[str] = field(
        default_factory=lambda: [
            "_proj_cam",
            "raymap_embedder",
            "delta_action_embedder",
            "delta_translation_embedder",
            "delta_pose_embedder",
            "delta_pose_proj",
            "plucker_embedder",
            "plucker_proj",
            "prope_proj",
            "rope_phase_",
        ]
    )
    max_steps: Optional[int] = None
    prefetch_factor: Optional[int] = None
    cam_branch_drop_prob: float = 0.0
    video_only_training_interval: int = 0
    train_batch_size_video_only: Optional[int] = None
    nocam_training_interval: int = 0
    train_batch_size_nocam: Optional[int] = None
    # Camera control validation settings
    camctrl_visualize: bool = False  # Enable camera control validation
    camctrl_val_data_path: str = "assets/camctrl_val_data.json"  # Path to validation data
    camctrl_val_cfg_scale: float = 6.0  # CFG scale for camera control validation
    camctrl_val_steps: int = 40  # Number of sampling steps for validation
    camctrl_val_wandb_scale: float = 1.5  # Upscale factor for wandb videos
    val_only: bool = False  # Run validation only and exit

    # When > 0, log model_kwargs shapes and isolated forward/backward times
    # for the first N AR steps on rank 0. Default 0 = no logging.
    ar_debug_shapes_n: int = 0

    # Synchronize AR ``K`` (and the ``T_lat`` clamp) across data-parallel ranks
    # before sampling so every rank uses the same ``T_active = K + G``.
    ar_sync_K_across_ranks: bool = True


@dataclass
class SanaVideoCamCtrlConfig(SanaVideoConfig):
    data: VideoDataCamCtrlConfig
    model: ModelVideoCamCtrlConfig
    train: TrainVideoCamCtrlConfig
    video_only_data: Optional[VideoDataCamCtrlConfig] = None
    nocam_data: Optional[VideoDataCamCtrlConfig] = None
    tracker_project_name: str = "sana-video-camctrl"


def model_video_camctrl_init_config(config: SanaVideoCamCtrlConfig, latent_size: int = 32):
    return {
        "camctrl_type": config.model.camctrl_type,
        "cam_attn_compress": config.model.cam_attn_compress,
        "init_cam_from_base": config.model.init_cam_from_base,
        "use_delta_actions": config.model.use_delta_actions,
        "delta_action_dim": config.model.delta_action_dim,
        "use_delta_translation": config.model.use_delta_translation,
        "fp32_norm": config.model.fp32_norm,
        "chunk_size": config.model.chunk_size,
        "chunk_split_strategy": config.model.chunk_split_strategy,
        "conv_kernel_size": config.model.conv_kernel_size,
        "k_conv_only": config.model.k_conv_only,
        "use_delta_pose_additive": config.model.use_delta_pose_additive,
        "delta_pose_additive_dim": config.model.delta_pose_additive_dim,
        "use_chunk_plucker_input": config.model.use_chunk_plucker_input,
        "use_chunk_plucker_post_attn": config.model.use_chunk_plucker_post_attn,
        "chunk_plucker_channels": config.model.chunk_plucker_channels,
        "chunk_plucker_post_attn_blocks": config.model.chunk_plucker_post_attn_blocks,
        "softmax_every_n": config.model.softmax_every_n,
        "use_autograd_kernel": config.model.use_autograd_kernel,
        **model_video_init_config(config, latent_size=latent_size),
    }
