import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from .config import BaseConfig, DataConfig, SchedulerConfig, TrainingConfig, VideoDataConfig


@dataclass
class WanModelConfig(BaseConfig):
    model: str = "Wan_T2V_1300M"
    from_pretrained: Optional[str] = None
    load_model_ckpt: Optional[str] = None
    init_patch_embedding: bool = False
    image_size: int = 256
    video_width: int = 832
    video_height: int = 480
    num_frames: int = 81
    patch_size: List[int] = field(default_factory=lambda: [1, 2, 2])
    dim: int = 1536
    ffn_dim: int = 8960
    freq_dim: int = 256
    num_heads: int = 12
    num_layers: int = 30
    window_size: Tuple[int, int] = field(default_factory=lambda: (-1, -1))
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6
    mixed_precision: str = "bf16"  # ['fp16', 'fp32', 'bf16']
    fp32_attention: bool = True
    load_from: Optional[str] = None
    resume_from: Optional[Union[Dict[str, Any], str]] = field(
        default_factory=lambda: {
            "checkpoint": None,
            "load_ema": False,
            "resume_lr_scheduler": True,
            "resume_optimizer": True,
        }
    )
    aspect_ratio_type: str = "ASPECT_RATIO_1024"
    multi_scale: bool = False
    class_dropout_prob: float = 0.0
    guidance_type: str = "classifier-free"
    mask: Optional[str] = None  # first, full, last mask, or no mask
    image_latent_mode: str = "video_zero"  # ["repeat", "zero", "video_zero"]
    linear_attn_idx: Optional[List[int]] = None
    self_attn_type: str = "flash"  # ["linear", "mllalinear", "flash"] this only used together with linear_attn_idx
    rope_after: bool = False
    power: float = 1.0
    ffn_type: str = "mlp"


@dataclass
class WanVAEConfig(BaseConfig):
    vae_type: str = "WanVAE"
    vae_latent_dim: int = 16
    vae_pretrained: str = "checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
    vae_stride: List[int] = field(default_factory=lambda: [4, 8, 8])
    weight_dtype: str = "float32"
    extra: Any = None
    cache_dir: Optional[str] = None
    if_cache: bool = False  # no more cache by default


@dataclass
class WanTextEncoderConfig(BaseConfig):
    t5_model: str = "umt5_xxl"
    t5_dtype: str = "bfloat16"
    text_len: int = 512
    t5_checkpoint: str = "checkpoints/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
    t5_tokenizer: str = "google/umt5-xxl"
    extra: Any = None
    caption_channels: int = 4096


@dataclass
class WanImageEncoderConfig(BaseConfig):
    image_encoder_type: Optional[str] = None
    image_encoder_pretrained: Optional[str] = None
    image_encoder_tokenizer: Optional[str] = None
    weight_dtype: str = "float32"
    extra: Any = None


@dataclass
class LoraConfig(BaseConfig):
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning"""

    use_lora: bool = False
    rank: int = 4  # Rank of LoRA adapters
    alpha: int = 4  # Scaling factor for LoRA
    target_modules: Optional[str] = "all-linear"  # Which modules to apply LoRA to
    dropout: float = 0.0  # Dropout for LoRA layers
    bias: str = "none"  # Bias handling: "none", "all", "lora_only"
    # Advanced LoRA settings
    init_lora_weights: str = "gaussian"  # "gaussian", "kaiming", "xavier"
    additional_trainable_layers: Optional[List[str]] = None  # Additional layers to keep trainable
    merge_weights: bool = False  # Whether to merge weights during training
    fan_in_fan_out: bool = False  # Set to True for certain transformer architectures


@dataclass
class FSDPConfig(BaseConfig):
    pass


@dataclass
class DistillConfig(BaseConfig):
    model: WanModelConfig
    distill_logit_weight: float = 0.0
    distill_attn_weight: float = 0.0


@dataclass
class WanTrainingConfig(TrainingConfig):
    sp_degree: int = 1  # sequence parallel degree
    fsdp_config: Optional[FSDPConfig] = None
    auto_lr: Optional[Dict[str, str]] = field(default_factory=lambda: {"rule": "sqrt"})
    validation_images: Optional[List[str]] = field(
        default_factory=lambda: [
            "dog",
            "portrait photo of a girl, photograph, highly detailed face, depth of field",
            "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
            "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
            "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
        ]
    )  # Path to validation images
    fsdp_inference: bool = False
    train_la_only: bool = False


@dataclass
class WanConfig(BaseConfig):
    data: VideoDataConfig
    model: WanModelConfig
    vae: WanVAEConfig
    text_encoder: WanTextEncoderConfig
    scheduler: SchedulerConfig
    train: WanTrainingConfig
    work_dir: str = "output/"
    resume_from: Optional[str] = None
    load_from: Optional[str] = None
    debug: bool = False
    caching: bool = False
    report_to: str = "wandb"
    tracker_project_name: str = "wan-video"
    name: str = "baseline"
    loss_report_name: str = "loss"
    task: str = "t2v"  # t2v or ti2v
    image_encoder: Optional[WanImageEncoderConfig] = None
    distill: Optional[DistillConfig] = None
    lora: Optional[LoraConfig] = None
    cfg_scale: float = 3.0
