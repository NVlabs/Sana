from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, ValidationInfo, field_validator, model_validator

from ltx_trainer.quantization import QuantizationOptions
from ltx_trainer.training_strategies.base_strategy import TrainingStrategyConfigBase
from ltx_trainer.training_strategies.flexible import FlexibleStrategyConfig
from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig
from ltx_trainer.training_strategies.video_to_video import VideoToVideoConfig

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Validation Condition Types
# =============================================================================


class FirstFrameConditionConfig(ConfigBaseModel):
    """First-frame conditioning (intrinsic, latent_idx=0). Always targets video.
    If image_or_video points to a video file, the first frame is automatically extracted.
    """

    type: Literal["first_frame"] = "first_frame"
    image_or_video: str | Path


class PrefixConditionConfig(ConfigBaseModel):
    """Prefix conditioning for temporal extension (intrinsic). Exactly one of video/audio must be set."""

    type: Literal["prefix"] = "prefix"
    video: str | None = None
    audio: str | None = None
    num_frames: int | None = Field(
        default=None,
        ge=1,
        description="Number of pixel frames for video prefix. Must be VAE-aligned: "
        "num_frames %% T == 1, where T is the VAE temporal compression factor (8 by default).",
    )
    duration: float | None = Field(default=None, gt=0, description="Duration in seconds for audio prefix")

    @model_validator(mode="after")
    def validate_exactly_one_modality(self) -> "PrefixConditionConfig":
        if (self.video is None) == (self.audio is None):
            raise ValueError("Exactly one of 'video' or 'audio' must be set for prefix condition")
        return self


class SuffixConditionConfig(ConfigBaseModel):
    """Suffix conditioning for temporal extension (intrinsic). Exactly one of video/audio must be set."""

    type: Literal["suffix"] = "suffix"
    video: str | None = None
    audio: str | None = None
    num_frames: int | None = Field(
        default=None,
        ge=1,
        description="Number of pixel frames for video suffix. Must be VAE-aligned: "
        "num_frames %% T == 0, where T is the VAE temporal compression factor (8 by default).",
    )
    duration: float | None = Field(default=None, gt=0, description="Duration in seconds for audio suffix")

    @model_validator(mode="after")
    def validate_exactly_one_modality(self) -> "SuffixConditionConfig":
        if (self.video is None) == (self.audio is None):
            raise ValueError("Exactly one of 'video' or 'audio' must be set for suffix condition")
        return self


class SpatialCropConditionConfig(ConfigBaseModel):
    """Spatial crop conditioning for outpainting (intrinsic, video only)."""

    type: Literal["spatial_crop"] = "spatial_crop"
    video: str
    spatial_region: tuple[int, int, int, int] = Field(
        ..., description="Spatial crop region as (y1, x1, y2, x2) in pixel coordinates"
    )


class MaskConditionConfig(ConfigBaseModel):
    """Mask-based conditioning for inpainting (intrinsic). Exactly one of video/audio must be set."""

    type: Literal["mask"] = "mask"
    video: str | None = None
    audio: str | None = None
    mask: str

    @model_validator(mode="after")
    def validate_exactly_one_modality(self) -> "MaskConditionConfig":
        if (self.video is None) == (self.audio is None):
            raise ValueError("Exactly one of 'video' or 'audio' must be set for mask condition")
        return self


class ReferenceConditionConfig(ConfigBaseModel):
    """Reference conditioning (IC-LoRA style concatenation). Exactly one of video/audio must be set."""

    type: Literal["reference"] = "reference"
    video: str | None = None
    audio: str | None = None
    downscale_factor: int = Field(default=1, ge=1)
    temporal_scale_factor: int = Field(default=1, ge=1)
    include_in_output: bool = False

    @model_validator(mode="after")
    def validate_exactly_one_modality(self) -> "ReferenceConditionConfig":
        if (self.video is None) == (self.audio is None):
            raise ValueError("Exactly one of 'video' or 'audio' must be set for reference condition")
        return self


class VideoToAudioConditionConfig(ConfigBaseModel):
    """Video-to-audio — video is provided as frozen cross-modal conditioning.
    The video is kept clean (sigma=0) and influences audio generation via cross-modal attention.
    """

    type: Literal["video_to_audio"] = "video_to_audio"
    video: str


class AudioToVideoConditionConfig(ConfigBaseModel):
    """Audio-to-video — audio is provided as frozen cross-modal conditioning.
    The audio is kept clean (sigma=0) and influences video generation via cross-modal attention.
    """

    type: Literal["audio_to_video"] = "audio_to_video"
    audio: str


ValidationCondition = Annotated[
    Union[
        FirstFrameConditionConfig,
        PrefixConditionConfig,
        SuffixConditionConfig,
        SpatialCropConditionConfig,
        MaskConditionConfig,
        ReferenceConditionConfig,
        VideoToAudioConditionConfig,
        AudioToVideoConditionConfig,
    ],
    Field(discriminator="type"),
]


def _condition_targets_video(cond: ValidationCondition) -> bool:
    """Check if a validation condition targets the video modality."""
    if cond.type in ("first_frame", "spatial_crop", "video_to_audio"):
        return True
    if cond.type in ("prefix", "suffix", "mask", "reference"):
        return getattr(cond, "video", None) is not None
    return False


def _condition_targets_audio(cond: ValidationCondition) -> bool:
    """Check if a validation condition targets the audio modality."""
    if cond.type == "audio_to_video":
        return True
    if cond.type in ("prefix", "suffix", "mask", "reference"):
        return getattr(cond, "audio", None) is not None
    return False


class ValidationSample(ConfigBaseModel):
    """Configuration for a single validation sample — fully self-describing."""

    prompt: str
    conditions: list[ValidationCondition] = Field(default_factory=list)

    video_dims: tuple[int, int, int] | None = Field(
        default=None,
        description="Per-sample override for (width, height, frames). None = inherit from ValidationConfig.",
    )
    seed: int | None = Field(
        default=None,
        description="Per-sample override for random seed. None = inherit from ValidationConfig.",
    )

    @model_validator(mode="after")
    def validate_frozen_modality_conflicts(self) -> "ValidationSample":
        frozen_types = {c.type for c in self.conditions if c.type in ("video_to_audio", "audio_to_video")}

        if "video_to_audio" in frozen_types and "audio_to_video" in frozen_types:
            raise ValueError(
                "Cannot have both video_to_audio and audio_to_video conditions — nothing would be generated"
            )

        if "video_to_audio" in frozen_types:
            for c in self.conditions:
                if c.type != "video_to_audio" and _condition_targets_video(c):
                    raise ValueError(
                        f"Cannot use video-targeting '{c.type}' condition when video is frozen (video_to_audio)"
                    )

        if "audio_to_video" in frozen_types:
            for c in self.conditions:
                if c.type != "audio_to_video" and _condition_targets_audio(c):
                    raise ValueError(
                        f"Cannot use audio-targeting '{c.type}' condition when audio is frozen (audio_to_video)"
                    )

        return self


class ModelConfig(ConfigBaseModel):
    """Configuration for the base model and training mode"""

    model_path: str | Path = Field(
        ...,
        description="Model path - local path to safetensors checkpoint file",
    )

    text_encoder_path: str | Path | None = Field(
        default=None,
        description="Path to text encoder (required for LTX-2/Gemma models, optional for LTXV/T5 models)",
    )

    training_mode: Literal["lora", "full"] = Field(
        default="lora",
        description="Training mode - either LoRA fine-tuning or full model fine-tuning",
    )

    load_checkpoint: str | Path | None = Field(
        default=None,
        description="Path to a checkpoint file or directory to load from. "
        "If a directory is provided, the latest checkpoint will be used.",
    )

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: str | Path) -> str | Path:
        """Validate that model_path is either a valid URL or an existing local path."""
        is_url = str(v).startswith(("http://", "https://"))

        if is_url:
            raise ValueError(f"Model path cannot be a URL: {v}")

        if not Path(v).exists():
            raise ValueError(f"Model path does not exist: {v}")

        return v


class LoraConfig(ConfigBaseModel):
    """Configuration for LoRA fine-tuning"""

    rank: int = Field(
        default=64,
        description="Rank of LoRA adaptation",
        ge=2,
    )

    alpha: int = Field(
        default=64,
        description="Alpha scaling factor for LoRA",
        ge=1,
    )

    dropout: float = Field(
        default=0.0,
        description="Dropout probability for LoRA layers",
        ge=0.0,
        le=1.0,
    )

    target_modules: list[str] = Field(
        default=["to_k", "to_q", "to_v", "to_out.0"],
        description="List of modules to target with LoRA",
    )


def _get_strategy_discriminator(v: dict | TrainingStrategyConfigBase) -> str:
    """Discriminator function for strategy config union."""
    if isinstance(v, dict):
        return v.get("name", "text_to_video")
    return v.name


# Union type for all strategy configs with discriminator
TrainingStrategyConfig = Annotated[
    Annotated[TextToVideoConfig, Tag("text_to_video")]
    | Annotated[VideoToVideoConfig, Tag("video_to_video")]
    | Annotated[FlexibleStrategyConfig, Tag("flexible")],
    Discriminator(_get_strategy_discriminator),
]


class OptimizationConfig(ConfigBaseModel):
    """Configuration for optimization parameters"""

    learning_rate: float = Field(
        default=5e-4,
        description="Learning rate for optimization",
    )

    steps: int = Field(
        default=3000,
        description="Number of training steps",
    )

    batch_size: int = Field(
        default=2,
        description="Batch size for training",
    )

    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of steps to accumulate gradients",
    )

    max_grad_norm: float = Field(
        default=1.0,
        description="Maximum gradient norm for clipping",
    )

    optimizer_type: Literal["adamw", "adamw8bit"] = Field(
        default="adamw",
        description="Type of optimizer to use for training",
    )

    scheduler_type: Literal[
        "constant",
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "step",
    ] = Field(
        default="linear",
        description="Type of scheduler to use for training",
    )

    scheduler_params: dict = Field(
        default_factory=dict,
        description="Parameters for the scheduler",
    )

    enable_gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to save memory at the cost of slower training",
    )


class AccelerationConfig(ConfigBaseModel):
    """Configuration for hardware acceleration and compute optimization"""

    mixed_precision_mode: Literal["no", "fp16", "bf16"] | None = Field(
        default="bf16",
        description="Mixed precision training mode",
    )

    quantization: QuantizationOptions | None = Field(
        default=None,
        description="Quantization precision to use",
    )

    load_text_encoder_in_8bit: bool = Field(
        default=False,
        description="Whether to load the text encoder in 8-bit precision to save memory",
    )

    offload_optimizer_during_validation: bool = Field(
        default=False,
        description="Offload optimizer state to CPU before validation video sampling and reload "
        "it afterwards, to free VRAM for inference. Useful when optimizer state is large "
        "(e.g. AdamW for full fine-tuning or high-rank LoRA) and validation OOMs because the "
        "VAE decoder + transformer + optimizer state cannot coexist on the GPU. Has no effect "
        "for FSDP (sharded state). Disabled by default.",
    )


class DataConfig(ConfigBaseModel):
    """Configuration for data loading and processing"""

    preprocessed_data_root: str = Field(
        description="Path to folder containing preprocessed training data",
    )

    num_dataloader_workers: int = Field(
        default=2,
        description="Number of background processes for data loading (0 means synchronous loading)",
        ge=0,
    )

    @field_validator("preprocessed_data_root")
    @classmethod
    def validate_preprocessed_data_root(cls, v: str) -> str:
        """Validate that preprocessed_data_root exists."""
        path = Path(v).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Dataset path does not exist: {v}")
        if not path.is_dir():
            raise ValueError(f"Dataset path is not a directory: {v}")
        return str(path)


class ValidationConfig(ConfigBaseModel):
    """Configuration for validation during training"""

    # Per-sample configuration (new format — preferred)
    samples: list[ValidationSample] = Field(
        default_factory=list,
        description="List of validation samples. Each sample is fully self-describing with its own "
        "prompt, conditions, and optional overrides. Replaces prompts/images/reference_videos.",
    )

    # Legacy fields (deprecated — converted to samples internally via convert_legacy_format)
    prompts: list[str] = Field(
        default_factory=list,
        description="[DEPRECATED: use 'samples' instead] List of prompts to use for validation",
    )

    negative_prompt: str = Field(
        default=DEFAULT_NEGATIVE_PROMPT,
        description="Negative prompt to use for validation examples",
    )

    images: list[str] | None = Field(
        default=None,
        description="[DEPRECATED: use 'samples' with first_frame conditions] "
        "List of image paths to use for validation. "
        "One image path must be provided for each validation prompt",
    )

    reference_videos: list[str] | None = Field(
        default=None,
        description="[DEPRECATED: use 'samples' with reference conditions] "
        "List of reference video paths to use for validation. "
        "One video path must be provided for each validation prompt",
    )

    reference_downscale_factor: int = Field(
        default=1,
        description="[DEPRECATED: use downscale_factor on ReferenceCondition] "
        "Downscale factor for reference videos in IC-LoRA validation. "
        "When > 1, reference videos are processed at 1/n resolution (e.g., 2 means half resolution). "
        "Must match the factor used during dataset preprocessing.",
        ge=1,
    )

    video_dims: tuple[int, int, int] = Field(
        default=(960, 544, 89),
        description="Dimensions of validation videos (width, height, frames). "
        "Width and height must be divisible by the VAE spatial factor (32 by default), and frames must "
        "satisfy frames % T == 1, where T is the VAE temporal compression factor (8 by default).",
    )

    frame_rate: float = Field(
        default=24.0,
        description="Frame rate for validation videos",
        gt=0,
    )

    seed: int = Field(
        default=42,
        description="Random seed used when sampling validation videos",
    )

    inference_steps: int = Field(
        default=30,
        description="Number of inference steps for validation",
        gt=0,
    )

    interval: int | None = Field(
        default=100,
        description="Number of steps between validation runs. If None, validation is disabled.",
        gt=0,
    )

    video_cfg_scale: float = Field(
        default=3.0,
        description="Video CFG guidance scale to use during validation",
        ge=1.0,
    )

    audio_cfg_scale: float = Field(
        default=7.0,
        description="Audio CFG guidance scale to use during validation",
        ge=1.0,
    )

    video_stg_scale: float = Field(
        default=1.0,
        description="Video STG (Spatio-Temporal Guidance) scale. 0.0 disables video STG.",
        ge=0.0,
    )

    audio_stg_scale: float = Field(
        default=1.0,
        description="Audio STG (Spatio-Temporal Guidance) scale. 0.0 disables audio STG.",
        ge=0.0,
    )

    stg_blocks: list[int] | None = Field(
        default_factory=lambda: [28],
        description="Which transformer blocks to perturb for STG. "
        "None means all blocks are perturbed. Recommended for LTX-2.3: [28].",
    )

    guidance_rescale: float = Field(
        default=0.7,
        description="Guidance rescale factor used after CFG/STG/modality guidance. "
        "Interpolates between the computed variance factor and 1.0. 0.0 disables rescaling.",
        ge=0.0,
        le=1.0,
    )

    video_modality_guidance_scale: float = Field(
        default=3.0,
        description="Video modality guidance scale for audio-to-video isolation guidance.",
        ge=1.0,
    )

    audio_modality_guidance_scale: float = Field(
        default=3.0,
        description="Audio modality guidance scale for video-to-audio isolation guidance.",
        ge=1.0,
    )

    generate_audio: bool = Field(
        default=True,
        description="Whether to generate audio in validation samples. "
        "Independent of training strategy setting - you can generate audio "
        "in validation even when not training the audio branch.",
    )

    generate_video: bool = Field(
        default=True,
        description="Whether to generate video in validation samples. "
        "Set to False for audio-only or v2a validation to save VRAM by skipping video VAE decoder loading. "
        "When False, validation will only generate audio (requires generate_audio=True).",
    )

    skip_initial_validation: bool = Field(
        default=False,
        description="Skip validation video sampling at step 0 (beginning of training)",
    )

    include_reference_in_output: bool = Field(
        default=False,
        description="[DEPRECATED: use include_in_output on ReferenceCondition] "
        "For video-to-video training: concatenate the original reference video side-by-side "
        "with the generated output. The reference comes from the input video, not from the model's output.",
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_deprecated_guidance_fields(cls, data: object) -> object:
        """Map deprecated scalar guidance fields to explicit per-modality fields."""
        if not isinstance(data, dict):
            return data

        values = dict(data)

        if "guidance_scale" in values:
            new_cfg_fields = {"video_cfg_scale", "audio_cfg_scale"} & values.keys()
            if new_cfg_fields:
                raise ValueError(
                    "guidance_scale is deprecated; use video_cfg_scale and audio_cfg_scale instead, "
                    "and do not combine old and new CFG fields."
                )
            guidance_scale = values.pop("guidance_scale")
            values["video_cfg_scale"] = guidance_scale
            values["audio_cfg_scale"] = guidance_scale

        legacy_stg_fields = {"stg_scale", "stg_mode"} & values.keys()
        if legacy_stg_fields:
            new_stg_fields = {"video_stg_scale", "audio_stg_scale"} & values.keys()
            if new_stg_fields:
                raise ValueError(
                    "stg_scale and stg_mode are deprecated; use video_stg_scale and audio_stg_scale instead, "
                    "and do not combine old and new STG fields."
                )
            stg_scale = values.pop("stg_scale", 1.0)
            stg_mode = values.pop("stg_mode", "stg_av")
            if stg_mode not in {"stg_av", "stg_v"}:
                raise ValueError("stg_mode must be either 'stg_av' or 'stg_v'")
            values["video_stg_scale"] = stg_scale
            values["audio_stg_scale"] = stg_scale if stg_mode == "stg_av" else 0.0

        return values

    @field_validator("images")
    @classmethod
    def validate_images(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        """Validate that number of images (if provided) matches number of prompts."""
        if v is None:
            return None

        num_prompts = len(info.data.get("prompts", []))
        if v is not None and len(v) != num_prompts:
            raise ValueError(f"Number of images ({len(v)}) must match number of prompts ({num_prompts})")

        for image_path in v:
            if not Path(image_path).exists():
                raise ValueError(f"Image path '{image_path}' does not exist")

        return v

    @field_validator("reference_videos")
    @classmethod
    def validate_reference_videos(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        """Validate that number of reference videos (if provided) matches number of prompts."""
        if v is None:
            return None

        num_prompts = len(info.data.get("prompts", []))
        if v is not None and len(v) != num_prompts:
            raise ValueError(f"Number of reference videos ({len(v)}) must match number of prompts ({num_prompts})")

        for video_path in v:
            if not Path(video_path).exists():
                raise ValueError(f"Reference video path '{video_path}' does not exist")

        return v

    @model_validator(mode="after")
    def convert_legacy_format(self) -> "ValidationConfig":
        """Convert deprecated prompts/images/reference_videos to the new samples format."""
        if self.prompts and not self.samples:
            samples = []
            for i, prompt in enumerate(self.prompts):
                conditions: list[ValidationCondition] = []
                if self.images and i < len(self.images):
                    conditions.append(FirstFrameConditionConfig(image_or_video=self.images[i]))
                if self.reference_videos and i < len(self.reference_videos):
                    conditions.append(
                        ReferenceConditionConfig(
                            video=self.reference_videos[i],
                            downscale_factor=self.reference_downscale_factor,
                            include_in_output=self.include_reference_in_output,
                        )
                    )
                samples.append(ValidationSample(prompt=prompt, conditions=conditions))
            self.samples = samples
        return self

    @model_validator(mode="after")
    def validate_output_modality_requirements(self) -> "ValidationConfig":
        """Validate output modality settings when validation is configured."""
        has_validation = bool(self.prompts) or bool(self.samples)
        if has_validation and not self.generate_video and not self.generate_audio:
            raise ValueError(
                "At least one of generate_video or generate_audio must be True when validation is configured."
            )
        return self


class CheckpointsConfig(ConfigBaseModel):
    """Configuration for model checkpointing during training"""

    interval: int | None = Field(
        default=None,
        description="Number of steps between checkpoint saves. If None, intermediate checkpoints are disabled.",
        gt=0,
    )

    keep_last_n: int = Field(
        default=1,
        description="Number of most recent checkpoints to keep. Set to -1 to keep all checkpoints.",
        ge=-1,
    )

    precision: Literal["bfloat16", "float32"] = Field(
        default="bfloat16",
        description="Precision to use when saving checkpoint weights. Options: 'bfloat16' or 'float32'.",
    )

    no_resume: bool = Field(
        default=False,
        description="When True, ignore any saved training state and start from step 0. "
        "Model weights from load_checkpoint are still loaded, but optimizer/scheduler "
        "state and step counter are reset.",
    )

    save_training_state: Literal["full", "minimal", "off"] = Field(
        default="minimal",
        description="Save training state alongside checkpoints for resume. "
        "'full': optimizer + scheduler + RNG + step (~800MB for LoRA, much larger for full fine-tuning). "
        "'minimal': scheduler + RNG + step only (~few KB, sufficient for LoRA). "
        "'off': nothing saved, resume not possible.",
    )


class HubConfig(ConfigBaseModel):
    """Configuration for Hugging Face Hub integration"""

    push_to_hub: bool = Field(default=False, description="Whether to push the model weights to the Hugging Face Hub")
    hub_model_id: str | None = Field(
        default=None, description="Hugging Face Hub repository ID (e.g., 'username/repo-name')"
    )

    @model_validator(mode="after")
    def validate_hub_config(self) -> "HubConfig":
        """Validate that hub_model_id is not None when push_to_hub is True."""
        if self.push_to_hub and not self.hub_model_id:
            raise ValueError("hub_model_id must be specified when push_to_hub is True")
        return self


class WandbConfig(ConfigBaseModel):
    """Configuration for Weights & Biases logging"""

    enabled: bool = Field(
        default=False,
        description="Whether to enable W&B logging",
    )

    project: str = Field(
        default="ltxv-trainer",
        description="W&B project name",
    )

    entity: str | None = Field(
        default=None,
        description="W&B username or team",
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Tags to add to the W&B run",
    )

    log_validation_videos: bool = Field(
        default=True,
        description="Whether to log validation videos to W&B",
    )


class FlowMatchingConfig(ConfigBaseModel):
    """Configuration for flow matching training"""

    timestep_sampling_mode: Literal["uniform", "shifted_logit_normal"] = Field(
        default="shifted_logit_normal",
        description="Mode to use for timestep sampling",
    )

    timestep_sampling_params: dict = Field(
        default_factory=dict,
        description="Parameters for timestep sampling",
    )


class LtxTrainerConfig(ConfigBaseModel):
    """Unified configuration for LTXV training"""

    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraConfig | None = Field(default=None)
    training_strategy: TrainingStrategyConfig = Field(
        default_factory=TextToVideoConfig,
        description="Training strategy configuration. Determines the training mode and its parameters.",
    )
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    data: DataConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    hub: HubConfig = Field(default_factory=HubConfig)
    flow_matching: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    # General configuration
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    output_dir: str = Field(
        default="outputs",
        description="Directory to save model outputs",
    )

    # noinspection PyNestedDecorators
    @field_validator("output_dir")
    @classmethod
    def expand_output_path(cls, v: str) -> str:
        """Expand user home directory in output path."""
        return str(Path(v).expanduser().resolve())

    def _validate_data_dirs_exist(self) -> None:
        """Verify that every directory declared by the training strategy exists under the data root."""
        data_root = Path(self.data.preprocessed_data_root)
        for dir_name in self.training_strategy.get_data_sources():
            dir_path = data_root / dir_name
            if not dir_path.is_dir():
                raise ValueError(
                    f"Required data directory '{dir_name}' does not exist under preprocessed_data_root: {dir_path}"
                )

    @model_validator(mode="after")
    def validate_strategy_compatibility(self) -> "LtxTrainerConfig":
        """Validate that training strategy and other configurations are compatible."""
        self._validate_data_dirs_exist()

        # Check that reference videos are provided when using video_to_video strategy
        if self.training_strategy.name == "video_to_video" and self.validation.interval:
            has_reference = bool(self.validation.reference_videos) or any(
                cond.type == "reference" for sample in self.validation.samples for cond in sample.conditions
            )
            if not has_reference:
                raise ValueError(
                    "reference_videos or samples with reference conditions must be provided "
                    "in validation config when using video_to_video strategy"
                )

        # Check that LoRA config is provided when training mode is lora
        if self.model.training_mode == "lora" and self.lora is None:
            raise ValueError("LoRA configuration must be provided when training_mode is 'lora'")

        # Check that LoRA config is provided when using video_to_video strategy
        if self.training_strategy.name == "video_to_video" and self.model.training_mode != "lora":
            raise ValueError("Training mode must be 'lora' when using video_to_video strategy")

        return self
