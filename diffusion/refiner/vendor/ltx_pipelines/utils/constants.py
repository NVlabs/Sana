import logging
from dataclasses import dataclass, field, replace

from safetensors import safe_open

from diffusion.refiner.vendor.ltx_core.components.guiders import MultiModalGuiderParams
from diffusion.refiner.vendor.ltx_core.types import SpatioTemporalScaleFactors

# =============================================================================
# Diffusion Schedule
# =============================================================================

# Noise schedule for the distilled pipeline. These sigma values control noise
# levels at each denoising step and were tuned to match the distillation process.
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

# Reduced schedule for super-resolution stage 2 (subset of distilled values)
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


# =============================================================================
# Pipeline Parameters
# =============================================================================


@dataclass(frozen=True)
class PipelineParams:
    seed: int = 10
    stage_1_height: int = 512
    stage_1_width: int = 768
    num_frames: int = 121
    frame_rate: float = 24.0
    num_inference_steps: int = 40
    video_guider_params: MultiModalGuiderParams = field(
        default_factory=lambda: MultiModalGuiderParams(
            cfg_scale=3.0,
            stg_scale=1.0,
            rescale_scale=0.7,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[29],
        )
    )
    audio_guider_params: MultiModalGuiderParams = field(
        default_factory=lambda: MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=1.0,
            rescale_scale=0.7,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[29],
        )
    )

    @property
    def stage_2_height(self) -> int:
        return int(self.stage_1_height * 2)

    @property
    def stage_2_width(self) -> int:
        return int(self.stage_1_width * 2)


# Default params for LTX-2.0 non-distilled models. These can be overridden by detecting from checkpoint metadata.
LTX_2_PARAMS = PipelineParams()

# Default params for LTX-2.3 non-distilled models. These override some of the LTX-2.0 defaults.
LTX_2_3_PARAMS = replace(
    LTX_2_PARAMS,
    num_inference_steps=30,
    video_guider_params=replace(LTX_2_PARAMS.video_guider_params, stg_blocks=[28]),
    audio_guider_params=replace(LTX_2_PARAMS.audio_guider_params, stg_blocks=[28]),
)
LTX_2_3_HQ_PARAMS = PipelineParams(
    num_inference_steps=15,
    stage_1_height=1088 // 2,
    stage_1_width=1920 // 2,
    video_guider_params=MultiModalGuiderParams(
        cfg_scale=3.0,
        stg_scale=0.0,
        rescale_scale=0.45,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[],
    ),
    audio_guider_params=MultiModalGuiderParams(
        cfg_scale=7.0,
        stg_scale=0.0,
        rescale_scale=1.0,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[],
    ),
)

DEFAULT_LORA_STRENGTH = 1.0
DEFAULT_IMAGE_CRF = 33
VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()
VIDEO_LATENT_CHANNELS = 128

_LTX_2_3_MODEL_VERSION_PREFIX = "2.3"


def detect_params(checkpoint_path: str) -> PipelineParams:
    """Detect pipeline params from checkpoint metadata.
    Reads the ``model_version`` field from the safetensors config metadata.
    Returns ``LTX_2_3_PARAMS`` when the version starts with "2.3",
    otherwise falls back to ``LTX_2_PARAMS``.
    """
    logger = logging.getLogger(__name__)

    try:
        with safe_open(checkpoint_path, framework="pt") as f:
            metadata = f.metadata() or {}
        version = metadata.get("model_version", "")
    except Exception:
        logger.warning("Could not read checkpoint metadata from %s, using LTX-2 defaults", checkpoint_path)
        return LTX_2_PARAMS

    if version.startswith(_LTX_2_3_MODEL_VERSION_PREFIX):
        return LTX_2_3_PARAMS

    logger.info("Using LTX_2_PARAMS for checkpoint (version=%s)", version or "unknown")
    return LTX_2_PARAMS


# =============================================================================
# Prompts
# =============================================================================

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
