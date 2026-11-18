import logging
from dataclasses import dataclass, field, replace

import torch
from safetensors import safe_open

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import parse_model_version
from ltx_core.types import SpatioTemporalScaleFactors

# =============================================================================
# Diffusion Schedule
# =============================================================================

# Noise schedule for the distilled pipeline. These sigma values control noise
# levels at each denoising step and were tuned to match the distillation process.
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

# Reduced schedule for super-resolution stage 2 (subset of distilled values)
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]

DISTILLED_SIGMAS = torch.tensor(DISTILLED_SIGMA_VALUES)
STAGE_2_DISTILLED_SIGMAS = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES)
# Stage 2 schedule for the tiled-data-parallel multi-GPU runner.
TDP_DISTILLED_SIGMAS = torch.tensor([0.625, 0.4, 0.0])


# =============================================================================
# Pipeline Parameters
# =============================================================================

# H.264 CRF an image conditioning is re-compressed at, matching the compression the model was
# trained against. This is a property of the model generation, not a code-level fallback -- reach it
# through ``PipelineParams.default_image_crf`` (see ``detect_params``), which is what a pipeline's
# ``ImageConditioner`` resolves an unset ``ImageConditioningInput.crf`` against.
DEFAULT_IMAGE_CRF = 33
LTX_2_4_IMAGE_CRF = 18


@dataclass(frozen=True)
class PipelineParams:
    seed: int = 10
    stage_1_height: int = 512
    stage_1_width: int = 768
    num_frames: int = 121
    frame_rate: float = 24.0
    num_inference_steps: int = 40
    default_image_crf: int = DEFAULT_IMAGE_CRF
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


# HQ preset: Res2s sampler, fewer steps, STG off. A plain constant on purpose -- it overrides
# every knob that varies between generations, so there is nothing for it to inherit from a
# detected checkpoint. The one generation-dependent value, ``default_image_crf``, is resolved from
# checkpoint by the pipeline's ``ImageConditioner``, not read off this object.
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
VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()
VIDEO_LATENT_CHANNELS = 128

# 2.4 continues the 2.3 lineage, so it inherits 2.3's knobs (30 steps, STG on block 28) and
# only moves the image CRF. Deriving it from LTX_2_PARAMS instead would silently hand a 2.4
# checkpoint the 2.0 step count and STG block.
LTX_2_4_PARAMS = replace(LTX_2_3_PARAMS, default_image_crf=LTX_2_4_IMAGE_CRF)

# Params per model generation, newest first. A checkpoint gets the params of the newest
# generation it is at or above, so an unrecognised *newer* version inherits the closest
# known one instead of silently falling back to the 2.0 defaults. Adding a generation is
# one row; anything older than every row falls through to LTX_2_PARAMS.
_PARAMS_SINCE_VERSION: tuple[tuple[tuple[int, ...], PipelineParams], ...] = (
    ((2, 4), LTX_2_4_PARAMS),
    ((2, 3), LTX_2_3_PARAMS),
)


def detect_model_version(checkpoint_path: str) -> tuple[int, ...]:
    """Read a checkpoint's ``model_version`` metadata as comparable numeric components.
    Returns ``()`` -- which compares below every real version -- when the field is unset,
    unparseable, or the file cannot be read, so callers get their oldest fallback.
    This is the single place that turns a checkpoint path into a version tuple: it owns both
    the metadata read and the separator normalization that ``parse_model_version`` documents
    but deliberately leaves to its callers. Use it for any "is this checkpoint at least
    generation X" question; ``detect_params`` is the same read specialised to per-generation
    parameter defaults.
    """
    logger = logging.getLogger(__name__)

    try:
        with safe_open(checkpoint_path, framework="pt") as f:
            metadata = f.metadata() or {}
        version = metadata.get("model_version", "")
    except Exception:
        logger.warning(
            "Could not read checkpoint metadata from %s, treating it as unversioned",
            checkpoint_path,
        )
        return ()

    # Pre-release tags come both dot- and hyphen-separated ("2.3.rc1", "2.4-rc2"); normalizing
    # the separator maps a release candidate onto the generation it is a candidate for.
    parsed = parse_model_version(version.replace("-", "."))
    logger.info("Checkpoint declares model_version=%s (parsed as %s)", version or "unknown", parsed)
    return parsed


def detect_params(checkpoint_path: str) -> PipelineParams:
    """Detect pipeline params from checkpoint metadata.
    Reads the ``model_version`` field from the safetensors config metadata and returns the
    params of the newest generation that version is at or above, falling back to
    ``LTX_2_PARAMS`` for older, unset, or unreadable versions.
    Does not log: ``detect_model_version`` already reports the version it read, and which
    generation's params that selects follows from it.
    """
    parsed = detect_model_version(checkpoint_path)
    for since, params in _PARAMS_SINCE_VERSION:
        if parsed >= since:
            return params

    return LTX_2_PARAMS


# =============================================================================
# Prompts
# =============================================================================

DEFAULT_NEGATIVE_PROMPT = (
    "has_subtitles, has_blurbox, transition from black, transition to black, speech_ending_short, "
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
