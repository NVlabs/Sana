import argparse
from pathlib import Path
from typing import NamedTuple

from diffusion.refiner.vendor.ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from diffusion.refiner.vendor.ltx_core.quantization import QuantizationPolicy
from diffusion.refiner.vendor.ltx_pipelines.utils.constants import (
    DEFAULT_IMAGE_CRF,
    DEFAULT_LORA_STRENGTH,
    DEFAULT_NEGATIVE_PROMPT,
    LTX_2_3_HQ_PARAMS,
    LTX_2_3_PARAMS,
    PipelineParams,
)


class ImageConditioningInput(NamedTuple):
    path: str
    frame_idx: int
    strength: float
    crf: int = DEFAULT_IMAGE_CRF


class VideoConditioningAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,  # noqa: ARG002
    ) -> None:
        path, strength_str = values
        resolved_path = resolve_path(path)
        strength = float(strength_str)
        current = getattr(namespace, self.dest) or []
        current.append((resolved_path, strength))
        setattr(namespace, self.dest, current)


class VideoMaskConditioningAction(argparse.Action):
    """Parse ``--conditioning-attention-mask PATH STRENGTH``.
    Stores a ``(mask_path, strength)`` tuple on the namespace.  The mask video
    should be grayscale with pixel values in [0, 1] controlling per-region
    conditioning attention strength.  The scalar *STRENGTH* is multiplied with
    the spatial mask before it is applied.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,
    ) -> None:
        if len(values) != 2:
            msg = f"{option_string} requires exactly 2 arguments (MASK_PATH STRENGTH), got {len(values)}"
            raise argparse.ArgumentError(self, msg)

        mask_path = resolve_path(values[0])
        strength = float(values[1])
        setattr(namespace, self.dest, (mask_path, strength))


class ImageAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,
    ) -> None:
        if len(values) not in (3, 4):
            msg = f"{option_string} requires 3 or 4 arguments (PATH FRAME_IDX STRENGTH [CRF]), got {len(values)}"
            raise argparse.ArgumentError(self, msg)

        conditioning = ImageConditioningInput(
            path=resolve_path(values[0]),
            frame_idx=int(values[1]),
            strength=float(values[2]),
            crf=int(values[3]) if len(values) > 3 else DEFAULT_IMAGE_CRF,
        )
        current = getattr(namespace, self.dest) or []
        current.append(conditioning)
        setattr(namespace, self.dest, current)


class LoraAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,
    ) -> None:
        if len(values) > 2:
            msg = f"{option_string} accepts at most 2 arguments (PATH and optional STRENGTH), got {len(values)} values"
            raise argparse.ArgumentError(self, msg)

        path = values[0]
        strength_str = values[1] if len(values) > 1 else str(DEFAULT_LORA_STRENGTH)

        resolved_path = resolve_path(path)
        strength = float(strength_str)

        current = getattr(namespace, self.dest) or []
        current.append(LoraPathStrengthAndSDOps(resolved_path, strength, LTXV_LORA_COMFY_RENAMING_MAP))
        setattr(namespace, self.dest, current)


def resolve_path(path: str) -> str:
    return str(Path(path).expanduser().resolve().as_posix())


QUANTIZATION_POLICIES = ("fp8-cast", "fp8-scaled-mm")


class QuantizationAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,
    ) -> None:
        if len(values) > 2:
            msg = (
                f"{option_string} accepts at most 2 arguments (POLICY and optional AMAX_PATH), got {len(values)} values"
            )
            raise argparse.ArgumentError(self, msg)

        policy_name = values[0]
        if policy_name not in QUANTIZATION_POLICIES:
            msg = f"Unknown quantization policy '{policy_name}'. Choose from: {', '.join(QUANTIZATION_POLICIES)}"
            raise argparse.ArgumentError(self, msg)

        if policy_name == "fp8-cast":
            if len(values) > 1:
                msg = f"{option_string} fp8-cast does not accept additional arguments"
                raise argparse.ArgumentError(self, msg)
            policy = QuantizationPolicy.fp8_cast()
        elif policy_name == "fp8-scaled-mm":
            amax_path = resolve_path(values[1]) if len(values) > 1 else None
            policy = QuantizationPolicy.fp8_scaled_mm(amax_path)

        setattr(namespace, self.dest, policy)


def detect_checkpoint_path(distilled: bool = False) -> str:
    """Pre-parse argv to extract the checkpoint path before building the full parser."""
    pre = argparse.ArgumentParser(add_help=False)
    flag = "--distilled-checkpoint-path" if distilled else "--checkpoint-path"
    pre.add_argument(flag, type=resolve_path, required=True)
    known, _ = pre.parse_known_args()
    return known.distilled_checkpoint_path if distilled else known.checkpoint_path


def basic_arg_parser(
    params: PipelineParams = LTX_2_3_PARAMS,
    distilled: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    if distilled:
        parser.add_argument(
            "--distilled-checkpoint-path",
            type=resolve_path,
            required=True,
            help="Path to LTX-2 distilled model checkpoint (.safetensors file).",
        )
    else:
        parser.add_argument(
            "--checkpoint-path",
            type=resolve_path,
            required=True,
            help="Path to LTX-2 model checkpoint (.safetensors file).",
        )
        parser.add_argument(
            "--num-inference-steps",
            type=int,
            default=params.num_inference_steps,
            help=(
                f"Number of denoising steps in the diffusion sampling process. "
                f"Higher values improve quality but increase generation time (default: {params.num_inference_steps})."
            ),
        )
    parser.add_argument(
        "--gemma-root",
        type=resolve_path,
        required=True,
        help="Path to the root directory containing the Gemma text encoder model files.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the desired video content to be generated by the model.",
    )
    parser.add_argument(
        "--output-path",
        type=resolve_path,
        required=True,
        help="Path to the output video file (MP4 format).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=params.seed,
        help=f"Random seed for reproducible generation (default: {params.seed}).",
    )
    parser.add_argument(
        "--lora",
        dest="lora",
        action=LoraAction,
        nargs="+",  # Accept 1-2 arguments per use (path and optional strength); validation is handled in LoraAction
        metavar=("PATH", "STRENGTH"),
        default=[],
        help=(
            "LoRA (Low-Rank Adaptation) model: path to model file and optional strength "
            f"(default strength: {DEFAULT_LORA_STRENGTH}). Can be specified multiple times. "
            "Example: --lora path/to/lora1.safetensors 0.8 --lora path/to/lora2.safetensors"
        ),
    )

    parser.add_argument("--enhance-prompt", action="store_true")

    def _positive_int(value: str) -> int:
        try:
            int_value = int(value)
            if int_value < 1:
                raise argparse.ArgumentTypeError("must be >= 1")
            return int_value
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"must be an integer, got {value}") from e

    # Layer streaming
    parser.add_argument(
        "--streaming-prefetch-count",
        type=_positive_int,
        default=None,
        metavar="N",
        help=(
            "Enable layer streaming prefetching N layers ahead. "
            "At most 1 + N layers reside on GPU at once. "
            "Must be >= 1. Example: --streaming-prefetch-count 2"
        ),
    )

    parser.add_argument(
        "--max-batch-size",
        type=_positive_int,
        default=1,
        metavar="N",
        help=(
            "Maximum batch size per transformer forward pass. "
            "Guided denoisers batch up to 4 guidance passes into a single call. "
            "Default 1 runs passes sequentially. Set to 4 to batch all passes "
            "together, which reduces layer-streaming PCIe transfers. "
            "Example: --max-batch-size 4"
        ),
    )

    parser.add_argument(
        "--quantization",
        dest="quantization",
        action=QuantizationAction,
        nargs="+",
        metavar=("POLICY", "AMAX_PATH"),
        default=None,
        help=(
            f"Quantization policy: {', '.join(QUANTIZATION_POLICIES)}. "
            "fp8-cast uses FP8 casting with upcasting during inference. "
            "fp8-scaled-mm uses FP8 scaled matrix multiplication (optionally provide amax calibration file path). "
            "Example: --quantization fp8-cast or --quantization fp8-scaled-mm /path/to/amax.json"
        ),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for transformer blocks to optimize performance.",
    )
    return parser


def new_video_gen_arg_parser(
    params: PipelineParams = LTX_2_3_PARAMS,
    distilled: bool = False,
) -> argparse.ArgumentParser:
    parser = basic_arg_parser(params=params, distilled=distilled)
    parser.add_argument(
        "--height",
        type=int,
        default=params.stage_1_height,
        help=f"Video height in pixels, divisible by 32 (default: {params.stage_1_height}).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=params.stage_1_width,
        help=f"Width of the generated video in pixels, should be divisible by 32 (default: {params.stage_1_width}).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=params.num_frames,
        help=f"Number of frames to generate in the output video sequence, num-frames = (8 x K) + 1, "
        f"where k is a non-negative integer (default: {params.num_frames}).",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=params.frame_rate,
        help=f"Frame rate of the generated video (fps) (default: {params.frame_rate}).",
    )
    parser.add_argument(
        "--image",
        dest="images",
        action=ImageAction,
        nargs="+",
        metavar="ARG",
        default=[],
        help=(
            "Image conditioning input: PATH FRAME_IDX STRENGTH [CRF]. "
            "PATH is the image file, FRAME_IDX is the target frame index, "
            "STRENGTH is the conditioning strength (all three required). "
            f"CRF is the optional H.264 compression quality (0=lossless, default: {DEFAULT_IMAGE_CRF}). "
            "Can be specified multiple times. Example: --image path/to/image1.jpg 0 0.8 "
            "--image path/to/image2.jpg 160 0.9 0"
        ),
    )

    return parser


def video_editing_arg_parser(
    distilled: bool = True,
) -> argparse.ArgumentParser:
    """Base argument parser for video-editing pipelines (retake, extension, inpainting, sticker movement).
    Uses the same actions and conventions as basic_arg_parser but only the args needed for editing
    (no height/width/num-frames; resolution comes from input video). Default is distilled checkpoint only.
    """
    parser = basic_arg_parser(distilled=distilled)
    parser.add_argument("--video-path", type=resolve_path, required=True, help="Path to the source video.")
    parser.add_argument("--start-time", type=float, required=True, help="Start time of the region to regenerate (s).")
    parser.add_argument("--end-time", type=float, required=True, help="End time of the region to regenerate (s).")
    return parser


def default_1_stage_arg_parser(params: PipelineParams = LTX_2_3_PARAMS) -> argparse.ArgumentParser:
    video_guider = params.video_guider_params
    audio_guider = params.audio_guider_params
    parser = new_video_gen_arg_parser(params=params)
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help=(
            "Negative prompt describing what should not appear in the generated video, "
            "used to guide the diffusion process away from unwanted content. "
            "Default: a comprehensive negative prompt covering common artifacts and quality issues."
        ),
    )
    parser.add_argument(
        "--video-cfg-guidance-scale",
        type=float,
        default=video_guider.cfg_scale,
        help=(
            f"Classifier-free guidance (CFG) scale controlling how strongly "
            f"the model adheres to the video prompt. Higher values increase prompt "
            f"adherence but may reduce diversity. 1.0 means no effect "
            f"(default: {video_guider.cfg_scale})."
        ),
    )
    parser.add_argument(
        "--video-stg-guidance-scale",
        type=float,
        default=video_guider.stg_scale,
        help=(
            f"STG (Spatio-Temporal Guidance) scale controlling how strongly "
            f"the model reacts to the perturbation of the video modality. Higher values increase "
            f"the effect but may reduce quality. 0.0 means no effect "
            f"(default: {video_guider.stg_scale})."
        ),
    )
    parser.add_argument(
        "--video-rescale-scale",
        type=float,
        default=video_guider.rescale_scale,
        help=(
            f"Rescale scale controlling how strongly "
            f"the model rescales the video modality after applying other guidance. Higher values tend to decrease "
            f"oversaturation effects. 0.0 means no effect (default: {video_guider.rescale_scale})."
        ),
    )
    parser.add_argument(
        "--video-stg-blocks",
        type=int,
        nargs="*",
        default=video_guider.stg_blocks,
        help=(f"Which transformer blocks to perturb for STG. Default: {video_guider.stg_blocks}."),
    )
    parser.add_argument(
        "--a2v-guidance-scale",
        type=float,
        default=video_guider.modality_scale,
        help=(
            f"A2V (Audio-to-Video) guidance scale controlling how strongly "
            f"the model reacts to the perturbation of the audio-to-video cross-attention. Higher values may increase "
            f"lipsync quality. 1.0 means no effect (default: {video_guider.modality_scale})."
        ),
    )
    parser.add_argument(
        "--video-skip-step",
        type=int,
        default=video_guider.skip_step,
        help=(
            "Video skip step N controls periodic skipping during the video diffusion process: "
            "only steps where step_index % (N + 1) == 0 are processed, all others are skipped "
            f"(e.g., 0 = no skipping; 1 = skip every other step; 2 = skip 2 of every 3 steps; "
            f"default: {video_guider.skip_step})."
        ),
    )
    parser.add_argument(
        "--audio-cfg-guidance-scale",
        type=float,
        default=audio_guider.cfg_scale,
        help=(
            f"Audio CFG (Classifier-free guidance) scale controlling how strongly "
            f"the model adheres to the audio prompt. Higher values increase prompt "
            f"adherence but may reduce diversity. 1.0 means no effect "
            f"(default: {audio_guider.cfg_scale})."
        ),
    )
    parser.add_argument(
        "--audio-stg-guidance-scale",
        type=float,
        default=audio_guider.stg_scale,
        help=(
            f"Audio STG (Spatio-Temporal Guidance) scale controlling how strongly "
            f"the model reacts to the perturbation of the audio modality. Higher values increase "
            f"the effect but may reduce quality. 0.0 means no effect "
            f"(default: {audio_guider.stg_scale})."
        ),
    )
    parser.add_argument(
        "--audio-rescale-scale",
        type=float,
        default=audio_guider.rescale_scale,
        help=(
            f"Audio rescale scale controlling how strongly "
            f"the model rescales the audio modality after applying other guidance. "
            f"Experimental. 0.0 means no effect (default: {audio_guider.rescale_scale})."
        ),
    )
    parser.add_argument(
        "--audio-stg-blocks",
        type=int,
        nargs="*",
        default=audio_guider.stg_blocks,
        help=(f"Which transformer blocks to perturb for Audio STG. Default: {audio_guider.stg_blocks}."),
    )
    parser.add_argument(
        "--v2a-guidance-scale",
        type=float,
        default=audio_guider.modality_scale,
        help=(
            f"V2A (Video-to-Audio) guidance scale controlling how strongly "
            f"the model reacts to the perturbation of the video-to-audio cross-attention. Higher values may increase "
            f"lipsync quality. 1.0 means no effect (default: {audio_guider.modality_scale})."
        ),
    )
    parser.add_argument(
        "--audio-skip-step",
        type=int,
        default=audio_guider.skip_step,
        help=(
            "Audio skip step N controls periodic skipping during the audio diffusion process: "
            "only steps where step_index % (N + 1) == 0 are processed, all others are skipped "
            f"(e.g., 0 = no skipping; 1 = skip every other step; 2 = skip 2 of every 3 steps; "
            f"default: {audio_guider.skip_step})."
        ),
    )
    return parser


def default_2_stage_arg_parser(params: PipelineParams = LTX_2_3_PARAMS) -> argparse.ArgumentParser:
    parser = default_1_stage_arg_parser(params=params)
    parser.set_defaults(height=params.stage_2_height, width=params.stage_2_width)
    # Update help text to reflect 2-stage defaults
    for action in parser._actions:
        if "--height" in action.option_strings:
            action.help = (
                f"Height of the generated video in pixels, should be divisible by 64 "
                f"(default: {params.stage_2_height})."
            )
        if "--width" in action.option_strings:
            action.help = (
                f"Width of the generated video in pixels, should be divisible by 64 (default: {params.stage_2_width})."
            )
    parser.add_argument(
        "--distilled-lora",
        dest="distilled_lora",
        action=LoraAction,
        nargs="+",  # Accept 1-2 arguments per use (path and optional strength); validation is handled in LoraAction
        metavar=("PATH", "STRENGTH"),
        required=True,
        help=(
            "Distilled LoRA (Low-Rank Adaptation) model used in the second stage (upscaling and refinement): "
            f"path to model file and optional strength (default strength: {DEFAULT_LORA_STRENGTH}). "
            "The second stage upsamples the video by 2x resolution and refines it using a distilled "
            "denoising schedule (fewer steps, no CFG). The distilled LoRA is specifically trained "
            "for this refinement process to improve quality at higher resolutions. "
            "Example: --distilled-lora path/to/distilled_lora.safetensors 0.8"
        ),
    )
    parser.add_argument(
        "--spatial-upsampler-path",
        type=resolve_path,
        required=True,
        help=(
            "Path to the spatial upsampler model used to increase the resolution "
            "of the generated video in the latent space."
        ),
    )
    return parser


def hq_2_stage_arg_parser(params: PipelineParams = LTX_2_3_HQ_PARAMS) -> argparse.ArgumentParser:
    parser = default_2_stage_arg_parser(params=params)
    parser.add_argument(
        "--distilled-lora-strength-stage-1",
        type=float,
        default=0.25,
        help=(f"Strength of the distilled LoRA used in the first stage (default: {0.25})."),
    )
    parser.add_argument(
        "--distilled-lora-strength-stage-2",
        type=float,
        default=0.5,
        help=(f"Strength of the distilled LoRA used in the second stage (default: {0.5})."),
    )
    return parser


def default_2_stage_distilled_arg_parser(params: PipelineParams = LTX_2_3_PARAMS) -> argparse.ArgumentParser:
    parser = new_video_gen_arg_parser(params=params, distilled=True)
    parser.set_defaults(height=params.stage_2_height, width=params.stage_2_width)
    # Update help text to reflect 2-stage defaults
    for action in parser._actions:
        if "--height" in action.option_strings:
            action.help = (
                f"Height of the generated video in pixels, should be divisible by 64 "
                f"(default: {params.stage_2_height})."
            )
        if "--width" in action.option_strings:
            action.help = (
                f"Width of the generated video in pixels, should be divisible by 64 (default: {params.stage_2_width})."
            )
    parser.add_argument(
        "--spatial-upsampler-path",
        type=resolve_path,
        required=True,
        help=(
            "Path to the spatial upsampler model used to increase the resolution "
            "of the generated video in the latent space."
        ),
    )
    return parser
