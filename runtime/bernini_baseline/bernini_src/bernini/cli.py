# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared command-line plumbing for the Bernini Renderer inference scripts."""

import argparse
import json
import logging

from transformers import PretrainedConfig

from .pipeline import BerniniRendererPipeline, BerniniPipeline
from .prompt_enhancer import get_system_prompt_for_task

# Standard Wan2.2 negative prompt; suppresses common quality / anatomy
# artefacts when no neg prompt is supplied on the command line.
DEFAULT_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
# Allowed --guidance_mode values, shared by argparse and the case-file loader.
GUIDANCE_MODES = ["rv2v", "v2v", "v2v_chain", "t2v", "r2v_apg", "v2v_apg", "t2v_apg", "vae_txt_vit_wapg", "rv2v_wapg"]

# Keys a case file under assets/testcases/ may set: one example's routing and
# inputs. Generation params (seed, num_frames, omega_*, ...) stay on the CLI.
CASE_KEYS = ("task_type", "guidance_mode", "prompt", "video", "image", "images", "output")


def add_common_args(parser):
    g = parser.add_argument_group("model")
    g.add_argument(
        "--config",
        default="configs/bernini_renderer_wan22",
        help="model config directory or you can pass the Diffusers directory to load model directly",
    )
    g.add_argument("--high_noise_ckpt", default=None, help="high-noise checkpoint (dir or HF repo)")
    g.add_argument("--low_noise_ckpt", default=None, help="low-noise checkpoint (dir or HF repo)")
    g.add_argument("--use_unipc", action=argparse.BooleanOptionalAction, default=True,
                   help="use the UniPC scheduler, required by the *_apg guidance modes "
                        "(default: on; pass --no-use_unipc to disable)")
    g.add_argument("--use_src_tgt_id", action=argparse.BooleanOptionalAction, default=True,
                   help="use source-id rotary embeddings, i.e. use_src_id_rotary_emb "
                        "(default: on; pass --no-use_src_tgt_id to disable)")
    g.add_argument("--interpolate_src_id", action=argparse.BooleanOptionalAction, default=True,
                   help="when reference sources exceed --max_trained_src_id, evenly map their "
                        "source ids into the trained range instead of extrapolating "
                        "(default: on; pass --no-interpolate_src_id to disable)")
    g.add_argument("--max_trained_src_id", type=int, default=5,
                   help="largest source_id seen during training; sources beyond this are "
                        "interpolated into [1, max_trained_src_id] (default: 5)")

    g = parser.add_argument_group("input")
    g.add_argument("--prompt", default=None, help="text prompt / editing instruction")
    g.add_argument("--task_type", default="v2v", help="task type (drives prompt enhancement)")
    g.add_argument("--video", nargs="+", default=None, help="source video path(s)")
    g.add_argument("--image", default=None, help="single source image for image editing")
    g.add_argument("--images", nargs="+", default=None, help="reference image(s)")
    g.add_argument("--output", default="outputs/output.mp4", help="output path")
    g.add_argument("--inputs", default=None, help="json/jsonl list of tasks (batch mode)")
    g.add_argument("--case", default=None,
                   help="json file describing a single example case (see assets/testcases/)")

    g = parser.add_argument_group("generation")
    g.add_argument("--neg_prompt", default=DEFAULT_NEG_PROMPT,
                   help="negative prompt (defaults to the standard Wan2.2 neg prompt)")
    g.add_argument("--system_prompt", default="",
                   help="system prompt prefix (default: auto-selected from --task_type)")
    g.add_argument("--num_frames", type=int, default=81)
    g.add_argument("--max_image_size", type=int, default=848)
    g.add_argument(
        "--height",
        type=int,
        default=480,
        help=(
            "output height. For Bernini: if --height/--width are positive, they override the output size; "
            "if both are 0, the output uses the input video's resolution"
        ),
    )
    g.add_argument(
        "--width",
        type=int,
        default=848,
        help=(
            "output width. For Bernini: if --height/--width are positive, they override the output size; "
            "if both are 0, the output uses the input video's resolution"
        ),
    )
    g.add_argument("--num_inference_steps", type=int, default=40)
    g.add_argument("--guidance_mode", default="rv2v", choices=GUIDANCE_MODES)
    g.add_argument("--omega_vid", type=float, default=1.25)
    g.add_argument("--omega_img", type=float, default=4.5)
    g.add_argument("--omega_txt", type=float, default=4.0)
    g.add_argument("--omega_tgt", type=float, default=0.5)
    g.add_argument("--omega_scale", type=float, default=0.8)
    g.add_argument("--planning_step", type=int, default=25)
    g.add_argument("--vit_txt_cfg", type=float, default=1.2)
    g.add_argument("--vit_img_cfg", type=float, default=1.0)
    g.add_argument("--vit_denoising_step", type=int, default=5)
    g.add_argument("--flow_shift", type=float, default=5.0)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--fps", type=int, default=16)
    g.add_argument("--eta", type=float, default=0.5)
    g.add_argument("--norm_threshold", type=float, nargs="+", default=[50.0, 50.0, 50.0])
    g.add_argument("--momentum", type=float, default=0)

    g = parser.add_argument_group("prompt enhancer")
    g.add_argument("--use_pe", action="store_true", help="enhance the prompt via an OpenAI-compatible endpoint")
    g.add_argument("--pe_model", default=None, help="prompt-enhancer model name")
    return parser


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_pipeline(args, device):
    config_dict, _ = PretrainedConfig.get_config_dict(args.config)
    model_type = config_dict.get("model_type")

    if model_type == "bernini":
        if args.high_noise_ckpt is not None or args.low_noise_ckpt is not None:
            raise ValueError(
                "Bernini uses the self-contained Bernini-Diffusers layout; "
                "pass that directory to --config and omit --high_noise_ckpt/--low_noise_ckpt"
            )
        logging.getLogger("bernini.cli").info(
            "Loading full Bernini directly from the Bernini-Diffusers dir '%s'", args.config
        )
        return BerniniPipeline.from_pretrained(
            args.config,
            device=device,
            use_unipc=args.use_unipc,
            use_src_id_rotary_emb=args.use_src_tgt_id,
            interpolate_src_id=args.interpolate_src_id,
            max_trained_src_id=args.max_trained_src_id,
        )

    if (args.high_noise_ckpt is None) != (args.low_noise_ckpt is None):
        raise ValueError(
            "--high_noise_ckpt and --low_noise_ckpt must be given together; "
            "got only one of them"
        )
    else:
        load_ckpt_weights = args.high_noise_ckpt is not None and args.low_noise_ckpt is not None
        if not load_ckpt_weights:
            logging.getLogger("bernini.cli").info(
                "no --high_noise_ckpt/--low_noise_ckpt given; loading Bernini weights directly "
                "from the diffusers-format dir '%s' (transformer/transformer_2)", args.config
            )
        return BerniniRendererPipeline.from_pretrained(
            args.config,
            high_noise_ckpt=args.high_noise_ckpt,
            low_noise_ckpt=args.low_noise_ckpt,
            device=device,
            load_ckpt_weights=load_ckpt_weights,
            use_unipc=args.use_unipc,
            shift=args.flow_shift,
            use_src_id_rotary_emb=args.use_src_tgt_id,
            interpolate_src_id=args.interpolate_src_id,
            max_trained_src_id=args.max_trained_src_id,
        )


def generation_kwargs(args) -> dict:
    """Pipeline kwargs shared by every task in a run."""
    return dict(
        neg_prompt=args.neg_prompt,
        num_frames=args.num_frames,
        max_image_size=args.max_image_size,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_mode=args.guidance_mode,
        omega_vid=args.omega_vid,
        omega_img=args.omega_img,
        omega_txt=args.omega_txt,
        omega_tgt=args.omega_tgt,
        omega_scale=args.omega_scale,
        planning_step=args.planning_step,
        vit_txt_cfg=args.vit_txt_cfg,
        vit_img_cfg=args.vit_img_cfg,
        vit_denoising_step=args.vit_denoising_step,
        flow_shift=args.flow_shift,
        seed=args.seed,
        fps=args.fps,
        eta=args.eta,
        norm_threshold=tuple(args.norm_threshold),
        momentum=args.momentum,
    )


def resolve_system_prompt(task: dict, args) -> str:
    """System-prompt prefix for one task.

    An explicit ``--system_prompt`` wins; otherwise it is auto-selected from
    the task type (per-task in batch mode, else the ``--task_type`` default).
    """
    if args.system_prompt:
        return args.system_prompt
    return get_system_prompt_for_task(task.get("task_type", args.task_type))


def apply_case_file(args) -> None:
    """Merge a single-example case file (``--case``) into ``args``.

    A case file under ``assets/testcases/`` bundles one example's routing and
    inputs -- task_type, guidance_mode, prompt, source media paths and the
    output path -- so a run is just ``--case assets/testcases/<task>/<task>.json``
    instead of a command line carrying a wall of prompt text. The file fully
    defines those fields; generation params (``--seed``, ``--num_frames``,
    ...) are still taken from the command line.
    """
    if not args.case:
        return
    if args.inputs:
        raise ValueError("--case (single example) and --inputs (batch) cannot be combined")
    with open(args.case) as f:
        case = json.load(f)
    if not isinstance(case, dict):
        raise ValueError(f"--case file '{args.case}' must hold a single JSON object")
    unknown = sorted(set(case) - set(CASE_KEYS))
    if unknown:
        raise ValueError(
            f"--case file '{args.case}' has unsupported keys {unknown}; "
            f"allowed keys: {list(CASE_KEYS)}"
        )
    if not case.get("prompt"):
        raise ValueError(f"--case file '{args.case}' must set 'prompt'")
    mode = case.get("guidance_mode")
    if mode is not None and mode not in GUIDANCE_MODES:
        raise ValueError(
            f"--case file '{args.case}': invalid guidance_mode '{mode}'; "
            f"choose from {GUIDANCE_MODES}"
        )
    for key, value in case.items():
        setattr(args, key, value)


def load_tasks(args) -> list:
    """Return task dicts from `--inputs`, or a single task built from CLI args."""
    if args.inputs:
        with open(args.inputs) as f:
            if args.inputs.endswith(".jsonl"):
                tasks = [json.loads(line) for line in f if line.strip()]
            else:
                tasks = json.load(f)
        return tasks
    if args.prompt is None:
        raise ValueError("provide --prompt, or --inputs for batch mode")
    return [
        {
            "prompt": args.prompt,
            "task_type": args.task_type,
            "video": args.video,
            "image": args.image,
            "images": args.images,
            "output": args.output,
        }
    ]
