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

"""Bernini / Bernini-R Gradio demo.

One task type per request; the guidance mode is derived automatically from
the task type (see ``GUIDANCE_MODE_BY_TASK`` below). Image tasks (``t2i`` /
``i2i``) force ``num_frames=1`` regardless of the UI slider.

Single-GPU:
    python gradio_demo.py \\
        --high_noise_ckpt <path-or-hf-repo> --low_noise_ckpt <path-or-hf-repo>

Single-GPU full Bernini:
    python gradio_demo.py \\
        --config ByteDance/Bernini-Diffusers

Multi-GPU with Ulysses sequence parallel (Open-VeOmni required):
    torchrun --nproc-per-node 8 gradio_demo.py --ulysses 8 \\
        --high_noise_ckpt <path-or-hf-repo> --low_noise_ckpt <path-or-hf-repo> \\
        --port 7860 --share

Multi-GPU full Bernini with Ulysses sequence parallel:
    torchrun --nproc-per-node 8 gradio_demo.py --ulysses 8 \\
        --config ByteDance/Bernini-Diffusers \\
        --port 7860 --share
"""

import argparse
import logging
import os
import tempfile
import traceback
from datetime import datetime, timedelta

import gradio as gr
import torch
import torch.distributed as dist

from bernini.cli import DEFAULT_NEG_PROMPT, GUIDANCE_MODES, build_pipeline
from bernini.pipeline import BerniniPipeline
from bernini.prompt_enhancer import get_system_prompt_for_task


logger = logging.getLogger("bernini.gradio")


# Task types backed by assets/testcases/. Selecting one of these in the UI
# fully determines the guidance mode below.
TASK_TYPE_CHOICES = ["t2i", "t2v", "i2i", "v2v", "mv2v", "r2v", "rv2v", "ads2v"]

# Derived from the case files under assets/testcases/<task>/*.json.
GUIDANCE_MODE_BY_TASK = {
    "t2i":   "t2v_apg",
    "t2v":   "t2v_apg",
    "i2i":   "v2v",
    "v2v":   "v2v_apg",
    "mv2v":  "v2v_apg",
    "r2v":   "r2v_apg",
    "rv2v":  "rv2v",
    "ads2v": "v2v_apg",
}

# Which media inputs each task consumes. `image_role` decides how the single-
# image tab is interpreted: "source" routes it to the pipeline's `image` slot
# (the editing source, used by i2i); "reference" merges it into `images` so
# the user can drop one ref image without opening the gallery tab; "none" is
# discarded.
TASK_INPUTS = {
    "t2i":   {"video": False, "image_role": "none",      "images": False},
    "t2v":   {"video": False, "image_role": "none",      "images": False},
    "i2i":   {"video": False, "image_role": "source",    "images": False},
    "v2v":   {"video": True,  "image_role": "none",      "images": False},
    "mv2v":  {"video": True,  "image_role": "none",      "images": False},
    "r2v":   {"video": False, "image_role": "reference", "images": True},
    "rv2v":  {"video": True,  "image_role": "reference", "images": True},
    "ads2v": {"video": True,  "image_role": "reference", "images": True},
}

IMAGE_TASKS = {"t2i", "i2i"}

BASE_TASK_DEFAULTS = {
    "max_image_size": 848,
    "num_inference_steps": 40,
    "num_frames": 81,
    "flow_shift": 5.0,
    "seed": 42,
    "fps": 16,
    "height": 480,
    "width": 848,
    "omega_vid": 1.25,
    "omega_img": 4.5,
    "omega_txt": 4.0,
    "omega_tgt": 0.5,
    "omega_scale": 0.8,
    "eta": 0.5,
    "momentum": 0.0,
    "planning_step": 25,
    "vit_txt_cfg": 1.2,
    "vit_img_cfg": 1.0,
    "vit_denoising_step": 5,
}

RENDERER_TASK_DEFAULTS = {
    "t2i": {"num_frames": 1, "guidance_mode": "t2v_apg"},
    "i2i": {"num_frames": 1, "guidance_mode": "v2v"},
    "t2v": {"guidance_mode": "t2v_apg"},
    "v2v": {"guidance_mode": "v2v_apg"},
    "mv2v": {"guidance_mode": "v2v_apg"},
    "r2v": {"guidance_mode": "r2v_apg"},
    "rv2v": {"guidance_mode": "rv2v"},
    "ads2v": {"guidance_mode": "v2v_apg"},
}

FULL_BERNINI_TASK_DEFAULTS = {
    "t2i": {
        "guidance_mode": "vae_txt_vit_wapg",
        "num_frames": 1,
        "max_image_size": 842,
        "height": 512,
        "width": 512,
        "num_inference_steps": 50,
        "omega_vid": 1.0,
        "omega_img": 1.0,
        "omega_txt": 4.0,
        "omega_tgt": 0.5,
        "omega_scale": 1.0,
    },
    "i2i": {
        "guidance_mode": "vae_txt_vit_wapg",
        "num_frames": 1,
        "max_image_size": 842,
        "height": 512,
        "width": 512,
        "num_inference_steps": 40,
        "omega_vid": 1.25,
        "omega_img": 1.25,
        "omega_txt": 4.0,
        "omega_tgt": 0.5,
        "omega_scale": 0.75,
    },
    "t2v": {
        "guidance_mode": "vae_txt_vit_wapg",
        "num_frames": 81,
        "max_image_size": 842,
        "height": 480,
        "width": 848,
        "num_inference_steps": 50,
        "omega_vid": 1.0,
        "omega_img": 1.0,
        "omega_txt": 4.0,
        "omega_tgt": 0.5,
        "omega_scale": 1.0,
    },
    "v2v": {
        "guidance_mode": "vae_txt_vit_wapg",
        "num_frames": 81,
        "max_image_size": 848,
        "num_inference_steps": 40,
        "omega_vid": 1.25,
        "omega_img": 1.25,
        "omega_txt": 4.0,
        "omega_tgt": 0.5,
        "omega_scale": 0.75,
    },
    "mv2v": {
        "guidance_mode": "vae_txt_vit_wapg",
        "num_frames": 81,
        "max_image_size": 848,
        "num_inference_steps": 40,
        "omega_vid": 1.25,
        "omega_img": 1.25,
        "omega_txt": 4.0,
        "omega_tgt": 0.5,
        "omega_scale": 0.75,
    },
    "rv2v": {
        "guidance_mode": "rv2v",
        "num_frames": 81,
        "max_image_size": 842,
        "num_inference_steps": 40,
        "omega_vid": 0.75,
        "omega_img": 3.0,
        "omega_txt": 4.0,
        "omega_tgt": 1.5,
        "omega_scale": 0.75,
    },
    "r2v": {
        "guidance_mode": "rv2v",
        "num_frames": 81,
        "max_image_size": 842,
        "num_inference_steps": 40,
        "omega_vid": 1.25,
        "omega_img": 2.5,
        "omega_txt": 4.0,
        "omega_tgt": 1.5,
        "omega_scale": 0.8,
    },
    "ads2v": {
        "guidance_mode": "vae_txt_vit_wapg",
        "num_frames": 81,
        "max_image_size": 848,
        "num_inference_steps": 40,
        "omega_vid": 1.25,
        "omega_img": 1.25,
        "omega_txt": 4.0,
        "omega_tgt": 0.5,
        "omega_scale": 0.75,
    },
}


# Filled in main().
PIPELINE = None
DEVICE = None
SAVE_BASE = None
REWRITER = None


def _is_full_bernini_pipeline() -> bool:
    return isinstance(PIPELINE, BerniniPipeline)


def _task_defaults(task_type: str) -> dict:
    defaults = dict(BASE_TASK_DEFAULTS)
    table = FULL_BERNINI_TASK_DEFAULTS if _is_full_bernini_pipeline() else RENDERER_TASK_DEFAULTS
    defaults.update(table.get(task_type, {}))
    defaults.setdefault("guidance_mode", GUIDANCE_MODE_BY_TASK.get(task_type))
    return defaults


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _coerce_video_paths(video_input):
    """gr.File(file_count='multiple') hands us a list of dicts / paths / None."""
    if not video_input:
        return None
    if isinstance(video_input, str):
        return [video_input]
    if isinstance(video_input, list):
        out = []
        for v in video_input:
            if v is None:
                continue
            if isinstance(v, str):
                out.append(v)
            elif hasattr(v, "name"):
                out.append(v.name)
            elif isinstance(v, dict) and v.get("path"):
                out.append(v["path"])
        return out or None
    return None


def _coerce_gallery_paths(gallery_input):
    """gr.Gallery returns a list of (path, caption) tuples."""
    if not gallery_input:
        return None
    out = []
    for item in gallery_input:
        if isinstance(item, (list, tuple)) and item:
            item = item[0]
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and item.get("path"):
            out.append(item["path"])
        elif hasattr(item, "name"):
            out.append(item.name)
    return out or None


def _output_path(task_type: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ext = "png" if task_type in IMAGE_TASKS else "mp4"
    return os.path.join(SAVE_BASE, f"{task_type}_{ts}.{ext}")


def _build_kwargs(
    prompt,
    task_type,
    video_input,
    image_input,
    gallery_input,
    guidance_mode,
    max_image_size,
    num_inference_steps,
    num_frames,
    flow_shift,
    seed,
    fps,
    height,
    width,
    omega_vid,
    omega_img,
    omega_txt,
    omega_tgt,
    omega_scale,
    eta,
    momentum,
    planning_step,
    vit_txt_cfg,
    vit_img_cfg,
    vit_denoising_step,
):
    needs = TASK_INPUTS[task_type]
    video = _coerce_video_paths(video_input) if needs["video"] else None
    images = _coerce_gallery_paths(gallery_input) if needs["images"] else None
    image = None
    if needs["image_role"] == "source":
        image = image_input or None
    elif needs["image_role"] == "reference" and image_input:
        # Treat the single-image tab as one more reference image so the user
        # does not have to use the gallery tab just for one ref.
        images = [image_input] + (images or [])

    # Image tasks must run with a single frame; surface that here instead of
    # silently honouring whatever the slider was set to.
    if task_type in IMAGE_TASKS:
        num_frames = 1

    task_defaults = _task_defaults(task_type)

    return dict(
        prompt=prompt or "",
        neg_prompt=DEFAULT_NEG_PROMPT,
        video=video,
        image=image,
        images=images,
        max_image_size=int(max_image_size if max_image_size is not None else task_defaults["max_image_size"]),
        num_inference_steps=int(num_inference_steps if num_inference_steps is not None else task_defaults["num_inference_steps"]),
        num_frames=int(num_frames if num_frames is not None else task_defaults["num_frames"]),
        flow_shift=float(flow_shift if flow_shift is not None else task_defaults["flow_shift"]),
        seed=int(seed if seed is not None else task_defaults["seed"]),
        fps=int(fps if fps is not None else task_defaults["fps"]),
        height=int(height if height is not None else task_defaults["height"]),
        width=int(width if width is not None else task_defaults["width"]),
        guidance_mode=guidance_mode or task_defaults["guidance_mode"],
        omega_vid=float(omega_vid if omega_vid is not None else task_defaults["omega_vid"]),
        omega_img=float(omega_img if omega_img is not None else task_defaults["omega_img"]),
        omega_txt=float(omega_txt if omega_txt is not None else task_defaults["omega_txt"]),
        omega_tgt=float(omega_tgt if omega_tgt is not None else task_defaults["omega_tgt"]),
        omega_scale=float(omega_scale if omega_scale is not None else task_defaults["omega_scale"]),
        eta=float(eta if eta is not None else task_defaults["eta"]),
        momentum=float(momentum if momentum is not None else task_defaults["momentum"]),
        planning_step=int(planning_step if planning_step is not None else task_defaults["planning_step"]),
        vit_txt_cfg=float(vit_txt_cfg if vit_txt_cfg is not None else task_defaults["vit_txt_cfg"]),
        vit_img_cfg=float(vit_img_cfg if vit_img_cfg is not None else task_defaults["vit_img_cfg"]),
        vit_denoising_step=int(vit_denoising_step if vit_denoising_step is not None else task_defaults["vit_denoising_step"]),
        system_prompt=get_system_prompt_for_task(task_type),
    )


def _broadcast(request):
    if dist.is_initialized() and dist.get_world_size() > 1:
        data = [request]
        dist.broadcast_object_list(data, src=0, device=torch.device("cpu"))


def _run_pipeline(task_type: str, *, write_output: bool, kwargs: dict):
    if _is_full_bernini_pipeline():
        full_kwargs = dict(kwargs)
        if isinstance(full_kwargs.get("video"), list):
            full_kwargs["video"] = [v for v in full_kwargs["video"] if v is not None]
            if len(full_kwargs["video"]) == 0:
                full_kwargs["video"] = None
            elif len(full_kwargs["video"]) == 1:
                full_kwargs["video"] = full_kwargs["video"][0]
        return PIPELINE(task_type, write_output=write_output, **full_kwargs)
    return PIPELINE(write_output=write_output, **kwargs)


def _run_pe_on_rank0(task_type, prompt, video, image, images):
    if REWRITER is None:
        return prompt
    try:
        rewritten = REWRITER(task_type, prompt, video=video, image=image, images=images)
    except Exception as e:  # noqa: BLE001
        logger.warning("PE failed (%s); using the raw prompt", e)
        return prompt
    return (rewritten or prompt).strip() or prompt


# ---------------------------------------------------------------------------
# rank 0 handler / worker loop
# ---------------------------------------------------------------------------

def generate_handler(
    prompt,
    task_type,
    video_input,
    image_input,
    gallery_input,
    use_pe,
    guidance_mode,
    max_image_size,
    num_inference_steps,
    num_frames,
    flow_shift,
    seed,
    fps,
    height,
    width,
    omega_vid,
    omega_img,
    omega_txt,
    omega_tgt,
    omega_scale,
    eta,
    momentum,
    planning_step,
    vit_txt_cfg,
    vit_img_cfg,
    vit_denoising_step,
    progress=gr.Progress(),
):
    if not task_type:
        gr.Warning("Please select a task type first!")
        return None, None, "", "Please select a task type first!"
    if not (prompt or "").strip():
        gr.Warning("Please enter a prompt!")
        return None, None, "", "Please enter a prompt!"

    kwargs = _build_kwargs(
        prompt, task_type, video_input, image_input, gallery_input,
        guidance_mode,
        max_image_size, num_inference_steps, num_frames, flow_shift,
        seed, fps, height, width,
        omega_vid, omega_img, omega_txt, omega_tgt, omega_scale, eta, momentum,
        planning_step, vit_txt_cfg, vit_img_cfg, vit_denoising_step,
    )

    if use_pe and REWRITER is not None:
        progress(0.02, desc=f"GPT prompt enhancement ({task_type})...")
        kwargs["prompt"] = _run_pe_on_rank0(
            task_type=task_type,
            prompt=kwargs["prompt"],
            video=kwargs["video"],
            image=kwargs["image"],
            images=kwargs["images"],
        )

    kwargs["output_path"] = _output_path(task_type)

    ws = dist.get_world_size() if dist.is_initialized() else 1
    progress(0.05, desc=f"Dispatching to {ws} GPU(s)...")

    logger.info(
        "[handler] task=%s guidance_mode=%s steps=%d frames=%d size=%dx%d",
        task_type, kwargs["guidance_mode"], kwargs["num_inference_steps"],
        kwargs["num_frames"], kwargs["width"], kwargs["height"],
    )

    _broadcast({"action": "generate", "task_type": task_type, "kwargs": dict(kwargs)})

    try:
        output_path = _run_pipeline(task_type, write_output=True, kwargs=kwargs)
    except Exception as e:  # noqa: BLE001
        logger.error("generate failed: %s\n%s", e, traceback.format_exc())
        return None, None, kwargs["prompt"], f"Generation failed: {e}"

    out_video = out_image = None
    if output_path:
        if output_path.endswith(".png") or task_type in IMAGE_TASKS:
            out_image = output_path
        else:
            out_video = output_path

    return out_video, out_image, kwargs["prompt"], f"Saved: {output_path}"


def worker_loop():
    while True:
        data = [None]
        dist.broadcast_object_list(data, src=0, device=torch.device("cpu"))
        request = data[0]
        if not request or request.get("action") == "shutdown":
            break
        if request.get("action") != "generate":
            continue
        try:
            _run_pipeline(request["task_type"], write_output=False, kwargs=request["kwargs"])
        except Exception as e:  # noqa: BLE001
            logger.error("[worker] generate error: %s\n%s", e, traceback.format_exc())


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _task_input_hint(task_type: str) -> str:
    if not task_type:
        return ""
    needs = TASK_INPUTS[task_type]
    bits = []
    if needs["video"]:
        bits.append("source video")
    if needs["image_role"] == "source":
        bits.append("single source image")
    if needs["image_role"] == "reference" or needs["images"]:
        bits.append("reference image(s)")
    extra = "inputs: " + ", ".join(bits) if bits else "text-only"
    frames = " | forced num_frames=1" if task_type in IMAGE_TASKS else ""
    return f"{extra}{frames}"


def _on_task_change(task_type: str):
    """Update the guidance_mode dropdown and the input-hint when the user
    picks a task. The dropdown is still user-editable afterwards."""
    if not task_type:
        return (
            gr.update(value=None), "", *(gr.update() for _ in range(19))
        )
    defaults = _task_defaults(task_type)
    return (
        gr.update(value=defaults.get("guidance_mode")),
        _task_input_hint(task_type),
        gr.update(value=defaults["max_image_size"]),
        gr.update(value=defaults["num_frames"]),
        gr.update(value=defaults["num_inference_steps"]),
        gr.update(value=defaults["flow_shift"]),
        gr.update(value=defaults["seed"]),
        gr.update(value=defaults["fps"]),
        gr.update(value=defaults["height"]),
        gr.update(value=defaults["width"]),
        gr.update(value=defaults["omega_vid"]),
        gr.update(value=defaults["omega_img"]),
        gr.update(value=defaults["omega_txt"]),
        gr.update(value=defaults["omega_tgt"]),
        gr.update(value=defaults["omega_scale"]),
        gr.update(value=defaults["eta"]),
        gr.update(value=defaults["momentum"]),
        gr.update(value=defaults["planning_step"]),
        gr.update(value=defaults["vit_txt_cfg"]),
        gr.update(value=defaults["vit_img_cfg"]),
        gr.update(value=defaults["vit_denoising_step"]),
    )


def create_ui():
    ws = dist.get_world_size() if dist.is_initialized() else 1
    demo_name = "Bernini Demo" if _is_full_bernini_pipeline() else "Bernini-R Demo"
    with gr.Blocks(title=demo_name) as demo:
        gr.Markdown(f"# 🎬 {demo_name}")
        gr.Markdown(
            f"**World size: {ws}** | Running **{'full Bernini' if _is_full_bernini_pipeline() else 'Bernini-R'}**. "
            "The selected task type auto-fills `guidance_mode`, chooses which media slots are used, "
            "and resets task-specific default inference parameters for the current pipeline."
        )

        with gr.Row():
            # ===== inputs =====
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Input")
                    prompt = gr.Textbox(
                        label="Prompt", lines=3,
                        placeholder="Describe the scene or the editing instruction...",
                    )

                    with gr.Tabs():
                        with gr.TabItem("Video"):
                            video_input = gr.File(
                                label="Upload video(s) — multi-file allowed; ads2v accepts source + ref",
                                file_count="multiple",
                                file_types=["video"],
                                type="filepath",
                            )
                        with gr.TabItem("Single image"):
                            image_input = gr.Image(
                                label="Upload an image (source for i2i, or a single reference)",
                                type="filepath",
                            )
                        with gr.TabItem("Multiple images"):
                            gallery_input = gr.Gallery(
                                label="Upload reference images (r2v / rv2v)",
                                columns=4, height="auto", interactive=True,
                            )

                with gr.Group():
                    gr.Markdown("### Task")
                    task_type = gr.Dropdown(
                        choices=TASK_TYPE_CHOICES,
                        value=None,
                        label="Task type (required)",
                        info="Auto-fills guidance_mode below; can still be overridden",
                    )
                    guidance_mode = gr.Dropdown(
                        choices=GUIDANCE_MODES,
                        value=None,
                        label="Guidance mode",
                        info="Rewritten whenever the task changes; edit manually if needed",
                    )
                    input_hint = gr.Markdown("")
                    use_pe = gr.Checkbox(
                        value=False,
                        label="Enable GPT prompt enhancer",
                        info="Requires BERNINI_PE_API_KEY / BERNINI_PE_BASE_URL (or OPENAI_API_KEY) and --use_pe at launch",
                    )

                with gr.Group():
                    gr.Markdown("### Basic parameters")
                    with gr.Row():
                        max_image_size = gr.Slider(256, 1280, value=848, step=16, label="Max image size")
                        num_frames = gr.Slider(1, 360, value=81, step=4, label="Num frames")
                    with gr.Row():
                        num_inference_steps = gr.Slider(10, 50, value=40, step=5, label="Inference steps")
                        flow_shift = gr.Slider(0.0, 12.0, value=5.0, step=0.5, label="Flow shift")
                    with gr.Row():
                        seed = gr.Number(value=42, precision=0, label="Seed")
                        fps = gr.Slider(1, 30, value=16, step=1, label="FPS")
                    with gr.Row():
                        height = gr.Number(value=480, precision=0, label="Height (text-only tasks)")
                        width = gr.Number(value=848, precision=0, label="Width (text-only tasks)")

                with gr.Accordion("Guidance (advanced)", open=False):
                    with gr.Row():
                        omega_vid = gr.Slider(0.0, 10.0, value=1.25, step=0.05, label="omega_vid")
                        omega_img = gr.Slider(0.0, 10.0, value=4.5, step=0.05, label="omega_img")
                        omega_txt = gr.Slider(0.0, 10.0, value=4.0, step=0.05, label="omega_txt")
                    with gr.Row():
                        omega_tgt = gr.Slider(0.0, 10.0, value=0.5, step=0.05, label="omega_tgt")
                        omega_scale = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="omega_scale")
                        eta = gr.Slider(0.0, 2.0, value=0.5, step=0.05, label="eta")
                    with gr.Row():
                        momentum = gr.Slider(-2.0, 2.0, value=0.0, step=0.05, label="momentum")
                        planning_step = gr.Slider(1, 50, value=25, step=1, label="planning_step")
                        vit_denoising_step = gr.Slider(1, 20, value=5, step=1, label="vit_denoising_step")
                    with gr.Row():
                        vit_txt_cfg = gr.Slider(0.0, 3.0, value=1.2, step=0.05, label="vit_txt_cfg")
                        vit_img_cfg = gr.Slider(0.0, 3.0, value=1.0, step=0.05, label="vit_img_cfg")

                generate_btn = gr.Button("Generate", variant="primary", size="lg")

            # ===== outputs =====
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                output_video = gr.Video(label="Generated video")
                output_image = gr.Image(label="Generated image")
                final_prompt = gr.Textbox(label="Prompt actually used", interactive=False, lines=3)
                output_status = gr.Textbox(label="Status", interactive=False, lines=2)

        # Auto-fill guidance_mode (still user-editable) + show input hint.
        task_type.change(
            fn=_on_task_change,
            inputs=task_type,
            outputs=[
                guidance_mode, input_hint,
                max_image_size, num_frames, num_inference_steps, flow_shift,
                seed, fps, height, width,
                omega_vid, omega_img, omega_txt, omega_tgt, omega_scale, eta, momentum,
                planning_step, vit_txt_cfg, vit_img_cfg, vit_denoising_step,
            ],
        )

        generate_btn.click(
            fn=generate_handler,
            inputs=[
                prompt, task_type, video_input, image_input, gallery_input,
                use_pe, guidance_mode,
                max_image_size, num_inference_steps, num_frames,
                flow_shift, seed, fps, height, width,
                omega_vid, omega_img, omega_txt, omega_tgt, omega_scale, eta, momentum,
                planning_step, vit_txt_cfg, vit_img_cfg, vit_denoising_step,
            ],
            outputs=[output_video, output_image, final_prompt, output_status],
        )

    return demo


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Bernini / Bernini-R Gradio demo")
    parser.add_argument("--config", default="configs/bernini_renderer_wan22")
    parser.add_argument("--high_noise_ckpt", default=None)
    parser.add_argument("--low_noise_ckpt", default=None)
    parser.add_argument("--use_unipc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_src_tgt_id", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interpolate_src_id", action=argparse.BooleanOptionalAction, default=True,
                        help="map reference source ids beyond --max_trained_src_id into the "
                             "trained range instead of extrapolating (default: on)")
    parser.add_argument("--max_trained_src_id", type=int, default=5,
                        help="largest source_id seen during training (default: 5)")
    parser.add_argument("--flow_shift", type=float, default=5.0)
    parser.add_argument("--ulysses", type=int, default=1)

    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="output root (default: a fresh tempdir)")

    parser.add_argument("--use_pe", action="store_true",
                        help="instantiate the prompt rewriter at startup")
    parser.add_argument("--pe_model", type=str, default=None)
    return parser.parse_args()


def main():
    global PIPELINE, DEVICE, SAVE_BASE, REWRITER

    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    DEVICE = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(DEVICE)

    if world_size > 1:
        dist.init_process_group(
            backend="cuda:nccl,cpu:gloo",
            timeout=timedelta(seconds=3600),
            rank=rank, world_size=world_size,
        )

    # Ulysses SP groups (no-op when ulysses=1).
    if args.ulysses > 1:
        from bernini.parallel import init_parallel_state
        init_parallel_state(ulysses_size=args.ulysses)

    PIPELINE = build_pipeline(args, DEVICE)

    SAVE_BASE = args.save_dir or tempfile.mkdtemp(prefix="bernini_gradio_")
    os.makedirs(SAVE_BASE, exist_ok=True)

    if rank == 0 and args.use_pe:
        from bernini.prompt_enhancer import PromptEnhancer
        REWRITER = PromptEnhancer(model=args.pe_model)

    if rank == 0:
        logger.info("output dir: %s", SAVE_BASE)
        logger.info(
            "loaded pipeline: %s",
            "BerniniPipeline" if _is_full_bernini_pipeline() else "BerniniRendererPipeline",
        )
        demo = create_ui()
        try:
            demo.queue(max_size=20, default_concurrency_limit=1).launch(
                server_name="0.0.0.0",
                server_port=args.port,
                share=args.share,
                allowed_paths=[os.path.abspath(SAVE_BASE)],
            )
        finally:
            if dist.is_initialized() and dist.get_world_size() > 1:
                _broadcast({"action": "shutdown"})
    else:
        worker_loop()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
