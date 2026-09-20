#!/usr/bin/env python
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
from __future__ import annotations

import argparse
import os
import random
import socket
import sqlite3
import time
import uuid
from datetime import datetime

import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image
from torchvision.utils import save_image
from transformers import AutoModelForCausalLM, AutoTokenizer

from app import safety_check
from app.sana_sprint_pipeline import SanaSprintPipeline

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "1") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "4096"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"
DEMO_PORT = int(os.getenv("DEMO_PORT", "15432"))
os.environ["GRADIO_EXAMPLES_CACHE"] = "./.gradio/cache"
COUNTER_DB = os.getenv("COUNTER_DB", ".count.db")
ROOT_PATH = os.getenv("ROOT_PATH", None)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, "
        "cinemascope, moody, epic, gorgeous, film grain, grainy",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, "
        "majestic, magical, fantasy art, cover art, dreamy",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, "
        "detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, "
        "ultra detailed, intricate, professional",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
    },
]

styles = {k["name"]: (k["prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"
SCHEDULE_NAME = ["scm"]
DEFAULT_SCHEDULE_NAME = "scm"
NUM_IMAGES_PER_PROMPT = 1
INFER_SPEED = 0


def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))
    return img


def open_db():
    db = sqlite3.connect(COUNTER_DB)
    db.execute("CREATE TABLE IF NOT EXISTS counter(app CHARS PRIMARY KEY UNIQUE, value INTEGER)")
    db.execute("INSERT OR IGNORE INTO counter(app, value) VALUES('Sana-sCM', 0)")
    return db


def read_inference_count():
    with open_db() as db:
        cur = db.execute("SELECT value FROM counter WHERE app='Sana-sCM'")
        db.commit()
    return cur.fetchone()[0]


def write_inference_count(count):
    count = max(0, int(count))
    with open_db() as db:
        db.execute(f"UPDATE counter SET value=value+{count} WHERE app='Sana-sCM'")
        db.commit()


def run_inference(num_imgs=1):
    write_inference_count(num_imgs)
    count = read_inference_count()

    return (
        f"<span style='font-size: 16px; font-weight: bold;'>Total inference runs: </span><span style='font-size: "
        f"16px; color:red; font-weight: bold;'>{count}</span>"
    )


def update_inference_count():
    count = read_inference_count()
    return (
        f"<span style='font-size: 16px; font-weight: bold;'>Total inference runs: </span><span style='font-size: "
        f"16px; color:red; font-weight: bold;'>{count}</span>"
    )


def apply_style(style_name: str, positive: str) -> tuple[str, str]:
    p = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument(
        "--model_path",
        nargs="?",
        default="hf://Efficient-Large-Model/Sana_Sprint_1.6B_1024px/checkpoints/Sana_Sprint_1.6B_1024px.pth",
        type=str,
        help="Path to the model file (positional)",
    )
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--shield_model_path",
        type=str,
        help="The path to shield model, we employ ShieldGemma-2B by default.",
        default="google/shieldgemma-2b",
    )

    return parser.parse_known_args()[0]


args = get_args()

if torch.cuda.is_available():
    model_path = args.model_path
    pipe = SanaSprintPipeline(args.config)
    pipe.from_pretrained(model_path)
    pipe.register_progress_bar(gr.Progress())

    # safety checker
    safety_checker_tokenizer = AutoTokenizer.from_pretrained(args.shield_model_path)
    safety_checker_model = AutoModelForCausalLM.from_pretrained(
        args.shield_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).to(device)


def save_image_sana(img, seed="", save_img=False):
    unique_name = f"{str(uuid.uuid4())}_{seed}.png"
    save_path = os.path.join(f"output/online_demo_img/{datetime.now().date()}")
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(save_path, exist_ok=True)
    unique_name = os.path.join(save_path, unique_name)
    if save_img:
        save_image(img, unique_name, nrow=1, normalize=True, value_range=(-1, 1))

    return unique_name


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


@torch.no_grad()
@torch.inference_mode()
@spaces.GPU(enable_queue=True)
def generate(
    prompt: str = None,
    style: str = DEFAULT_STYLE_NAME,
    num_imgs: int = 1,
    seed: int = 0,
    height: int = 1024,
    width: int = 1024,
    guidance_scale: float = 4.5,
    num_inference_steps: int = 20,
    max_timesteps: float = 1.56830,
    intermediate_timesteps: float = 1.3,
    timesteps: str = None,
    randomize_seed: bool = False,
    use_resolution_binning: bool = True,
):
    global INFER_SPEED
    box = run_inference(num_imgs)
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"PORT: {DEMO_PORT}, model_path: {model_path}")
    if safety_check.is_dangerous(safety_checker_tokenizer, safety_checker_model, prompt, threshold=0.2):
        prompt = "A red heart."

    print(prompt)

    # few steps setting
    pipe.config.max_timesteps = max_timesteps
    pipe.config.intermediate_timesteps = intermediate_timesteps
    if isinstance(timesteps, str):
        custom_steps = [float(t.strip()) for t in timesteps.split(",") if t.strip()]
        custom_steps.append(0.0)
        pipe.config.timesteps = custom_steps
    prompt = apply_style(style, prompt)

    pipe.progress_fn(0, desc="Sana Start")

    time_start = time.time()
    images = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_imgs,
        generator=generator,
        use_resolution_binning=use_resolution_binning,
    )

    pipe.progress_fn(1.0, desc="Sana End")
    INFER_SPEED = (time.time() - time_start) / num_imgs

    save_img = False
    if save_img:
        img = [save_image_sana(img, seed, save_img=save_image) for img in images]
        print(img)
    else:
        img = [
            Image.fromarray(
                norm_ip(img, -1, 1)
                .mul(255)
                .add_(0.5)
                .clamp_(0, 255)
                .permute(1, 2, 0)
                .to("cpu", torch.uint8)
                .numpy()
                .astype(np.uint8)
            )
            for img in images
        ]

    torch.cuda.empty_cache()

    return (
        img,
        seed,
        f"<span style='font-size: 16px; font-weight: bold;'>Inference Speed: {INFER_SPEED:.3f} s/Img</span>",
        box,
    )


model_size = "1.6" if "1.6B" in args.model_path else "0.6"
title = f"""
    <div style='display: flex; align-items: center; justify-content: center; text-align: center;'>
        <img src="https://nvlabs.github.io/Sana/Sprint/asset/SANA-Sprint.png" width="50%" alt="logo"/>
    </div>
"""
DESCRIPTION = f"""
        <p><span style="font-size: 36px; font-weight: bold;">SANA-Sprint-{model_size}B</span><span style="font-size: 20px; font-weight: bold;">{args.image_size}px</span></p>
        <p style="font-size: 16px; font-weight: bold;">SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation</p>
        <p><span style="font-size: 16px;"><a href="https://arxiv.org/abs/2503.09641">[Paper]</a></span> <span style="font-size: 16px;"><a href="https://github.com/NVlabs/Sana">[Github]</a></span> <span style="font-size: 16px;"><a href="https://nvlabs.github.io/Sana">[Project]</a></span</p>
        <p style="font-size: 16px; font-weight: bold;">Powered by <a href="https://hanlab.mit.edu/projects/dc-ae">DC-AE</a> with 32x latent space, </p>running on node {socket.gethostname()}.
        <p style="font-size: 16px; font-weight: bold;">Unsafe word will give you a 'Red Heart' in the image instead.</p>
        """
if model_size == "0.6":
    DESCRIPTION += "\n<p>0.6B model's text rendering ability is limited.</p>"
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

examples = [
    "a tiny astronaut hatching from an egg on the moon",
    "A very detailed and realistic full body photo set of a tall, slim, and athletic Shiba Inu in a white oversized straight t-shirt, white shorts, and short white shoes.",
    "Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, art nouveau style, illustration art artwork by SenseiJaye, intricate detail.",
    "portrait photo of a girl, photograph, highly detailed face, depth of field",
    "🐶 Wearing 🕶 flying on the 🌈",
    "👧 with 🌹 in the ❄️",
    "an old rusted robot wearing pants and a jacket riding skis in a supermarket.",
    "professional portrait photo of an anthropomorphic cat wearing fancy gentleman hat and jacket walking in autumn forest.",
    "Astronaut in a jungle, cold color palette, muted colors, detailed",
    "a stunning and luxurious bedroom carved into a rocky mountainside seamlessly blending nature with modern design with a plush earth-toned bed textured stone walls circular fireplace massive uniquely shaped window framing snow-capped mountains dense forests",
]

css = """
.gradio-container{max-width: 640px !important}
body{align-items: center;}
h1{text-align:center}
"""
with gr.Blocks(css=css, title="SANA-Sprint") as demo:
    gr.Markdown(title)
    gr.HTML(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    info_box = gr.Markdown(
        value=f"<span style='font-size: 16px; font-weight: bold;'>Total inference runs: </span><span style='font-size: 16px; color:red; font-weight: bold;'>{read_inference_count()}</span>"
    )
    demo.load(fn=update_inference_count, outputs=info_box)  # update the value when re-loading the page
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(label="Result", show_label=False, columns=NUM_IMAGES_PER_PROMPT, format="webp", height=600)
    speed_box = gr.Markdown(
        value=f"<span style='font-size: 16px; font-weight: bold;'>Inference speed: {INFER_SPEED} s/Img</span>"
    )
    with gr.Accordion("Advanced options", open=False):
        with gr.Group():
            with gr.Row(visible=True):
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=args.image_size,
                )
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=args.image_size,
                )
                use_resolution_binning = gr.Checkbox(label="Use resolution binning", value=True)
            with gr.Row():
                num_inference_steps = gr.Slider(
                    label="Sampling steps",
                    minimum=1,
                    maximum=40,
                    step=1,
                    value=2,
                )
                guidance_scale = gr.Slider(
                    label="CFG Guidance scale",
                    minimum=1,
                    maximum=30,
                    step=0.5,
                    value=4.5,
                )
            with gr.Row(visible=True):
                max_timesteps = gr.Dropdown(
                    label="max timesteps",
                    choices=[1.57080, 1.56830, 1.56580, 1.56454, 1.56246, 1.55830, 1.55413, 1.55080, 1.54580],
                    value=1.57080,
                )
                intermediate_timesteps = gr.Dropdown(
                    label="intermediate timesteps",
                    choices=[1.0, 1.1, 1.2, 1.3, 1.4],
                    value=1.3,
                )
                timesteps = gr.Textbox(
                    label="Custom timesteps",
                    placeholder="eg: 1.4,1.3,1.1",
                    info="Separate by commas",
                )
            style_selection = gr.Radio(
                show_label=True,
                container=True,
                interactive=True,
                choices=STYLE_NAMES,
                value=DEFAULT_STYLE_NAME,
                label="Image Style",
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            with gr.Row(visible=True):
                schedule = gr.Radio(
                    show_label=True,
                    container=True,
                    interactive=True,
                    choices=SCHEDULE_NAME,
                    value=DEFAULT_SCHEDULE_NAME,
                    label="Sampler Schedule",
                    visible=True,
                )
                num_imgs = gr.Slider(
                    label="Num Images",
                    minimum=1,
                    maximum=6,
                    step=1,
                    value=1,
                )

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result, seed],
        fn=generate if CACHE_EXAMPLES else None,
        cache_examples=CACHE_EXAMPLES,
    )

    gr.on(
        triggers=[
            prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            style_selection,
            num_imgs,
            seed,
            height,
            width,
            guidance_scale,
            num_inference_steps,
            max_timesteps,
            intermediate_timesteps,
            timesteps,
            randomize_seed,
            use_resolution_binning,
        ],
        outputs=[result, seed, speed_box, info_box],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0", server_port=DEMO_PORT, debug=False, share=args.share, root_path=ROOT_PATH
    )
