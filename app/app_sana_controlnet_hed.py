# Changed from https://github.com/GaParmar/img2img-turbo/blob/main/gradio_sketch2image.py
import argparse
import os
import random
import socket
import tempfile
import time

import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from app import safety_check
from app.sana_controlnet_pipeline import SanaControlNetPipeline

MAX_SEED = 1000000000
DEFAULT_SKETCH_GUIDANCE = 0.28
DEMO_PORT = int(os.getenv("DEMO_PORT", "15432"))
ROOT_PATH = os.getenv("ROOT_PATH", None)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gr.set_static_paths(paths=["asset"])

blank_image = Image.new("RGB", (1024, 1024), (255, 255, 255))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument(
        "--model_path",
        nargs="?",
        default="hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth",
        type=str,
        help="Path to the model file (positional)",
    )
    parser.add_argument("--output", default="./", type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--cfg_scale", default=5.0, type=float)
    parser.add_argument("--pag_scale", default=2.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--step", default=-1, type=int)
    parser.add_argument("--custom_image_size", default=None, type=int)
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
    pipe = SanaControlNetPipeline(args.config)
    pipe.from_pretrained(model_path)
    pipe.register_progress_bar(gr.Progress())

    # safety checker
    safety_checker_tokenizer = AutoTokenizer.from_pretrained(args.shield_model_path)
    safety_checker_model = AutoModelForCausalLM.from_pretrained(
        args.shield_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).to(device)


def save_image(img):
    if isinstance(img, dict):
        return img["composite"]
    if isinstance(img, dict):
        img = img["composite"]
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(temp_file.name)
    return temp_file.name


def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))
    return img


@torch.no_grad()
@torch.inference_mode()
def run(
    image,
    prompt: str,
    sketch_thickness: int,
    guidance_scale: float,
    inference_steps: int,
    seed: int,
    blend_alpha: float,
) -> tuple[Image, str]:
    print(f"Prompt: {prompt}, cond: {image['composite']}")
    image["composite"] = Image.open(image["composite"]).convert("RGB")
    image_numpy = np.array(image["composite"])

    if prompt.strip() == "" and (np.sum(image_numpy == 255) >= 3145628 or np.sum(image_numpy == 0) >= 3145628):
        return blank_image, "Please input the prompt or draw something."

    if safety_check.is_dangerous(safety_checker_tokenizer, safety_checker_model, prompt, threshold=0.2):
        prompt = "A red heart."

    pipe.set_blend_alpha(blend_alpha)
    start_time = time.time()
    images = pipe(
        prompt=prompt,
        ref_image=image["composite"],
        guidance_scale=guidance_scale,
        num_inference_steps=inference_steps,
        num_images_per_prompt=1,
        sketch_thickness=sketch_thickness,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    latency = time.time() - start_time

    if latency < 1:
        latency = latency * 1000
        latency_str = f"{latency:.2f}ms"
    else:
        latency_str = f"{latency:.2f}s"
    torch.cuda.empty_cache()

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
    img = img[0]
    return img, latency_str


model_size = "1.6" if "1600M" in args.model_path else "0.6"
title = f"""
    <div style='display: flex; align-items: center; justify-content: center; text-align: center;'>
        <img src="https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/main/asset/logo.png" width="25%" alt="logo"/>
    </div>
"""
DESCRIPTION = f"""
        <p><span style="font-size: 36px; font-weight: bold;">Sana-ControlNet-{model_size}B</span><span style="font-size: 20px; font-weight: bold;">{args.image_size}px</span></p>
        <p style="font-size: 18px; font-weight: bold;">Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer</p>
        <p><span style="font-size: 16px;"><a href="https://arxiv.org/abs/2410.10629">[Paper]</a></span> <span style="font-size: 16px;"><a href="https://github.com/NVlabs/Sana">[Github]</a></span> <span style="font-size: 16px;"><a href="https://nvlabs.github.io/Sana">[Project]</a></span</p>
        <p style="font-size: 18px; font-weight: bold;">Powered by <a href="https://hanlab.mit.edu/projects/dc-ae">DC-AE</a> with 32x latent space, </p>running on node {socket.gethostname()}.
        <p style="font-size: 16px; font-weight: bold;">Unsafe word will give you a 'Red Heart' in the image instead.</p>
        """
if model_size == "0.6":
    DESCRIPTION += "\n<p>0.6B model's text rendering ability is limited.</p>"
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"


with gr.Blocks(css_paths="asset/app_styles/controlnet_app_style.css", title=f"Sana Sketch-to-Image Demo") as demo:
    gr.Markdown(title)
    gr.HTML(DESCRIPTION)

    with gr.Row(elem_id="main_row"):
        with gr.Column(elem_id="column_input"):
            gr.Markdown("## INPUT", elem_id="input_header")
            with gr.Group():
                canvas = gr.ImageEditor(
                    value=blank_image,
                    height=640,
                    image_mode="RGB",
                    sources=["upload", "clipboard"],
                    type="filepath",
                    label="Sketch",
                    show_label=False,
                    show_download_button=True,
                    interactive=True,
                    transforms=[],
                    canvas_size=(1024, 1024),
                    scale=1,
                    brush=gr.Brush(default_size=3, colors=["#000000"], color_mode="fixed"),
                    format="webp",
                    layers=False,
                )
                with gr.Row():
                    prompt = gr.Text(label="Prompt", placeholder="Enter your prompt", scale=6)
                    run_button = gr.Button("Run", scale=1, elem_id="run_button")
            download_sketch = gr.DownloadButton("Download Sketch", scale=1, elem_id="download_sketch")

            with gr.Row():
                sketch_thickness = gr.Slider(
                    label="Sketch",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=2,
                )
                seed = gr.Slider(label="Seed", show_label=True, minimum=0, maximum=MAX_SEED, value=233, step=1, scale=4)
                randomize_seed = gr.Button("Random Seed", scale=1, min_width=50, elem_id="random_seed")
            with gr.Row():
                inference_steps = gr.Slider(
                    label="Sampling steps",
                    minimum=5,
                    maximum=40,
                    step=1,
                    value=20,
                )
                guidance_scale = gr.Slider(
                    label="CFG Guidance scale",
                    minimum=1,
                    maximum=10,
                    step=0.1,
                    value=4.5,
                )
                blend_alpha = gr.Slider(
                    label="Blend Alpha",
                    minimum=0,
                    maximum=1,
                    step=0.1,
                    value=0,
                )

        with gr.Column(elem_id="column_output"):
            gr.Markdown("## OUTPUT", elem_id="output_header")
            with gr.Group():
                result = gr.Image(
                    format="webp",
                    height=640,
                    image_mode="RGB",
                    type="pil",
                    label="Result",
                    show_label=False,
                    show_download_button=True,
                    interactive=False,
                    elem_id="output_image",
                )
                latency_result = gr.Text(label="Inference Latency", show_label=True)

            download_result = gr.DownloadButton("Download Result", elem_id="download_result")
            gr.Markdown("### Instructions")
            gr.Markdown("**1**. Enter a text prompt (e.g. a cat)")
            gr.Markdown("**2**. Start sketching or upload a reference image")
            gr.Markdown("**3**. Try different seeds to generate different results")

    run_inputs = [canvas, prompt, sketch_thickness, guidance_scale, inference_steps, seed, blend_alpha]
    run_outputs = [result, latency_result]
    randomize_seed.click(
        lambda: random.randint(0, MAX_SEED),
        inputs=[],
        outputs=seed,
        api_name=False,
        queue=False,
    )

    gr.on(
        triggers=[run_button.click],
        fn=run,
        inputs=run_inputs,
        outputs=run_outputs,
        api_name=False,
    )

    gr.Examples(
        examples=[
            [
                "asset/mit-logo.jpg",
                "Ethereal fantasy concept art of  A logo of MIT. magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy.",
            ],
            [
                "asset/robot.png",
                "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts.",
            ],
            [
                "asset/apple.webp",
                "A blue melting apple.",
            ],
        ],
        inputs=[canvas, prompt],
    )

    download_sketch.click(fn=save_image, inputs=canvas, outputs=download_sketch)
    download_result.click(fn=save_image, inputs=result, outputs=download_result)
    gr.Markdown("MIT Accessibility: https://accessibility.mit.edu/", elem_id="accessibility")


if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0", server_port=DEMO_PORT, debug=False, share=args.share, root_path=ROOT_PATH
    )
