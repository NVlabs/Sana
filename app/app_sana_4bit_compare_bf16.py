# Changed from https://huggingface.co/spaces/playgroundai/playground-v2.5/blob/main/app.py
import argparse
import os
import random
import time
from datetime import datetime

import GPUtil

# import gradio last to avoid conflicts with other imports
import gradio as gr
import safety_check
import spaces
import torch
from diffusers import SanaPipeline
from nunchaku.models.transformer_sana import NunchakuSanaTransformer2DModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_IMAGE_SIZE = 2048
MAX_SEED = 1000000000

DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024

# num_inference_steps, guidance_scale, seed
EXAMPLES = [
    [
        "🐶 Wearing 🕶 flying on the 🌈",
        1024,
        1024,
        20,
        5,
        2,
    ],
    [
        "大漠孤烟直, 长河落日圆",
        1024,
        1024,
        20,
        5,
        23,
    ],
    [
        "Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, "
        "volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, "
        "art nouveau style, illustration art artwork by SenseiJaye, intricate detail.",
        1024,
        1024,
        20,
        5,
        233,
    ],
    [
        "A photo of a Eurasian lynx in a sunlit forest, with tufted ears and a spotted coat. The lynx should be "
        "sharply focused, gazing into the distance, while the background is softly blurred for depth. Use cinematic "
        "lighting with soft rays filtering through the trees, and capture the scene with a shallow depth of field "
        "for a natural, peaceful atmosphere. 8K resolution, highly detailed, photorealistic, "
        "cinematic lighting, ultra-HD.",
        1024,
        1024,
        20,
        5,
        2333,
    ],
    [
        "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. "
        "She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. "
        "She wears sunglasses and red lipstick. She walks confidently and casually. "
        "The street is damp and reflective, creating a mirror effect of the colorful lights. "
        "Many pedestrians walk about.",
        1024,
        1024,
        20,
        5,
        23333,
    ],
    [
        "Cozy bedroom with vintage wooden furniture and a large circular window covered in lush green vines, "
        "opening to a misty forest. Soft, ambient lighting highlights the bed with crumpled blankets, a bookshelf, "
        "and a desk. The atmosphere is serene and natural. 8K resolution, highly detailed, photorealistic, "
        "cinematic lighting, ultra-HD.",
        1024,
        1024,
        20,
        5,
        233333,
    ],
]


def hash_str_to_int(s: str) -> int:
    """Hash a string to an integer."""
    modulus = 10**9 + 7  # Large prime modulus
    hash_int = 0
    for char in s:
        hash_int = (hash_int * 31 + ord(char)) % modulus
    return hash_int


def get_pipeline(
    precision: str, use_qencoder: bool = False, device: str | torch.device = "cuda", pipeline_init_kwargs: dict = {}
) -> SanaPipeline:
    if precision == "int4":
        assert torch.device(device).type == "cuda", "int4 only supported on CUDA devices"
        transformer = NunchakuSanaTransformer2DModel.from_pretrained("mit-han-lab/svdq-int4-sana-1600m")

        pipeline_init_kwargs["transformer"] = transformer
        if use_qencoder:
            raise NotImplementedError("Quantized encoder not supported for Sana for now")
    else:
        assert precision == "bf16"
    pipeline = SanaPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        variant="bf16",
        torch_dtype=torch.bfloat16,
        **pipeline_init_kwargs,
    )

    pipeline = pipeline.to(device)
    return pipeline


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--precisions",
        type=str,
        default=["int4"],
        nargs="*",
        choices=["int4", "bf16"],
        help="Which precisions to use",
    )
    parser.add_argument("--use-qencoder", action="store_true", help="Whether to use 4-bit text encoder")
    parser.add_argument("--no-safety-checker", action="store_true", help="Disable safety checker")
    parser.add_argument("--count-use", action="store_true", help="Whether to count the number of uses")
    return parser.parse_args()


args = get_args()


pipelines = []
pipeline_init_kwargs = {}
for i, precision in enumerate(args.precisions):

    pipeline = get_pipeline(
        precision=precision,
        use_qencoder=args.use_qencoder,
        device="cuda",
        pipeline_init_kwargs={**pipeline_init_kwargs},
    )
    pipelines.append(pipeline)
    if i == 0:
        pipeline_init_kwargs["vae"] = pipeline.vae
        pipeline_init_kwargs["text_encoder"] = pipeline.text_encoder

# safety checker
safety_checker_tokenizer = AutoTokenizer.from_pretrained(args.shield_model_path)
safety_checker_model = AutoModelForCausalLM.from_pretrained(
    args.shield_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
).to(pipeline.device)


@spaces.GPU(enable_queue=True)
def generate(
    prompt: str = None,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 4,
    guidance_scale: float = 0,
    seed: int = 0,
):
    print(f"Prompt: {prompt}")
    is_unsafe_prompt = False
    if safety_check.is_dangerous(safety_checker_tokenizer, safety_checker_model, prompt, threshold=0.2):
        prompt = "A peaceful world."
    images, latency_strs = [], []
    for i, pipeline in enumerate(pipelines):
        progress = gr.Progress(track_tqdm=True)
        start_time = time.time()
        image = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]
        end_time = time.time()
        latency = end_time - start_time
        if latency < 1:
            latency = latency * 1000
            latency_str = f"{latency:.2f}ms"
        else:
            latency_str = f"{latency:.2f}s"
        images.append(image)
        latency_strs.append(latency_str)
    if is_unsafe_prompt:
        for i in range(len(latency_strs)):
            latency_strs[i] += " (Unsafe prompt detected)"
    torch.cuda.empty_cache()

    if args.count_use:
        if os.path.exists("use_count.txt"):
            with open("use_count.txt") as f:
                count = int(f.read())
        else:
            count = 0
        count += 1
        current_time = datetime.now()
        print(f"{current_time}: {count}")
        with open("use_count.txt", "w") as f:
            f.write(str(count))
        with open("use_record.txt", "a") as f:
            f.write(f"{current_time}: {count}\n")

    return *images, *latency_strs


with open("./assets/description.html") as f:
    DESCRIPTION = f.read()
gpus = GPUtil.getGPUs()
if len(gpus) > 0:
    gpu = gpus[0]
    memory = gpu.memoryTotal / 1024
    device_info = f"Running on {gpu.name} with {memory:.0f} GiB memory."
else:
    device_info = "Running on CPU 🥶 This demo does not work on CPU."
notice = f'<strong>Notice:</strong>&nbsp;We will replace unsafe prompts with a default prompt: "A peaceful world."'

with gr.Blocks(
    css_paths=[f"assets/frame{len(args.precisions)}.css", "assets/common.css"],
    title=f"SVDQuant SANA-1600M Demo",
) as demo:

    def get_header_str():

        if args.count_use:
            if os.path.exists("use_count.txt"):
                with open("use_count.txt") as f:
                    count = int(f.read())
            else:
                count = 0
            count_info = (
                f"<div style='display: flex; justify-content: center; align-items: center; text-align: center;'>"
                f"<span style='font-size: 18px; font-weight: bold;'>Total inference runs: </span>"
                f"<span style='font-size: 18px; color:red; font-weight: bold;'>&nbsp;{count}</span></div>"
            )
        else:
            count_info = ""
        header_str = DESCRIPTION.format(device_info=device_info, notice=notice, count_info=count_info)
        return header_str

    header = gr.HTML(get_header_str())
    demo.load(fn=get_header_str, outputs=header)

    with gr.Row():
        image_results, latency_results = [], []
        for i, precision in enumerate(args.precisions):
            with gr.Column():
                gr.Markdown(f"# {precision.upper()}", elem_id="image_header")
                with gr.Group():
                    image_result = gr.Image(
                        format="png",
                        image_mode="RGB",
                        label="Result",
                        show_label=False,
                        show_download_button=True,
                        interactive=False,
                    )
                    latency_result = gr.Text(label="Inference Latency", show_label=True)
                    image_results.append(image_result)
                    latency_results.append(latency_result)
    with gr.Row():
        prompt = gr.Text(
            label="Prompt", show_label=False, max_lines=1, placeholder="Enter your prompt", container=False, scale=4
        )
        run_button = gr.Button("Run", scale=1)

    with gr.Row():
        seed = gr.Slider(label="Seed", show_label=True, minimum=0, maximum=MAX_SEED, value=233, step=1, scale=4)
        randomize_seed = gr.Button("Random Seed", scale=1, min_width=50, elem_id="random_seed")
    with gr.Accordion("Advanced options", open=False):
        with gr.Group():
            height = gr.Slider(label="Height", minimum=256, maximum=4096, step=32, value=1024)
            width = gr.Slider(label="Width", minimum=256, maximum=4096, step=32, value=1024)
        with gr.Group():
            num_inference_steps = gr.Slider(label="Sampling Steps", minimum=10, maximum=50, step=1, value=20)
            guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=10, step=0.1, value=5)

    input_args = [prompt, height, width, num_inference_steps, guidance_scale, seed]

    gr.Examples(examples=EXAMPLES, inputs=input_args, outputs=[*image_results, *latency_results], fn=generate)

    gr.on(
        triggers=[prompt.submit, run_button.click],
        fn=generate,
        inputs=input_args,
        outputs=[*image_results, *latency_results],
        api_name="run",
    )
    randomize_seed.click(
        lambda: random.randint(0, MAX_SEED), inputs=[], outputs=seed, api_name=False, queue=False
    ).then(fn=generate, inputs=input_args, outputs=[*image_results, *latency_results], api_name=False, queue=False)

    gr.Markdown("MIT Accessibility: https://accessibility.mit.edu/", elem_id="accessibility")


if __name__ == "__main__":
    demo.queue(max_size=20).launch(server_name="0.0.0.0", debug=True, share=True)
