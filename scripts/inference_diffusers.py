# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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

"""Generate an image with a Sana model from Hugging Face using diffusers."""

import argparse

import torch
from diffusers import SanaPipeline


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        help="Hugging Face model repository.",
    )
    parser.add_argument("--prompt", default='a cyberpunk cat with a neon sign that says "Sana"')
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="sana.png")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Sana Diffusers inference requires a CUDA device.")

    device = "cuda"
    pipe = SanaPipeline.from_pretrained(
        args.model,
        variant="fp16",
        torch_dtype=torch.float16,
    ).to(device)
    pipe.vae.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)

    result = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    )
    result.images[0].save(args.output)


if __name__ == "__main__":
    main()
