# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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

"""Text-to-video inference for the released Sana-Video 2.0 models."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("DISABLE_XFORMERS", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import imageio
import pyrallis
import torch
from accelerate import init_empty_weights

from diffusion import DPMS
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode
from diffusion.model.utils import get_weight_dtype
from diffusion.utils.config import SanaVideoConfig, model_video_init_config, validate_sana_video2_config
from tools.download import find_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a Sana-Video 2.0 YAML config.")
    parser.add_argument("--model-path", required=True, help="Local path or hf:// URI for the model checkpoint.")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="A single text prompt.")
    prompt_group.add_argument("--prompt-file", help="UTF-8 text file containing one prompt per line.")
    parser.add_argument("--output-dir", default="output/sana_video2_samples")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt is not None:
        prompt = args.prompt.strip()
        if not prompt:
            raise ValueError("--prompt cannot be empty.")
        return [prompt]
    with open(args.prompt_file, encoding="utf-8") as prompt_file:
        prompts = [line.strip() for line in prompt_file if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompt_file}.")
    return prompts


def normalize_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    if "generator" in checkpoint:
        checkpoint = checkpoint["generator"]
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "state_dict_ema" in checkpoint:
        state_dict = checkpoint["state_dict_ema"]
    else:
        state_dict = checkpoint

    normalized = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key != "pos_embed":
            normalized[key] = value
    return normalized


def load_model_weights(model: torch.nn.Module, model_path: str) -> None:
    state_dict = normalize_state_dict(find_model(model_path))
    assign = any(parameter.is_meta for parameter in model.parameters())
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=assign)
    missing = [key for key in missing if key != "pos_embed"]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match the selected Sana-Video 2.0 config. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}."
        )


def encode_positive_prompt(prompt: str, config: SanaVideoConfig, tokenizer, text_encoder, device):
    if config.text_encoder.chi_prompt:
        prefix = "\n".join(config.text_encoder.chi_prompt)
        prompt = prefix + prompt
        max_length = len(tokenizer.encode(prefix)) + config.text_encoder.model_max_length - 2
    else:
        max_length = config.text_encoder.model_max_length
    tokens = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    selected = [0] + list(range(-config.text_encoder.model_max_length + 1, 0))
    embeddings = text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0]
    return embeddings[:, None, selected], tokens.attention_mask[:, selected]


def encode_negative_prompt(prompt: str, config: SanaVideoConfig, tokenizer, text_encoder, device):
    tokens = tokenizer(
        prompt,
        max_length=config.text_encoder.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    embeddings = text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0]
    return embeddings[:, None], tokens.attention_mask


def validate_dimensions(args: argparse.Namespace, config: SanaVideoConfig) -> None:
    temporal_stride, height_stride, width_stride = config.vae.vae_stride
    if args.height <= 0 or args.width <= 0 or args.num_frames <= 0:
        raise ValueError("height, width, and num_frames must be positive.")
    if args.height % height_stride or args.width % width_stride:
        raise ValueError(
            f"height and width must be divisible by the VAE stride "
            f"({height_stride}, {width_stride}); got ({args.height}, {args.width})."
        )
    if (args.num_frames - 1) % temporal_stride:
        raise ValueError(
            f"num_frames must satisfy (num_frames - 1) % {temporal_stride} == 0; " f"got {args.num_frames}."
        )
    if args.steps <= 0 or args.fps <= 0 or args.cfg_scale <= 0:
        raise ValueError("steps, fps, and cfg-scale must be positive.")


@torch.inference_mode()
def generate(
    prompt: str,
    config: SanaVideoConfig,
    args: argparse.Namespace,
    model: torch.nn.Module,
    vae,
    tokenizer,
    text_encoder,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    positive, positive_mask = encode_positive_prompt(prompt, config, tokenizer, text_encoder, device)
    negative = None
    condition_mask = positive_mask
    if args.cfg_scale > 1.0:
        negative, negative_mask = encode_negative_prompt(
            args.negative_prompt,
            config,
            tokenizer,
            text_encoder,
            device,
        )
        condition_mask = torch.cat([negative_mask, positive_mask], dim=0)

    temporal_stride, height_stride, width_stride = config.vae.vae_stride
    latent_frames = (args.num_frames - 1) // temporal_stride + 1
    latent_height = args.height // height_stride
    latent_width = args.width // width_stride
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        1,
        config.vae.vae_latent_dim,
        latent_frames,
        latent_height,
        latent_width,
        generator=generator,
        device=device,
        dtype=get_weight_dtype(config.model.mixed_precision),
    )
    model_kwargs = {
        "mask": condition_mask,
        "data_info": {"img_hw": torch.tensor([[args.height, args.width]], dtype=torch.float32, device=device)},
    }
    solver = DPMS(
        model,
        condition=positive,
        uncondition=negative,
        cfg_scale=args.cfg_scale,
        model_type="flow",
        guidance_type="classifier-free",
        model_kwargs=model_kwargs,
        schedule="FLOW",
    )
    flow_shift = (
        args.flow_shift
        if args.flow_shift is not None
        else config.scheduler.inference_flow_shift or config.scheduler.flow_shift
    )
    samples = solver.sample(
        latents,
        steps=args.steps,
        order=2,
        skip_type="time_uniform_flow",
        method="multistep",
        flow_shift=flow_shift,
    )
    vae_dtype = next(vae.parameters()).dtype
    decoded = vae_decode(config.vae.vae_type, vae, samples.to(vae_dtype))
    if isinstance(decoded, list):
        decoded = torch.stack(decoded)
    return torch.clamp(127.5 * decoded + 127.5, 0, 255).permute(0, 2, 3, 4, 1).to("cpu", dtype=torch.uint8)[0]


def save_video(video: torch.Tensor, output_path: Path, fps: int) -> None:
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    try:
        for frame in video.numpy():
            writer.append_data(frame)
    finally:
        writer.close()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as config_file:
        config = pyrallis.load(SanaVideoConfig, config_file)
    validate_sana_video2_config(config)
    validate_dimensions(args, config)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested, but CUDA is not available.")
    weight_dtype = get_weight_dtype(config.model.mixed_precision)
    torch.manual_seed(args.seed)

    tokenizer, text_encoder = get_tokenizer_and_text_encoder(
        name=config.text_encoder.text_encoder_name,
        device=device,
    )
    text_encoder.eval().requires_grad_(False)

    # The causal encoder is training-only; inference only needs the Diffusers decoder.
    config.vae.use_causal_encode = False
    vae = get_vae(
        config.vae.vae_type,
        config.vae.vae_pretrained,
        device=device,
        dtype=get_weight_dtype(config.vae.weight_dtype),
        config=config.vae,
    )
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()

    model_kwargs = model_video_init_config(
        config,
        latent_size=config.model.image_size // config.vae.vae_stride[-1],
    )
    model_kwargs["config"] = None
    with init_empty_weights():
        model = build_model(
            config.model.model,
            use_fp32_attention=config.model.fp32_attention,
            **model_kwargs,
        )
    load_model_weights(model, args.model_path)
    model.to(device=device, dtype=weight_dtype).eval()
    model.set_cross_attention_xformers(False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = read_prompts(args)
    for index, prompt in enumerate(prompts):
        video = generate(
            prompt,
            config,
            args,
            model,
            vae,
            tokenizer,
            text_encoder,
            device,
            seed=args.seed + index,
        )
        output_path = output_dir / f"{index:04d}.mp4"
        save_video(video, output_path, args.fps)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
