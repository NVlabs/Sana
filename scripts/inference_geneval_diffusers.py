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
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import datasets
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from pytorch_lightning import seed_everything
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from tqdm import tqdm, trange

from diffusion.utils.logger import get_root_logger

_CITATION = """\
@article{ghosh2024geneval,
  title={Geneval: An object-focused framework for evaluating text-to-image alignment},
  author={Ghosh, Dhruba and Hajishirzi, Hannaneh and Schmidt, Ludwig},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
"""

_DESCRIPTION = (
    "We demonstrate the advantages of evaluating text-to-image models using existing object detection methods, "
    "to produce a fine-grained instance-level analysis of compositional capabilities."
)


def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)


@torch.inference_mode()
def visualize():

    tqdm_desc = f"{save_root.split('/')[-1]} Using GPU: {args.gpu_id}: {args.start_index}-{args.end_index}"
    for index, metadata in tqdm(list(enumerate(metadatas)), desc=tqdm_desc, position=args.gpu_id, leave=True):
        metadata["include"] = (
            metadata["include"] if isinstance(metadata["include"], list) else eval(metadata["include"])
        )
        seed_everything(args.seed)
        index += args.start_index

        outpath = os.path.join(save_root, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        prompt = metadata["prompt"]
        # print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0

        with torch.no_grad():
            all_samples = list()
            for _ in range((args.n_samples + batch_size - 1) // batch_size):
                #
                # check exists
                save_path = os.path.join(sample_path, f"{sample_count:05}.png")
                if os.path.exists(save_path):
                    continue

                else:
                    # Generate images
                    samples = model(
                        prompt,
                        height=None,
                        width=None,
                        num_inference_steps=50,
                        guidance_scale=9.0,
                        num_images_per_prompt=min(batch_size, args.n_samples - sample_count),
                        negative_prompt=None,
                    ).images
                    for sample in samples:
                        sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                        sample_count += 1
                    if not args.skip_grid:
                        all_samples.append(torch.stack([ToTensor()(sample) for sample in samples], 0))

            if not args.skip_grid and all_samples:
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, "n b c h w -> (n b) c h w")
                grid = make_grid(grid, nrow=n_rows, normalize=True, value_range=(-1, 1))

                # to image
                grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                grid.save(os.path.join(outpath, f"grid.png"))
                del grid
        del all_samples

    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser()
    # GenEval
    parser.add_argument("--dataset", default="GenEval", type=str)
    parser.add_argument("--model_path", default=None, type=str, help="Path to the model file (optional)")
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default="outputs")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="number of samples",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="how many samples can be produced simultaneously",
    )
    parser.add_argument(
        "--diffusers",
        action="store_true",
        help="if use diffusers pipeline",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="skip saving grid",
    )

    parser.add_argument("--work_dir", default=None, type=str)
    parser.add_argument("--sample_nums", default=553, type=int)
    parser.add_argument("--add_label", default="", type=str)
    parser.add_argument("--exist_time_prefix", default="", type=str)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=553)
    parser.add_argument(
        "--if_save_dirname",
        action="store_true",
        help="if save img save dir name at wor_dir/metrics/tmp_time.time().txt for metric testing",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_env(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_root_logger()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    n_rows = batch_size = args.n_samples
    assert args.batch_size == 1, ValueError(f"{batch_size} > 1 is not available in GenEval")

    from diffusers import DiffusionPipeline, StableDiffusionPipeline

    model = DiffusionPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    )
    model.enable_xformers_memory_efficient_attention()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.enable_attention_slicing()

    # dataset
    metadatas = datasets.load_dataset(
        "scripts/inference_geneval.py", trust_remote_code=True, split=f"train[{args.start_index}:{args.end_index}]"
    )
    logger.info(f"Eval {len(metadatas)} samples")

    # save path
    if args.work_dir is None:
        work_dir = (
            f"/{os.path.join(*args.model_path.split('/')[:-1])}"
            if args.model_path.startswith("/")
            else os.path.join(*args.model_path.split("/")[:-1])
        )
    else:
        work_dir = args.work_dir
    args.work_dir = work_dir
    img_save_dir = os.path.join(str(work_dir), "vis")

    os.umask(0o000)
    os.makedirs(img_save_dir, exist_ok=True)

    save_root = (
        os.path.join(
            img_save_dir,
            f"{args.dataset}_{model.config['_class_name']}_bs{batch_size}_seed{args.seed}_imgnums{args.sample_nums}",
        )
        + args.add_label
    )
    print(f"images save at: {img_save_dir}")
    os.makedirs(save_root, exist_ok=True)

    if args.if_save_dirname and args.gpu_id == 0:
        # save at work_dir/metrics/tmp_xxx.txt for metrics testing
        os.makedirs(f"{work_dir}/metrics", exist_ok=True)
        with open(f"{work_dir}/metrics/tmp_geneval_{time.time()}.txt", "w") as f:
            print(f"save tmp file at {work_dir}/metrics/tmp_geneval_{time.time()}.txt")
            f.write(os.path.basename(save_root))

    visualize()
