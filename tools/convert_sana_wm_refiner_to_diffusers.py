#!/usr/bin/env python3
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

"""Convert the Sana-WM LTX-2 refiner checkpoint to diffusers components.

The Sana-WM PR currently vendors large parts of LTX-2 only to load the
refiner. Diffusers already ships official LTX-2 model code and key remapping
for the transformer, so this utility materializes the checkpoint as
diffusers-style component folders:

  output_dir/
    transformer/config.json
    transformer/diffusion_pytorch_model.safetensors
    connectors/config.json
    connectors/diffusion_pytorch_model.safetensors

The video VAE and Gemma text encoder can stay in their existing diffusers /
Transformers folders; this script focuses on the refiner checkpoint file.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Callable

WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
DEFAULT_CONFIG_REPO = "Lightricks/LTX-2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Sana-WM's LTX-2 refiner safetensors to diffusers component folders."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Local .safetensors path or hf://repo_id/path/to/file.safetensors.",
    )
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--config_repo",
        default=DEFAULT_CONFIG_REPO,
        help="Diffusers repo/local dir to read transformer and connector config.json files from.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only inspect key counts; do not load tensors or write output files.",
    )
    return parser.parse_args()


def resolve_checkpoint(path_or_uri: str) -> Path:
    if not path_or_uri.startswith("hf://"):
        return Path(path_or_uri).expanduser().resolve()

    parts = path_or_uri[len("hf://") :].split("/")
    if len(parts) < 3:
        raise ValueError("hf:// paths must look like hf://org/repo/path/to/file.safetensors")

    from huggingface_hub import hf_hub_download

    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    return Path(hf_hub_download(repo_id=repo_id, filename=filename)).resolve()


def safetensor_keys(path: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return list(handle.keys())


def load_selected_tensors(path: Path, predicate: Callable[[str], bool], component: str) -> dict[str, object]:
    from safetensors import safe_open

    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = [key for key in handle.keys() if predicate(key)]
        total = len(keys)
        for index, key in enumerate(keys, start=1):
            if predicate(key):
                tensors[key] = handle.get_tensor(key)
            if index == total or index % 250 == 0:
                print(f"loaded {component} tensors: {index}/{total}", flush=True)
    return tensors


def is_transformer_key(key: str) -> bool:
    if not key.startswith("model.diffusion_model."):
        return False
    return "embeddings_connector" not in key


def is_connector_key(key: str) -> bool:
    return key.startswith(
        (
            "text_embedding_projection.aggregate_embed.",
            "model.diffusion_model.video_embeddings_connector.",
            "model.diffusion_model.audio_embeddings_connector.",
        )
    )


def convert_connectors_to_diffusers(checkpoint: dict[str, object]) -> dict[str, object]:
    """Map original LTX-2 connector keys to diffusers LTX2TextConnectors keys."""
    rename_pairs = (
        ("text_embedding_projection.aggregate_embed.", "text_proj_in."),
        ("model.diffusion_model.video_embeddings_connector.", "video_connector."),
        ("model.diffusion_model.audio_embeddings_connector.", "audio_connector."),
        ("transformer_1d_blocks.", "transformer_blocks."),
        ("q_norm.", "norm_q."),
        ("k_norm.", "norm_k."),
    )

    converted = {}
    unsupported = [
        key
        for key in checkpoint
        if key.startswith(
            (
                "text_embedding_projection.video_aggregate_embed.",
                "text_embedding_projection.audio_aggregate_embed.",
            )
        )
    ]
    if unsupported:
        raise NotImplementedError(
            "Found LTX-2 V2 dual aggregate connector keys. "
            "This converter currently handles the 19B/V1 aggregate_embed layout used by LTX-2."
        )

    for key, value in checkpoint.items():
        new_key = key
        for old, new in rename_pairs:
            new_key = new_key.replace(old, new)
        converted[new_key] = value
    return converted


def write_component_config(model_cls, config_repo: str, subfolder: str, output_dir: Path) -> None:
    config = model_cls.load_config(config_repo, subfolder=subfolder)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_component(name: str, config_repo: str, state_dict: dict[str, object], output_dir: Path) -> None:
    component_dir = output_dir / name
    if name == "transformer":
        from diffusers import LTX2VideoTransformer3DModel

        write_component_config(LTX2VideoTransformer3DModel, config_repo, "transformer", component_dir)
    elif name == "connectors":
        from diffusers.pipelines.ltx2 import LTX2TextConnectors

        write_component_config(LTX2TextConnectors, config_repo, "connectors", component_dir)
    else:
        raise ValueError(f"Unknown component: {name}")

    from safetensors.torch import save_file

    save_file(state_dict, component_dir / WEIGHTS_NAME, metadata={"format": "pt"})


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    keys = safetensor_keys(checkpoint)

    transformer_keys = [key for key in keys if is_transformer_key(key)]
    connector_keys = [key for key in keys if is_connector_key(key)]
    print(f"checkpoint: {checkpoint}")
    print(f"transformer keys: {len(transformer_keys)}")
    print(f"connector keys: {len(connector_keys)}")

    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from diffusers.loaders.single_file_utils import convert_ltx2_transformer_to_diffusers

    transformer_state = load_selected_tensors(checkpoint, is_transformer_key, "transformer")
    print("converting transformer keys", flush=True)
    transformer_state = convert_ltx2_transformer_to_diffusers(transformer_state)
    print("writing transformer component", flush=True)
    write_component("transformer", args.config_repo, transformer_state, args.output_dir)
    print(f"wrote transformer to {args.output_dir / 'transformer'}")
    del transformer_state
    gc.collect()

    connector_state = load_selected_tensors(checkpoint, is_connector_key, "connectors")
    print("converting connector keys", flush=True)
    connector_state = convert_connectors_to_diffusers(connector_state)
    print("writing connectors component", flush=True)
    write_component("connectors", args.config_repo, connector_state, args.output_dir)
    print(f"wrote connectors to {args.output_dir / 'connectors'}")


if __name__ == "__main__":
    main()
