# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained Sana models
"""
import argparse
import os

import torch
from torchvision.datasets.utils import download_url

pretrained_models = {}


def find_model(model_name):
    """
    Finds a pre-trained G.pt model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained G.pt checkpoints
        return download_model(model_name)
    else:  # Load a custom Sana checkpoint:
        assert os.path.isfile(model_name), f"Could not find Sana checkpoint at {model_name}"
        return torch.load(model_name, map_location=lambda storage, loc: storage)


def download_model(model_name):
    """
    Downloads a pre-trained Sana model from the web.
    """
    assert model_name in pretrained_models
    local_path = f"output/pretrained_models/{model_name}"
    if not os.path.isfile(local_path):
        hf_endpoint = os.environ.get("HF_ENDPOINT")
        if hf_endpoint is None:
            hf_endpoint = "https://huggingface.co"
        os.makedirs("output/pretrained_models", exist_ok=True)
        web_path = f"{hf_endpoint}/xxx/resolve/main/{model_name}"
        download_url(web_path, "output/pretrained_models/")
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", nargs="+", type=str, default=pretrained_models)
    args = parser.parse_args()
    model_names = args.model_names
    model_names = set(model_names)

    # Download Sana checkpoints
    for model in model_names:
        download_model(model)
    print("Done.")
