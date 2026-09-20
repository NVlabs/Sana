#!/bin/bash
set -e

pip install --upgrade "huggingface-hub<1.0"

# download test data
mkdir -p data/data_public
hf download Efficient-Large-Model/sana_data_public --repo-type dataset --local-dir ./data/data_public
hf download Efficient-Large-Model/toy_data --repo-type dataset --local-dir ./data/toy_data
hf download Efficient-Large-Model/video_toy_data --repo-type dataset --local-dir ./data/video_toy_data

mkdir -p output/pretrained_models
hf download Wan-AI/Wan2.1-T2V-1.3B --repo-type model --local-dir ./output/pretrained_models/Wan2.1-T2V-1.3B
