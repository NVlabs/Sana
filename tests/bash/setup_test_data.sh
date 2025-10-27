#!/bin/bash
set -e

# download test data
mkdir -p data/data_public
huggingface-cli download Efficient-Large-Model/sana_data_public --repo-type dataset --local-dir ./data/data_public --local-dir-use-symlinks False
huggingface-cli download Efficient-Large-Model/toy_data --repo-type dataset --local-dir ./data/toy_data --local-dir-use-symlinks False
huggingface-cli download Efficient-Large-Model/video_toy_data --repo-type dataset --local-dir ./data/video_toy_data --local-dir-use-symlinks False

mkdir -p output/pretrained_models
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --repo-type model --local-dir ./output/pretrained_models/Wan2.1-T2V-1.3B --local-dir-use-symlinks False
