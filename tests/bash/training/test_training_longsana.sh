#!/bin/bash
set -e

echo "Setting up test data..."

mkdir -p data/longsana
hf download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir data/longsana

echo "Testing LongSANA training"

torchrun --nproc_per_node=2 train_video_scripts/train_longsana.py \
    --config_path configs/sana_video_config/longsana/480ms/self_forcing.yaml \
    --logdir output/debug_480p_self_forcing --disable-wandb --max_iters=10

torchrun --nproc_per_node=2 train_video_scripts/train_longsana.py \
    --config_path configs/sana_video_config/longsana/480ms/longsana.yaml \
    --logdir output/debug_480p_longsana --disable-wandb --max_iters=10
