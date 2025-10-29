#!/bin/bash
set -e

echo "Setting up test data..."
# bash tests/bash/setup_test_data.sh
hf download Efficient-Large-Model/video_toy_data --repo-type dataset --local-dir ./data/video_toy_data

mkdir -p output/pretrained_models
hf download Wan-AI/Wan2.1-T2V-1.3B --repo-type model --local-dir ./output/pretrained_models/Wan2.1-T2V-1.3B

echo "Testing FSDP video training"
bash train_video_scripts/train_video_ivjoint.sh configs/sana_video_config/Sana_2000M_256px_AdamW_fsdp.yaml --np=2 --train.num_epochs=1 --train.log_interval=1 --train.train_batch_size=1 --train.joint_training_interval=0
