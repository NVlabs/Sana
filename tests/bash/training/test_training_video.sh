#!/bin/bash
set -e

echo "Setting up test data..."
bash tests/bash/setup_test_data.sh

echo "Testing FSDP video training"
bash train_video_scripts/train.sh configs/sana_video_config/Sana_2000M_256px_AdamW_fsdp.yaml --np=2 --train.num_epochs=1 --train.log_interval=1 --train.train_batch_size=1
