#!/bin/bash
set -e

echo "Setting up test data..."
bash tests/bash/setup_test_data.sh

echo "Testing FSDP training"
bash train_scripts/train.sh configs/sana1-5_config/1024ms/Sana_1600M_1024px_AdamW_fsdp.yaml --np=2 --data.data_dir="[data/toy_data]" --data.load_vae_feat=true --train.num_epochs=1 --train.log_interval=1
