#!/bin/bash
set -e

echo "Setting up test data..."

mkdir -p data/data_public
hf download Efficient-Large-Model/toy_data --repo-type dataset --local-dir ./data/toy_data

echo "Testing SANA-Sprint(sCM + LADD) training"
bash train_scripts/train_scm_ladd.sh configs/sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml --np=4 --data.data_dir="[data/toy_data]" --data.load_vae_feat=true --train.num_epochs=1 --train.log_interval=1

echo "Testing FSDP training"
bash train_scripts/train.sh configs/sana1-5_config/1024ms/Sana_1600M_1024px_AdamW_fsdp.yaml --np=2 --data.data_dir="[data/toy_data]" --data.load_vae_feat=true --train.num_epochs=1 --train.log_interval=1
