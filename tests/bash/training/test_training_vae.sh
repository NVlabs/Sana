#!/bin/bash
set -e

echo "Setting up test data..."
# bash tests/bash/setup_test_data.sh
mkdir -p data/data_public
hf download Efficient-Large-Model/sana_data_public --repo-type dataset --local-dir ./data/data_public

echo "Testing offline VAE feature"
bash train_scripts/train.sh configs/sana_config/512ms/ci_Sana_600M_img512.yaml --np=4 --data.load_vae_feat=true

echo "Testing online VAE feature"
bash train_scripts/train.sh configs/sana_config/512ms/ci_Sana_600M_img512.yaml --np=4 --data.data_dir="[asset/example_data]" --data.type=SanaImgDataset --model.multi_scale=false
