#!/bin/bash
set -e

echo "Setting up test data..."
bash tests/bash/setup_test_data.sh

echo "Testing offline VAE feature"
bash train_scripts/train.sh configs/sana_config/512ms/ci_Sana_600M_img512.yaml --np=4 --data.load_vae_feat=true
