#!/bin/bash
set -e

echo "Setting up test data..."
bash tests/bash/setup_test_data.sh

echo "Testing SANA-Sprint(sCM + LADD) training"
bash train_scripts/train_scm_ladd.sh configs/sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml --np=4 --data.data_dir="[data/toy_data]" --data.load_vae_feat=true --train.num_epochs=1 --train.log_interval=1
