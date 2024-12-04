#!/bin/bash
set -e

python scripts/inference.py \
    --config=configs/sana_config/1024ms/Sana_600M_img1024.yaml \
    --model_path=hf://Efficient-Large-Model/Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth


python scripts/inference.py \
    --config=configs/sana_config/1024ms/Sana_1600M_img1024.yaml \
    --model_path=hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth
