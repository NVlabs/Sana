#! /bin/bash

export MODEL_NAME="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
export INSTANCE_DIR="data/dreambooth/dog"
export OUTPUT_DIR="trained-sana-lora"

accelerate launch --num_processes 4 --main_process_port 29500 --gpu_ids 0,1,2,3 \
  train_scripts/train_dreambooth_lora_sana.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a pond, yarn art style" \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub
