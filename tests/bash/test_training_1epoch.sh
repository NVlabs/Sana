#/bin/bash
set -e

mkdir -p data/data_public
huggingface-cli download Efficient-Large-Model/sana_data_public --repo-type dataset --local-dir ./data/data_public --local-dir-use-symlinks False
huggingface-cli download Efficient-Large-Model/toy_data --repo-type dataset --local-dir ./data/toy_data --local-dir-use-symlinks False

bash train_scripts/train.sh configs/sana_config/512ms/ci_Sana_600M_img512.yaml --data.load_vae_feat=true

bash train_scripts/train.sh configs/sana_config/512ms/ci_Sana_600M_img512.yaml --data.data_dir="[asset/example_data]" --data.type=SanaImgDataset --model.multi_scale=false

bash train_scripts/train.sh configs/sana1-5_config/1024ms/Sana_1600M_1024px_AdamW_fsdp.yaml --data.data_dir="[data/toy_data]" --data.load_vae_feat=true --train.num_epochs=1 --train.log_interval=1
