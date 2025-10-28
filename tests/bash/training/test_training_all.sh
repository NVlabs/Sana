#/bin/bash
set -e

mkdir -p data/data_public
hf download Efficient-Large-Model/sana_data_public --repo-type dataset --local-dir ./data/data_public
hf download Efficient-Large-Model/toy_data --repo-type dataset --local-dir ./data/toy_data
hf download Efficient-Large-Model/video_toy_data --repo-type dataset --local-dir ./data/video_toy_data

mkdir -p output/pretrained_models
hf download Efficient-Large-Model/Wan2.1-T2V-1.3B --repo-type model --local-dir ./output/pretrained_models/Wan2.1-T2V-1.3B

# test offline vae feature
bash train_scripts/train.sh configs/sana_config/512ms/ci_Sana_600M_img512.yaml --data.load_vae_feat=true

# test online vae feature
bash train_scripts/train.sh configs/sana_config/512ms/ci_Sana_600M_img512.yaml --data.data_dir="[asset/example_data]" --data.type=SanaImgDataset --model.multi_scale=false

# test FSDP training
bash train_scripts/train.sh configs/sana1-5_config/1024ms/Sana_1600M_1024px_AdamW_fsdp.yaml --data.data_dir="[data/toy_data]" --data.load_vae_feat=true --train.num_epochs=1 --train.log_interval=1

# test SANA-Sprint(sCM + LADD) training
bash train_scripts/train_scm_ladd.sh configs/sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml  --data.data_dir="[data/toy_data]" --data.load_vae_feat=true --train.num_epochs=1 --train.log_interval=1

# test FSDP video training
bash train_video_scripts/train_video_ivjoint.sh configs/sana_video_config/Sana_2000M_256px_AdamW_fsdp.yaml --np=2 --train.num_epochs=1 --train.log_interval=1 --train.train_batch_size=1
