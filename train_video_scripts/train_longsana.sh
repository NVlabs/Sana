torchrun --nproc_per_node=8 train_video_scripts/train_longsana.py \
    --config_path configs/sana_video_config/longsana/480ms/ode.yaml \
    --wandb_name debug_480p_ode --logdir output/debug_480p_ode

torchrun --nproc_per_node=8 train_video_scripts/train_longsana.py \
    --config_path configs/sana_video_config/longsana/480ms/self_forcing.yaml \
    --wandb_name debug_480p_self_forcing --logdir output/debug_480p_self_forcing


torchrun --nproc_per_node=8 train_video_scripts/train_longsana.py \
    --config_path configs/sana_video_config/longsana/480ms/longsana.yaml \
    --wandb_name debug_480p_longsana --logdir output/debug_480p_longsana

# inference longsana
accelerate launch --mixed_precision=bf16 \
    inference_video_scripts/inference_sana_video.py \
    --config=configs/sana_video_config/Sana_2000M_480px_adamW_fsdp_longsana.yaml \
    --model_path=hf://Efficient-Large-Model/SANA-Video_2B_480p_LongLive/checkpoints/SANA_Video_2B_480p_LongLive.pth \
    --work_dir=output/inference/longsana_480p \
    --txt_file=asset/samples/video_prompts_samples.txt \
    --dataset=samples --cfg_scale=1.0 --num_frames 161
