#!/bin/bash
set -e

python scripts/inference.py \
    --config=configs/sana_config/1024ms/Sana_600M_img1024.yaml \
    --model_path=hf://Efficient-Large-Model/Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth

python scripts/inference.py \
    --config=configs/sana_config/1024ms/Sana_1600M_img1024.yaml \
    --model_path=hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth

mkdir -p tools/controlnet/annotator/ckpts
hf download lllyasviel/Annotators ControlNetHED.pth --local-dir tools/controlnet/annotator/ckpts

python tools/controlnet/inference_controlnet.py \
    --config=configs/sana_controlnet_config/Sana_600M_img1024_controlnet.yaml \
    --model_path=hf://Efficient-Large-Model/Sana_600M_1024px_ControlNet_HED/checkpoints/Sana_600M_1024px_ControlNet_HED.pth \
    --json_file=asset/controlnet/samples_controlnet.json

python scripts/inference_sana_sprint.py \
    --config=configs/sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml \
    --model_path=hf://Lawrence-cj/Sana_Sprint_1600M_1024px/Sana_Sprint_1600M_1024px_36K.pth \
    --txt_file=asset/samples/samples_mini.txt

python inference_video_scripts/inference_sana_video.py \
    --config=configs/sana_video_config/Sana_2000M_256px_AdamW_fsdp.yaml \
    --model_path=hf://Efficient-Large-Model/SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth \
    --debug=true

python inference_video_scripts/inference_sana_video.py \
    --config=configs/sana_video_config/Sana_2000M_480px_adamW_fsdp_longsana.yaml \
    --model_path=hf://Efficient-Large-Model/SANA-Video_2B_480p_LongLive/checkpoints/SANA_Video_2B_480p_LongLive.pth \
    --cfg_scale=1.0 --debug=true

python inference_video_scripts/wm/inference_sana_wm.py \
    --image=asset/sana_wm/demo_0.png \
    --prompt=asset/sana_wm/demo_0.txt \
    --action=w-641 \
    --output_dir=results/sana_wm_ci \
    --name=demo_0 \
    --num_frames=641 \
    --step=4

python inference_video_scripts/wm/inference_sana_wm.py \
    --config=configs/sana_wm/sana_wm_chunk_causal_1600m_720p.yaml \
    --model_path=hf://Efficient-Large-Model/SANA-WM_chunk_causal/dit/sana_wm_chunk_causal_1600m_720p.safetensors \
    --image=asset/sana_wm/demo_0.png \
    --prompt=asset/sana_wm/demo_0.txt \
    --action=w-25 \
    --intrinsics=asset/sana_wm/demo_0_intrinsics.npy \
    --output_dir=results/sana_wm_chunk_causal_ci \
    --name=demo_0_chunk_causal \
    --num_frames=25 \
    --step=4 \
    --no_refiner

python inference_video_scripts/wm/inference_sana_wm_streaming.py \
    --image=asset/sana_wm/demo_0.png \
    --prompt=asset/sana_wm/demo_0.txt \
    --action=w-25 \
    --intrinsics=asset/sana_wm/demo_0_intrinsics.npy \
    --output_dir=results/sana_wm_streaming_ci \
    --name=demo_0_streaming \
    --num_frames=25 \
    --no_compile \
    --streaming_preset=ultrafast

python inference_video_scripts/v2v/inference_sana_streaming.py \
    --mode=bidirectional_short \
    --config=configs/sana_streaming/sana_streaming_bidirectional_2b_720p.yaml \
    --model_path=hf://Efficient-Large-Model/SANA-Streaming_bidirectional/dit/sana_bidirectional_short.pth \
    --prompt="Remove the thick, textured gold hoop earrings from the woman's ears. Carefully reconstruct the exposed earlobes to match her natural skin tone and texture. Ensure the lighting and soft shadows on the newly bare ears blend seamlessly with the rest of her face, leaving no trace or reflection of the metallic jewelry behind." \
    --video_path=hf://Efficient-Large-Model/SANA-Streaming/source/00_local_editing_source.mp4 \
    --output_dir=results/sana_streaming_bidirectional_ci

python inference_video_scripts/v2v/inference_sana_streaming.py \
    --mode=long_streaming \
    --config=configs/sana_streaming/sana_streaming_2b_720p.yaml \
    --model_path=hf://Efficient-Large-Model/SANA-Streaming/dit/sana_streaming_ar.pth \
    --prompt="Transform the entire scene into a breathtaking Sci-Fi Art digital painting." \
    --video_path=hf://Efficient-Large-Model/SANA-Streaming/source/09_style_transfer_source.mp4 \
    --output_dir=results/sana_streaming_long_ci
