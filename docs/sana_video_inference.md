## Inference CLI

### Inference SANA-Video

```bash
python app/sana_video_pipeline.py \
        --config configs/sana_video_config/480ms/Sana_1600M_480px_adamW_fsdp.yaml \
        --model_path "hf://Efficient-Large-Model/SanaVideo_willquant/checkpoints/model.pth" \
        --save_path sana_video.mp4 \
        --prompt "In a whimsical forest setting, a small deer with antlers stands amidst oversized mushrooms and scattered carrots. The scene is vibrant with lush green moss and rocks, creating a magical atmosphere. The deer appears curious, moving slowly across the ground, surrounded by the towering fungi and colorful vegetables. The sky above is clear and bright, adding to the enchanting ambiance. A low-angle shot captures the deer's gentle exploration of this fantastical landscape."
```

### Inference SANA-Video Chunked Version

```bash
python app/sana_video_pipeline.py \
        --config configs/sana_video_config/480ms/Sana_1600M_480px_adamW_fsdp_chunk.yaml \
        --model_path "hf://Efficient-Large-Model/SanaVideo_chunk/checkpoints/model.pth" \
        --save_path sana_video_chunk_i2v.mp4 \
        --interval_k 0.2 \
        --image_path output/tmp_videos/wan_goodcase_i2v_eval/00000000_video_001.jpg \
        --prompt "In a whimsical forest setting, a small deer with antlers stands amidst oversized mushrooms and scattered carrots. The scene is vibrant with lush green moss and rocks, creating a magical atmosphere. The deer appears curious, moving slowly across the ground, surrounded by the towering fungi and colorful vegetables. The sky above is clear and bright, adding to the enchanting ambiance. A low-angle shot captures the deer's gentle exploration of this fantastical landscape."
```
