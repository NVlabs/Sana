#!/usr/bin/env bash
# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail
export NCCL_NET_PLUGIN=none
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export FSDP=1
CASE_PATH=${CASE_PATH:-assets/testcases/v2v/v2v_case3.json}

for CASE_PATH in assets/testcases/v2v/v2v_case1.json assets/testcases/v2v/v2v_case2.json assets/testcases/v2v/v2v_case3.json;
do

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
ULYSSES=${ULYSSES:-8}
NEG_PROMPT=${NEG_PROMPT:-"vivid tones, overexposed, static, blurry details, subtitles, style, artwork, painting, image, motionless, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, still frame, cluttered background, three legs, too many people in the background, walking backwards"}
BERNINI_CONFIG=${BERNINI_CONFIG:-ByteDance/Bernini-Diffusers}
torchrun --standalone --nproc-per-node "$NPROC_PER_NODE" infer_multi_gpu.py \
    --config "$BERNINI_CONFIG" \
    --use_unipc \
    --use_src_tgt_id \
    --ulysses "$ULYSSES" \
    --num_frames 81 \
    --max_image_size 848 \
    --num_inference_steps 40 \
    --flow_shift 5.0 \
    --height 0 \
    --width 0 \
    --seed 42 \
    --fps 16 \
    --omega_txt 4 \
    --omega_tgt 0.5 \
    --omega_img 1.25 \
    --omega_vid 1.25 \
    --omega_scale 0.75 \
    --vit_denoising_step 5 \
    --vit_txt_cfg 1.2 \
    --vit_img_cfg 1.0 \
    --guidance_mode vae_txt_vit_wapg \
    --system_prompt "You are a helpful assistant specialized in video editing." \
    --neg_prompt "$NEG_PROMPT" \
    --case "$CASE_PATH"
done
