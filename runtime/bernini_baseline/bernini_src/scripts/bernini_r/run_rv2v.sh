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
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
ULYSSES=${ULYSSES:-8}
BERNINI_R_CONFIG=${BERNINI_R_CONFIG:-./pretrained_models/Bernini-R-Diffusers}

# Case 1 is reference-image-guided video editing — replacing a garment in the source video with one from a reference image:

torchrun --nproc-per-node "$NPROC_PER_NODE" infer_multi_gpu.py \
    --config "$BERNINI_R_CONFIG" --ulysses "$ULYSSES" \
    --case assets/testcases/rv2v/rv2v_case1.json --guidance_mode rv2v

# Case 2 is a video-insertion example — inserting content into the source video. It is run at 720p / 24fps to show the insertion result more clearly:

torchrun --nproc-per-node "$NPROC_PER_NODE" infer_multi_gpu.py \
    --config "$BERNINI_R_CONFIG" --ulysses "$ULYSSES" \
    --case assets/testcases/rv2v/rv2v_case2.json \
    --num_frames 121 --fps 24 --max_image_size 1280 --guidance_mode rv2v