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

torchrun --nproc-per-node "$NPROC_PER_NODE" gradio_demo.py --ulysses "$ULYSSES" \
        --config "$BERNINI_R_CONFIG" \
        --port 7860 --share
