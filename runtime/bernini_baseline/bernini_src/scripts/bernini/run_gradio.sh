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
BERNINI_CONFIG=${BERNINI_CONFIG:-ByteDance/Bernini-Diffusers}

torchrun --nproc-per-node 8 gradio_demo.py --ulysses 8 \
        --config "$BERNINI_CONFIG" \
        --port 9500 --share
