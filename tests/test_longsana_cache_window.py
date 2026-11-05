# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from diffusion.longsana.pipeline.sana_inference_interactive_pipeline import SanaInferenceInteractivePipeline
from diffusion.longsana.pipeline.sana_inference_interactive_pipeline_long_chunk import (
    SanaInferenceInteractivePipelineLongChunk,
)
from diffusion.longsana.pipeline.sana_inference_pipeline import SanaInferencePipeline
from diffusion.longsana.pipeline.sana_training_pipeline import SanaTrainingPipeline
from diffusion.scheduler.longlive_flow_euler_sampler import LongLiveFlowEuler


@pytest.mark.parametrize(
    "cache_type",
    [
        LongLiveFlowEuler,
        SanaInferencePipeline,
        SanaInferenceInteractivePipeline,
        SanaInferenceInteractivePipelineLongChunk,
        SanaTrainingPipeline,
    ],
    ids=lambda cls: cls.__name__,
)
@pytest.mark.parametrize(
    "num_chunks, chunk_idx, window",
    [
        (3, 1, 4),  # Negative indices revisit the first chunk without clamping.
        (3, 1, 5),  # A still larger window indexes outside the cache list.
        (4, 2, 8),
        (4, 3, 1),
        (4, 3, 2),
        (4, 3, 4),
        (4, 3, 0),
        (4, 3, -1),
        (1, 0, 4),
    ],
)
def test_cache_window_uses_available_history(cache_type, num_chunks, chunk_idx, window):
    owner = cache_type.__new__(cache_type)
    owner.num_model_blocks = 2
    owner.num_cached_blocks = window
    cache = [[[None, None, None] for _ in range(owner.num_model_blocks)] for _ in range(num_chunks)]
    for i in range(chunk_idx):
        for block in range(owner.num_model_blocks):
            value = 10 * block + i + 1
            cache[i][block] = [
                torch.full((1, 2, 3, 3), float(value)),
                torch.full((1, 2, 1, 3), float(value + 1)),
                torch.tensor([float(value + 2)]),
            ]
    history = [[tuple(t.clone() for t in block) for block in chunk] for chunk in cache[:chunk_idx]]

    actual = owner._accumulate_kv_cache(cache, chunk_idx)

    assert actual is cache[chunk_idx]
    if chunk_idx == 0:
        assert actual == [[None, None, None], [None, None, None]]
        return
    selected = history[-window:] if window > 0 else history
    for block in range(owner.num_model_blocks):
        for slot in (0, 1):
            expected = torch.stack([chunk[block][slot] for chunk in selected]).sum(dim=0)
            torch.testing.assert_close(actual[block][slot], expected, rtol=0, atol=0)
        assert actual[block][2] is cache[chunk_idx - 1][block][2]
    for before, after in zip(history, cache[:chunk_idx]):
        for before_block, after_block in zip(before, after):
            for expected, value in zip(before_block, after_block):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
