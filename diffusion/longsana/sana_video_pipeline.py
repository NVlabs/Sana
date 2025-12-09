# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
import warnings
from dataclasses import dataclass, field
from typing import Optional

import imageio
import pyrallis
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")  # ignore warning

from diffusion.utils.config import SanaVideoConfig


@dataclass
class LongSANAVideoInference(SanaVideoConfig):
    config: Optional[str] = "configs/sana_video_config/longsana/480ms/self_forcing.yaml"  # config
    model_path: str = field(
        default="hf://Efficient-Large-Model/LongSANA_2B_480p_self_forcing/checkpoints/LongSANA_2B_480p_self_forcing.pt",
        metadata={"help": "Path to the model file (positional)"},
    )
    prompt: Optional[str] = None
    output: str = "./output"
    task: str = "t2v"
    bs: int = 1
    num_inference_steps: int = 50
    image_size: int = 480
    sampling_algo: str = "self_forcing_flow_euler"  # "flow_dpm-solver"
    skip_type: str = "time_uniform_flow"  # time_uniform_flow, linear_quadratic
    cfg_scale: float = 1.0
    guidance_type: str = "classifier-free"
    flow_shift: Optional[float] = None
    seed: int = 42
    step: int = -1
    custom_image_size: Optional[int] = None
    high_motion: bool = False
    motion_score: int = 10
    negative_prompt: str = (
        "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."
    )
    save_path: Optional[str] = None
    # chunkcausal setting
    unified_noise: bool = False
    interval_k: float = 1.0
    base_model_frames: int = 40
    num_frames: int = 81
