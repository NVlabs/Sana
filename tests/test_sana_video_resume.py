# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from inference_video_scripts import inference_sana_video as inference


@pytest.mark.parametrize("index", [0, 9, 99, 100])
def test_video_inference_resumes_its_own_outputs(monkeypatch, tmp_path, index):
    calls = {"text": 0, "model": 0, "writer": 0}
    config = SimpleNamespace(
        task="t2v",
        vae=SimpleNamespace(vae_downsample_rate=8, vae_latent_dim=3, vae_type="stub"),
        text_encoder=SimpleNamespace(chi_prompt=None, model_max_length=4, text_encoder_name="Qwen"),
    )
    args = SimpleNamespace(
        seed=7,
        gpu_id=0,
        start_index=index,
        end_index=index + 1,
        image_size=16,
        sampling_algo="flow_euler",
        prompt_split_token="<split>",
        stg_applied_layers=[],
        stg_scale=0.0,
        fps=24,
    )

    class TextHandler:
        def get_prompt_embeds(self, prompts, max_length):
            calls["text"] += 1
            return torch.ones(len(prompts), max_length, 3), torch.ones(len(prompts), max_length)

    def model(latents):
        calls["model"] += 1
        return torch.zeros_like(latents)

    class Sampler:
        def __init__(self, model, **_kwargs):
            self.model = model

        def sample(self, latents, steps):
            return self.model(latents)

    def get_writer(path, **_kwargs):
        calls["writer"] += 1
        frames = []
        return SimpleNamespace(
            append_data=lambda frame: frames.append(frame.tobytes()),
            close=lambda: Path(path).write_bytes(b"test-video" + b"".join(frames)),
        )

    chunk = {"prompt": ["a cat"], "global_idx": torch.tensor([index]), "key": ["first-key"]}
    values = {
        "rank": 0,
        "device": torch.device("cpu"),
        "save_root": str(tmp_path),
        "prompts_dataloader": [chunk],
        "base_ratios": {},
        "motion_prompt": "",
        "latent_size_t": 2,
        "weight_dtype": torch.float32,
        "vae_dtype": torch.float32,
        "vae": None,
        "flow_shift": 3.0,
        "apg": None,
        "text_handler": TextHandler(),
        "negative_caption_embs": torch.zeros(1, 4, 3),
        "negative_caption_embs_mask": torch.ones(1, 4),
        "FlowEuler": Sampler,
        "vae_decode": lambda _vae_type, _vae, sample: sample,
        "prepare_prompt_ar": lambda prompt, *_a, **_k: (prompt, None, torch.tensor([[16, 16]]), None, None),
    }
    for name, value in values.items():
        monkeypatch.setattr(inference, name, value, raising=False)
    monkeypatch.setattr(inference.imageio, "get_writer", get_writer)
    monkeypatch.setattr(inference.os, "umask", lambda mask: 0)

    def run():
        inference.visualize(config, args, model, items=None, bs=1, sample_steps=1, cfg_scale=1.0)

    run()
    output = tmp_path / f"{index:03d}. first-key.mp4"
    assert output.is_file()
    original = output.read_bytes(), output.stat().st_mtime_ns
    assert calls == {"text": 1, "model": 1, "writer": 1}

    run()
    assert calls == {"text": 1, "model": 1, "writer": 1}
    assert (output.read_bytes(), output.stat().st_mtime_ns) == original

    # A different prompt key at the same global index still needs its own output.
    chunk["key"] = ["second-key"]
    run()
    assert calls == {"text": 2, "model": 2, "writer": 2}
    assert (tmp_path / f"{index:03d}. second-key.mp4").is_file()
    assert (output.read_bytes(), output.stat().st_mtime_ns) == original
