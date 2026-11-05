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
from types import SimpleNamespace

import pytest
import torch

from inference_video_scripts import inference_sana_video as inference


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
    ]
)
def device(request):
    return torch.device(request.param)


class _ConstantVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, latents, timesteps, prompt_embeds, **kwargs):
        self.calls.append((latents.shape, timesteps.clone(), kwargs["chunk_index"]))
        velocity = 1 + 2 * prompt_embeds[:, 0, 0, 0]
        return velocity.reshape(-1, 1, 1, 1, 1).expand_as(latents)


def _inference_setup(monkeypatch, tmp_path, interval_k, device):
    config = SimpleNamespace(
        task="t2v",
        model=SimpleNamespace(chunk_index=[0, 11]),
        vae=SimpleNamespace(vae_downsample_rate=8, vae_latent_dim=2, vae_type="stub"),
        text_encoder=SimpleNamespace(chi_prompt=None, model_max_length=4, text_encoder_name="Qwen"),
    )
    args = SimpleNamespace(
        seed=7,
        gpu_id=0,
        start_index=0,
        end_index=1,
        image_size=16,
        sampling_algo="chunk_flow_euler",
        prompt_split_token="<split>",
        stg_applied_layers=[],
        stg_scale=0.0,
        interval_k=interval_k,
        fps=24,
    )
    outputs = []

    def decode(_vae_type, _vae, latent):
        outputs.append(latent.clone())
        return latent

    class TextHandler:
        def get_prompt_embeds(self, prompts, max_length):
            return torch.ones(len(prompts), max_length, 3, device=device), torch.ones(
                len(prompts), max_length, device=device
            )

    values = {
        "rank": 0,
        "device": device,
        "save_root": str(tmp_path),
        "prompts_dataloader": [{"prompt": ["a cat"], "global_idx": [0], "key": ["test"]}],
        "base_ratios": {},
        "motion_prompt": "",
        "latent_size_t": 21,
        "weight_dtype": torch.float32,
        "vae_dtype": torch.float32,
        "vae": None,
        "flow_shift": 7.0,
        "apg": None,
        "text_handler": TextHandler(),
        "negative_caption_embs": torch.zeros(1, 4, 3, device=device),
        "negative_caption_embs_mask": torch.ones(1, 4, device=device),
        "vae_decode": decode,
        "prepare_prompt_ar": lambda prompt, *_a, **_k: (prompt, None, torch.tensor([[16, 16]]), None, None),
    }
    for name, value in values.items():
        monkeypatch.setattr(inference, name, value, raising=False)
    writer = SimpleNamespace(append_data=lambda frame: None, close=lambda: None)
    monkeypatch.setattr(inference.imageio, "get_writer", lambda *_a, **_k: writer)
    monkeypatch.setenv("DPM_TQDM", "True")
    return config, args, outputs


@pytest.mark.parametrize("cfg_scale", [1.0, 2.0])
@pytest.mark.parametrize("interval_k", [0.0, 1.0])
def test_chunk_config_runs_video_inference(monkeypatch, tmp_path, cfg_scale, interval_k, device):
    config, args, outputs = _inference_setup(monkeypatch, tmp_path, interval_k, device)
    model = _ConstantVelocity()
    initial = torch.randn(
        1, 2, 21, 2, 2, device=device, generator=torch.Generator(device=device).manual_seed(args.seed)
    )

    inference.visualize(config, args, model, items=None, bs=1, sample_steps=4, cfg_scale=cfg_scale)

    assert len(outputs) == 1
    # Each frame integrates from sigma=1 to 0. This also checks the per-token
    # scheduler's velocity sign and the unconditional/conditional CFG ordering.
    expected_velocity = 3.0 if cfg_scale == 1 else 5.0
    torch.testing.assert_close(outputs[0], initial - expected_velocity, rtol=1e-5, atol=2e-6)
    start_step = 2 if interval_k == 0 else 4
    assert len(model.calls) == start_step + 4
    for step, (shape, timesteps, chunk_index) in enumerate(model.calls):
        expected_frames = 11 if step < start_step else 21
        expected_batch = 1 if cfg_scale == 1 else 2
        assert shape == (expected_batch, 2, expected_frames, 2, 2)
        assert timesteps.shape == (expected_batch, 1, expected_frames)
        assert chunk_index == ([0] if step < start_step else [0, 11])
    assert torch.all(model.calls[start_step][1][..., 11:] == 1000)
    assert torch.all(model.calls[4][1][..., :11] == 0)


def test_chunk_sampler_requires_chunk_config(monkeypatch, tmp_path, device):
    config, args, _ = _inference_setup(monkeypatch, tmp_path, 0.0, device)
    config.model.chunk_index = None
    with pytest.raises(ValueError, match="requires model.chunk_size or model.chunk_index"):
        inference.visualize(config, args, _ConstantVelocity(), items=None, bs=1, sample_steps=4, cfg_scale=1.0)
