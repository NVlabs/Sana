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
from types import SimpleNamespace

import torch

from app import sana_pipeline
from app.sana_pipeline import SanaPipeline

SEQ_LEN = 4
EMBED_VALUE = {"": 0.0, "a cat": 1.0, "blurry": 7.0}


class _Tokenizer:
    def __call__(self, text, **_kwargs):
        texts = text if isinstance(text, list) else [text]
        input_ids = torch.stack([torch.full((SEQ_LEN,), EMBED_VALUE[t]) for t in texts])
        tokens = SimpleNamespace(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        tokens.to = lambda *_a, **_k: tokens
        return tokens


def _text_encoder(input_ids, _attention_mask):
    return (input_ids[..., None].expand(-1, -1, 3).clone(),)


def _make_pipeline(unconditional_values):
    def model(latents, _timestep, prompt_embeds, **_kwargs):
        unconditional_values.append(prompt_embeds[0, 0, 0, 0].item())
        return torch.zeros_like(latents)

    pipe = SanaPipeline.__new__(SanaPipeline)
    torch.nn.Module.__init__(pipe)
    pipe.config = pipe.args = SimpleNamespace(
        vae=SimpleNamespace(vae_downsample_rate=8, vae_latent_dim=4, vae_type="stub"),
        model=SimpleNamespace(attn_type="linear"),
        text_encoder=SimpleNamespace(chi_prompt=None, model_max_length=SEQ_LEN),
    )
    pipe.device = torch.device("cpu")
    pipe.weight_dtype = pipe.vae_dtype = torch.float32
    pipe.tokenizer, pipe.text_encoder, pipe.model, pipe.vae = _Tokenizer(), _text_encoder, model, None
    pipe.base_ratios = {"1.0": [64, 64]}
    pipe.vis_sampler = "flow_euler"
    pipe.flow_shift = 3.0
    pipe.guidance_type = "classifier-free"
    pipe.image_size = 64
    pipe.max_sequence_length = SEQ_LEN
    pipe.null_caption_embs = _text_encoder(_Tokenizer()("").input_ids, None)[0]
    return pipe


def test_negative_prompt_does_not_leak_into_later_calls(monkeypatch):
    monkeypatch.setattr(sana_pipeline, "vae_decode", lambda _vae_type, _vae, sample: sample)
    monkeypatch.setattr(sana_pipeline, "prepare_prompt_ar", lambda prompt, *_a, **_k: (prompt, None))

    unconditional_values = []
    pipe = _make_pipeline(unconditional_values)
    for negative_prompt in ["blurry", ""]:
        pipe(
            prompt="a cat",
            height=64,
            width=64,
            negative_prompt=negative_prompt,
            num_inference_steps=1,
            use_resolution_binning=False,
        )

    assert unconditional_values == [EMBED_VALUE["blurry"], EMBED_VALUE[""]]
