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

from pathlib import Path

import pyrallis
import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKLLTX2Video

from diffusion.model.builder import _build_ltx2_causal_encoder
from diffusion.model.nets.sana_video2 import (
    BlockAttentionResidual,
    SanaVideo2,
    SanaVideo2_5B,
    SanaVideo2_14B,
)
from diffusion.model.nets.sana_video2_blocks import SanaVideo2Block
from diffusion.utils.config import SanaVideoConfig


def test_causal_encoder_reuses_the_diffusers_encoder_weights():
    diffusers_vae = AutoencoderKLLTX2Video(
        latent_channels=4,
        block_out_channels=(8, 8, 8, 8),
        decoder_block_out_channels=(8, 8, 8),
        layers_per_block=(1, 1, 1, 1, 1),
        decoder_layers_per_block=(1, 1, 1, 1),
        patch_size=1,
    )
    encoder_keys = {
        key
        for key in diffusers_vae.state_dict()
        if key.startswith("encoder.") or key in {"latents_mean", "latents_std"}
    }

    vae = _build_ltx2_causal_encoder(diffusers_vae, device="cpu", dtype=torch.float32)

    assert vae.decoder is None
    assert set(vae.state_dict()) == encoder_keys
    assert not any(parameter.is_meta for parameter in vae.parameters())


def test_released_model_dimensions_and_parameter_counts():
    with init_empty_weights():
        model_5b = SanaVideo2_5B()
        model_14b = SanaVideo2_14B()

    assert (model_5b.depth, model_5b.hidden_size) == (32, 2560)
    assert (model_14b.depth, model_14b.hidden_size) == (40, 4096)
    assert model_5b.softmax_layer_indices == list(range(3, 32, 4))
    assert model_14b.softmax_layer_indices == list(range(3, 40, 4))
    assert sum(parameter.numel() for parameter in model_5b.parameters()) == 4_466_980_960
    assert sum(parameter.numel() for parameter in model_14b.parameters()) == 14_246_716_224


def test_attention_residual_buffer_matches_training_path():
    module = BlockAttentionResidual(hidden_size=8)
    completed = [torch.randn(2, 5, 8), torch.randn(2, 5, 8)]
    partial = torch.randn(2, 5, 8)
    expected = module.attend(module.attn_proj, completed, partial)

    values = torch.empty(4, 2, 5, 8)
    keys = torch.empty_like(values)
    for index, value in enumerate(completed):
        values[index] = value
        keys[index] = module.key_norm(value.unsqueeze(0)).squeeze(0)
    actual = module.attend_buffer(module.attn_proj, values, keys, len(completed), partial)
    torch.testing.assert_close(actual, expected)


def test_adaln_modulation_includes_the_learned_table():
    block = SanaVideo2Block(
        hidden_size=32,
        num_heads=4,
        attention="linear",
        linear_head_dim=8,
        softmax_head_dim=16,
    )
    with torch.no_grad():
        block.scale_shift_table.copy_(torch.arange(6 * 32).reshape(6, 32))

    modulation = block._modulation(torch.zeros(1, 6 * 32), batch=1)
    for index, value in enumerate(modulation):
        torch.testing.assert_close(value, block.scale_shift_table[index][None, None])


def test_scalar_and_per_token_attnres_paths_match_in_eval():
    model = SanaVideo2(
        input_size=2,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        caption_channels=16,
        model_max_length=4,
        class_dropout_prob=0.0,
        linear_head_dim=8,
        softmax_head_dim=16,
        softmax_ratio=0.5,
    )
    model.set_cross_attention_xformers(False)
    latents = torch.randn(1, 4, 1, 2, 2)
    captions = torch.randn(1, 1, 4, 16)
    mask = torch.ones(1, 1, 1, 4)

    for timestep in (torch.tensor([317.0]), torch.full((1, 1, 1, 2, 2), 317.0)):
        model.train()
        training_output = model(latents, timestep, captions, mask=mask)
        model.eval()
        inference_output = model(latents, timestep, captions, mask=mask)
        torch.testing.assert_close(inference_output, training_output)


def test_checkpoint_parameter_names_are_preserved():
    model = SanaVideo2(
        input_size=2,
        in_channels=4,
        hidden_size=64,
        depth=2,
        num_heads=4,
        caption_channels=32,
        model_max_length=4,
        class_dropout_prob=0.0,
        linear_head_dim=16,
        softmax_head_dim=32,
        softmax_ratio=0.5,
    )
    keys = set(model.state_dict())
    assert "blocks.0.attn.beta_proj.weight" in keys
    assert "blocks.1.attn.output_gate.weight" in keys
    assert "blocks.0.mlp.gate_proj.weight" in keys
    assert "attn_res.attn_proj.weight" in keys
    assert "attn_res.mlp_proj.weight" in keys
    assert "attn_res.final_proj.weight" in keys


def test_null_caption_embedding_can_be_loaded(tmp_path):
    null_caption = torch.randn(4, 32)
    null_embed_path = tmp_path / "null_embed.pth"
    torch.save({"uncond_prompt_embeds": null_caption.unsqueeze(0)}, null_embed_path)

    model = SanaVideo2(
        input_size=2,
        in_channels=4,
        hidden_size=64,
        depth=2,
        num_heads=4,
        caption_channels=32,
        model_max_length=4,
        class_dropout_prob=0.0,
        linear_head_dim=16,
        softmax_head_dim=32,
        softmax_ratio=0.5,
        null_embed_path=str(null_embed_path),
    )

    torch.testing.assert_close(model.y_embedder.y_embedding, null_caption)


def test_public_configs_select_released_models_and_video_only_training():
    repo_root = Path(__file__).resolve().parents[1]
    expected = {
        "SanaVideo2_5B_480p.yaml": ("SanaVideo2_5B", 480, 81, 16),
        "SanaVideo2_5B_720p.yaml": ("SanaVideo2_5B", 720, 193, 24),
        "SanaVideo2_14B_480p.yaml": ("SanaVideo2_14B", 480, 81, 16),
    }
    for filename, (model_name, image_size, num_frames, target_fps) in expected.items():
        with open(repo_root / "configs" / "sana_video2" / filename, encoding="utf-8") as stream:
            config = pyrallis.load(SanaVideoConfig, stream)
        assert config.model.model == model_name
        assert config.model.image_size == image_size
        assert config.data.num_frames == num_frames
        assert config.data.target_fps == target_fps
        assert config.model.softmax_ratio == 0.25
        assert config.model.attn_res_block_size == 8
        assert config.vae.use_causal_encode
        assert config.train.joint_training_interval == 0
