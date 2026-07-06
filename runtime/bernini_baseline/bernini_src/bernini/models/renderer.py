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

"""Bernini Renderer model: a UMT5 text encoder + a Wan2.2 dual-expert diffusion decoder.

Inference only. Transformer weights are loaded separately from a Bernini
Renderer checkpoint (see ``bernini.weights``); the Wan2.2 base supplies the text
encoder and the transformer architecture.
"""

import torch
from transformers import UMT5EncoderModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from .wan_diffusion import GEN_Wanx22


class BerniniRendererConfig(PretrainedConfig):
    model_type = "bernini_renderer"

    def __init__(
        self,
        wan22_base: str = None,
        skip_transformer_1: bool = False,
        skip_transformer_2: bool = False,
        switch_dit_boundary: float = 0.875,
        max_sequence_length: int = 512,
        shift: float = 3.0,
        use_unipc: bool = True,
        use_src_id_rotary_emb: bool = True,
        interpolate_src_id: bool = True,
        max_trained_src_id: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.wan22_base = wan22_base
        self.skip_transformer_1 = skip_transformer_1
        self.skip_transformer_2 = skip_transformer_2
        self.switch_dit_boundary = switch_dit_boundary
        self.max_sequence_length = max_sequence_length
        self.shift = shift
        self.use_unipc = use_unipc
        self.use_src_id_rotary_emb = use_src_id_rotary_emb
        # When the number of reference sources exceeds `max_trained_src_id`
        # (the largest source_id seen in training), evenly map their ids into
        # the trained range [1, max_trained_src_id] instead of extrapolating to
        # unseen integer ids. The noisy target keeps source_id 0.
        self.interpolate_src_id = interpolate_src_id
        self.max_trained_src_id = max_trained_src_id
        self.architectures = ["BerniniRendererModel"]


class BerniniRendererModel(PreTrainedModel):
    config_class = BerniniRendererConfig

    def __init__(self, config: BerniniRendererConfig):
        super().__init__(config)
        self.max_sequence_length = config.max_sequence_length
        self.t5_text_encoder = UMT5EncoderModel.from_pretrained(
            config.wan22_base, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        self.diff_dec = GEN_Wanx22(config)
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    def encode_prompt(self, input_ids, attention_mask):
        """Encode token ids into padded T5 embeddings `[B, max_len, hidden]`."""
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        with torch.no_grad():
            prompt_embeds = self.t5_text_encoder(input_ids, attention_mask).last_hidden_state
        prompt_embeds = [u[: min(v, self.max_sequence_length)] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(self.max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )
        return prompt_embeds

    @torch.no_grad()
    def sample(
        self,
        input_ids,
        attention_mask,
        uncond_input_ids,
        uncond_attention_mask,
        image_vae_latents=None,
        multi_video_vae_latents=None,
        multi_image_vae_latents=None,
        num_frames: int = 1,
        width: int = 832,
        height: int = 480,
        num_inference_steps: int = 50,
        guidance_mode: str = "rv2v",
        omega_vid: float = 3.0,
        omega_img: float = 3.0,
        omega_txt: float = 4.0,
        omega_scale: float = 0.75,
        flow_shift: float = 5.0,
        seed: int = 42,
        device="cuda",
        eta: float = 0.5,
        norm_threshold=(50.0, 50.0),
        momentum: float = -0.5,
    ):
        self.t5_text_encoder = self.t5_text_encoder.to(device)
        prompt_embeds = self.encode_prompt(input_ids, attention_mask)
        uncond_prompt_embeds = (
            self.encode_prompt(uncond_input_ids, uncond_attention_mask)
            if uncond_input_ids is not None
            else None
        )
        self.t5_text_encoder = self.t5_text_encoder.to("cpu")
        torch.cuda.empty_cache()

        return self.diff_dec.sample(
            prompt_embeds=prompt_embeds,
            uncond_prompt_embeds=uncond_prompt_embeds,
            image_vae_latents=image_vae_latents,
            multi_video_vae_latents=multi_video_vae_latents,
            multi_image_vae_latents=multi_image_vae_latents,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_mode=guidance_mode,
            omega_vid=omega_vid,
            omega_img=omega_img,
            omega_txt=omega_txt,
            omega_scale=omega_scale,
            flow_shift=flow_shift,
            seed=seed,
            device=device,
            eta=eta,
            norm_threshold=norm_threshold,
            momentum=momentum,
        )
