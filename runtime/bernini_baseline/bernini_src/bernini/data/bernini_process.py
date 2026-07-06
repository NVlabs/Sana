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
import io
import copy
import json
import random
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from einops import rearrange

if TYPE_CHECKING:
    from transformers import ProcessorMixin
    from veomni.data.chat_template import ChatTemplate


def filt_out_source_vae(
    inputs,
    vae_latents,
    vae_latent_shape,
    target_mask,
    drop_vision,
    vae_latent_key,
    vae_shape_key,
    vae_mask_key,
):
    vae_latents_filted, vae_latents_filted_shape, vae_mask = [], [], []
    if drop_vision:
        for (vae_emb, shape, is_target) in zip(vae_latents, vae_latent_shape, target_mask):
            if is_target:
                vae_mask.extend([is_target])
                vae_latents_filted.append(vae_emb)
                vae_latents_filted_shape.append(shape)
    else:
        vae_latents_filted = vae_latents
        vae_latents_filted_shape = vae_latent_shape
        vae_mask = target_mask

    inputs[vae_latent_key] = vae_latents_filted
    inputs[vae_mask_key] = torch.tensor(vae_mask)
    inputs[vae_shape_key] = vae_latents_filted_shape


def get_drop_condition(
    text_dropout_rate: float,
    img_dropout_rate: float,
    video_dropout_rate: float,
):
    drop_text, drop_video, drop_img = 0, 0, 0
    text_drop, img_drop, video_drop = random.random(), random.random(), random.random()
    if text_drop < text_dropout_rate:
        drop_text = 1
    if img_drop < img_dropout_rate:
        drop_img = 1
    if video_drop < video_dropout_rate:
        drop_video = 1
    return drop_text, drop_video, drop_img


def rearrange_vae_feature(vae_emb):
    pt, ph, pw = 1, 2, 2
    patched_vae_emb = rearrange(
        vae_emb, 'c (t pt) (h ph) (w pw) -> (t h w) c pt ph pw', pt=pt, ph=ph, pw=pw
    )
    return patched_vae_emb


def packing_vae(
    vae_rope_func: "Callable",
    vae_type_list: "List[int]",
    image_inputs: "Dict[str, Any]",
    video_inputs: "Dict[str, Any]",
    noise_sigma: float,
    max_vae_frames: int = None,
    interpolate_src_id: bool = True,
    max_trained_src_id: int = 5,
):
    image_vae_masks = list(image_inputs.pop('image_vae_mask', []))
    video_vae_masks = list(video_inputs.pop('video_vae_mask', []))

    image_vae_list = iter(image_inputs.pop('image_vae_latents', []))
    image_vae_shape_list = iter(image_inputs.pop('image_vae_shape', []))
    image_vae_mask_list = iter(image_vae_masks)

    video_vae_list = iter(video_inputs.pop('video_vae_latents', []))
    video_vae_shape_list = iter(video_inputs.pop('video_vae_shape', []))
    video_vae_mask_list = iter(video_vae_masks)

    # Source ids for the conditioning segments (the target keeps source_id 0).
    # Training assigns position-based integer ids; when more conditioning
    # segments are given than the model saw in training (`max_trained_src_id`),
    # evenly spread their ids across the trained range [1, max_trained_src_id]
    # so the rotary phases stay inside the trained manifold instead of
    # extrapolating to unseen integer ids.
    num_src = sum(1 for m in image_vae_masks + video_vae_masks if not m)
    src_sids = None
    if interpolate_src_id and num_src > max_trained_src_id:
        src_sids = torch.linspace(1.0, float(max_trained_src_id), num_src).tolist()
    src_ptr = 0  # cursor into src_sids

    input_vae_latents = []
    input_vae_shape = []
    input_vae_rope = []
    vae_latents_mask = []
    target_velocity = []
    target_lens = []

    for idx, vae_type in enumerate(vae_type_list):
        if vae_type == 0:
            vae_emb = next(image_vae_list)
            vae_shape = next(image_vae_shape_list)
            vae_mask = next(image_vae_mask_list)
        else:
            vae_emb = next(video_vae_list)
            vae_shape = next(video_vae_shape_list)
            vae_mask = next(video_vae_mask_list)

        if max_vae_frames is not None and vae_emb.shape[1] > max_vae_frames:
            vae_emb = vae_emb[:, :max_vae_frames, :, :]
            vae_shape[0] = max_vae_frames

        if not vae_mask:
            src_sid = src_sids[src_ptr] if src_sids is not None else float(idx + 1)
            src_ptr += 1

        if vae_rope_func is None:
            vae_rope = None
        elif vae_mask:
            vae_rope = vae_rope_func(vae_emb.unsqueeze(0), source_id=0).squeeze(0)
        else:
            vae_rope = vae_rope_func(vae_emb.unsqueeze(0), source_id=src_sid).squeeze(0)
        input_vae_rope.append(vae_rope)
        input_vae_shape.append(vae_shape)
        vae_emb = rearrange_vae_feature(vae_emb)

        vae_latents_mask.extend([vae_mask] * vae_emb.shape[0])
        if vae_mask:
            noise_sigma = noise_sigma.to(device=vae_emb.device, dtype=vae_emb.dtype)
            noise = torch.randn_like(vae_emb, dtype=torch.float32)
            input_noise_latent = (1 - noise_sigma) * vae_emb + noise_sigma * noise
            target_velocity.append(noise - vae_emb.float())
            target_lens.append(vae_emb.shape[0])
            input_vae_latents.append(input_noise_latent)
        else:
            input_vae_latents.append(vae_emb)
    input_vae_shape = torch.tensor(input_vae_shape)
    vae_latents_mask = torch.tensor(vae_latents_mask)
    packed_vae_latents = {
        "input_vae_shape": input_vae_shape,
        "vae_latents_mask": vae_latents_mask,
    }
    if vae_rope_func is not None and len(input_vae_rope) > 0:
        input_vae_rope = torch.cat(input_vae_rope, dim=1)
        input_vae_rope = input_vae_rope.permute(1, 0, 2)
        packed_vae_latents['input_vae_rope'] = input_vae_rope

    if len(input_vae_latents) > 0:
        input_vae_latents = torch.cat(input_vae_latents, dim=0)
        vae_seqlen = input_vae_latents.shape[0]
        target_lens = torch.tensor(target_lens)
        packed_vae_latents['input_vae_latents'] = input_vae_latents
    else:
        vae_seqlen = 0
        target_lens = torch.tensor([])

    packed_vae_latents['vae_seqlen'] = torch.tensor([vae_seqlen])
    packed_vae_latents['target_lens'] = target_lens

    if len(target_velocity) > 0:
        target_velocity = torch.cat(target_velocity, dim=0)
        packed_vae_latents['target_velocity'] = target_velocity

    return packed_vae_latents


def bernini_process_sample(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    vae_rope_func: Callable,
    vae_latent_mean: torch.Tensor,
    vae_latent_std: torch.Tensor,
    text_dropout_rate: float,
    img_dropout_rate: float,
    video_dropout_rate: float,
    max_vae_frames: int = None,  
    noise_sigma=torch.tensor(0),
    noise_timestep=torch.tensor(0), 
    noise_sigma_low=None,
    noise_timestep_low=None,
    vit_mask_ratio: float = 1.0, # mask all tokens when inference
    interpolate_src_id: bool = True,
    max_trained_src_id: int = 5,
    **kwargs,
):
    """
    Processes multimodal example with qwen2_5_vl's pre-processor.
    """
    
    task_name = kwargs.get("source_name", "").split("$")[0].lower()
    drop_text, drop_video, drop_img = get_drop_condition(
        text_dropout_rate,
        img_dropout_rate,
        video_dropout_rate
    )

    conversations = json.loads(sample["inputs"])

    token_num_inputs = {}
    if "image_embeds" in sample and sample['image_embeds'] is not None and len(sample['image_embeds']) > 0:
        raw_image_embeds = []
        raw_image_grid_thw = []
        for vit_emb, thw in zip(sample['image_embeds'], sample['image_grid_thw']):
            buffer = io.BytesIO(vit_emb)
            buffer.seek(0)
            vit_emb = torch.load(buffer)
            raw_image_embeds.append(vit_emb)
            raw_image_grid_thw.append(thw)

        merge_length = processor.image_processor.merge_size**2
        token_num_inputs["image"] = (
            torch.tensor(raw_image_grid_thw).prod(dim=-1) // merge_length
        )

    if "video_embeds" in sample and sample['video_embeds'] is not None and len(sample['video_embeds']) > 0:
        raw_video_embeds = []
        raw_video_grid_thw = []
        for vit_emb, thw in zip(sample['video_embeds'], sample['video_grid_thw']):
            buffer = io.BytesIO(vit_emb)
            buffer.seek(0)
            vit_emb = torch.load(buffer)
            raw_video_embeds.append(vit_emb)
            raw_video_grid_thw.append(thw)

        merge_length = processor.image_processor.merge_size**2
        token_num_inputs["video"] = (
            torch.tensor(raw_video_grid_thw).prod(dim=-1) // merge_length
        )

    tokenized_example = chat_template.encode_messages(
        conversations,
        token_num_inputs,
        task_name,
        drop_text=drop_text,
        drop_video=drop_video,
        drop_img=drop_img,
        vit_mask_ratio=vit_mask_ratio,
        **kwargs,
    )

    for k, v in tokenized_example.items():
        if isinstance(v, str):
            continue
        if torch.is_tensor(v):
            tokenized_example[k] = v
        else:
            tokenized_example[k] = torch.as_tensor(v)
    
    # Packing vit embeds
    vit_type_list, vit_img_and_vid_id_list = tokenized_example.pop(
        'vit_type_list'), tokenized_example.pop('vit_img_and_vid_id_list')
    visual_embeds = []
    image_grid_thw, video_grid_thw = [], []
    for vit_type, vit_id in zip(vit_type_list, vit_img_and_vid_id_list):
        if vit_type == 0: # image
            image_grid_thw.append(raw_image_grid_thw[vit_id])
            visual_embeds.append(raw_image_embeds[vit_id])
        elif vit_type == 1: # video
            video_grid_thw.append(raw_video_grid_thw[vit_id])
            visual_embeds.append(raw_video_embeds[vit_id])
    
    if len(image_grid_thw) > 0:
        image_grid_thw = torch.tensor(image_grid_thw)
    if len(video_grid_thw) > 0:
        video_grid_thw = torch.tensor(video_grid_thw)
    if len(visual_embeds) > 0:
        visual_embeds = torch.cat(visual_embeds, dim=0)
        tokenized_example['visual_embeds'] = visual_embeds
    else:
        visual_embeds = torch.randn(0, 3584)

    
    input_ids = tokenized_example["input_ids"]
    tokenized_example["position_ids"] = position_id_func(
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=image_grid_thw if len(image_grid_thw) > 0 else None,
        video_grid_thw=video_grid_thw if len(video_grid_thw) > 0 else None,
        attention_mask=tokenized_example["attention_mask"].unsqueeze(0),
    )[0].squeeze(1).clone()  # (dim, l)
    tokenized_example["mllm_seqlen"] = tokenized_example["attention_mask"].sum().reshape(1)

    image_inputs, video_inputs = {}, {}
    # Packing vae latents
    image_target_mask, video_target_mask = tokenized_example.pop(
        'image_target_mask'), tokenized_example.pop('video_target_mask')

    if sample.get('image_vae_latents', None) is not None and len(sample['image_vae_latents']) > 0:
        image_vae_latents = []
        image_vae_shape = []
        for vae_emb, _ in zip(sample['image_vae_latents'], image_target_mask):
            buffer = io.BytesIO(vae_emb)
            buffer.seek(0)
            vae_emb = torch.load(buffer)
            _, _, t, h, w = vae_emb.shape
            image_vae_shape.append([t, h, w])
            vae_emb = DiagonalGaussianDistribution(vae_emb).mode()
            vae_emb = vae_emb.squeeze(0)
            vae_emb = (vae_emb - vae_latent_mean) / vae_latent_std
            image_vae_latents.append(vae_emb)
        filt_out_source_vae(
            image_inputs,
            image_vae_latents,
            image_vae_shape,
            image_target_mask,
            drop_img,
            'image_vae_latents',
            'image_vae_shape',
            'image_vae_mask'
        )

    if "video_vae_latents" in sample and len(sample['video_vae_latents']) > 0:
        video_vae_latents = []
        video_vae_shape = []
        for vae_emb, _ in zip(sample['video_vae_latents'], video_target_mask):
            buffer = io.BytesIO(vae_emb)
            buffer.seek(0)
            vae_emb = torch.load(buffer)
            _, _, t, h, w = vae_emb.shape
            video_vae_shape.append([t, h, w])
            vae_emb = DiagonalGaussianDistribution(vae_emb).sample()
            vae_emb = vae_emb.squeeze(0)
            vae_emb = (vae_emb - vae_latent_mean) / vae_latent_std
            video_vae_latents.append(vae_emb)
        filt_out_source_vae(
            video_inputs,
            video_vae_latents,
            video_vae_shape,
            video_target_mask,
            drop_video,
            'video_vae_latents',
            'video_vae_shape',
            'video_vae_mask'
        )

    vae_type_list = tokenized_example.pop('vae_type_list')

    if noise_sigma_low is not None:
        image_inputs_copy = copy.deepcopy(image_inputs)
        video_inputs_copy = copy.deepcopy(video_inputs)
        packed_vae_latents_low = packing_vae(
            vae_rope_func,
            vae_type_list,
            image_inputs_copy,
            video_inputs_copy,
            noise_sigma_low,
            max_vae_frames,
            interpolate_src_id=interpolate_src_id,
            max_trained_src_id=max_trained_src_id,
        )
        for k, v in packed_vae_latents_low.items():
            tokenized_example[k + '_low'] = v

    packed_vae_latents = packing_vae(
        vae_rope_func,
        vae_type_list,
        image_inputs,
        video_inputs,
        noise_sigma,
        max_vae_frames,
        interpolate_src_id=interpolate_src_id,
        max_trained_src_id=max_trained_src_id,
    )
    tokenized_example.update(packed_vae_latents)
    tokenized_example['timesteps'] = torch.tensor([noise_timestep])
    if noise_timestep_low is not None:
        tokenized_example['timesteps_low'] = torch.tensor([noise_timestep_low])

    tokenized_example["num_tokens"] = torch.tensor(
        [tokenized_example["vae_seqlen"][0] + tokenized_example["mllm_seqlen"][0]]
    )

    tokenized_example['task_name'] = task_name
    return [tokenized_example]
