#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import torch


def _as_prompt_list(prompt):
    return [prompt] if isinstance(prompt, str) else prompt


def _move_to_device(value, device):
    if value is None or device is None:
        return value
    return value.to(device)


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = _as_prompt_list(prompt)
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    elif text_input_ids is None:
        raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt,
    device=None,
    text_input_ids=None,
    num_images_per_prompt=1,
):
    prompt = _as_prompt_list(prompt)
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    elif text_input_ids is None:
        raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds, pooled_prompt_embeds


def encode_sd3_prompt(
    text_encoders,
    tokenizers,
    prompt,
    max_sequence_length,
    device=None,
    num_images_per_prompt=1,
    text_input_ids_list=None,
):
    prompt = _as_prompt_list(prompt)
    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []

    for idx, (tokenizer, text_encoder) in enumerate(zip(tokenizers[:2], text_encoders[:2])):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[idx] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds,
        (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    target_device = device if device is not None else prompt_embeds.device
    return _move_to_device(prompt_embeds, target_device), _move_to_device(pooled_prompt_embeds, target_device)


def encode_flux_prompt(
    pipeline,
    prompt,
    max_sequence_length,
    device=None,
    num_images_per_prompt=1,
    prompt_2=None,
    lora_scale=None,
):
    prompt = _as_prompt_list(prompt)
    prompt_2 = prompt if prompt_2 is None else _as_prompt_list(prompt_2)

    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    target_device = device if device is not None else prompt_embeds.device
    return (
        _move_to_device(prompt_embeds, target_device),
        _move_to_device(pooled_prompt_embeds, target_device),
        _move_to_device(text_ids, target_device),
    )


def encode_sana_prompt(
    pipeline,
    prompt,
    max_sequence_length,
    device=None,
    negative_prompt="",
    do_classifier_free_guidance=True,
):
    prompt = _as_prompt_list(prompt)
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
        pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=device,
            max_sequence_length=max_sequence_length,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
    )

    target_device = device if device is not None else prompt_embeds.device
    return (
        _move_to_device(prompt_embeds, target_device),
        _move_to_device(prompt_attention_mask, target_device),
        _move_to_device(negative_prompt_embeds, target_device),
        _move_to_device(negative_prompt_attention_mask, target_device),
    )
