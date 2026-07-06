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
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import os
import torch
import numpy as np
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
try:
    from veomni.utils.constants import IGNORE_INDEX
except ModuleNotFoundError:
    IGNORE_INDEX = -100
from veomni.utils import logging
from .utils.attention_utils import build_custom_attention_mask

logger = logging.get_logger(__name__)


class T5TextTokenizer:
    def __init__(self, t5_tokenizer):
        self.t5_tokenizer = t5_tokenizer

    def extract_text_prompt(
        self, conversations: Sequence[Dict], drop_text: int = 0
    ) -> str:
        if drop_text:
            return ""
        text_parts = []
        for message in conversations:
            msg_type = message.get("type", "")
            has_loss = message.get("has_loss", 0)
            if msg_type == "text" and has_loss == 0:
                text_parts.append(message.get("text", ""))
        return " ".join(text_parts).strip()

    def tokenize(
        self,
        conversations: Sequence[Dict],
        drop_text: int = 0,
        max_length: Optional[int] = None,
        preprocess_fn=None,
    ) -> Dict[str, torch.Tensor]:
        text_prompt = self.extract_text_prompt(conversations, drop_text)

        if preprocess_fn is not None:
            text_prompt = preprocess_fn(text_prompt)
        else:
            text_prompt = prompt_clean(text_prompt)

        tokenizer_kwargs = {
            "add_special_tokens": True,
            "return_attention_mask": True,
            "return_tensors": "pt",
        }
        if max_length is not None:
            tokenizer_kwargs["max_length"] = max_length
            tokenizer_kwargs["truncation"] = True

        text_inputs = self.t5_tokenizer([text_prompt], **tokenizer_kwargs)
        input_ids = text_inputs.input_ids.squeeze(0)
        attention_mask = text_inputs.attention_mask.squeeze(0)

        return {
            "t5_input_ids": input_ids,
            "t5_attention_mask": attention_mask,
            "t5_input_lens": torch.tensor([input_ids.shape[0]]),
        }


class Qwen2VLTemplate:
    """Minimal local Qwen2VL template base for inference.

    Importing ``veomni.data`` eagerly imports torchcodec, which fails on hosts
    without system FFmpeg shared libraries. Bernini inference only needs these
    tokenizer helpers from the VeOmni template base.
    """

    def __init__(self, tokenizer, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.image_pad = "<|image_pad|>"
        self.video_pad = "<|video_pad|>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_pad)
        self.image_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.image_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.eos = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        self.bos = self.tokenizer.encode("<|im_start|>", add_special_tokens=False)
        self.cfg_ratio = kwargs.get("cfg_ratio", None)

    def image_pattern(self, token_num):
        return "<|vision_start|>" + self.image_pad * token_num + "<|vision_end|>"

    def video_pattern(self, token_num):
        return "<|vision_start|>" + self.video_pad * token_num + "<|vision_end|>"

SYSTEM_PROMPT = {
    "default": "You are a helpful assistant.",
    "t2i": "You are a helpful assistant specialized in text-to-image generation.",
    "t2v": "You are a helpful assistant specialized in text-to-video generation.",
    "i2i": "You are a helpful assistant specialized in image editing.",
    "v2v": "You are a helpful assistant specialized in video editing.",
    "r2v": "You are a helpful assistant specialized in subject-to-video generation.",
    "rv2v": "You are a helpful assistant specialized in video editing with reference.",
}

class BerniniTemplate(Qwen2VLTemplate):
    system_prompt = SYSTEM_PROMPT
    def __init__(self, tokenizer, t5_tokenizer=None, **kwargs) -> None:
        super().__init__(tokenizer, **kwargs)
        self.t5_text_tokenizer = T5TextTokenizer(t5_tokenizer) if t5_tokenizer else None

        self.image_pad_id = 151655
        self.video_pad_id = 151656
        add_special_tokens = kwargs.get("add_special_tokens", [])
        self.max_image_or_video_inter_num = kwargs.get("max_image_or_video_inter_num", 64)
        # Image/Video INPUT vit tokens with item id
        self.visual_input_token_pads = [f"<|visual_input_token_pad_{i}|>" for i in range(self.max_image_or_video_inter_num)]
        add_special_tokens.extend(self.visual_input_token_pads)
        # Image/Video OUTPUT vit tokens with item id
        self.visual_output_token_pads = [f"<|visual_output_token_pad_{i}|>" for i in range(self.max_image_or_video_inter_num)]
        add_special_tokens.extend(self.visual_output_token_pads)
        
        self.tokenizer.add_special_tokens({"additional_special_tokens": add_special_tokens})
        self.visual_input_token_pad_ids = self.tokenizer.convert_tokens_to_ids(self.visual_input_token_pads)
        self.visual_output_token_pad_ids = self.tokenizer.convert_tokens_to_ids(self.visual_output_token_pads)
    
    def visual_input_token_pattern(self, token_num, item_id):
        return "<|vision_start|>" + self.visual_input_token_pads[item_id] * token_num + "<|vision_end|>"

    def visual_output_token_pattern(self, token_num, item_id):
        return "<|vision_start|>" + self.visual_output_token_pads[item_id] * token_num + "<|vision_end|>"

    def _get_system_mesage(self, task_name):
        if task_name not in self.system_prompt:
            task_name = "default"
        role = "system"
        system_message = {
            "role": role,
            "content": self.system_prompt[task_name],
            "loss_mask": 0,
        }
        return system_message

    def format_message(self, content, has_loss):
        return {
            "role": 'user' if has_loss == 0 else 'assistant',
            "content": content,
            "loss_mask": 0 if has_loss == 0 else 1,
        }

    def encode_messages(
        self, 
        conversations: Sequence[Dict[str, str]],
        num_tokens: Dict[str, List[int]] = defaultdict(list),
        task_name: str = "",
        drop_text: int = 0,
        drop_video: int = 0,
        drop_img: int = 0,
        vit_mask_ratio: float = 1.0,
        neg_prompt: Optional[str] = '',
        **kwargs
    ) -> Dict[str, List[int]]:
        sys_msg = self._get_system_mesage(task_name)
        messages = [] if sys_msg is None else [sys_msg]
        image_token_num_list = iter(num_tokens.get("image", []))
        video_token_num_list = iter(num_tokens.get("video", []))

        content = ""
        text_content = ""
        pre_has_loss = 0
        visual_id_to_type = dict({})
        visual_id, img_id, vid_id = 0, 0, 0
        indicator_id = 2
        visual_indicator_maps = {}
        image_target_mask, video_target_mask = [], []
        vae_type_list, vit_type_list = [], [] # 0 for image, 1 for video
        vit_img_and_vid_id_list = [] 
        for message in conversations:
            if message['type'] == 'special_token':
                continue

            if message['type'] == 'cot_text':
                assert 'has_loss' in message
                message['has_loss'] = 0
            
            if 'has_loss' not in message:
                if message['type'] == 'video_gen':
                    message['has_loss'] = 1
                else:
                    message['has_loss'] = 0
            
            if pre_has_loss != message['has_loss']:
                messages.append(self.format_message(content, pre_has_loss))
                content = ""
                pre_has_loss = message['has_loss']
            
            if message['type'] in ['text', 'cot_text']:
                if len(neg_prompt) > 1: 
                    text_content += neg_prompt
                    content += neg_prompt

                elif not drop_text:
                    text_content += message['text']
                    content += message['text']
            
            elif message['type'] in ['image', 'image_gen']:
                token_num = next(image_token_num_list)
                if message['has_loss'] == 1: # image_gen
                    
                    content += self.visual_output_token_pattern(token_num, visual_id)
                    vit_img_and_vid_id_list.append(img_id)
                    vit_type_list.append(0)
                    indicator_id += 1
                    visual_indicator_maps[self.tokenizer.convert_tokens_to_ids(self.visual_output_token_pads[visual_id])] = indicator_id
                elif message['has_loss'] == 0:  # image
                    if not drop_img:
                        
                        content += self.visual_input_token_pattern(token_num, visual_id)
                        vit_img_and_vid_id_list.append(img_id)
                        vit_type_list.append(0)
                        visual_indicator_maps[self.tokenizer.convert_tokens_to_ids(self.visual_input_token_pads[visual_id])] = indicator_id
                
                visual_id_to_type[visual_id] = 0
                img_id += 1
                visual_id += 1
                indicator_id += 1
                
                image_target_mask.extend([message['has_loss']])
                if not drop_img or message['has_loss'] == 1:
                    vae_type_list.append(0)
            
            elif message['type'] in ['video', 'frame_gen', 'video_gen']:
                token_num = next(video_token_num_list)
                if message['has_loss'] == 1:  # frame_gen or video_gen
                    
                    content += self.visual_output_token_pattern(token_num, visual_id)
                    vit_img_and_vid_id_list.append(vid_id)
                    vit_type_list.append(1)
                    indicator_id += 1
                    visual_indicator_maps[self.tokenizer.convert_tokens_to_ids(self.visual_output_token_pads[visual_id])] = indicator_id
                elif message['has_loss'] == 0:  # video
                    if not drop_video:
                        
                        content += self.visual_input_token_pattern(token_num, visual_id)
                        vit_img_and_vid_id_list.append(vid_id)
                        vit_type_list.append(1)
                        visual_indicator_maps[self.tokenizer.convert_tokens_to_ids(self.visual_input_token_pads[visual_id])] = indicator_id

                visual_id_to_type[visual_id] = 1
                vid_id += 1
                visual_id += 1
                indicator_id += 1
                
                video_target_mask.extend([message['has_loss']])
                if not drop_video or message['has_loss'] == 1:
                    vae_type_list.append(1)
            else:
                raise ValueError(f"Unknown value type: {message['type']}")

        messages.append(self.format_message(content, pre_has_loss))

        input_ids, attention_mask, labels = [], [], []
        for i, message in enumerate(messages):
            content_str = message["content"].strip()
            if not content_str:
                continue
            loss_mask = message["loss_mask"]

            message_ids = self.tokenizer.encode("<|im_start|>" + message["role"] + "\n", add_special_tokens=False)
            content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            message_ids += content_ids

            input_ids += message_ids
            attention_mask += [1] * len(message_ids)
            if loss_mask == 1:
                labels += message_ids
            else:
                labels += [IGNORE_INDEX] * len(message_ids)

        tokenized_example = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels, 
            # items for vit embeds
            "vit_type_list": vit_type_list,
            "vit_img_and_vid_id_list": vit_img_and_vid_id_list,
            # items for vae latents
            "image_target_mask": image_target_mask,
            "video_target_mask": video_target_mask,
            "vae_type_list": vae_type_list,
        }
        tokenized_example = {k: torch.tensor(v)
                             for k, v in tokenized_example.items()}
        tokenized_example['text_content'] = text_content

        vision_start_indices = []
        token_types = torch.zeros_like(tokenized_example["labels"], dtype=torch.int) 
        flex_token_types = -torch.ones_like(tokenized_example["labels"], dtype=torch.int)
        token_segment_ids = torch.tensor(range(len(tokenized_example["labels"]))) 

        visual_input_token_mask = torch.zeros_like(tokenized_example["labels"], dtype=torch.bool)
        visual_output_token_mask = torch.zeros_like(tokenized_example["labels"], dtype=torch.bool)
        
        for visual_id, input_vit_id in enumerate(self.visual_input_token_pad_ids):
            input_vit_mask = tokenized_example["input_ids"] == input_vit_id
            if input_vit_mask.sum() > 0:
                token_types[input_vit_mask] = 2
                visual_input_token_mask[input_vit_mask] = True
                token_segment_ids[input_vit_mask] = visual_id + 1
                
                vision_start_indices.append(input_vit_mask.nonzero().min().item())
                mllm_visual_pad = self.image_pad_id if visual_id_to_type[visual_id] == 0 else self.video_pad_id
                tokenized_example["input_ids"][input_vit_mask] = mllm_visual_pad

        
        for visual_id, output_vit_id in enumerate(self.visual_output_token_pad_ids):
            output_vit_mask = tokenized_example["input_ids"] == output_vit_id
            if output_vit_mask.sum() > 0:
                token_types[output_vit_mask] = 3
                flex_token_types[output_vit_mask] = visual_indicator_maps[output_vit_id]
                visual_output_token_mask[output_vit_mask] = True
                token_segment_ids[output_vit_mask] = visual_id + 1
                
                vision_start_indices.append(output_vit_mask.nonzero().min().item())
                mllm_visual_pad = self.image_pad_id if visual_id_to_type[visual_id] == 0 else self.video_pad_id
                tokenized_example["input_ids"][output_vit_mask] = mllm_visual_pad

        
        tokenized_example["vision_start_indices"] = sorted(vision_start_indices)
        tokenized_example["visual_input_token_mask"] = visual_input_token_mask
        tokenized_example["visual_output_token_mask"] = visual_output_token_mask

        # the label will be filled in decoder.
        tokenized_example["labels"][visual_input_token_mask] = IGNORE_INDEX
        tokenized_example["labels"][visual_output_token_mask] = IGNORE_INDEX

        # Some tasks should not calculate MLLM text loss
        labels = tokenized_example["labels"]
        if task_name in ['t2i', 't2v', 'i2i', 'v2v', 'v2v_trans', 'i2v_trans', 'i2v', 'iv2v', 'rv2v', 'r2v']:
            labels[labels != IGNORE_INDEX] = IGNORE_INDEX
            tokenized_example["labels"] = labels
        
        # Process masks
        all_target_vit_token_num = visual_output_token_mask.sum()
        if all_target_vit_token_num > 0:
            mask_vit_token_num = int(np.ceil(all_target_vit_token_num * vit_mask_ratio))
            all_tgt_vit_token_idx = list(range(all_target_vit_token_num))
            np.random.shuffle(all_tgt_vit_token_idx)
            tgt_vit_mask_idx = all_tgt_vit_token_idx[:mask_vit_token_num]
            tgt_vit_mask = torch.zeros(all_target_vit_token_num)
            tgt_vit_mask[tgt_vit_mask_idx] = 1
            tokenized_example["tgt_vit_mask"] = tgt_vit_mask

        # Build the MLLM attention mask here
        mllm_attn_mask = build_custom_attention_mask(
            token_type=token_types.unsqueeze(0),
            token_segment_ids=token_segment_ids.unsqueeze(0),
        )
        tokenized_example["attention_mask_4d"] = mllm_attn_mask
        # shift labels for causal LM
        labels = tokenized_example["labels"]
        labels = torch.cat(
            [labels[1:], labels.new_full((1,), IGNORE_INDEX)],
            dim=0
        )
        tokenized_example["labels"] = labels
        tokenized_example["flex_token_types"] = flex_token_types

        # T5 tokenize
        if self.t5_text_tokenizer is not None:
            t5_outputs = self.t5_text_tokenizer.tokenize(conversations, drop_text)
            tokenized_example.update(t5_outputs)

        return tokenized_example
