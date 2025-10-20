from typing import List, Optional, Tuple, Union

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class QwenVLEmbedder:
    """
    A class to generate prompt embeddings from the Qwen2.5-VL model,
    specifically isolating the embeddings for the user's text within a larger template.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: Optional[str] = None):
        """
        Initializes the tokenizer and text encoder model.

        Args:
            model_id (str): The model identifier from the Hugging Face Hub.
            device (Optional[str]): The device to run the model on ('cuda', 'cpu', etc.).
                                     Automatically detects CUDA if available.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # --- 核心改动: 使用 AutoProcessor 替换 AutoTokenizer ---
        # AutoProcessor 包含了 Tokenizer，并且还能处理图像，因此对两个功能都适用。
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16  # Use bfloat16 for efficiency
        ).to(self.device)
        self.processor.tokenizer.padding_side = "left"

        # --- Configuration from your code ---
        self.tokenizer_max_length = 300
        # Template used to guide the model. The `{}` is where the user prompt goes.
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        # The number of tokens at the start of the template to discard,
        # so we only get the embeddings for the user's actual prompt text.
        self.prompt_template_encode_start_idx = 34

        # image+prompt template
        self.image_prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        self.image_prompt_template_encode_start_idx = 64

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
        """
        Helper function to extract non-padded token embeddings from a batch.

        Args:
            hidden_states (torch.Tensor): The output embeddings from the text encoder
                                          (batch_size, seq_len, hidden_size).
            attention_mask (torch.Tensor): The attention mask for the batch
                                           (batch_size, seq_len).

        Returns:
            List[torch.Tensor]: A list of tensors, where each tensor contains the
                                non-padded embeddings for one sequence in the batch.
        """
        split_hidden_states = []
        for i in range(hidden_states.shape[0]):
            # Get the indices of non-padded tokens (where mask is 1)
            mask_indices = attention_mask[i].nonzero(as_tuple=False).squeeze()
            # Select the corresponding hidden states
            extracted_states = hidden_states[i, mask_indices, :]
            split_hidden_states.append(extracted_states)
        return split_hidden_states

    # ===================================================================
    # function: get_prompt_embeds
    # ===================================================================
    def get_prompt_embeds(
        self, prompt: Union[str, List[str]], max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates embeddings and an attention mask for the given prompt(s),
        stripping the template part.

        Args:
            prompt (Union[str, List[str]]): A single prompt string or a list of prompts.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - prompt_embeds (torch.Tensor): The final prompt embeddings.
                - encoder_attention_mask (torch.Tensor): The corresponding attention mask.
        """
        dtype = self.text_encoder.dtype
        prompts = [prompt] if isinstance(prompt, str) else prompt

        # 1. Format prompts with the full template
        txt_with_template = [self.prompt_template_encode.format(p) for p in prompts]

        # 2. Tokenize the formatted text
        # We add `drop_idx` to max_length to ensure the user's text isn't truncated prematurely
        txt_tokens = self.processor.tokenizer(
            txt_with_template,
            max_length=self.tokenizer_max_length + self.prompt_template_encode_start_idx,
            padding="max_length",  # Pad to the longest sequence in the batch or max_length
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # 3. Get hidden states from the text encoder
        encoder_outputs = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        # We need the last hidden state
        hidden_states = encoder_outputs.hidden_states[-1]

        # 4. Remove padding from the batch
        unpadded_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)

        # 5. For each sequence, drop the template's prefix embeddings
        prompt_only_states = [e[self.prompt_template_encode_start_idx :] for e in unpadded_hidden_states]

        # 6. Create new attention masks for the prompt-only embeddings
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in prompt_only_states]

        # 7. Pad the sequences back into a single tensor for batch processing
        if max_length is None:
            max_seq_len = max([e.size(0) for e in prompt_only_states])
        else:
            max_seq_len = max_length

        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in prompt_only_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        return prompt_embeds.to(dtype=dtype, device=self.device), encoder_attention_mask

    # ===================================================================
    # function: get_image_prompt_embeds
    # ===================================================================
    def get_image_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        image: Union[Image.Image, List[Image.Image]],
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates embeddings and an attention mask for the given prompt(s) and image(s),
        stripping the template part.

        Args:
            prompt (Union[str, List[str]]): A single prompt string or a list of prompts.
            image (Union[Image.Image, List[Image.Image]]): A single PIL Image object or a list of PIL Image objects.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - prompt_embeds (torch.Tensor): The final prompt embeddings.
                - encoder_attention_mask (torch.Tensor): The corresponding attention mask.
        """
        prompts = [prompt] if isinstance(prompt, str) else prompt
        images = [image] if isinstance(image, Image.Image) else image

        template = self.image_prompt_template_encode
        drop_idx = self.image_prompt_template_encode_start_idx
        txt = [template.format(p) for p in prompts]

        model_inputs = self.processor(
            text=txt,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.device, self.torch_dtype)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        if max_length is None:
            max_seq_len = max([e.size(0) for e in split_hidden_states])
        else:
            max_seq_len = max_length

        prompt_embeds = torch.stack(
            [torch.nn.functional.pad(u, (0, 0, 0, max_seq_len - u.size(0))) for u in split_hidden_states]
        )
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        encoder_attention_mask = torch.stack(
            [torch.nn.functional.pad(u, (0, max_seq_len - u.size(0))) for u in attn_mask_list]
        )

        return prompt_embeds.to(dtype=self.torch_dtype, device=self.device), encoder_attention_mask

    # ===================================================================
    # function: chat
    # ===================================================================
    def chat(self, prompt: str, image_path: Optional[str] = None) -> str:
        """
        Chatbot function to generate responses based on the given prompt and optional image.
        """
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "image",
        #                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        #             },
        #             {"type": "text", "text": "Describe this image."},
        #         ],
        #     }
        # ]
        prompt = "Generate a beautiful image of a cat."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt_template_encode.format(prompt)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device, self.torch_dtype)

        with torch.no_grad():
            generated_ids = self.text_encoder.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]


if __name__ == "__main__":
    # Initialize the handler
    handler = QwenVLEmbedder()

    # --- Case 1: A single prompt ---
    print("--- Processing a single prompt ---")
    single_prompt = ""
    prompt_embeds, attention_mask = handler.get_prompt_embeds(single_prompt)

    print(f"Prompt Text: '{single_prompt}'")
    print(f"Shape of Prompt Embeddings: {prompt_embeds.shape}")
    print(f"Shape of Attention Mask: {attention_mask.shape}")
    print("-" * 20)
