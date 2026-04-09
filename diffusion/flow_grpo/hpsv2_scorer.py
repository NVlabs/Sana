import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from hpsv2.src.open_clip import create_model, get_tokenizer
from PIL import Image
from torchvision.transforms import Compose, InterpolationMode, Normalize, ToTensor

from diffusion.flow_grpo.reward_ckpt_path import CKPT_PATH

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class ResizeMaxSize(nn.Module):
    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn="max", fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == "min" else min  # Note: both 'min' and 'max' map to min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            # Assuming NCHW, get H and W from the last two dimensions
            height, width = img.shape[-2:]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], fill=self.fill)
        return img


class MaskAwareNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.normalize = Normalize(mean=mean, std=std)

    def forward(self, tensor):
        # Assuming NCHW, check the channel dimension
        if tensor.shape[1] == 4:
            # Process each image in the batch
            normalized_parts = []
            for i in range(tensor.shape[0]):
                img_slice = tensor[i]
                normalized_rgb = self.normalize(img_slice[:3])
                alpha_channel = img_slice[3:]
                normalized_parts.append(torch.cat([normalized_rgb, alpha_channel], dim=0))
            return torch.stack(normalized_parts, dim=0)
        else:
            return self.normalize(tensor)


def image_transform_tensor(
    image_size: int,
    mean: tuple = None,
    std: tuple = None,
    fill_color: int = 0,
):
    mean = mean or OPENAI_DATASET_MEAN
    std = std or OPENAI_DATASET_STD

    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    normalize = MaskAwareNormalize(mean=mean, std=std)

    transforms = [
        ResizeMaxSize(image_size, fill=fill_color),
        normalize,
    ]
    return Compose(transforms)


class HPSv2Scorer(nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device
        model = create_model(
            "ViT-H-14",
            os.path.join(CKPT_PATH, "open_clip_pytorch_model.bin"),
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            output_dict=True,
        )

        image_mean = getattr(model.visual, "image_mean", None)
        image_std = getattr(model.visual, "image_std", None)
        image_size = model.visual.image_size
        if isinstance(image_size, tuple):
            image_size = image_size[0]
        preprocess_val = image_transform_tensor(
            image_size,
            mean=image_mean,
            std=image_std,
        )

        self.model = model.to(device)
        self.preprocess_val = preprocess_val
        checkpoint = torch.load(os.path.join(CKPT_PATH, "HPS_v2.1_compressed.pt"), map_location="cpu")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.processor = get_tokenizer("ViT-H-14")
        self.eval()

    @torch.no_grad()
    def __call__(self, images, prompts):
        image = self.preprocess_val(images.to(self.dtype).to(device=self.device, non_blocking=True))
        # Process the prompt
        text = self.processor(prompts).to(device=self.device, non_blocking=True)
        outputs = self.model(image, text)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits_per_image = image_features @ text_features.T
        hps_score = torch.diagonal(logits_per_image, 0)
        return hps_score.contiguous()


def main():
    scorer = HPSv2Scorer(dtype=torch.float32, device="cuda")

    images = [
        "test_cases/nasa.jpg",
        "test_cases/hello world.jpg",
    ]
    pil_images = [Image.open(img) for img in images]
    prompts = [
        'An astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
        'New York Skyline with "Hello World" written with fireworks on the sky',
    ]
    images = [np.array(img) for img in pil_images]
    images = np.array(images)
    images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    images = torch.tensor(images, dtype=torch.uint8) / 255.0
    print(scorer(images, prompts))


if __name__ == "__main__":
    main()
