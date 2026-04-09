# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py

import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from diffusion.flow_grpo.reward_ckpt_path import CKPT_PATH


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP().to(device)
        state_dict = torch.load(os.path.join(CKPT_PATH, "sac+logos+ava1-l14-linearMSE.pth"), map_location="cpu")
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.device = device
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(self.device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)


# Usage example
def main():
    scorer = AestheticScorer(device="cuda", dtype=torch.float32)

    images = [
        "test_cases/nasa.jpg",
    ]
    pil_images = np.stack([np.array(Image.open(img)) for img in images])
    images = pil_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    images = torch.tensor(images, dtype=torch.uint8)
    print(scorer(images))


if __name__ == "__main__":
    main()
