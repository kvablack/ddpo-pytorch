# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py

from importlib import resources
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, CLIPProcessor

ASSETS_PATH = resources.files("ddpo_pytorch.assets")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Identity(),
            nn.Linear(1024, 128),
            nn.Identity(),
            nn.Linear(128, 64),
            nn.Identity(),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

        state_dict = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"))
        self.load_state_dict(state_dict)

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()

    @torch.no_grad()
    def __call__(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / embed.norm(dim=-1, keepdim=True)
        return self.mlp(embed)
