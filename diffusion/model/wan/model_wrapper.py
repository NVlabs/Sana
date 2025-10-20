import torch.nn as nn

from .model import WanAttentionBlock, WanLinearAttentionModel, WanModel


class SanaVideoMSBlock(WanAttentionBlock):
    pass


class SanaWanModel(WanModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cross_attn_type = "t2v_cross_attn" if self.model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                SanaVideoMSBlock(
                    cross_attn_type,
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    self.window_size,
                    self.qk_norm,
                    self.cross_attn_norm,
                    self.eps,
                )
                for _ in range(self.num_layers)
            ]
        )


class SanaWanLinearAttentionModel(WanLinearAttentionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cross_attn_type = "t2v_cross_attn" if self.model_type == "t2v" else "i2v_cross_attn"
        self_attn_types = ["flash"] * self.num_layers
        ffn_types = ["mlp"] * self.num_layers

        self.blocks = nn.ModuleList(
            [
                SanaVideoMSBlock(
                    cross_attn_type,
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    self.window_size,
                    self.qk_norm,
                    self.cross_attn_norm,
                    self.eps,
                    self_attn_types[i],
                    self.rope_after,
                    self.power,
                    ffn_types[i],
                )
                for i in range(self.num_layers)
            ]
        )
