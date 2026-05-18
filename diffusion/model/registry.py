"""Registries for Sana attention / FFN / norm / pos-embed / generic components."""

from mmcv import Registry

ATTENTION_BLOCKS = Registry("attention_blocks")
FFN_BLOCKS = Registry("ffn_blocks")
NORM_LAYERS = Registry("norm_layers")
POS_EMBEDS = Registry("pos_embeds")
COMPONENTS = Registry("components")
