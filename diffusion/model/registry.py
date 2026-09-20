# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Registries for Sana attention / FFN / norm / pos-embed / generic components."""

from mmcv import Registry

ATTENTION_BLOCKS = Registry("attention_blocks")
FFN_BLOCKS = Registry("ffn_blocks")
NORM_LAYERS = Registry("norm_layers")
POS_EMBEDS = Registry("pos_embeds")
COMPONENTS = Registry("components")
