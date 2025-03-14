# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from diffusion.model.builder import MODELS

from .ladd_blocks import DiscHead
from .sana_multi_scale import SanaMSCM


@MODELS.register_module()
class SanaMSCMDiscriminator(nn.Module):
    def __init__(self, pretrained_model: SanaMSCM, is_multiscale=False, head_block_ids=None):
        super().__init__()
        self.transformer = pretrained_model
        self.transformer.requires_grad_(False)

        if head_block_ids is None or len(head_block_ids) == 0:
            self.block_hooks = {2, 8, 14, 20, 27} if is_multiscale else {self.transformer.depth - 1}
        else:
            self.block_hooks = head_block_ids

        heads = []
        for i in range(len(self.block_hooks)):
            heads.append(DiscHead(self.transformer.hidden_size, 0, 0))
        self.heads = nn.ModuleList(heads)

    def get_head_inputs(self):
        return self.head_inputs

    def forward(self, x, timestep, y=None, data_info=None, mask=None, **kwargs):
        feat_list = []
        self.head_inputs = []

        def get_features(module, input, output):
            feat_list.append(output)
            return output

        hooks = []
        for i, block in enumerate(self.transformer.blocks):
            if i in self.block_hooks:
                hooks.append(block.register_forward_hook(get_features))

        self.transformer(x, timestep, y=y, mask=mask, data_info=data_info, return_logvar=False, **kwargs)

        for hook in hooks:
            hook.remove()

        res_list = []
        for feat, head in zip(feat_list, self.heads):
            B, N, C = feat.shape
            feat = feat.transpose(1, 2)  # [B, C, N]
            self.head_inputs.append(feat)
            res_list.append(head(feat, None).reshape(feat.shape[0], -1))

        concat_res = torch.cat(res_list, dim=1)

        return concat_res

    @property
    def model(self):
        return self.transformer

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)


class DiscHeadModel:
    def __init__(self, disc):
        self.disc = disc

    def state_dict(self):
        return {name: param for name, param in self.disc.state_dict().items() if not name.startswith("transformer.")}

    def __getattr__(self, name):
        return getattr(self.disc, name)
