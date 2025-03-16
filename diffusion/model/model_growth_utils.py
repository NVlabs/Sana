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

import inspect
from enum import Enum
from typing import Any, Dict, Optional

import torch

from diffusion.utils.logger import get_root_logger


class InitStrategy(str, Enum):
    """Init strategy enum"""

    CYCLIC = "cyclic"
    BLOCK_EXPAND = "block_expand"
    PROGRESSIVE = "progressive"
    INTERPOLATION = "interpolation"
    RANDOM = "random"
    CONSTANT = "constant"


class ModelGrowthInitializer:
    """Model growth initializer"""

    def __init__(self, target_model, config):
        """
        Args:
            target_model: target model (deeper model)
            config: ModelGrowthConfig config object
        """
        self.target_model = target_model
        self.pretrained_state = torch.load(config.pretrained_ckpt_path, map_location="cpu")
        if "state_dict" in self.pretrained_state:
            self.pretrained_state = self.pretrained_state["state_dict"]
        self.target_state = target_model.state_dict()

        # get model layers
        self.pretrained_layers = self._get_num_layers_from_state_dict(self.pretrained_state)
        self.target_layers = self._get_num_layers_from_state_dict(self.target_state)

        # verify layers
        assert (
            config.source_num_layers <= self.pretrained_layers
        ), f"config source layers({config.source_num_layers}) must be less than pretrained model layers({self.pretrained_layers})"
        assert (
            self.target_layers == config.target_num_layers
        ), f"target model layers({self.target_layers}) must be equal to config target layers({config.target_num_layers})"

        if config.source_num_layers < self.pretrained_layers:
            self.pretrained_layers = config.source_num_layers

        self.logger = get_root_logger()
        self.logger.info(f"init strategy: {config.init_strategy}")

    def initialize(self, strategy: str, **kwargs) -> torch.nn.Module:
        """

        Args:
            strategy: init strategy name
            **kwargs: strategy specific parameters

        Returns:
            initialized model

        Raises:
            ValueError: when strategy name is invalid
        """
        try:
            strategy = InitStrategy(strategy.lower())
        except ValueError:
            raise ValueError(
                f"unsupported init strategy: {strategy}, " f"supported strategies: {[s.value for s in InitStrategy]}"
            )

        # strategy mapping
        strategy_map = {
            InitStrategy.CYCLIC: self.init_cyclic,
            InitStrategy.BLOCK_EXPAND: self.init_block_expand,
            InitStrategy.PROGRESSIVE: self.init_progressive,
            InitStrategy.INTERPOLATION: self.init_interpolation,
            InitStrategy.RANDOM: self.init_random,
            InitStrategy.CONSTANT: self.init_constant,
        }

        # get init method
        init_method = strategy_map[strategy]

        # get method parameters
        valid_params = inspect.signature(init_method).parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params} if kwargs else {}

        # only pass method parameters
        return init_method(**filtered_kwargs)

    def _get_num_layers_from_state_dict(self, state_dict):
        """get model layers from state dict"""
        return (
            max(
                [
                    int(key.split(".")[1])
                    for key in state_dict.keys()
                    if key.startswith("blocks.") and key.split(".")[1].isdigit()
                ]
            )
            + 1
        )

    def _copy_non_transformer_params(self):
        """copy non-transformer params, skip specific params"""
        skip_keys = {
            "pos_embed",  # position embedding
            # if there are other params to skip, add them here
        }

        for key in self.pretrained_state:
            if "blocks." not in key and key not in skip_keys:
                self.logger.info(f"copy non-transformer params: {key}")
                self.target_state[key] = self.pretrained_state[key]

    def init_cyclic(self, zero_gate=False):
        """cyclic copy strategy

        Args:
            zero_gate: whether to initialize the gate params of the non-first-appearing repeated layers to 0

        For example: for 20 layers to 60 layers, when zero_gate=True:
        Original model layer 0 → New model layer 0 (keep original), layer 20 (zero-gate), layer 40 (zero-gate)
        Original model layer 1 → New model layer 1 (keep original), layer 21 (zero-gate), layer 41 (zero-gate)
        And so on.
        """
        self._copy_non_transformer_params()

        for i in range(self.target_layers):
            src_layer_idx = i % self.pretrained_layers
            # check if it is a repeated layer
            is_repeated = i >= self.pretrained_layers

            for key in self.pretrained_state:
                if f"blocks.{src_layer_idx}." in key:
                    new_key = key.replace(f"blocks.{src_layer_idx}.", f"blocks.{i}.")

                    # use zero-gate strategy for repeated layers
                    if zero_gate and is_repeated:
                        if "scale_shift_table" in new_key:
                            # set the entire scale_shift_table to 0
                            self.target_state[new_key] = torch.zeros_like(self.pretrained_state[key])
                            self.logger.info(f"zero init entire scale_shift_table: {new_key}")
                        elif "cross_attn.proj.weight" in new_key or "cross_attn.proj.bias" in new_key:
                            # set the output projection layer of cross attention to 0
                            self.target_state[new_key] = torch.zeros_like(self.pretrained_state[key])
                            self.logger.info(f"zero init cross attn proj: {new_key}")
                        elif "attn.proj.weight" in new_key or "attn.proj.bias" in new_key:
                            # set the output projection layer of self attention to 0
                            self.target_state[new_key] = torch.zeros_like(self.pretrained_state[key])
                            self.logger.info(f"zero init self attn proj: {new_key}")
                        elif "mlp.point_conv.conv.weight" in new_key or "mlp.point_conv.conv.bias" in new_key:
                            # set the last point_conv of mlp to 0
                            self.target_state[new_key] = torch.zeros_like(self.pretrained_state[key])
                            self.logger.info(f"zero init mlp point conv: {new_key}")
                        else:
                            # other params copy normally
                            self.target_state[new_key] = self.pretrained_state[key]
                            self.logger.info(f"copy transformer params: {key} -> {new_key}")
                    else:
                        # the first appearing layer or not using zero_gate
                        self.target_state[new_key] = self.pretrained_state[key]
                        self.logger.info(f"copy transformer params: {key} -> {new_key}")

        self.target_model.load_state_dict(self.target_state)
        return self.target_model

    def init_progressive(self, noise_scale=0.01):
        """progressive init strategy (with noise)"""
        self._copy_non_transformer_params()

        # copy pretrained layers
        for i in range(self.pretrained_layers):
            for key in self.pretrained_state:
                if f"blocks.{i}." in key:
                    new_key = key.replace(f"blocks.{i}.", f"blocks.{i}.")
                    self.target_state[new_key] = self.pretrained_state[key]

        # progressive init new layers
        for i in range(self.pretrained_layers, self.target_layers):
            prev_layer = i - 1
            for key in self.target_state:
                if f"blocks.{i}." in key:
                    prev_key = key.replace(f"blocks.{i}.", f"blocks.{prev_layer}.")
                    # add random noise
                    noise = torch.randn_like(self.target_state[prev_key]) * noise_scale
                    self.target_state[key] = self.target_state[prev_key] + noise

        self.target_model.load_state_dict(self.target_state)
        return self.target_model

    def init_interpolation(self):
        """interpolation init strategy"""
        self._copy_non_transformer_params()

        # copy pretrained layers
        for i in range(self.pretrained_layers):
            for key in self.pretrained_state:
                if f"blocks.{i}." in key:
                    new_key = key.replace(f"blocks.{i}.", f"blocks.{i}.")
                    self.target_state[new_key] = self.pretrained_state[key]

        # interpolate new layers
        for i in range(self.pretrained_layers, self.target_layers):
            # find the nearest two pretrained layers
            lower_idx = (i * self.pretrained_layers) // self.target_layers
            upper_idx = min(lower_idx + 1, self.pretrained_layers - 1)
            alpha = (i * self.pretrained_layers) / self.target_layers - lower_idx

            for key in self.target_state:
                if f"blocks.{i}." in key:
                    lower_key = key.replace(f"blocks.{i}.", f"blocks.{lower_idx}.")
                    upper_key = key.replace(f"blocks.{i}.", f"blocks.{upper_idx}.")
                    # linear interpolation
                    lower_value = self.pretrained_state[lower_key]
                    upper_value = self.pretrained_state[upper_key]
                    self.target_state[key] = (1 - alpha) * lower_value + alpha * upper_value

        self.target_model.load_state_dict(self.target_state)
        return self.target_model

    def init_constant(self, scale_spec=0.0, scale_others=0.02):
        """partial random init strategy: keep the first N layers, random init the remaining layers"""
        self._copy_non_transformer_params()

        # copy pretrained layers
        for i in range(self.pretrained_layers):
            self.logger.info(f"*********copy pretrained layer {i}")
            for key in self.pretrained_state:
                if f"blocks.{i}." in key:
                    new_key = key.replace(f"blocks.{i}.", f"blocks.{i}.")
                    self.target_state[new_key] = self.pretrained_state[key]

        # iterate new layers
        for i in range(self.pretrained_layers, self.target_layers):
            self.logger.info(f"*********init new layer {i}")
            # iterate all params in the current layer
            for key in self.target_state:
                # only process the params in the current layer
                if f"blocks.{i}." in key:
                    # initialize specific weight params (cross attention, self attention, point-wise conv) to 0
                    if any(
                        x in key for x in ["cross_attn.proj.weight", "attn.proj.weight", "mlp.point_conv.conv.weight"]
                    ):
                        self.target_state[key] = torch.randn_like(self.target_state[key]) * scale_spec  # *0
                    elif "q_norm.weight" in key or "k_norm.weight" in key:
                        self.target_state[key] = torch.ones_like(self.target_state[key])
                    elif (
                        "attn.proj.bias" in key
                        or "cross_attn.q_linear.bias" in key
                        or "cross_attn.kv_linear.bias" in key
                        or "cross_attn.proj.bias" in key
                    ):
                        self.target_state[key] = torch.zeros_like(self.target_state[key])
                    elif "mlp.depth_conv.conv.weight" in key or "mlp.depth_conv.conv.bias" in key:
                        self.target_state[key] = torch.randn_like(self.target_state[key]) * 0.2
                    # initialize other params with smaller std (0.02)
                    else:
                        self.target_state[key] = torch.randn_like(self.target_state[key]) * scale_others  # *0.02

        self.target_model.load_state_dict(self.target_state)
        return self.target_model

    def init_random(self):
        """partial random init strategy: keep the first N layers, random init the remaining layers"""
        self._copy_non_transformer_params()

        # copy pretrained layers
        for i in range(self.pretrained_layers):
            for key in self.pretrained_state:
                if f"blocks.{i}." in key:
                    new_key = key.replace(f"blocks.{i}.", f"blocks.{i}.")
                    self.target_state[new_key] = self.pretrained_state[key]

        # keep the remaining layers random init (do not process, use the model original init)
        self.target_model.load_state_dict(self.target_state)
        return self.target_model

    def init_block_expand(self, expand_ratio=3, zero_gate=False):
        """block expand strategy: each layer is expanded to continuous multiple layers

        Args:
            expand_ratio: expand ratio, default is 3, i.e., each layer is expanded to continuous 3 layers
            zero_gate: whether to initialize the gate params of the subsequent layers in the expanded group to 0

        For example: for 20 layers to 60 layers, when zero_gate=True:
        Original model layer 0 → New model layer 0 (keep original), layer 1-2 (zero-gate)
        Original model layer 1 → New model layer 3 (keep original), layer 4-5 (zero-gate)
        Original model layer 2 → New model layer 6 (keep original), layer 7-8 (zero-gate)
        And so on.
        """
        assert (
            self.target_layers == self.pretrained_layers * expand_ratio
        ), f"target layers({self.target_layers}) must be {expand_ratio} times of source layers({self.pretrained_layers})"

        self._copy_non_transformer_params()

        # expand each layer
        for src_layer_idx in range(self.pretrained_layers):
            # calculate the start index of the target model
            target_start_idx = src_layer_idx * expand_ratio

            # copy the params of the source layer to the continuous expand_ratio layers
            for offset in range(expand_ratio):
                target_layer_idx = target_start_idx + offset

                for key in self.pretrained_state:
                    if f"blocks.{src_layer_idx}." in key:
                        new_key = key.replace(f"blocks.{src_layer_idx}.", f"blocks.{target_layer_idx}.")

                        # only set zero-gate for the subsequent layers in the expanded group (offset > 0)
                        if zero_gate and offset > 0:
                            if "scale_shift_table" in new_key:
                                # set the entire scale_shift_table to 0
                                self.target_state[new_key] = torch.zeros_like(self.pretrained_state[key])
                                self.logger.info(f"zero init entire scale_shift_table: {new_key}")
                            elif "cross_attn.proj.weight" in new_key or "cross_attn.proj.bias" in new_key:
                                # set the output projection layer of cross attention to 0
                                self.target_state[new_key] = torch.zeros_like(self.pretrained_state[key])
                                self.logger.info(f"zero init cross attn proj: {new_key}")
                            elif "attn.proj.weight" in new_key or "attn.proj.bias" in new_key:
                                # set the output projection layer of self attention to 0
                                self.target_state[new_key] = torch.zeros_like(self.pretrained_state[key])
                                self.logger.info(f"zero init self attn proj: {new_key}")
                            elif "mlp.point_conv.conv.weight" in new_key or "mlp.point_conv.conv.bias" in new_key:
                                # set the last point_conv of mlp to 0
                                self.target_state[new_key] = torch.zeros_like(self.pretrained_state[key])
                                self.logger.info(f"zero init mlp point conv: {new_key}")
                            else:
                                # other params copy normally
                                self.target_state[new_key] = self.pretrained_state[key]
                                self.logger.info(f"copy transformer params: {key} -> {new_key}")
                        else:
                            # original layer or not using zero_gate
                            self.target_state[new_key] = self.pretrained_state[key]
                            self.logger.info(f"copy transformer params: {key} -> {new_key}")

        self.target_model.load_state_dict(self.target_state)
        return self.target_model
