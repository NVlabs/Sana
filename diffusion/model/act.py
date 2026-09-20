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

import copy

import torch.nn as nn

__all__ = ["build_act", "get_act_name"]

# register activation function here
#   name: module, kwargs with default values
REGISTERED_ACT_DICT: dict[str, tuple[type, dict[str, any]]] = {
    "relu": (nn.ReLU, {"inplace": True}),
    "relu6": (nn.ReLU6, {"inplace": True}),
    "hswish": (nn.Hardswish, {"inplace": True}),
    "hsigmoid": (nn.Hardsigmoid, {"inplace": True}),
    "swish": (nn.SiLU, {"inplace": True}),
    "silu": (nn.SiLU, {"inplace": True}),
    "tanh": (nn.Tanh, {}),
    "sigmoid": (nn.Sigmoid, {}),
    "gelu": (nn.GELU, {"approximate": "tanh"}),
    "mish": (nn.Mish, {"inplace": True}),
    "identity": (nn.Identity, {}),
}


def build_act(name: str or None, **kwargs) -> nn.Module or None:
    if name in REGISTERED_ACT_DICT:
        act_cls, default_args = copy.deepcopy(REGISTERED_ACT_DICT[name])
        for key in default_args:
            if key in kwargs:
                default_args[key] = kwargs[key]
        return act_cls(**default_args)
    elif name is None or name.lower() == "none":
        return None
    else:
        raise ValueError(f"do not support: {name}")


def get_act_name(act: nn.Module or None) -> str or None:
    if act is None:
        return None
    module2name = {}
    for key, config in REGISTERED_ACT_DICT.items():
        module2name[config[0].__name__] = key
    return module2name.get(type(act).__name__, "unknown")
