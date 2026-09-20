# Copyright 2024 MIT Han Lab
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

import os
import warnings
from typing import Tuple

import torch


def export_onnx(
    model: torch.nn.Module,
    input_shape: Tuple[int],
    export_path: str,
    opset: int,
    export_dtype: torch.dtype,
    export_device: torch.device,
) -> None:
    model.eval()

    dummy_input = {"x": torch.randn(input_shape, dtype=export_dtype, device=export_device)}
    dynamic_axes = {
        "x": {0: "batch_size"},
    }

    # _ = model(**dummy_input)

    output_names = ["image_embeddings"]

    export_dir = os.path.dirname(export_path)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        print(f"Exporting onnx model to {export_path}...")
        with open(export_path, "wb") as f:
            torch.onnx.export(
                model,
                tuple(dummy_input.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_input.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
