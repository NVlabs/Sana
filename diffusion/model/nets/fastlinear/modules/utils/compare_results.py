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

import torch


def compare_results(name: str, result: torch.Tensor, ref_result: torch.Tensor):
    print(f"comparing {name}")
    diff = (result - ref_result).abs().view(-1)
    max_error_pos = diff.argmax()
    print(f"max error: {diff.max()}, mean error: {diff.mean()}")
    print(f"max error pos: {result.view(-1)[max_error_pos]} {ref_result.view(-1)[max_error_pos]}")
