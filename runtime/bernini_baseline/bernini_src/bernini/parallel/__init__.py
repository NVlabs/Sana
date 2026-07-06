# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
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

"""Data-parallel + Ulysses sequence-parallel layer.

Single-GPU inference works with no extra dependency; multi-GPU sequence
parallel delegates communication to Open-VeOmni.
"""

from .ops import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    gen_cu_seqlens_for_cross_attn,
    pad_tensor,
    padding_tensor_for_seqeunce_parallel,
    slice_input_tensor,
    slice_input_tensor_scale_grad,
    unpad_tensor,
)
from .state import ParallelState, get_parallel_state, init_parallel_state

__all__ = [
    "ParallelState",
    "get_parallel_state",
    "init_parallel_state",
    "gather_heads_scatter_seq",
    "gather_outputs",
    "gather_seq_scatter_heads",
    "gen_cu_seqlens_for_cross_attn",
    "pad_tensor",
    "padding_tensor_for_seqeunce_parallel",
    "slice_input_tensor",
    "slice_input_tensor_scale_grad",
    "unpad_tensor",
]
