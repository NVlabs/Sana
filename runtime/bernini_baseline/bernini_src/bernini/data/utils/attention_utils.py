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
import torch

def build_custom_attention_mask(token_type, token_segment_ids):
    """
    Build a custom attention mask.
    
    Args:
        token_type (torch.Tensor): Shape (B, L), with values 0(t), 1(p), 2(i), 3(o).
        token_segment_ids (torch.Tensor): Shape (B, L), with segment ids k (for example, 1 in t1).
        
    Returns:
        torch.Tensor: Shape (B, L, L), where visible positions are 0.0 and invisible positions are -inf.
    """
    B, L = token_type.shape
    device = token_type.device
    
    # 1. Expand dimensions to build the (B, L, L) matrix via broadcasting.
    # q_*: Query (row), shape (B, L, 1)
    # k_*: Key (column), shape (B, 1, L)
    q_type = token_type.unsqueeze(2)
    k_type = token_type.unsqueeze(1)
    q_id = token_segment_ids.unsqueeze(2)
    k_id = token_segment_ids.unsqueeze(1)
    
    # 2. Build the base boolean condition matrices.
    # Causal relation matrix (lower triangle is True).
    # Note: tril includes the diagonal by default (j <= i).
    causal_mask = torch.tril(torch.ones((L, L), device=device, dtype=torch.bool))
    # Expand to the batch dimension: (1, L, L) -> (B, L, L).
    # Broadcasting would handle this automatically; we unsqueeze explicitly for clarity.
    causal_mask = causal_mask.unsqueeze(0) 

    # Key type checks.
    k_is_ti = (k_type == 0) | (k_type == 2) # Key is t or i
    k_is_p  = (k_type == 1)                 # Key is p
    k_is_o  = (k_type == 3)                 # Key is o

    # Whether ids match (used for bidirectional attention).
    ids_match = (q_id == k_id)

    # 3. Define the visibility rules.
    # Shared rule: any Query can see previous t/i tokens.
    visible_base_ti = causal_mask & k_is_ti
    
    # Bidirectional rule for p: can see p with the same id.
    visible_p_bidirectional = k_is_p & ids_match
    
    # Bidirectional rule for o: can see o with the same id.
    visible_o_bidirectional = k_is_o & ids_match
    
    # 4. Combine the final boolean mask, where True means visible.
    # Initialize everything to False (fully invisible).
    final_bool_mask = torch.zeros((B, L, L), device=device, dtype=torch.bool)
    
    # Rule A: Query is t(0) or i(2), and can only see visible_base_ti.
    q_is_ti = (q_type == 0) | (q_type == 2)
    final_bool_mask = final_bool_mask | (q_is_ti & visible_base_ti)
    
    # Rule B: Query is p(1), so it can see visible_base_ti OR visible_p_bidirectional.
    q_is_p = (q_type == 1)
    final_bool_mask = final_bool_mask | (q_is_p & (visible_base_ti | visible_p_bidirectional))
    
    # Rule C: Query is o(3), so it can see visible_base_ti OR visible_o_bidirectional.
    q_is_o = (q_type == 3)
    final_bool_mask = final_bool_mask | (q_is_o & (visible_base_ti | visible_o_bidirectional))

    # 5. Convert to a float mask (0.0 / -inf).
    # The dtype here should usually follow the model precision
    # (float32 or float16/bfloat16).
    attention_mask = torch.zeros((B, L, L), device=device, dtype=torch.float32)
    attention_mask.masked_fill_(~final_bool_mask, float('-inf'))
    
    return attention_mask
