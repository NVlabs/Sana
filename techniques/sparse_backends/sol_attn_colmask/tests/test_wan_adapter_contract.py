import pytest
import torch

from integrations.wan import adapter


def test_qkv_contract_rejects_wrong_block_size_before_cuda():
    tensor = torch.empty((1, 1, 64, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="block_size=64"):
        adapter._validate_qkv(tensor, tensor, tensor, 32)


def test_qkv_contract_rejects_wrong_head_dim_before_cuda():
    tensor = torch.empty((1, 1, 64, 64), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head_dim=128"):
        adapter._validate_qkv(tensor, tensor, tensor, 64)


def test_qkv_contract_rejects_non_bf16_before_cuda():
    tensor = torch.empty((1, 1, 64, 128), dtype=torch.float32)
    with pytest.raises(TypeError, match="BF16"):
        adapter._validate_qkv(tensor, tensor, tensor, 64)
