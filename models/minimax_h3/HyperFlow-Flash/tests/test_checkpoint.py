"""checkpoint.py finds the model code by feature in a folder shaped like the public checkpoint (fake code, no weights)."""
import json

import pytest
import torch

from sol_hyperflow_flash import checkpoint as CK

INT8 = '''
KERNEL_OPTIONS = {"int8_gemm": "auto", "rotate_quant": "dense"}
def set_kernel_options(*, int8_gemm=None, rotate_quant=None):
    if int8_gemm is not None:
        KERNEL_OPTIONS["int8_gemm"] = int8_gemm
    if rotate_quant is not None:
        KERNEL_OPTIONS["rotate_quant"] = rotate_quant
def resolve_int8_gemm(device):
    return "torch"
class Int8RotLinear:
    pass
'''
TRANSFORMER = '''
from .model_int8 import Int8RotLinear
CALLS = []
class DroppedBlock:
    pass
class Block:
    pass
class FakeTransformer:
    def __init__(self):
        self.transformer_blocks = [Block(), DroppedBlock(), Block()]
    @classmethod
    def from_pretrained(cls, path, torch_dtype=None):
        CALLS.append(("from_pretrained", path, torch_dtype))
        return cls()
    def eval(self):
        return self
def set_lowrank_rank_multiple(model, multiple):
    CALLS.append(("rank_multiple", multiple))
    return 3
def set_ff_chunk(model, tokens):
    CALLS.append(("ff_chunk", tokens))
'''


def _folder(tmp_path, sub=None):
    root = tmp_path / "ckpt"
    part = root / (sub or "")
    part.mkdir(parents=True)
    (part / "config.json").write_text(json.dumps({"_class_name": "FakeTransformer",
                                                  "auto_map": {"AutoModel": "model_transformer.FakeTransformer"}}))
    (part / "model_transformer.py").write_text(TRANSFORMER)
    (part / "model_int8.py").write_text(INT8)
    return root, part


def test_resolve_local_root_and_partition(tmp_path):
    root, part = _folder(tmp_path, CK.PARTITIONS["ref2va"])
    assert CK.resolve(str(root), "ref2va") == part
    assert CK.resolve(str(part), "ref2va") == part
    with pytest.raises(ValueError):
        CK.resolve(str(root), "fl2va")
    with pytest.raises(FileNotFoundError):
        CK.resolve(str(root), "t2va")


def test_load_transformer_by_feature(tmp_path):
    _, part = _folder(tmp_path, CK.PARTITIONS["t2va"])
    loaded = CK.load_transformer(part, torch)
    tmod = loaded.module
    assert ("from_pretrained", str(part), torch.bfloat16) in tmod.CALLS
    assert ("rank_multiple", 64) in tmod.CALLS and ("ff_chunk", 0) in tmod.CALLS
    assert loaded.int8.KERNEL_OPTIONS == {"int8_gemm": "auto", "rotate_quant": "dense"}
    assert sum(isinstance(b, loaded.dropped_cls) for b in loaded.model.transformer_blocks) == 1
    assert loaded.info["transformer_class"] == "FakeTransformer" and loaded.info["repadded_linears"] == 3


def test_pins():
    assert CK.HF_REPO == "VideoRebirthAI/hyperflow-h3-flash"
    assert len(CK.HF_REVISION) == 40 and int(CK.HF_REVISION, 16) >= 0
    assert CK.PARTITIONS == {"t2va": "fl2va_int8_convrot", "ref2va": "ref2va_int8_convrot"}
