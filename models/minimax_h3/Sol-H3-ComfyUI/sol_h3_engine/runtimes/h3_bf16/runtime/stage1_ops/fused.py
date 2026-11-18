"""BSHD selected attention and a fused merge preserving BF16 product rounding."""
import ast
import functools
import hashlib
import inspect
import os
from pathlib import Path
import textwrap

NATIVE_SHA = "b7c6d4e470ee232b50d16d71c2b21806f414f9baa4447434b2da66c94dc0fab9"


@functools.lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def merge(Out, Gate, Compressed, Result, N: tl.constexpr, SEQ: tl.constexpr,
              HEADS: tl.constexpr, DIM: tl.constexpr, TILE: tl.constexpr,
              CB: tl.constexpr, CT: tl.constexpr, CH: tl.constexpr, CD: tl.constexpr,
              BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = idx < N
        dim = idx % DIM
        head = idx // DIM % HEADS
        row = idx // (DIM * HEADS) % SEQ
        batch = idx // (DIM * HEADS * SEQ)
        c_idx = batch * CB + (row // TILE) * CT + head * CH + dim * CD
        a = tl.load(Out + idx, valid, 0).to(tl.float32)
        gate = tl.load(Gate + idx, valid, 0).to(tl.float32)
        c = tl.load(Compressed + c_idx, valid, 0).to(tl.float32)
        # Eager PyTorch materializes BF16 multiply BEFORE the BF16 addition.
        # Preserve that rounding boundary; do not replace it by FP32 FMA.
        product = (c * gate).to(tl.bfloat16).to(tl.float32)
        result = a + product
        tl.store(Result + idx, result.to(tl.bfloat16), valid)
    return merge


def fused_gate_merge(out, gate, compressed, tile_elems=64):
    import torch
    import triton
    if (out.dtype != torch.bfloat16 or gate.dtype != out.dtype or compressed.dtype != out.dtype
            or out.shape != gate.shape or out.ndim != 4 or not out.is_contiguous()
            or not gate.is_contiguous() or out.device != gate.device or out.device != compressed.device
            or out.device.type != "cuda" or out.shape[1] % tile_elems
            or compressed.shape != (out.shape[0], out.shape[1] // tile_elems, out.shape[2], out.shape[3])):
        raise ValueError("fused merge requires contiguous CUDA BSHD BF16 out/gate and matching tiled compression")
    result = torch.empty_like(out)
    _kernel()[(triton.cdiv(out.numel(), 1024),)](
        out, gate, compressed, result, out.numel(), out.shape[1], out.shape[2], out.shape[3],
        tile_elems, *compressed.stride(), BLOCK=1024, num_warps=4, enable_fp_fusion=False)
    return result


def optimized_tree(source, *, bshd, fused_merge):
    """Replace only four layout assignments and/or final BF16 merge statement."""
    counts = {"layout_qkv": 0, "layout_output": 0, "gate_merge": 0}
    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                return node
            name, value = node.targets[0].id, ast.unparse(node.value)
            if bshd and name in ("q_bhsd", "k_bhsd", "v_bhsd") and ".transpose(1, 2)" in value:
                original = {"q_bhsd": "query", "k_bhsd": "key", "v_bhsd": "value"}[name]
                node.value = ast.Name(original, ast.Load())
                counts["layout_qkv"] += 1
            elif bshd and name == "out" and "out_bhsd.transpose(1, 2)" in value:
                node.value = ast.Name("out_bhsd", ast.Load())
                counts["layout_output"] += 1
            elif fused_merge and name == "out" and "out_tiled +" in value:
                node.value = ast.Call(ast.Name("_optimized_merge", ast.Load()),
                                      [ast.Name(x, ast.Load()) for x in ("out", "logical_gate", "out_c", "tile_elems")], [])
                counts["gate_merge"] += 1
            return node
    tree = Rewrite().visit(ast.parse(textwrap.dedent(source)))
    ast.fix_missing_locations(tree)
    expected = {"layout_qkv": 3 if bshd else 0, "layout_output": 1 if bshd else 0,
                "gate_merge": 1 if fused_merge else 0}
    if counts != expected:
        raise RuntimeError("native optimization seam drift: " + str(counts))
    return tree


def build_forward(native, selected, *, bshd=True, fused_merge=True):
    """Return an unbound optimized forward; caller supplies the SAME index policy.

    selected signature remains (q,k,v,mask,block_sizes)->(out,lse), but uses
    BSHD input/output iff bshd=True. No globals or installed native files change.
    Tile64/no-grad SM121 accepts odd/even logical tiles, never an SM100 partner.
    """
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != NATIVE_SHA:
        raise RuntimeError("optimized requires verified native H3 VSA source")
    if os.environ.get("FASTVIDEO_VSA_SM100A", "0") != "0":
        raise RuntimeError("SM100A branch is outside this SM121 optimized")
    namespace = dict(native.__dict__, _optimized_merge=fused_gate_merge,
                     block_sparse_attn_64_bhsd=selected)
    source = inspect.getsource(native.MiniMaxH3VSAImpl.forward)
    exec(compile(optimized_tree(source, bshd=bshd, fused_merge=fused_merge),
                 native.__file__ + ":bandwidth_optimized", "exec"), namespace)
    return namespace["forward"]
