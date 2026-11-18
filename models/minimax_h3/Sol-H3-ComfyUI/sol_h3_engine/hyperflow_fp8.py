"""HyperFlow-8 MiniMax-H3 DiT with the LoRA merged and the 300 block linears in FP8 (W8A8).

Loads the checkpoint written by scripts/hyperflow/convert_fp8.py: every adapted block linear holds
`W + B @ A` (exact merge, alpha == rank) quantised per tensor to float8_e4m3fn with an fp32 scale. The model is
built on the meta device and filled with `assign=True`, so host RAM holds about 31 GB for the DiT instead of 62 GB.
The two time embedders keep their small fp32 LoRA, merged here into `time_embedder` and its endpoint copy, and
HyperFlow's TwoTimeEmbedder (gate) and sigma grid are installed exactly as `load_hyperflow_lora` does.
"""
import json
from pathlib import Path

import torch
from torch import nn

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max


def _quantize_rows(x2):
    """Per-tensor dynamic e4m3 quantisation of the activation (amax → scale → cast)."""
    x_scale = (x2.abs().amax().float() / FP8_MAX).clamp(min=1e-12)
    return (x2 / x_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8), x_scale.reshape(1)


# One fused kernel pair instead of abs/amax/div/clamp/cast launched separately on every linear (300 per step).
# Set SOLH3_FP8_COMPILE=0 to run the eager version.
_quantize = _quantize_rows
if __import__("os").environ.get("SOLH3_FP8_COMPILE", "1") != "0":
    # emulate_precision_casts: bit-identical to eager (without it ~3% of the e4m3 values differ: x / s is computed
    # in fp32 instead of BF16).
    _quantize = torch.compile(_quantize_rows, dynamic=True, fullgraph=True, options={"emulate_precision_casts": True})


class Fp8Linear(nn.Module):
    """y = x W^T with W in e4m3 (per-tensor scale) and x quantised per tensor on the fly: torch._scaled_mm."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=FP8), requires_grad=False)
        self.scale = nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        shape = x.shape
        xq, x_scale = _quantize(x.reshape(-1, shape[-1]))
        y = torch._scaled_mm(xq, self.weight.t(), scale_a=x_scale, scale_b=self.scale, out_dtype=x.dtype)
        return y.reshape(*shape[:-1], self.out_features)


TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")


def build_fp8_transformer(model_dir, fp8_dir, lora_path, adaln_table=None):
    """Return (transformer, metadata) for the merged FP8 HyperFlow DiT, on the CPU."""
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3Transformer3DModel
    from safetensors import safe_open
    from hyperflow_h3 import TwoTimeEmbedder, read_metadata

    fp8_dir = Path(fp8_dir)
    config = MiniMaxH3Transformer3DModel.load_config(str(fp8_dir))
    with init_empty_weights():
        model = MiniMaxH3Transformer3DModel.from_config(config)
    swapped = 0
    for block in model.transformer_blocks:
        for path in TARGETS:
            parent_path, _, leaf = path.rpartition(".")
            parent = block.get_submodule(parent_path)
            old = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]
            new = Fp8Linear(old.in_features, old.out_features)
            if leaf.isdigit():
                parent[int(leaf)] = new
            else:
                setattr(parent, leaf, new)
            swapped += 1
    if swapped != 300:
        raise RuntimeError(f"expected 300 FP8 block linears, got {swapped}")
    table = None
    if adaln_table:
        # Exact precomputed modulation rows replace the 26 GB of adaln_proj weights (hyperflow_adaln.py).
        from hyperflow_adaln import AdaLNTable
        table = torch.load(adaln_table, map_location="cpu", weights_only=True)
        hidden = model.transformer_blocks[0].adaln_proj.hidden_size
        starts, offset = {}, 0
        for key in table["keys"]:
            starts[tuple(tuple(map(float, p)) for p in key)] = (offset, len(key))
            offset += len(key)
        for block, rows in zip(model.transformer_blocks, table["tables"]):
            block.adaln_proj = AdaLNTable(hidden, starts, rows)
    index = json.loads((fp8_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())
    state = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(str(fp8_dir / shard), "pt") as f:
            for key in f.keys():
                if table is not None and ".adaln_proj." in key:
                    continue
                state[key] = f.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    missing = [k for k in missing if not (table is not None and ".adaln_proj." in k)]
    if missing or unexpected:
        raise RuntimeError(f"FP8 checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    del state
    # Time embedders: the fp32 LoRA stays exact, merged into a base copy and an endpoint copy.
    metadata = read_metadata(Path(lora_path))
    scale = metadata.lora_alpha / metadata.lora_rank
    base = model.time_embedder
    wrapper = TwoTimeEmbedder.wrap(model, metadata.gate)
    with safe_open(str(lora_path), "pt") as f, torch.no_grad():
        for prefix, module in (("time_embedder", wrapper.base), ("endpoint_time_embedder", wrapper.endpoint)):
            for lin in ("linear_1", "linear_2"):
                layer = getattr(module, lin)
                a = f.get_tensor(f"transformer.{prefix}.{lin}.lora_A.weight").float()
                b = f.get_tensor(f"transformer.{prefix}.{lin}.lora_B.weight").float()
                layer.weight.copy_((layer.weight.float() + scale * (b @ a)).to(layer.weight.dtype))
    assert wrapper.base is base
    if table is not None:
        from hyperflow_adaln import route_rows
        route_rows(model)
    model.eval().requires_grad_(False)
    return model, metadata
