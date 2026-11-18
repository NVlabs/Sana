"""Open-H3-Regenerate-2K adapter: one merged LoRA plus the Sharp curve AdaLN branches.

`lora/open_h3_regenerate_2k_lora.safetensors` is the exact rank-concatenation of the refiner (step5000 EMA),
LightX2V FL2VA Turbo v1.2 (x1.0, alpha/rank 8/128 folded) and LMS Sharp v1.0 (x1.0): for every targeted linear,
B @ A equals the sum of the three scaled deltas, so it is applied as a plain LoRA with scale 1.

LMS Sharp also adds 50 curve branches to the AdaLN projections. Their input is an 8-D curve interpolated from the
timestep (`adaln_t_table`, 1025x8, fetched separately by download.py), so they are kept as extra modules instead of
being folded into weights.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

LORA_FILE = "lora/open_h3_regenerate_2k_lora.safetensors"
CURVE_FILE = "lora/sharp_curve_branches.safetensors"
CURVE_TABLE = "lora/adaln_t_table.bin"


def load_lora(path):
    """{module name: (A fp32 [rank, in], B fp32 [out, rank])} for the transformer, scale 1."""
    raw = load_file(str(path), device="cpu")
    pairs = {}
    for key, value in raw.items():
        if key.endswith(".lora_A.weight"):
            name = key[len("transformer."):-len(".lora_A.weight")]
            pairs[name] = (value.float(), raw[key[:-len(".lora_A.weight")] + ".lora_B.weight"].float())
    if len(pairs) * 2 != len(raw):
        raise ValueError(f"{path}: expected only lora_A / lora_B pairs")
    return pairs


class CurveContext:
    """Interpolates the Sharp curve coordinates from the model timestep once per forward."""

    def __init__(self, table_path):
        self.table = torch.frombuffer(bytearray(Path(table_path).read_bytes()), dtype=torch.float32).reshape(1025, 8).clone()
        self.coords = None
        self.strength = 0.0

    def pre(self, module, args, kwargs):
        t = kwargs["timestep"].float()
        assert t.min() >= 0 and t.max() <= 1
        table = self.table.to(t.device)
        pos = t * (len(table) - 1)
        lo = pos.floor().long().clamp(max=len(table) - 2)
        self.coords = torch.lerp(table[lo], table[lo + 1], (pos - lo).unsqueeze(-1))


class CurveLinear(torch.nn.Module):
    """AdaLN projection plus the Sharp curve branch: base(x) + strength * B(A(curve(t)))."""

    def __init__(self, base, a, b, context):
        super().__init__()
        self.base = base
        self.register_buffer("curve_a", a)
        self.register_buffer("curve_b", b)
        self.context = context
        assert a.shape[1] == 8 and b.shape == (base.out_features, a.shape[0])

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        out = self.base(x.to(self.base.weight.dtype))
        if self.context.strength:
            c = self.context.coords
            assert c is not None and c.shape[0] == x.shape[0]
            delta = F.linear(F.linear(c, self.curve_a), self.curve_b)
            out = out + delta.to(out.dtype) * self.context.strength
        return out


def attach(model, weights_dir):
    """Validate the merged LoRA against the model, install the Sharp curve branches (strength 0 until a stage enables
    them) and return (lora pairs, curve context)."""
    weights_dir = Path(weights_dir)
    pairs = load_lora(weights_dir / LORA_FILE)
    for name, (a, b) in pairs.items():
        layer = model.get_submodule(name)
        if tuple(layer.weight.shape) != (b.shape[0], a.shape[1]) or a.shape[0] != b.shape[1]:
            raise ValueError(f"LoRA does not match the transformer: {name}")
    curves = load_file(str(weights_dir / CURVE_FILE), device="cpu")
    context = CurveContext(weights_dir / CURVE_TABLE)
    roots = sorted(k[len("transformer."):-len(".curve_A.weight")] for k in curves if k.endswith(".curve_A.weight"))
    if len(roots) != 50:
        raise ValueError("All 50 Sharp curve AdaLN branches are required")
    for name in roots:
        a = curves[f"transformer.{name}.curve_A.weight"].float()
        b = curves[f"transformer.{name}.curve_B.weight"].float()
        parent, key = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), key, CurveLinear(model.get_submodule(name), a, b, context))
    model.register_forward_pre_hook(context.pre, with_kwargs=True)
    return pairs, context


class StageMerger:
    """Physical LoRA merge per stage onto one resident model.

    Every targeted weight keeps a pristine copy in pinned host memory; switching to a stage rebuilds it as
    cast(W_fp32 + sum scale * B @ A) in 512-row chunks. With `resident=True` (sol_engine_fwd) the merged weights of
    each stage are cached on the GPU after the first switch, so later switches are device copies / pointer swaps.
    Entries are keyed by weight identity, so aliases (e.g. CurveLinear.weight) resolve to one entry."""

    def __init__(self, model, stages, device, resident=False):
        self.targets = {}
        self.resident, self.resident_done = resident, set()
        for stage, items in stages.items():
            for adapters, scale in items:
                if not scale:
                    continue
                for name, (a, b) in adapters.items():
                    weight = model.get_submodule(name).weight
                    entry = self.targets.setdefault(id(weight), {"weight": weight, "stages": {}, "state": "base"})
                    entry["stages"].setdefault(stage, []).append(
                        (a.to(device, torch.float32), b.to(device, torch.float32), float(scale)))
        for entry in self.targets.values():
            entry["host"] = entry["weight"].detach().to("cpu").pin_memory()
        torch.cuda.empty_cache()

    def attach_quantized(self, layers):
        """layers: MXLinear modules. Their BF16 weights only exist while merging (merge -> refresh -> release)."""
        by_weight = {id(layer.base.weight): layer for layer in layers}
        for key, entry in self.targets.items():
            entry["mx"] = by_weight.get(key)
        for layer in layers:
            if id(layer.base.weight) in self.targets:
                layer.refresh()
                _release(layer.base.weight)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def merge(self, stage):
        if self.resident and stage in self.resident_done:
            for entry in self.targets.values():
                if entry["state"] == stage:
                    continue
                if entry.get("mx") is not None:
                    entry["mx"].activate(stage)
                else:
                    entry["weight"].copy_(entry["cache"][stage])
                entry["state"] = stage
            return
        self._merge(stage)
        if self.resident:
            for entry in self.targets.values():
                if entry.get("mx") is None:
                    entry.setdefault("cache", {})[stage] = entry["weight"].detach().clone()
            self.resident_done.add(stage)

    def _merge(self, stage):
        for entry in self.targets.values():
            if entry["state"] == stage:
                continue
            weight, active, mx = entry["weight"], entry["stages"].get(stage), entry.get("mx")
            if mx is not None and weight.data.numel() == 0:
                weight.data = torch.empty(entry["host"].shape, dtype=entry["host"].dtype, device=weight.device)
            if not active:
                if entry["state"] != "base" or mx is not None:
                    weight.copy_(entry["host"], non_blocking=True)
                entry["state"] = "base" if mx is None else stage
                if mx is not None:
                    mx.refresh(stage if self.resident else None)
                    _release(weight)
                continue
            source = entry["host"].to(weight.device, non_blocking=True)
            for offset in range(0, len(weight), 512):
                n = min(512, len(weight) - offset)
                base = source[offset:offset + n].float()
                for a, b, scale in active:
                    base.addmm_(b[offset:offset + n], a, beta=1., alpha=scale)
                weight[offset:offset + n].copy_(base.to(weight.dtype))
            entry["state"] = stage
            if mx is not None:
                mx.refresh(stage if self.resident else None)
                _release(weight)


def _release(weight):
    """Drop a parameter's device storage, keeping the Parameter object (and its id) alive."""
    weight.data = weight.data.new_empty(0)
