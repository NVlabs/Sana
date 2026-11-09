"""LMS Sharp v1.0 (Alissonerdx/Minimax-H3-ComfyUI, loras/minimax_h3_lms_v1.0_r64) merged into the Ref2VA base weights
(after HyperFlow, before the IC LoRA), with its 50 AdaLN curve branches driven by the [1025, 8] adaln_t_table.

Conversion follows the SoL-Refiner-H3 release (h3-release-5000-hyperflow-20261019-v1/package/runtime/adapters.py):
ComfyUI keys -> Diffusers module names, fused attn.qkv_proj split into to_q/to_k/to_v, mlp.fc1 rows reordered from
[gate, value] to [value, gate] for ff.net.0.proj, and the 50 AdaLN branches whose input is the 8-D curve kept as extra
modules (CurveLinear), their curve interpolated from the published [1025, 8] adaln_t_table at the model timestep.
Linear deltas are added as W += strength * B @ A (no alpha factor, as in the release), in fp32 on `device`.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def convert(path):
    raw = load_file(str(path), device='cpu')
    roots = sorted(k[:-len('.lora_A.weight')] for k in raw if k.endswith('.lora_A.weight'))
    assert set(raw) == {r + s for r in roots for s in ('.lora_A.weight', '.lora_B.weight')}
    linears, curves = {}, {}
    for src in roots:
        a, b = raw[src + '.lora_A.weight'].float(), raw[src + '.lora_B.weight'].float()
        name = src.removeprefix('diffusion_model.').replace('token_refiner.blocks.', 'token_refiner.refiner_blocks.')
        if name.startswith('blocks.'):
            name = 'transformer_blocks.' + name[len('blocks.'):]
        name = name.replace('final_layer.adaln_proj.linear', 'norm_out.linear').replace('.attn.out_proj', '.attn.to_out.0')
        if name.endswith('.attn.qkv_proj'):
            for tag, part in zip(('to_q', 'to_k', 'to_v'), b.chunk(3, dim=0)):
                linears[name[:-len('qkv_proj')] + tag] = (a, part.contiguous())
        elif name.endswith('.mlp.fc1'):
            gate, value = b.chunk(2, dim=0)
            linears[name[:-len('.mlp.fc1')] + '.ff.net.0.proj'] = (a, torch.cat([value, gate], dim=0))
        elif name.endswith('.mlp.fc2'):
            linears[name[:-len('.mlp.fc2')] + '.ff.net.2'] = (a, b)
        elif '.adaln_proj.linear' in name and a.shape[1] == 8:
            curves[name] = (a, b)
        else:
            linears[name] = (a, b)
    return linears, curves


class CurveContext:
    def __init__(self, path, strength):
        self.table = torch.frombuffer(bytearray(Path(path).read_bytes()), dtype=torch.float32).reshape(1025, 8).clone()
        self.strength = float(strength)
        self.coords = None

    def pre(self, module, args, kwargs):
        t = kwargs['timestep'].float()
        assert t.min() >= 0 and t.max() <= 1
        table = self.table.to(t.device)
        pos = t * (len(table) - 1)
        lo = pos.floor().long().clamp(max=len(table) - 2)
        self.coords = torch.lerp(table[lo], table[lo + 1], (pos - lo).unsqueeze(-1))


class CurveLinear(torch.nn.Module):
    """AdaLN projection plus the Sharp curve branch: base(x) + strength * B(A(curve(t)))."""

    def __init__(self, base, a, b, context):
        super().__init__()
        self.base, self.context = base, context
        self.register_buffer('curve_a', a)
        self.register_buffer('curve_b', b)
        assert a.shape[1] == 8 and b.shape == (base.out_features, a.shape[0])

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        out = self.base(x.to(self.base.weight.dtype))
        c = self.context.coords
        assert c is not None and c.shape[0] == x.shape[0], (None if c is None else c.shape, x.shape)
        delta = F.linear(F.linear(c.to(self.curve_a.device), self.curve_a), self.curve_b)
        return out + delta.to(out.dtype) * self.context.strength


def install(model, sharp_path, curve_path, strength, device='cpu'):
    linears, curves = convert(sharp_path)
    assert len(curves) == 50, len(curves)
    context = CurveContext(curve_path, strength)
    with torch.no_grad():
        for name, (a, b) in linears.items():
            layer = model.get_submodule(name)
            w = layer.weight
            assert tuple(w.shape) == (b.shape[0], a.shape[1]), (name, tuple(w.shape), tuple(b.shape), tuple(a.shape))
            w.copy_((w.to(device).float() + strength * (b.to(device) @ a.to(device))).to(w.dtype).to(w.device))
    for name, (a, b) in curves.items():
        parent, key = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), key, CurveLinear(model.get_submodule(name), a, b, context))
    model.register_forward_pre_hook(context.pre, with_kwargs=True)
    return dict(linears=len(linears), curves=len(curves), strength=float(strength))
