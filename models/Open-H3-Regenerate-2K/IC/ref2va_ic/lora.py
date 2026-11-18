"""The IC LoRA on the Ref2VA transformer, applied unmerged: y = W x + (alpha / rank) * B (A x), with
A and B (FP32 masters) cast to the activation dtype.  alpha = rank = 384, so the scale is 1.

File layout (as Open-H3-Regenerate-2K): "transformer.<module>.lora_A.weight" [384, in] and ".lora_B.weight"
[out, 384], FP32, 312 linears (attention q/k/v/out and MLP of the 2 token-refiner and 50 transformer blocks).
Merging B @ A into the weights in FP32 and casting once gives the same function up to the weight dtype's rounding.
"""
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

A, B = '.lora_A.weight', '.lora_B.weight'


class LoRALinear(nn.Module):
    def __init__(self, base, a, b, scale):
        super().__init__()
        self.base, self.scale = base, float(scale)
        # parameters (not buffers), so FSDP shards them and gathers them in its compute dtype
        self.lora_A = nn.Parameter(a, requires_grad=False)
        self.lora_B = nn.Parameter(b, requires_grad=False)

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        out = self.base(x)
        update = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return out.add_(update.to(out.dtype), alpha=self.scale)


def install(model, path, alpha=384.):
    """Wrap every LoRA target linear of `model` (the Ref2VA transformer).  Returns the number of linears."""
    state = load_file(str(path), device='cpu')
    roots = sorted(k[:-len(A)] for k in state if k.endswith(A))
    assert len(roots) == 312 and set(state) == {r + s for r in roots for s in (A, B)}, len(roots)
    for root in roots:
        name = root[len('transformer.'):]
        parent_name, child = name.rsplit('.', 1)
        parent = model.get_submodule(parent_name)
        base = parent[int(child)] if child.isdigit() else getattr(parent, child)
        a, b = state[root + A].float(), state[root + B].float()
        assert isinstance(base, nn.Linear) and a.shape == (a.shape[0], base.in_features) \
            and b.shape == (base.out_features, a.shape[0]), name
        wrapped = LoRALinear(base, a.to(base.weight.device), b.to(base.weight.device), alpha / a.shape[0])
        if child.isdigit():
            parent[int(child)] = wrapped
        else:
            setattr(parent, child, wrapped)
    return len(roots)
