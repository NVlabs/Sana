# /// script
# requires-python = ">=3.10"
# dependencies = ["kernels[curated]>=0.16.0", "torch>=2.10"]
# ///

import os

import torch
import torch.nn.functional as F
from kernels import get_kernel


repo_id = os.environ.get(
    "SOL_ATTN_HF_REPO",
    "Efficient-Large-Model/sol-attn",
)
kernel = get_kernel(repo_id, version=1, trust_remote_code=True)

torch.manual_seed(42)
q = torch.randn(1, 256, 4, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)

expected = F.scaled_dot_product_attention(
    q.transpose(1, 2),
    k.transpose(1, 2),
    v.transpose(1, 2),
).transpose(1, 2)
actual = kernel.sol_attn(
    q,
    k,
    v,
    tau=1.0,
    thresh_type="exact",
    sink_start=0,
    sink_tokens=q.shape[1],
)
torch.testing.assert_close(actual, expected, atol=2e-2, rtol=3e-2)

sparse = kernel.sol_attn(
    q,
    k,
    v,
    tau=1.0,
    thresh_type="exact",
)
assert sparse.shape == q.shape
assert sparse.dtype == q.dtype
assert torch.isfinite(sparse).all()

print(
    {
        "repo_id": repo_id,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "max_abs_full_sink": (actual.float() - expected.float()).abs().max().item(),
    }
)
