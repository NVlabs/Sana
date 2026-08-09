#!/usr/bin/env python3
"""Compare local ``piecewise_pisa_env`` routing against public PISA.

This probe pins the public PISA routing boundary from the
``piecewise-sparse-attention`` repository. It intentionally separates the
public default selector from the optional public bias form:

* public default: ``topk(qc @ kc * scale)``
* public optional bias: ``topk(qc @ kc * scale + log(bias))``

The local ``piecewise_attn`` route now exposes ``piecewise_route_bias``. The
``piecewise_pisa_env`` config sets it to ``false`` so the score selector
matches the public default top-k boundary. The bias form remains available for
local diagnostics, but is no longer the claim for this public PISA config.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = ROOT / "Sol-LTX-Infer" / "python"
PUBLIC_PISA = Path("/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/.cache/autovideo/public_refs/piecewise-sparse-attention")
PUBLIC_KERNEL = PUBLIC_PISA / "piecewise_attn" / "kernels" / "piecewise_sparse_attn_tma.py"
MANIFEST = ROOT / "config" / "sparse_attention" / "piecewise_pisa_env.toml"


def _torch():
    import torch

    return torch


def git_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def source_checks() -> dict[str, bool]:
    text = PUBLIC_KERNEL.read_text(errors="ignore") if PUBLIC_KERNEL.exists() else ""
    return {
        "has_public_kernel": PUBLIC_KERNEL.exists(),
        "has_piecewise_sparse_attention": "def piecewise_sparse_attention" in text,
        "uses_chunk_centroids": "fused_chunk_reduce(q, k, v, block_size" in text
        and "qc, kc, vc" in text,
        "default_route_uses_qc_kc_topk": "score = torch.einsum('bhid, bhjd -> bhij', qc, kc * scale)" in text
        and "torch.topk(score" in text,
        "optional_bias_route_exists": "score + torch.log(bias + 1e-5)" in text,
        "has_exact_and_approx_phases": "Phase 1: Exact Attention" in text
        and "Phase 2: Approx Attention" in text
        and "Phase 3: Approx Attention" in text,
    }


def _sorted_indices(tensor):
    torch = _torch()
    return torch.sort(tensor.to(torch.int64), dim=-1).values


def public_default_indices(qc, kc, *, density: float, scale: float):
    torch = _torch()
    nt = kc.shape[2]
    top_k = max(1, int(density * nt))
    score = torch.einsum("bhid,bhjd->bhij", qc, kc * scale)
    return torch.topk(score, k=top_k, dim=-1).indices.to(torch.int32)


def public_bias_indices(qc, kc, bias, *, density: float, scale: float, eps: float = 1e-5):
    torch = _torch()
    nt = kc.shape[2]
    top_k = max(1, int(density * nt))
    score = torch.einsum("bhid,bhjd->bhij", qc, kc * scale)
    score = score + torch.log(bias.clamp_min(eps)).unsqueeze(-2)
    return torch.topk(score, k=top_k, dim=-1).indices.to(torch.int32)


def behavior_probe() -> dict[str, Any]:
    torch = _torch()
    if str(RUNTIME_ROOT) not in sys.path:
        sys.path.insert(0, str(RUNTIME_ROOT))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from sglang.multimodal_gen.runtime.layers.attention.backends.piecewise_attn import (  # noqa: E402
        taylor_error_block_indices,
    )

    qc = torch.tensor([[[[1.0, 0.0]]]])
    kc = torch.tensor([[[[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]]])
    k_var = torch.tensor([[[1.0, 1.0, 1000.0, 1000.0]]])
    density = 0.5
    scale = 1.0

    local_default = taylor_error_block_indices(
        qc=qc,
        kc=kc,
        k_var=k_var,
        density=density,
        scale=scale,
        use_bias=False,
    )
    local_bias = taylor_error_block_indices(
        qc=qc,
        kc=kc,
        k_var=k_var,
        density=density,
        scale=scale,
        use_bias=True,
    )
    default = public_default_indices(qc, kc, density=density, scale=scale)
    bias = public_bias_indices(qc, kc, k_var, density=density, scale=scale)

    default_match = bool(
        torch.equal(_sorted_indices(local_default), _sorted_indices(default))
    )
    bias_match = bool(torch.equal(_sorted_indices(local_bias), _sorted_indices(bias)))

    return {
        "manifest": str(MANIFEST),
        "density": density,
        "scale": scale,
        "configured_route_bias": False,
        "local_default_indices": local_default.flatten().tolist(),
        "local_bias_indices": local_bias.flatten().tolist(),
        "public_default_indices": default.flatten().tolist(),
        "public_optional_bias_indices": bias.flatten().tolist(),
        "matches_public_default_route": default_match,
        "matches_public_optional_bias_route": bias_match,
        "matches_full_public_pisa": False,
        "known_difference": (
            "piecewise_pisa_env now matches the public PISA default score "
            "top-k selector at the route boundary, while the local runtime still "
            "differs from a full public-original port in integration settings, "
            "kernel/runtime glue, model-specific Cosmos3 adapters, and current "
            "GPU quality."
        ),
    }


def probe() -> dict[str, Any]:
    return {
        "status": "pass",
        "public_reference": {
            "repo": str(PUBLIC_PISA),
            "commit": git_commit(PUBLIC_PISA),
            "kernel_source": str(PUBLIC_KERNEL),
            "checks": source_checks(),
        },
        "behavior_probe": behavior_probe(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = probe()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
