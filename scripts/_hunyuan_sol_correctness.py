"""GPU correctness checks for the released Hunyuan Sol-Attn adapter.

Run on H100 or B200 with ``SOL_ATTN_STRICT=1``. The test verifies the public
all-exact limit, dense text-query rows, padded-row handling, and finite output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")

from techniques.sparse_backends import sol_attn_backend as backend  # noqa: E402


def rel_l2(actual, expected) -> float:
    return float(
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1.0e-12)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()

    os.environ["SOL_ATTN_STRICT"] = "1"
    assert torch.cuda.is_available(), "requires H100 or B200"
    assert torch.cuda.get_device_capability() in {(9, 0), (10, 0)}

    batch, heads, dim = 1, 2, 128
    video_tokens = 192
    valid_text_tokens = 65
    padded_text_tokens = 31
    tokens = video_tokens + valid_text_tokens + padded_text_tokens

    torch.manual_seed(20260728)
    q = torch.randn(
        batch,
        heads,
        tokens,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    key_valid = torch.ones(batch, tokens, dtype=torch.bool, device="cuda")
    key_valid[:, video_tokens + valid_text_tokens :] = False

    effective_tokens = video_tokens + valid_text_tokens
    q_joint = q[:, :, :effective_tokens].transpose(1, 2).contiguous()
    k_joint = k[:, :, :effective_tokens].transpose(1, 2).contiguous()
    v_joint = v[:, :, :effective_tokens].transpose(1, 2).contiguous()
    dense_joint = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, :effective_tokens],
        k[:, :, :effective_tokens],
        v[:, :, :effective_tokens],
    ).transpose(1, 2)
    all_exact = backend._run_sol_attn_bthd(
        q_joint,
        k_joint,
        v_joint,
        tau=1.0,
        thresh_type="diag",
        kv_splits="auto",
        sink_start=0,
        sink_tokens=effective_tokens,
    )

    dense_text = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, video_tokens:effective_tokens],
        k[:, :, :effective_tokens],
        v[:, :, :effective_tokens],
    )
    actual = backend._run_mmdit_sol_attn_bthd(
        q,
        k,
        v,
        video_len=video_tokens,
        key_valid=key_valid,
        tau=1.0,
        thresh_type="diag",
        kv_splits="auto",
    )

    def unexpected_dense_dispatch(*_args, **_kwargs):
        raise AssertionError("eligible Hunyuan dispatch unexpectedly fell back")

    dispatch = backend.make_mmdit_sol_attn_dispatch(
        unexpected_dense_dispatch,
        video_len=video_tokens,
        tau=1.0,
        thresh_type="diag",
        kv_splits="auto",
    )
    dispatch_mask = torch.zeros(
        batch,
        1,
        1,
        tokens,
        dtype=q.dtype,
        device=q.device,
    )
    dispatch_mask[..., effective_tokens:] = float("-inf")
    backend.reset_sol_attn_state()
    backend.sol_attn_begin_forward()
    dispatched = dispatch(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=dispatch_mask,
    ).transpose(1, 2)

    all_exact_error = rel_l2(all_exact, dense_joint)
    text_error = rel_l2(
        actual[:, :, video_tokens:effective_tokens],
        dense_text,
    )
    padded_abs = float(actual[:, :, effective_tokens:].abs().max())
    dispatch_error = rel_l2(dispatched, actual)
    stats = backend.get_sol_attn_stats()
    result = {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "vendor_module": str(sys.modules["sol_attn"].__file__),
        "all_exact_dense_rel_l2": all_exact_error,
        "text_query_dense_rel_l2": text_error,
        "padded_query_max_abs": padded_abs,
        "dispatch_direct_rel_l2": dispatch_error,
        "stats": stats,
        "finite": bool(torch.isfinite(actual).all()),
    }
    print(result)
    assert all_exact_error < 0.01
    assert text_error == 0.0
    assert padded_abs == 0.0
    assert dispatch_error == 0.0
    assert stats == {
        "dispatch_calls": 1,
        "kernel_calls": 1,
        "hunyuan_calls": 1,
        "dense_guard_calls": 0,
    }
    assert torch.isfinite(actual).all()
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print("HUNYUAN_SOL_ATTN_CORRECTNESS_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
