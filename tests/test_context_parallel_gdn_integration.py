"""GPU integration parity for real Triton GDN CP dispatch.

This is intentionally separate from ``test_context_parallel.py``:

* ``test_context_parallel.py`` proves the CP primitives against eager CPU
  references.
* This file instantiates the real Triton GDN blocks, runs the non-CP full
  sequence and the CP temporal split, then compares gathered CP outputs (and
  training-path gradients) against the non-CP reference.

Run this test on a GPU allocation.
"""

from __future__ import annotations

import datetime as _datetime
import os
import queue
import socket
import traceback
import unittest
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@dataclass(frozen=True)
class _BlockCase:
    name: str
    block_cls_name: str
    use_autograd_kernel: bool
    check_backward: bool
    chunk_size: int | None = None


_CASES: tuple[_BlockCase, ...] = (
    _BlockCase(
        name="raw_forward_training_autograd",
        block_cls_name="RawForwardGDN",
        use_autograd_kernel=True,
        check_backward=True,
    ),
    _BlockCase(
        name="raw_reverse_training_autograd",
        block_cls_name="RawReverseGDN",
        use_autograd_kernel=True,
        check_backward=True,
    ),
    _BlockCase(
        name="bidirectional_inference_fused",
        block_cls_name="BidirectionalGDNTriton",
        use_autograd_kernel=False,
        check_backward=False,
    ),
    _BlockCase(
        name="bidirectional_training_autograd",
        block_cls_name="BidirectionalGDNTriton",
        use_autograd_kernel=True,
        check_backward=True,
    ),
    _BlockCase(
        name="chunk_causal_training_autograd",
        block_cls_name="ChunkCausalGDNTriton",
        use_autograd_kernel=True,
        check_backward=True,
        chunk_size=3,
    ),
)


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
    atol: float = 5e-3,
    rtol: float = 5e-3,
) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().item())
    denom = expected.float().abs().clamp_min(1e-8)
    max_rel = float((diff / denom).max().item())
    if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
        raise AssertionError(
            f"{name} mismatch: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, "
            f"actual_shape={tuple(actual.shape)}, expected_shape={tuple(expected.shape)}"
        )
    return max_abs, max_rel


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _integration_dtype() -> torch.dtype:
    dtype_name = os.environ.get("CP_GDN_TEST_DTYPE", "float32").strip().lower()
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def _make_block(case: _BlockCase, *, device: torch.device) -> torch.nn.Module:
    from diffusion.model.nets.sana_gdn_blocks_triton import (
        BidirectionalGDNTriton,
        ChunkCausalGDNTriton,
    )

    cls_by_name = {
        "BidirectionalGDNTriton": BidirectionalGDNTriton,
        "ChunkCausalGDNTriton": ChunkCausalGDNTriton,
    }
    cls = cls_by_name[case.block_cls_name]
    torch.manual_seed(12345)
    block = cls(
        in_dim=64,
        out_dim=64,
        heads=2,
        dim=32,
        eps=1e-6,
        use_bias=True,
        qk_norm=True,
        use_output_gate=True,
        conv_kernel_size=0,
        k_conv_only=True,
        use_autograd_kernel=case.use_autograd_kernel,
    ).to(device=device, dtype=_integration_dtype())
    block.train(case.check_backward)
    return block


def _make_full_input(*, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20240608)
    x = torch.randn(1, 6 * 4, 64, generator=generator, dtype=torch.float32) * 0.2
    return x.to(device=device, dtype=_integration_dtype())


def _slice_temporal_tokens(x: torch.Tensor, rank: int, world: int, *, spatial_tokens: int) -> torch.Tensor:
    frames_total = x.shape[1] // spatial_tokens
    assert frames_total % world == 0
    local_frames = frames_total // world
    start = rank * local_frames * spatial_tokens
    end = (rank + 1) * local_frames * spatial_tokens
    return x[:, start:end].contiguous()


def _gather_token_shards(local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local.contiguous(), group=group)
    return torch.cat(gathered, dim=1)


def _gather_dim_shards(local: torch.Tensor, group: dist.ProcessGroup, *, dim: int) -> torch.Tensor:
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local.contiguous(), group=group)
    return torch.cat(gathered, dim=dim)


def _allreduce_parameter_grads(module: torch.nn.Module, group: dist.ProcessGroup) -> None:
    for param in module.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=group)


def _allreduce_tensor_grad(tensor: torch.Tensor, group: dist.ProcessGroup) -> None:
    if tensor.grad is not None:
        dist.all_reduce(tensor.grad, op=dist.ReduceOp.SUM, group=group)


def _grad_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().item())
    denom = expected.float().abs().clamp_min(1e-8)
    max_rel = float((diff / denom).max().item())
    return max_abs, max_rel


def _assert_parameter_grads_close(
    cp_block: torch.nn.Module,
    ref_block: torch.nn.Module,
    *,
    case_name: str,
    atol: float = 8e-3,
    rtol: float = 8e-3,
    norm_rtol: float = 5e-3,
) -> tuple[float, float]:
    worst_abs = 0.0
    worst_rel = 0.0
    for (cp_name, cp_param), (ref_name, ref_param) in zip(cp_block.named_parameters(), ref_block.named_parameters()):
        assert cp_name == ref_name
        if cp_param.grad is None and ref_param.grad is None:
            continue
        if cp_param.grad is None or ref_param.grad is None:
            raise AssertionError(f"{case_name}:{cp_name} grad presence mismatch")
        max_abs, max_rel = _grad_error(cp_param.grad, ref_param.grad)
        if not torch.allclose(cp_param.grad.float(), ref_param.grad.float(), atol=atol, rtol=rtol):
            diff_norm = float((cp_param.grad.float() - ref_param.grad.float()).norm().item())
            ref_norm = float(ref_param.grad.float().norm().item())
            if diff_norm > norm_rtol * max(ref_norm, 1e-8):
                raise AssertionError(
                    f"{case_name}:{cp_name}.grad mismatch: max_abs={max_abs:.6e}, "
                    f"max_rel={max_rel:.6e}, diff_norm={diff_norm:.6e}, "
                    f"ref_norm={ref_norm:.6e}, actual_shape={tuple(cp_param.grad.shape)}, "
                    f"expected_shape={tuple(ref_param.grad.shape)}"
                )
        worst_abs = max(worst_abs, max_abs)
        worst_rel = max(worst_rel, max_rel)
    return worst_abs, worst_rel


def _case_weight(case: _BlockCase, *, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(9000 + len(case.name))
    return torch.randn(1, 6 * 4, 64, generator=generator, dtype=torch.float32).to(
        device=device,
        dtype=_integration_dtype(),
    )


def _make_raw_inputs(*, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(777123)
    B, T, S, H, D = 1, 6, 4, 2, 32
    N = T * S
    C = H * D
    qkv = (torch.randn(B, N, 3, H, D, generator=generator, dtype=torch.float32) * 0.2).to(device)
    beta = torch.sigmoid(torch.randn(B, H, T, S, generator=generator, dtype=torch.float32) * 0.3).to(device)
    decay = torch.sigmoid(torch.randn(B, H, T, generator=generator, dtype=torch.float32) * 0.2 + 1.0).to(device)
    q_norm_weight = (1.0 + torch.randn(C, generator=generator, dtype=torch.float32) * 0.05).to(device)
    k_norm_weight = (1.0 + torch.randn(C, generator=generator, dtype=torch.float32) * 0.05).to(device)
    weight_num = torch.randn(B, N, H, D, generator=generator, dtype=torch.float32).to(device)
    weight_den = torch.randn(B, H, N, generator=generator, dtype=torch.float32).to(device)
    return {
        "qkv": qkv,
        "beta": beta,
        "decay": decay,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "weight_num": weight_num,
        "weight_den": weight_den,
    }


def _slice_frames(tensor: torch.Tensor, rank: int, world: int, *, frame_dim: int, frames_total: int) -> torch.Tensor:
    assert frames_total % world == 0
    local_frames = frames_total // world
    start = rank * local_frames
    end = (rank + 1) * local_frames
    return tensor.narrow(frame_dim, start, local_frames).contiguous()


def _slice_raw_tokens(tensor: torch.Tensor, rank: int, world: int, *, spatial_tokens: int) -> torch.Tensor:
    frames_total = tensor.shape[1] // spatial_tokens
    assert frames_total % world == 0
    local_frames = frames_total // world
    start = rank * local_frames * spatial_tokens
    return tensor.narrow(1, start, local_frames * spatial_tokens).contiguous()


def _raw_rope_tables(*, device: torch.device, tokens: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(tokens, device=device, dtype=torch.float32)
    dims = torch.arange(head_dim // 2, device=device, dtype=torch.float32)
    inv_freq = torch.pow(torch.tensor(10000.0, device=device), -2.0 * dims / head_dim)
    angles = positions[:, None] * inv_freq[None, :]
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    rope_cos = cos_half.repeat_interleave(2, dim=-1)
    rope_sin = torch.stack([-sin_half, sin_half], dim=-1).reshape(tokens, head_dim)
    return rope_cos.contiguous(), rope_sin.contiguous()


def _run_raw_forward_case(rank: int, world: int, group: dist.ProcessGroup) -> dict[str, float]:
    from diffusion.distributed.context_parallel.config import (
        CpRuntimeConfig,
        set_cp_runtime_config,
    )
    from diffusion.model.ops.fused_gdn_chunkwise_stateful_raw import (
        fused_gdn_chunkwise_stateful_raw_autograd,
    )
    from diffusion.model.ops.fused_gdn_cp import cp_fused_gdn_chunkwise_raw_autograd

    device = torch.device("cuda", rank)
    B, T, S, H, D = 1, 6, 4, 2, 32
    N = T * S
    local_T = T // world
    k_scale = (D**-0.5) * (S**-0.5)

    set_cp_runtime_config(
        CpRuntimeConfig(
            triton_block_fusion=True,
            scan_backend=os.environ.get("CP_GDN_SCAN_BACKEND", "torch"),
            allgather_impl="collective",
            halo_impl="collective",
        )
    )

    raw = _make_raw_inputs(device=device)
    ref_qkv = raw["qkv"].detach().clone().requires_grad_(True)
    ref_beta = raw["beta"].detach().clone().requires_grad_(True)
    ref_decay = raw["decay"].detach().clone().requires_grad_(True)
    ref_q_norm_weight = raw["q_norm_weight"].detach().clone().requires_grad_(True)
    ref_k_norm_weight = raw["k_norm_weight"].detach().clone().requires_grad_(True)

    cp_qkv = _slice_raw_tokens(raw["qkv"], rank, world, spatial_tokens=S).detach().clone().requires_grad_(True)
    cp_beta = _slice_frames(raw["beta"], rank, world, frame_dim=2, frames_total=T).detach().clone().requires_grad_(True)
    cp_decay = _slice_frames(raw["decay"], rank, world, frame_dim=2, frames_total=T).detach().clone().requires_grad_(True)
    cp_q_norm_weight = raw["q_norm_weight"].detach().clone().requires_grad_(True)
    cp_k_norm_weight = raw["k_norm_weight"].detach().clone().requires_grad_(True)

    rope_cos, rope_sin = _raw_rope_tables(device=device, tokens=N, head_dim=D)
    local_rope_cos = _slice_raw_tokens(rope_cos.unsqueeze(0), rank, world, spatial_tokens=S).squeeze(0).contiguous()
    local_rope_sin = _slice_raw_tokens(rope_sin.unsqueeze(0), rank, world, spatial_tokens=S).squeeze(0).contiguous()

    ref_num, ref_den = fused_gdn_chunkwise_stateful_raw_autograd(
        ref_qkv,
        ref_beta,
        ref_decay,
        ref_q_norm_weight,
        ref_k_norm_weight,
        rope_cos,
        rope_sin,
        F=T,
        S=S,
        k_scale=k_scale,
        norm_eps=1e-6,
        dot_precision=2,
        direction=1,
    )
    cp_res = cp_fused_gdn_chunkwise_raw_autograd(
        cp_qkv,
        cp_beta,
        cp_decay,
        cp_q_norm_weight,
        cp_k_norm_weight,
        local_rope_cos,
        local_rope_sin,
        F=local_T,
        S=S,
        group=group,
        k_scale=k_scale,
        norm_eps=1e-6,
        eps=1e-6,
        dot_precision=2,
        reverse_rank_order=False,
        truncate_to_active=None,
    )

    cp_num_full = _gather_dim_shards(cp_res.num, group, dim=1)
    cp_den_full = _gather_dim_shards(cp_res.den, group, dim=2)
    num_abs, num_rel = _assert_close(cp_num_full, ref_num.detach(), name="raw_forward_training_autograd:num")
    den_abs, den_rel = _assert_close(cp_den_full, ref_den.detach(), name="raw_forward_training_autograd:den")

    local_weight_num = _slice_raw_tokens(raw["weight_num"], rank, world, spatial_tokens=S)
    local_weight_den = _slice_raw_tokens(raw["weight_den"].transpose(1, 2), rank, world, spatial_tokens=S).transpose(1, 2)
    ref_loss = (ref_num * raw["weight_num"]).sum() + (ref_den * raw["weight_den"]).sum()
    cp_loss = (cp_res.num * local_weight_num).sum() + (cp_res.den * local_weight_den).sum()
    ref_loss.backward()
    cp_loss.backward()

    _allreduce_tensor_grad(cp_q_norm_weight, group)
    _allreduce_tensor_grad(cp_k_norm_weight, group)

    expected_qkv_grad = _slice_raw_tokens(ref_qkv.grad, rank, world, spatial_tokens=S)
    qkv_abs, qkv_rel = _assert_close(
        cp_qkv.grad,
        expected_qkv_grad,
        name="raw_forward_training_autograd:qkv_grad",
        atol=8e-3,
        rtol=8e-3,
    )
    expected_beta_grad = _slice_frames(ref_beta.grad, rank, world, frame_dim=2, frames_total=T)
    beta_abs, beta_rel = _assert_close(
        cp_beta.grad,
        expected_beta_grad,
        name="raw_forward_training_autograd:beta_grad",
        atol=8e-3,
        rtol=8e-3,
    )
    expected_decay_grad = _slice_frames(ref_decay.grad, rank, world, frame_dim=2, frames_total=T)
    decay_abs, decay_rel = _assert_close(
        cp_decay.grad,
        expected_decay_grad,
        name="raw_forward_training_autograd:decay_grad",
        atol=8e-3,
        rtol=8e-3,
    )
    qnw_abs, qnw_rel = _assert_close(
        cp_q_norm_weight.grad,
        ref_q_norm_weight.grad,
        name="raw_forward_training_autograd:q_norm_weight_grad",
        atol=8e-3,
        rtol=8e-3,
    )
    knw_abs, knw_rel = _assert_close(
        cp_k_norm_weight.grad,
        ref_k_norm_weight.grad,
        name="raw_forward_training_autograd:k_norm_weight_grad",
        atol=8e-3,
        rtol=8e-3,
    )

    return {
        "output_max_abs": max(num_abs, den_abs),
        "output_max_rel": max(num_rel, den_rel),
        "input_grad_max_abs": max(qkv_abs, beta_abs, decay_abs),
        "input_grad_max_rel": max(qkv_rel, beta_rel, decay_rel),
        "param_grad_max_abs": max(qnw_abs, knw_abs),
        "param_grad_max_rel": max(qnw_rel, knw_rel),
    }


def _run_raw_reverse_case(rank: int, world: int, group: dist.ProcessGroup) -> dict[str, float]:
    from diffusion.distributed.context_parallel.config import (
        CpRuntimeConfig,
        set_cp_runtime_config,
    )
    from diffusion.distributed.context_parallel.halo_exchange import cp_halo_exchange
    from diffusion.model.ops.fused_gdn_chunkwise_stateful_raw import (
        fused_gdn_chunkwise_stateful_raw_autograd,
    )
    from diffusion.model.ops.fused_gdn_cp import cp_fused_gdn_chunkwise_raw_autograd

    device = torch.device("cuda", rank)
    B, T, S, H, D = 1, 6, 4, 2, 32
    N = T * S
    local_T = T // world
    k_scale = (D**-0.5) * (S**-0.5)

    set_cp_runtime_config(
        CpRuntimeConfig(
            triton_block_fusion=True,
            scan_backend=os.environ.get("CP_GDN_SCAN_BACKEND", "torch"),
            allgather_impl="collective",
            halo_impl="collective",
        )
    )

    raw = _make_raw_inputs(device=device)
    ref_qkv = raw["qkv"].detach().clone().requires_grad_(True)
    ref_beta = raw["beta"].detach().clone().requires_grad_(True)
    ref_decay = raw["decay"].detach().clone().requires_grad_(True)
    ref_q_norm_weight = raw["q_norm_weight"].detach().clone().requires_grad_(True)
    ref_k_norm_weight = raw["k_norm_weight"].detach().clone().requires_grad_(True)

    cp_qkv = _slice_raw_tokens(raw["qkv"], rank, world, spatial_tokens=S).detach().clone().requires_grad_(True)
    cp_beta = _slice_frames(raw["beta"], rank, world, frame_dim=2, frames_total=T).detach().clone().requires_grad_(True)
    cp_decay = _slice_frames(raw["decay"], rank, world, frame_dim=2, frames_total=T).detach().clone().requires_grad_(True)
    cp_q_norm_weight = raw["q_norm_weight"].detach().clone().requires_grad_(True)
    cp_k_norm_weight = raw["k_norm_weight"].detach().clone().requires_grad_(True)

    rope_cos, rope_sin = _raw_rope_tables(device=device, tokens=N, head_dim=D)
    local_rope_cos = _slice_raw_tokens(rope_cos.unsqueeze(0), rank, world, spatial_tokens=S).squeeze(0).contiguous()
    local_rope_sin = _slice_raw_tokens(rope_sin.unsqueeze(0), rank, world, spatial_tokens=S).squeeze(0).contiguous()

    ref_num, ref_den = fused_gdn_chunkwise_stateful_raw_autograd(
        ref_qkv,
        ref_beta,
        ref_decay,
        ref_q_norm_weight,
        ref_k_norm_weight,
        rope_cos,
        rope_sin,
        F=T,
        S=S,
        k_scale=k_scale,
        norm_eps=1e-6,
        dot_precision=2,
        direction=2,
    )

    def _cp_flip_and_shift(tensors: list[torch.Tensor], shift_vals: list[float]) -> list[torch.Tensor]:
        is_last = rank == world - 1
        results = []
        for tensor, shift_val in zip(tensors, shift_vals):
            first_frame = tensor[:, :, :1, ...].contiguous()
            haloed = cp_halo_exchange(first_frame, left_size=0, right_size=1, dim=2, group=group)
            boundary = haloed[:, :, 1:2, ...]
            if is_last and shift_val != 0.0:
                boundary = boundary.mul(0.0).add(shift_val)
            flipped = torch.flip(tensor, dims=[2])
            body = flipped[:, :, : tensor.shape[2] - 1, ...]
            results.append(torch.cat([boundary, body], dim=2))
        return results

    def _bnhd_to_frame(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.permute(0, 2, 3, 1).reshape(B, H, D, local_T, S).permute(0, 1, 3, 2, 4).contiguous()

    def _frame_to_bnhd(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.permute(0, 2, 4, 1, 3).reshape(B, local_T * S, H, D).contiguous()

    q_raw_f = _bnhd_to_frame(cp_qkv[:, :, 0])
    k_raw_f = _bnhd_to_frame(cp_qkv[:, :, 1])
    v_raw_f = _bnhd_to_frame(cp_qkv[:, :, 2])
    q_bwd_f = torch.flip(q_raw_f, dims=[2])
    k_bwd_f, v_bwd_f = _cp_flip_and_shift([k_raw_f, v_raw_f], [0.0, 0.0])
    qkv_bwd = torch.stack(
        [
            _frame_to_bnhd(q_bwd_f),
            _frame_to_bnhd(k_bwd_f),
            _frame_to_bnhd(v_bwd_f),
        ],
        dim=2,
    )

    beta_f = cp_beta.unsqueeze(3)
    decay_f = cp_decay.view(B, H, local_T, 1, 1)
    beta_bwd_f, decay_bwd_f = _cp_flip_and_shift([beta_f, decay_f], [0.0, 1.0])
    beta_bwd = beta_bwd_f.squeeze(3).contiguous()
    decay_bwd = decay_bwd_f.squeeze(-1).squeeze(-1).contiguous()

    def _rope_to_frame(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(local_T, S, D).permute(0, 2, 1).reshape(1, 1, local_T, D, S).contiguous()

    def _rope_from_frame(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(local_T, D, S).permute(0, 2, 1).reshape(local_T * S, D).contiguous()

    rope_cos_f = _rope_to_frame(local_rope_cos)
    rope_sin_f = _rope_to_frame(local_rope_sin)
    rope_cos_q = _rope_from_frame(torch.flip(rope_cos_f, dims=[2]))
    rope_sin_q = _rope_from_frame(torch.flip(rope_sin_f, dims=[2]))
    rope_cos_k_f, rope_sin_k_f = _cp_flip_and_shift([rope_cos_f, rope_sin_f], [1.0, 0.0])
    rope_cos_k = _rope_from_frame(rope_cos_k_f)
    rope_sin_k = _rope_from_frame(rope_sin_k_f)

    cp_res = cp_fused_gdn_chunkwise_raw_autograd(
        qkv_bwd,
        beta_bwd,
        decay_bwd,
        cp_q_norm_weight,
        cp_k_norm_weight,
        rope_cos_k,
        rope_sin_k,
        F=local_T,
        S=S,
        group=group,
        k_scale=k_scale,
        norm_eps=1e-6,
        eps=1e-6,
        dot_precision=2,
        reverse_rank_order=True,
        truncate_to_active=None,
        rope_cos_q=rope_cos_q,
        rope_sin_q=rope_sin_q,
    )

    cp_num_flipped = cp_res.num.reshape(B, local_T, S, H, D).permute(0, 3, 1, 4, 2).contiguous()
    cp_den_flipped = cp_res.den.reshape(B, H, local_T, S).unsqueeze(3).contiguous()
    cp_num = torch.flip(cp_num_flipped, dims=[2]).permute(0, 2, 4, 1, 3).reshape(B, local_T * S, H, D)
    cp_den = torch.flip(cp_den_flipped, dims=[2]).squeeze(3).reshape(B, H, local_T * S)

    cp_num_full = _gather_dim_shards(cp_num, group, dim=1)
    cp_den_full = _gather_dim_shards(cp_den, group, dim=2)
    num_abs, num_rel = _assert_close(cp_num_full, ref_num.detach(), name="raw_reverse_training_autograd:num")
    den_abs, den_rel = _assert_close(cp_den_full, ref_den.detach(), name="raw_reverse_training_autograd:den")

    local_weight_num = _slice_raw_tokens(raw["weight_num"], rank, world, spatial_tokens=S)
    local_weight_den = _slice_raw_tokens(raw["weight_den"].transpose(1, 2), rank, world, spatial_tokens=S).transpose(1, 2)
    ref_loss = (ref_num * raw["weight_num"]).sum() + (ref_den * raw["weight_den"]).sum()
    cp_loss = (cp_num * local_weight_num).sum() + (cp_den * local_weight_den).sum()
    ref_loss.backward()
    cp_loss.backward()

    _allreduce_tensor_grad(cp_q_norm_weight, group)
    _allreduce_tensor_grad(cp_k_norm_weight, group)

    expected_qkv_grad = _slice_raw_tokens(ref_qkv.grad, rank, world, spatial_tokens=S)
    qkv_abs, qkv_rel = _assert_close(
        cp_qkv.grad,
        expected_qkv_grad,
        name="raw_reverse_training_autograd:qkv_grad",
        atol=8e-3,
        rtol=8e-3,
    )
    expected_beta_grad = _slice_frames(ref_beta.grad, rank, world, frame_dim=2, frames_total=T)
    beta_abs, beta_rel = _assert_close(
        cp_beta.grad,
        expected_beta_grad,
        name="raw_reverse_training_autograd:beta_grad",
        atol=8e-3,
        rtol=8e-3,
    )
    expected_decay_grad = _slice_frames(ref_decay.grad, rank, world, frame_dim=2, frames_total=T)
    decay_abs, decay_rel = _assert_close(
        cp_decay.grad,
        expected_decay_grad,
        name="raw_reverse_training_autograd:decay_grad",
        atol=8e-3,
        rtol=8e-3,
    )
    qnw_abs, qnw_rel = _assert_close(
        cp_q_norm_weight.grad,
        ref_q_norm_weight.grad,
        name="raw_reverse_training_autograd:q_norm_weight_grad",
        atol=8e-3,
        rtol=8e-3,
    )
    knw_abs, knw_rel = _assert_close(
        cp_k_norm_weight.grad,
        ref_k_norm_weight.grad,
        name="raw_reverse_training_autograd:k_norm_weight_grad",
        atol=8e-3,
        rtol=8e-3,
    )

    return {
        "output_max_abs": max(num_abs, den_abs),
        "output_max_rel": max(num_rel, den_rel),
        "input_grad_max_abs": max(qkv_abs, beta_abs, decay_abs),
        "input_grad_max_rel": max(qkv_rel, beta_rel, decay_rel),
        "param_grad_max_abs": max(qnw_abs, knw_abs),
        "param_grad_max_rel": max(qnw_rel, knw_rel),
    }


def _run_one_case(case: _BlockCase, rank: int, world: int, group: dist.ProcessGroup) -> dict[str, float]:
    from diffusion.distributed.context_parallel.config import (
        CpRuntimeConfig,
        set_cp_group,
        set_cp_runtime_config,
    )

    if case.block_cls_name == "RawForwardGDN":
        return _run_raw_forward_case(rank, world, group)
    if case.block_cls_name == "RawReverseGDN":
        return _run_raw_reverse_case(rank, world, group)

    device = torch.device("cuda", rank)
    spatial_tokens = 4
    global_hw = (6, 2, 2)
    local_hw = (global_hw[0] // world, global_hw[1], global_hw[2])

    set_cp_runtime_config(
        CpRuntimeConfig(
            triton_block_fusion=True,
            scan_backend=os.environ.get("CP_GDN_SCAN_BACKEND", "torch"),
            allgather_impl="collective",
            halo_impl="collective",
        )
    )

    ref_block = _make_block(case, device=device)
    cp_block = _make_block(case, device=device)
    cp_block.load_state_dict(ref_block.state_dict())

    full_x = _make_full_input(device=device).requires_grad_(case.check_backward)
    local_x = _slice_temporal_tokens(full_x.detach(), rank, world, spatial_tokens=spatial_tokens)
    local_x = local_x.contiguous().requires_grad_(case.check_backward)
    weight = _case_weight(case, device=device)
    local_weight = _slice_temporal_tokens(weight, rank, world, spatial_tokens=spatial_tokens)

    forward_kwargs: dict[str, Any] = {}
    if case.chunk_size is not None:
        forward_kwargs.update(chunk_size=case.chunk_size, chunk_split_strategy="uniform")

    set_cp_group(None)
    if case.check_backward:
        ref_out = ref_block(full_x, HW=global_hw, rotary_emb=None, **forward_kwargs)
    else:
        with torch.no_grad():
            ref_out = ref_block(full_x.detach(), HW=global_hw, rotary_emb=None, **forward_kwargs)

    set_cp_group(group)
    if case.check_backward:
        cp_out_local = cp_block(local_x, HW=local_hw, rotary_emb=None, **forward_kwargs)
    else:
        with torch.no_grad():
            cp_out_local = cp_block(local_x.detach(), HW=local_hw, rotary_emb=None, **forward_kwargs)

    cp_out_full = _gather_token_shards(cp_out_local, group)
    out_abs, out_rel = _assert_close(cp_out_full, ref_out.detach(), name=f"{case.name}:output")

    stats = {
        "output_max_abs": out_abs,
        "output_max_rel": out_rel,
        "input_grad_max_abs": 0.0,
        "input_grad_max_rel": 0.0,
        "param_grad_max_abs": 0.0,
        "param_grad_max_rel": 0.0,
    }

    if case.check_backward:
        ref_loss = (ref_out * weight).sum()
        cp_loss = (cp_out_local * local_weight).sum()
        ref_loss.backward()
        cp_loss.backward()
        _allreduce_parameter_grads(cp_block, group)

        expected_local_grad = _slice_temporal_tokens(full_x.grad, rank, world, spatial_tokens=spatial_tokens)
        grad_abs, grad_rel = _assert_close(
            local_x.grad,
            expected_local_grad,
            name=f"{case.name}:input_grad",
            atol=8e-3,
            rtol=8e-3,
        )
        param_abs, param_rel = _assert_parameter_grads_close(cp_block, ref_block, case_name=case.name)
        stats.update(
            input_grad_max_abs=grad_abs,
            input_grad_max_rel=grad_rel,
            param_grad_max_abs=param_abs,
            param_grad_max_rel=param_rel,
        )

    dist.barrier(group=group)
    return stats


def _worker(
    rank: int,
    world: int,
    init_method: str,
    result_queue: Any,
    case_names: tuple[str, ...],
) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world,
            timeout=_datetime.timedelta(seconds=600),
        )
        selected = [case for case in _CASES if case.name in case_names]
        for case in selected:
            stats = _run_one_case(case, rank, world, dist.group.WORLD)
            if rank == 0:
                result_queue.put(("case", case.name, stats))
        dist.barrier()
        result_queue.put(("ok", rank, ""))
    except BaseException:
        result_queue.put(("error", rank, traceback.format_exc()))
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_distributed_gpu(*, world: int = 2, case_names: tuple[str, ...] | None = None) -> list[tuple[str, dict[str, float]]]:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required for Triton GDN CP integration parity")
    if torch.cuda.device_count() < world:
        raise unittest.SkipTest(f"need at least {world} visible CUDA devices, got {torch.cuda.device_count()}")

    case_names = case_names or tuple(case.name for case in _CASES)
    port = _free_tcp_port()
    init_method = f"tcp://127.0.0.1:{port}"
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(rank, world, init_method, result_queue, case_names))
        for rank in range(world)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=900)

    alive = [proc for proc in procs if proc.is_alive()]
    for proc in alive:
        proc.terminate()
    for proc in alive:
        proc.join(timeout=5)
    if alive:
        raise AssertionError(f"GPU integration parity timed out with {len(alive)} live workers")

    messages: list[tuple[str, Any, Any]] = []
    while True:
        try:
            messages.append(result_queue.get_nowait())
        except queue.Empty:
            break

    errors = [(rank, tb) for status, rank, tb in messages if status == "error"]
    bad_exits = [(idx, proc.exitcode) for idx, proc in enumerate(procs) if proc.exitcode != 0]
    if errors or bad_exits:
        details = "\n".join(f"rank {rank}:\n{tb}" for rank, tb in errors)
        raise AssertionError(f"GPU integration parity failed; exits={bad_exits}\n{details}")

    ok_ranks = sorted(rank for status, rank, _ in messages if status == "ok")
    assert ok_ranks == list(range(world)), f"missing worker completions: got {ok_ranks}"
    return [(case_name, stats) for status, case_name, stats in messages if status == "case"]


def test_triton_gdn_cp_matches_non_cp_full_sequence() -> None:
    selected = os.environ.get("CP_GDN_CASES")
    case_names = tuple(name.strip() for name in selected.split(",") if name.strip()) if selected else None
    stats = _run_distributed_gpu(case_names=case_names)
    expected = len(case_names) if case_names is not None else len(_CASES)
    assert len(stats) == expected
