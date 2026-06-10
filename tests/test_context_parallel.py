"""Deterministic equivalence tests for context-parallel primitives.

These tests run with a local CPU ``gloo`` process group, so they validate the
distributed math without requiring a GPU allocation.
"""

from __future__ import annotations

import datetime as _datetime
import os
import queue
import socket
import sys
import traceback
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _assert_raises(exc_type: type[BaseException], fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except BaseException as exc:  # pragma: no cover - failure path
        raise AssertionError(f"expected {exc_type.__name__}, got {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"expected {exc_type.__name__}, but no exception was raised")


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 1e-6, rtol: float = 1e-6) -> None:
    if torch.allclose(actual, expected, atol=atol, rtol=rtol):
        return
    diff = (actual - expected).abs()
    raise AssertionError(
        f"tensor mismatch: max_abs={diff.max().item():.6e}, "
        f"actual_shape={tuple(actual.shape)}, expected_shape={tuple(expected.shape)}"
    )


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _distributed_worker(
    rank: int,
    world_size: int,
    case_name: str,
    init_method: str,
    result_queue: Any,
    kwargs: dict[str, Any],
) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=_datetime.timedelta(seconds=180),
        )
        _DIST_CASES[case_name](rank, world_size, dist.group.WORLD, **kwargs)
        dist.barrier()
        result_queue.put(("ok", rank, ""))
    except BaseException:
        result_queue.put(("error", rank, traceback.format_exc()))
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_distributed(case_name: str, *, world_size: int = 3, timeout_s: int = 240, **kwargs: Any) -> None:
    assert dist.is_available(), "torch.distributed is not available"
    if sys.platform != "win32":
        if case_name == "core_data_utils":
            import diffusion.distributed.context_parallel.data_utils  # noqa: F401
        elif case_name == "halo_exchange":
            import diffusion.distributed.context_parallel.halo_exchange  # noqa: F401
        elif case_name == "frame_gdn_scan":
            import diffusion.distributed.context_parallel.distributed_scan  # noqa: F401

    port = _free_tcp_port()
    init_method = f"tcp://127.0.0.1:{port}"
    ctx = mp.get_context("spawn" if sys.platform == "win32" else "fork")
    result_queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_distributed_worker,
            args=(rank, world_size, case_name, init_method, result_queue, kwargs),
        )
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()

    deadline = _datetime.datetime.now() + _datetime.timedelta(seconds=timeout_s)
    for proc in procs:
        remaining = max(1.0, (deadline - _datetime.datetime.now()).total_seconds())
        proc.join(timeout=remaining)

    alive = [proc for proc in procs if proc.is_alive()]
    for proc in alive:
        proc.terminate()
    for proc in alive:
        proc.join(timeout=5)
    if alive:
        raise AssertionError(f"distributed case {case_name!r} timed out with {len(alive)} live workers")

    messages: list[tuple[str, int, str]] = []
    while True:
        try:
            messages.append(result_queue.get_nowait())
        except queue.Empty:
            break

    errors = [msg for msg in messages if msg[0] == "error"]
    bad_exits = [(idx, proc.exitcode) for idx, proc in enumerate(procs) if proc.exitcode != 0]
    if errors or bad_exits:
        details = "\n".join(f"rank {rank}:\n{tb}" for _, rank, tb in errors)
        raise AssertionError(f"distributed case {case_name!r} failed; exits={bad_exits}\n{details}")

    ok_ranks = sorted(rank for status, rank, _ in messages if status == "ok")
    assert ok_ranks == list(range(world_size)), f"missing worker completions: got {ok_ranks}"


def _gather_local_chunks(local: torch.Tensor, group: dist.ProcessGroup, dim: int) -> torch.Tensor:
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local.contiguous(), group=group)
    return torch.cat(gathered, dim=dim)


def _case_core_data_utils(rank: int, world: int, group: dist.ProcessGroup) -> None:
    from diffusion.distributed.context_parallel.data_utils import (
        cp_broadcast_tensor,
        cp_reduce_loss,
        cp_split_temporal,
    )

    x = torch.arange(2 * 3 * (world * 4) * 5, dtype=torch.float32).reshape(2, 3, world * 4, 5)
    local = cp_split_temporal(x, dim=2, group=group)
    assert tuple(local.shape) == (2, 3, 4, 5)
    _assert_close(_gather_local_chunks(local, group, dim=2), x)

    bcast = torch.arange(7, dtype=torch.float32) if rank == 0 else torch.full((7,), -1.0)
    cp_broadcast_tensor(bcast, group)
    _assert_close(bcast, torch.arange(7, dtype=torch.float32))

    param = torch.tensor(float(rank + 2), dtype=torch.float32, requires_grad=True)
    local_loss = param.square()
    tokens = torch.tensor(float(rank + 1), dtype=torch.float32)
    reduced = cp_reduce_loss(local_loss, group, tokens)
    token_sum = float(world * (world + 1) // 2)
    expected_forward = sum(float(r + 1) * float(r + 2) ** 2 for r in range(world)) / token_sum
    _assert_close(reduced.detach(), torch.tensor(expected_forward))
    reduced.backward()
    mean_tokens = token_sum / world
    expected_grad = 2.0 * float(rank + 2) * float(rank + 1) / mean_tokens
    _assert_close(param.grad, torch.tensor(expected_grad))

    param_equal = torch.tensor(float(rank + 3), dtype=torch.float32, requires_grad=True)
    reduced_equal = cp_reduce_loss(3.0 * param_equal, group)
    expected_equal_forward = sum(3.0 * float(r + 3) for r in range(world)) / world
    _assert_close(reduced_equal.detach(), torch.tensor(expected_equal_forward))
    reduced_equal.backward()
    _assert_close(param_equal.grad, torch.tensor(3.0))


def _case_halo_exchange(rank: int, world: int, group: dist.ProcessGroup) -> None:
    from diffusion.distributed.context_parallel.config import CpRuntimeConfig, set_cp_runtime_config
    from diffusion.distributed.context_parallel.halo_exchange import cp_halo_exchange

    batch, local_t, channels = 2, 4, 3
    left_size, right_size = 2, 1
    full = torch.arange(batch * world * local_t * channels, dtype=torch.float32).reshape(
        batch, world * local_t, channels
    )
    start = rank * local_t
    local_base = full[:, start : start + local_t].contiguous()

    expected_left = (
        full[:, start - left_size : start].contiguous()
        if rank > 0
        else torch.zeros(batch, left_size, channels, dtype=torch.float32)
    )
    expected_right = (
        full[:, start + local_t : start + local_t + right_size].contiguous()
        if rank < world - 1
        else torch.zeros(batch, right_size, channels, dtype=torch.float32)
    )
    expected_out = torch.cat([expected_left, local_base, expected_right], dim=1)

    for impl in ("collective", "p2p"):
        set_cp_runtime_config(CpRuntimeConfig(halo_impl=impl))
        local = local_base.detach().clone().requires_grad_(True)
        out = cp_halo_exchange(local, left_size=left_size, right_size=right_size, dim=1, group=group)
        _assert_close(out, expected_out)
        out.sum().backward()

        expected_grad = torch.ones_like(local)
        if rank < world - 1:
            expected_grad[:, local_t - left_size : local_t].add_(1.0)
        if rank > 0:
            expected_grad[:, :right_size].add_(1.0)
        _assert_close(local.grad, expected_grad)
        dist.barrier(group=group)


def _eager_cumulative_right(W: torch.Tensor) -> torch.Tensor:
    values = [W[:, 0]]
    for idx in range(1, W.shape[1]):
        values.append(torch.matmul(values[-1], W[:, idx]))
    return torch.stack(values, dim=1)


def _eager_cumulative_left(W: torch.Tensor) -> torch.Tensor:
    values = [W[:, 0]]
    for idx in range(1, W.shape[1]):
        values.append(torch.matmul(W[:, idx], values[-1]))
    return torch.stack(values, dim=1)


def _reference_scan(
    W_kv: torch.Tensor,
    U_kv: torch.Tensor,
    W_z: torch.Tensor,
    U_z: torch.Tensor,
    S_init_kv: torch.Tensor | None = None,
    S_init_z: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    bh, timesteps, dim, _ = W_kv.shape
    S_kv = torch.zeros(bh, dim, dim, dtype=W_kv.dtype, device=W_kv.device) if S_init_kv is None else S_init_kv
    S_z = torch.zeros(bh, dim, dtype=U_z.dtype, device=U_z.device) if S_init_z is None else S_init_z
    S_kv_all: list[torch.Tensor] = []
    S_z_all: list[torch.Tensor] = []
    for idx in range(timesteps):
        S_kv = torch.matmul(S_kv, W_kv[:, idx]) + U_kv[:, idx]
        S_z = torch.bmm(W_z[:, idx], S_z.unsqueeze(-1)).squeeze(-1) + U_z[:, idx]
        S_kv_all.append(S_kv)
        S_z_all.append(S_z)
    return torch.stack(S_kv_all, dim=1), torch.stack(S_z_all, dim=1)


class _EagerScan:
    @staticmethod
    def apply(
        W_kv: torch.Tensor,
        U_kv: torch.Tensor,
        W_z: torch.Tensor,
        U_z: torch.Tensor,
        S_init_kv: torch.Tensor | None = None,
        S_init_z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _reference_scan(W_kv, U_kv, W_z, U_z, S_init_kv, S_init_z)


def _make_scan_inputs(world: int, local_t: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(20240608)
    bh, dim, total_t = 2, 3, world * local_t
    eye = torch.eye(dim, dtype=torch.float32).view(1, 1, dim, dim)
    W_kv = eye + 0.025 * torch.randn(bh, total_t, dim, dim, dtype=torch.float32)
    W_z = eye + 0.025 * torch.randn(bh, total_t, dim, dim, dtype=torch.float32)
    U_kv = 0.05 * torch.randn(bh, total_t, dim, dim, dtype=torch.float32)
    U_z = 0.05 * torch.randn(bh, total_t, dim, dtype=torch.float32)
    return W_kv, U_kv, W_z, U_z


def _logical_order(x: torch.Tensor, world: int, local_t: int, reverse: bool) -> torch.Tensor:
    if not reverse:
        return x
    bh = x.shape[0]
    rest = x.shape[2:]
    return x.reshape(bh, world, local_t, *rest).flip(1).reshape(bh, world * local_t, *rest)


def _apply_active_mask(
    W_kv: torch.Tensor,
    U_kv: torch.Tensor,
    W_z: torch.Tensor,
    U_z: torch.Tensor,
    active_t: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dim = W_kv.shape[-1]
    total_t = W_kv.shape[1]
    valid = (torch.arange(total_t, device=W_kv.device) < active_t).to(W_kv.dtype)
    valid_kv = valid.view(1, total_t, 1, 1)
    valid_z = valid.view(1, total_t, 1)
    eye = torch.eye(dim, dtype=W_kv.dtype, device=W_kv.device).view(1, 1, dim, dim)
    W_kv_masked = valid_kv * W_kv + (1.0 - valid_kv) * eye
    U_kv_masked = valid_kv * U_kv
    W_z_masked = valid_kv * W_z + (1.0 - valid_kv) * eye
    U_z_masked = valid_z * U_z
    return W_kv_masked, U_kv_masked, W_z_masked, U_z_masked


def _case_frame_gdn_scan(rank: int, world: int, group: dist.ProcessGroup) -> None:
    from diffusion.distributed.context_parallel.config import CpRuntimeConfig, set_cp_runtime_config
    from diffusion.distributed.context_parallel import distributed_scan as scan_mod

    scan_mod._cumulative_matmul_right = _eager_cumulative_right
    scan_mod._cumulative_matmul_left = _eager_cumulative_left
    scan_mod._get_local_scan_cls = lambda device_is_cuda: _EagerScan

    local_t = 3
    start = rank * local_t

    for allgather_impl in ("collective", "list", "p2p"):
        for reverse in (False, True):
            set_cp_runtime_config(CpRuntimeConfig(allgather_impl=allgather_impl, scan_backend="torch"))
            full_inputs = _make_scan_inputs(world, local_t)
            local_inputs = [
                tensor[:, start : start + local_t].contiguous().detach().clone().requires_grad_(True)
                for tensor in full_inputs
            ]
            out_kv, out_z = scan_mod.cp_frame_gdn_scan(*local_inputs, group=group, reverse=reverse)

            ref_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in full_inputs]
            logical_inputs = [_logical_order(tensor, world, local_t, reverse) for tensor in ref_inputs]
            ref_kv, ref_z = _reference_scan(*logical_inputs)
            logical_rank = world - 1 - rank if reverse else rank
            ref_local_kv = ref_kv[:, logical_rank * local_t : (logical_rank + 1) * local_t]
            ref_local_z = ref_z[:, logical_rank * local_t : (logical_rank + 1) * local_t]
            _assert_close(out_kv, ref_local_kv, atol=1e-5, rtol=1e-5)
            _assert_close(out_z, ref_local_z, atol=1e-5, rtol=1e-5)

            loss = out_kv.sum() + 0.7 * out_z.sum()
            loss.backward()
            ref_loss = ref_kv.sum() + 0.7 * ref_z.sum()
            ref_loss.backward()
            names = ("W_kv", "U_kv", "W_z", "U_z")
            for name, local_tensor, ref_tensor in zip(names, local_inputs, ref_inputs):
                expected_grad = ref_tensor.grad[:, start : start + local_t]
                _assert_close(local_tensor.grad, expected_grad, atol=3e-4, rtol=3e-4)
            dist.barrier(group=group)

    set_cp_runtime_config(CpRuntimeConfig(allgather_impl="collective", scan_backend="torch"))
    active_t = world * local_t - 2
    full_inputs = _make_scan_inputs(world, local_t)
    local_inputs = [
        tensor[:, start : start + local_t].contiguous().detach().clone().requires_grad_(True)
        for tensor in full_inputs
    ]
    result = scan_mod.cp_frame_gdn_scan(*local_inputs, group=group, reverse=False, truncate_to_active=active_t)
    masked_inputs = _apply_active_mask(*full_inputs, active_t=active_t)
    ref_kv, ref_z = _reference_scan(*masked_inputs)
    _assert_close(result.S_kv_all, ref_kv[:, start : start + local_t], atol=1e-5, rtol=1e-5)
    _assert_close(result.S_z_all, ref_z[:, start : start + local_t], atol=1e-5, rtol=1e-5)
    terminal_idx = active_t - 1
    _assert_close(result.terminal_state_kv, ref_kv[:, terminal_idx], atol=1e-5, rtol=1e-5)
    _assert_close(result.terminal_state_z, ref_z[:, terminal_idx], atol=1e-5, rtol=1e-5)


_DIST_CASES: dict[str, Callable[..., None]] = {
    "core_data_utils": _case_core_data_utils,
    "halo_exchange": _case_halo_exchange,
    "frame_gdn_scan": _case_frame_gdn_scan,
}


def test_padding_layout_and_validation() -> None:
    from diffusion.distributed.context_parallel.data_utils import (
        cp_build_frame_valid_mask,
        cp_right_pad_size,
        cp_right_pad_temporal,
    )

    assert cp_right_pad_size(10, 3) == 2
    assert cp_right_pad_size(12, 3) == 0
    _assert_raises(ValueError, cp_right_pad_size, 10, 0)

    x = torch.arange(2 * 3 * 5 * 2, dtype=torch.float32).reshape(2, 3, 5, 2)
    padded = cp_right_pad_temporal(x, dim=2, pad_size=1, value=-9.0)
    assert tuple(padded.shape) == (2, 3, 6, 2)
    _assert_close(padded[:, :, :5], x)
    _assert_close(padded[:, :, 5], torch.full((2, 3, 2), -9.0))

    mask = cp_build_frame_valid_mask(padded, pad_frames=1)
    assert tuple(mask.shape) == (2, 1, 6, 1, 1)
    _assert_close(mask[:, :, :5], torch.ones_like(mask[:, :, :5]))
    _assert_close(mask[:, :, 5:], torch.zeros_like(mask[:, :, 5:]))
    _assert_raises(ValueError, cp_build_frame_valid_mask, torch.zeros(2, 3), 0)
    _assert_raises(ValueError, cp_build_frame_valid_mask, padded, 7)


def test_cp_data_utils_split_broadcast_and_loss() -> None:
    _run_distributed("core_data_utils")


def test_cp_halo_exchange_forward_backward_collective_and_p2p() -> None:
    _run_distributed("halo_exchange")


def test_cp_frame_gdn_scan_forward_reverse_truncate_and_gradients() -> None:
    _run_distributed("frame_gdn_scan", timeout_s=300)
