#!/usr/bin/env python3
"""Freeze Triton BF16 PISA2 semantics and gate a CuTe BF16 candidate.

The Triton aligned kernel is an immutable oracle in this program.  It prepares
BF16 K/V centroids once, freezes the exact FP32 threshold tensor, materializes
the device route predicate, and runs the prepared attention main.  A candidate
receives those exact same prepared tensors; it is never allowed to recalibrate
tau or rebuild the threshold.

Candidate modules may expose the Triton-shaped protocol::

    make_prepared_runner(q, k, v, kc, vc, global_thresh, *,
                         group_size, block_size, scale) -> callable
    materialize_route_mask(q, kc, global_thresh, *,
                           group_size, block_size, scale) -> uint8[B,H,N,N]

Alternatively ``--candidate module:factory`` calls ``factory`` with the same
named tensors plus ``trace_route_masks=True``.  The returned callable must
publish ``route_mask_trace`` with layout
``[B,H,N,ceil(N/64),3]`` = ``(word0, word1, popcount)``.  This mirrors the
existing native SM100 diagnostic trace without putting route indices in the
production kernel ABI.

For cross-architecture replay, first use ``--input-mode write`` and
``--oracle-mode write`` on H100.  Copy both directories to B200, then use
``--input-mode load`` and ``--oracle-mode check``.  Check mode consumes the
frozen Kc/Vc/threshold rather than a freshly reduced substitute.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CONTRACT = "pisa2_bf16_triton_bitwise_oracle_cutedsl_gate_v1"
INPUT_CONTRACT = "pisa2_bf16_frozen_qkv_v1"
BLOCK_SIZE = 64
HEAD_DIM = 128
GROUP_SIZE = 64
VALID_THRESHOLD_MODES = ("computed", "all_exact", "local_exact", "strict_tie")
VALID_INPUT_PROFILES = ("random", "zero_qk")
OUTPUT_LIMITS = {"max_abs": 0.08, "mean_abs": 0.01, "rel_l2": 0.01}
LSE_LIMITS = {"max_abs": 0.05, "mean_abs": 0.005, "rel_l2": 0.005}


@dataclass(frozen=True)
class CaseSpec:
    name: str
    batch: int
    heads: int
    tokens: int
    threshold_mode: str
    input_profile: str = "random"

    @property
    def num_blocks(self) -> int:
        return (self.tokens + BLOCK_SIZE - 1) // BLOCK_SIZE

    @property
    def route_tiles(self) -> int:
        return (self.num_blocks + GROUP_SIZE - 1) // GROUP_SIZE

    def validate(self) -> "CaseSpec":
        if not self.name or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for ch in self.name):
            raise ValueError("case name must use lowercase letters, digits, '_' or '-'")
        if min(self.batch, self.heads, self.tokens) <= 0:
            raise ValueError("case batch/heads/tokens must be positive")
        if self.threshold_mode not in VALID_THRESHOLD_MODES:
            raise ValueError("invalid threshold mode: %s" % self.threshold_mode)
        if self.input_profile not in VALID_INPUT_PROFILES:
            raise ValueError("invalid input profile: %s" % self.input_profile)
        if self.threshold_mode == "strict_tie" and self.input_profile != "zero_qk":
            raise ValueError("strict_tie requires the zero_qk input profile")
        return self


SMOKE_CASES = (
    CaseSpec("computed_full", 1, 1, 256, "computed"),
)

EDGE_CASES = (
    CaseSpec("computed_full", 1, 1, 256, "computed"),
    CaseSpec("computed_multi_bh", 2, 2, 320, "computed"),
    CaseSpec("computed_g64_boundary", 1, 1, 4096, "computed"),
    CaseSpec("computed_g64_plus_one", 1, 1, 4160, "computed"),
    CaseSpec("computed_token_tail", 1, 1, 4113, "computed"),
    CaseSpec("all_exact", 1, 1, 256, "all_exact"),
    CaseSpec("local_exact", 1, 1, 256, "local_exact"),
    CaseSpec("strict_tie_tail", 1, 1, 273, "strict_tie", "zero_qk"),
)


def suite_cases(name: str) -> Tuple[CaseSpec, ...]:
    if name == "smoke":
        return tuple(case.validate() for case in SMOKE_CASES)
    if name == "edge":
        return tuple(case.validate() for case in EDGE_CASES)
    raise ValueError("unknown suite %r" % name)


def signed_i32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def unsigned_i32(value: int) -> int:
    return value & 0xFFFFFFFF


def pack_selected_offsets(offsets: Iterable[int]) -> Tuple[int, int, int]:
    words = [0, 0]
    unique = sorted(set(int(offset) for offset in offsets))
    for offset in unique:
        if not 0 <= offset < GROUP_SIZE:
            raise ValueError("route offset is outside [0,64): %r" % offset)
        words[offset // 32] |= 1 << (offset % 32)
    return signed_i32(words[0]), signed_i32(words[1]), len(unique)


def unpack_packet(word0: int, word1: int, count: int) -> Tuple[int, ...]:
    offsets = tuple(
        offset
        for offset in range(GROUP_SIZE)
        if (unsigned_i32(word0 if offset < 32 else word1) >> (offset % 32)) & 1
    )
    if len(offsets) != int(count):
        raise ValueError("packet count %d disagrees with popcount %d" % (count, len(offsets)))
    return offsets


def local_exact_count(num_blocks: int) -> int:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    return sum(
        min(num_blocks - 1, query + 1) - max(0, query - 1) + 1
        for query in range(num_blocks)
    )


def parse_candidate_spec(value: str) -> Tuple[str, Optional[str]]:
    value = value.strip()
    if value in ("none", "triton"):
        return value, None
    module, separator, factory = value.partition(":")
    if not module or any(not part.isidentifier() for part in module.split(".")):
        raise ValueError("candidate must be none, triton, or module[:factory]")
    if separator and not factory.isidentifier():
        raise ValueError("candidate factory must be a Python identifier")
    return module, factory if separator else None


def artifact_filename(case: CaseSpec) -> str:
    case.validate()
    return case.name + ".pt"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _tensor_sha256(tensor: Any) -> str:
    raw = (
        tensor.detach()
        .contiguous()
        .view(__import__("torch").uint8)
        .cpu()
        .numpy()
        .tobytes()
    )
    return _sha256_bytes(raw)


def _error_stats(reference: Any, candidate: Any) -> Dict[str, float]:
    torch = __import__("torch")
    reference_f32 = reference.float()
    candidate_f32 = candidate.float()
    diff = candidate_f32 - reference_f32
    denominator = max(float(torch.linalg.vector_norm(reference_f32).item()), 1.0e-12)
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rel_l2": float(torch.linalg.vector_norm(diff).item()) / denominator,
    }


def _stats_pass(stats: Mapping[str, float], limits: Mapping[str, float]) -> bool:
    return all(float(stats[name]) <= float(limit) for name, limit in limits.items())


def _runtime_record(torch: Any, triton: Any, arch: str) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    expected = {"sm90": (9, 0), "sm100": (10, 0)}.get(arch)
    if expected is not None and capability != expected:
        raise RuntimeError("--arch %s requires CC%s, got %s" % (arch, expected, capability))
    return {
        "device": torch.cuda.get_device_name(0),
        "capability": list(capability),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "triton": str(triton.__version__),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }


def _install_tma_allocator(torch: Any, triton: Any) -> None:
    def allocate(size: int, alignment: int, stream: Any) -> Any:
        del alignment, stream
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(allocate)


def _source_identity(candidate_module: Optional[Any]) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__).resolve(),
        root / "kernels/online_piecewise_sparse_attn_bf16_aligned.py",
        root / "kernels/online_piecewise_sparse_attn_bf16_legacy.py",
        root / "experiments/pisa2/prepared_reference.py",
    ]
    core_paths = list(paths)
    candidate_paths = []
    if candidate_module is not None:
        source = inspect.getsourcefile(candidate_module)
        if source is not None:
            candidate_path = Path(source).resolve()
            if candidate_path not in paths:
                paths.append(candidate_path)
                candidate_paths.append(candidate_path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing semantic-gate source: %s" % missing)
    files = {}
    for path in paths:
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = str(path)
        files[label] = _sha256_file(path)
    core_labels = {}
    for path in core_paths:
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = str(path)
        core_labels[label] = files[label]
    return {
        "root": str(root),
        "files": files,
        "core_files": core_labels,
        "candidate_files": [str(path) for path in candidate_paths],
        # Candidate SM90 and SM100 adapters intentionally differ.  Only the
        # immutable oracle/harness sources participate in cross-arch replay.
        "mapping_sha256": _mapping_sha256(core_labels),
    }


def _input_path(directory: Path, case: CaseSpec) -> Path:
    return directory / artifact_filename(case)


def _make_or_load_inputs(torch: Any, case: CaseSpec, seed: int, mode: str, directory: Optional[Path]) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    shape = (case.batch, case.heads, case.tokens, HEAD_DIM)
    path = _input_path(directory, case) if directory is not None else None
    if mode == "load":
        if path is None or not path.is_file():
            raise FileNotFoundError("frozen input is missing: %s" % path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("contract") != INPUT_CONTRACT or payload.get("case") != asdict(case):
            raise RuntimeError("frozen input contract/case mismatch: %s" % path)
        q = payload["q"].to(device="cuda")
        k = payload["k"].to(device="cuda")
        v = payload["v"].to(device="cuda")
    else:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        if case.input_profile == "zero_qk":
            q = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
            k = torch.zeros_like(q)
        else:
            q = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
        if mode == "write":
            if path is None:
                raise ValueError("--input-mode write requires --input-dir")
            if path.exists():
                raise FileExistsError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "contract": INPUT_CONTRACT,
                    "case": asdict(case),
                    "seed": int(seed),
                    "q": q.cpu(),
                    "k": k.cpu(),
                    "v": v.cpu(),
                },
                path,
            )
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape:
            raise RuntimeError("invalid frozen %s: %s %s" % (name, tensor.dtype, tuple(tensor.shape)))
    hashes = {name: _tensor_sha256(tensor) for name, tensor in (("q", q), ("k", k), ("v", v))}
    return q.contiguous(), k.contiguous(), v.contiguous(), {
        "mode": mode,
        "path": str(path) if path is not None else None,
        "hashes": hashes,
        "mapping_sha256": _mapping_sha256(hashes),
    }


def _threshold_for_mode(torch: Any, computed: Any, case: CaseSpec) -> Any:
    if case.threshold_mode == "computed":
        return computed.detach().clone().contiguous()
    if case.threshold_mode == "all_exact":
        return torch.full_like(computed, -1.0e9)
    if case.threshold_mode == "local_exact":
        return torch.full_like(computed, 1.0e9)
    if case.threshold_mode == "strict_tie":
        return torch.zeros_like(computed)
    raise AssertionError(case.threshold_mode)


def _pack_dense_mask(torch: Any, mask: Any) -> Any:
    if mask.ndim != 4:
        raise ValueError("dense route mask must have rank four")
    batch, heads, queries, keys = (int(value) for value in mask.shape)
    route_tiles = (keys + GROUP_SIZE - 1) // GROUP_SIZE
    padded = torch.zeros(
        (batch, heads, queries, route_tiles * GROUP_SIZE),
        device=mask.device,
        dtype=torch.bool,
    )
    padded[..., :keys] = mask.to(torch.bool)
    grouped = padded.reshape(batch, heads, queries, route_tiles, GROUP_SIZE)
    shifts = torch.arange(32, device=mask.device, dtype=torch.int64)
    weights = torch.bitwise_left_shift(torch.ones_like(shifts), shifts)
    word0 = (grouped[..., :32].to(torch.int64) * weights).sum(dim=-1).to(torch.int32)
    word1 = (grouped[..., 32:].to(torch.int64) * weights).sum(dim=-1).to(torch.int32)
    count = grouped.sum(dim=-1).to(torch.int32)
    return torch.stack((word0, word1, count), dim=-1)


def _unpack_trace(torch: Any, trace: Any, num_blocks: int) -> Tuple[Any, Dict[str, Any]]:
    if trace.dtype != torch.int32 or trace.ndim != 5 or int(trace.shape[-1]) != 3:
        raise TypeError("route trace must be int32 [B,H,N,R,3], got %s %s" % (trace.dtype, tuple(trace.shape)))
    expected_tiles = (num_blocks + GROUP_SIZE - 1) // GROUP_SIZE
    if int(trace.shape[2]) != num_blocks or int(trace.shape[3]) != expected_tiles:
        raise ValueError("route trace extent mismatch: %s" % (tuple(trace.shape),))
    shifts = torch.arange(32, device=trace.device, dtype=torch.int64)
    unsigned0 = trace[..., 0].to(torch.int64) & 0xFFFFFFFF
    unsigned1 = trace[..., 1].to(torch.int64) & 0xFFFFFFFF
    bits0 = torch.bitwise_and(torch.bitwise_right_shift(unsigned0[..., None], shifts), 1)
    bits1 = torch.bitwise_and(torch.bitwise_right_shift(unsigned1[..., None], shifts), 1)
    grouped = torch.cat((bits0, bits1), dim=-1).to(torch.uint8)
    popcount = grouped.sum(dim=-1).to(torch.int32)
    count_mismatch = trace[..., 2] != popcount
    dense_padded = grouped.flatten(start_dim=-2)
    padding = dense_padded[..., num_blocks:]
    dense = dense_padded[..., :num_blocks].contiguous()
    evidence = {
        "count_mismatch_packets": int(count_mismatch.sum().item()),
        "padding_set_bits": int(padding.sum().item()) if padding.numel() else 0,
        "packet_count": int(trace[..., 2].numel()),
        "trace_sha256": _tensor_sha256(trace),
        "dense_sha256": _tensor_sha256(dense),
    }
    evidence["passes"] = evidence["count_mismatch_packets"] == 0 and evidence["padding_set_bits"] == 0
    return dense, evidence


def _route_census(torch: Any, mask: Any) -> Dict[str, Any]:
    counts = mask.sum(dim=-1, dtype=torch.int32)
    keys = int(mask.shape[-1])
    route_tiles = (keys + GROUP_SIZE - 1) // GROUP_SIZE
    padded = torch.zeros((*mask.shape[:-1], route_tiles * GROUP_SIZE), device=mask.device, dtype=torch.uint8)
    padded[..., :keys] = mask
    group_counts = padded.reshape(*mask.shape[:-1], route_tiles, GROUP_SIZE).sum(dim=-1, dtype=torch.int32)
    return {
        "shape": list(mask.shape),
        "mask_sha256": _tensor_sha256(mask),
        "exact_count": int(mask.sum().item()),
        "slot_count": int(mask.numel()),
        "density": float(mask.float().mean().item()),
        "per_query_min": int(counts.min().item()),
        "per_query_max": int(counts.max().item()),
        "per_query_counts_sha256": _tensor_sha256(counts),
        "per_group_counts_sha256": _tensor_sha256(group_counts),
    }


def _route_comparison(torch: Any, oracle: Any, candidate: Any) -> Dict[str, Any]:
    if tuple(candidate.shape) != tuple(oracle.shape):
        return {"shape_matches": False, "candidate_shape": list(candidate.shape), "oracle_shape": list(oracle.shape), "passes": False}
    candidate = candidate.to(torch.uint8)
    mismatch = candidate != oracle
    indices = torch.nonzero(mismatch, as_tuple=False)[:16].cpu().tolist()
    examples = []
    for coordinate in indices:
        index = tuple(coordinate)
        examples.append({
            "coordinate": coordinate,
            "oracle": int(oracle[index].item()),
            "candidate": int(candidate[index].item()),
        })
    result = {
        "shape_matches": True,
        "bit_mismatch_count": int(mismatch.sum().item()),
        "false_positive_count": int(((candidate == 1) & (oracle == 0)).sum().item()),
        "false_negative_count": int(((candidate == 0) & (oracle == 1)).sum().item()),
        "first_mismatches": examples,
        "candidate_census": _route_census(torch, candidate),
        "oracle_census": _route_census(torch, oracle),
    }
    result["passes"] = result["bit_mismatch_count"] == 0
    return result


def _oracle_path(directory: Path, case: CaseSpec) -> Path:
    return directory / artifact_filename(case)


def _load_frozen_oracle(torch: Any, directory: Path, case: CaseSpec, input_record: Mapping[str, Any], source: Mapping[str, Any]) -> Tuple[Dict[str, Any], Path]:
    path = _oracle_path(directory, case)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("contract") != CONTRACT or payload.get("case") != asdict(case):
        raise RuntimeError("oracle contract/case mismatch: %s" % path)
    manifest = payload.get("manifest", {})
    if manifest.get("input_hashes") != input_record["hashes"]:
        raise RuntimeError("oracle QKV identity does not match frozen input")
    if manifest.get("source_mapping_sha256") != source["mapping_sha256"]:
        raise RuntimeError("oracle source identity changed")
    stored_manifest_hash = manifest.get("mapping_sha256")
    manifest_without_hash = {
        key: value for key, value in manifest.items() if key != "mapping_sha256"
    }
    if stored_manifest_hash != _mapping_sha256(manifest_without_hash):
        raise RuntimeError("oracle manifest checksum is invalid")
    tensor_hashes = {
        "kc": _tensor_sha256(payload["kc"]),
        "vc": _tensor_sha256(payload["vc"]),
        "threshold": _tensor_sha256(payload["threshold"]),
    }
    if tensor_hashes != manifest.get("prepared_hashes"):
        raise RuntimeError("oracle prepared tensor checksum is invalid")
    if _tensor_sha256(payload["route_mask"]) != manifest.get("route_mask_sha256"):
        raise RuntimeError("oracle route-mask checksum is invalid")
    if _tensor_sha256(payload["route_trace"]) != manifest.get("route_trace_sha256"):
        raise RuntimeError("oracle route-trace checksum is invalid")
    repacked = _pack_dense_mask(torch, payload["route_mask"])
    if not torch.equal(repacked.cpu(), payload["route_trace"].cpu()):
        raise RuntimeError("oracle dense mask and packed trace disagree")
    return payload, path


def _write_frozen_oracle(torch: Any, directory: Path, case: CaseSpec, payload: Mapping[str, Any]) -> Path:
    path = _oracle_path(directory, case)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)
    return path


def _resolve_candidate(spec: str, aligned: Any) -> Tuple[Optional[Any], Optional[str], str]:
    module_name, factory_name = parse_candidate_spec(spec)
    if module_name == "none":
        return None, None, "none"
    if module_name == "triton":
        return aligned, None, "triton"
    return importlib.import_module(module_name), factory_name, module_name


def _normalise_factory_result(result: Any) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(result, tuple):
        if len(result) != 2:
            raise TypeError("candidate factory tuple must be (runner, metadata)")
        runner, metadata = result
    elif isinstance(result, Mapping):
        runner = result.get("runner")
        metadata = result.get("metadata", {})
    else:
        runner, metadata = result, {}
    if not callable(runner):
        raise TypeError("candidate factory must return a callable runner")
    if not isinstance(metadata, Mapping):
        raise TypeError("candidate metadata must be a mapping")
    return runner, dict(metadata)


def _snapshot_runner_value(value: Any, runner: Any) -> Tuple[Any, Optional[Any]]:
    output = value[0] if isinstance(value, tuple) else value
    lse = (
        value[1]
        if isinstance(value, tuple) and len(value) > 1
        else getattr(runner, "lse", None)
    )
    if output is None:
        raise RuntimeError("candidate runner returned no output")
    return output.clone(), lse.clone() if lse is not None else None


def _candidate_run_and_route(torch: Any, module: Any, factory_name: Optional[str], q: Any, k: Any, v: Any, kc: Any, vc: Any, threshold: Any, scale: float) -> Dict[str, Any]:
    metadata = {}
    if factory_name is not None:
        factory = getattr(module, factory_name)
        runner, metadata = _normalise_factory_result(
            factory(
                q=q,
                k=k,
                v=v,
                kc=kc,
                vc=vc,
                global_thresh=threshold,
                group_size=GROUP_SIZE,
                block_size=BLOCK_SIZE,
                scale=scale,
                trace_route_masks=True,
            )
        )
        runner()
        torch.cuda.synchronize()
        first_value = runner()
        torch.cuda.synchronize()
        first_output, first_lse = _snapshot_runner_value(first_value, runner)
        trace_first = getattr(runner, "route_mask_trace", None)
        if trace_first is None:
            raise RuntimeError("factory candidate did not publish route_mask_trace")
        trace_first = trace_first.clone()
        second_value = runner()
        torch.cuda.synchronize()
        second_output, second_lse = _snapshot_runner_value(second_value, runner)
        trace_second = getattr(runner, "route_mask_trace").clone()
        if not torch.equal(trace_first, trace_second):
            raise RuntimeError("candidate route trace is not repeatable")
        dense, trace_evidence = _unpack_trace(torch, trace_second, int(kc.shape[2]))
    else:
        if not hasattr(module, "make_prepared_runner") or not hasattr(module, "materialize_route_mask"):
            raise TypeError("aligned candidate protocol requires make_prepared_runner and materialize_route_mask")
        runner = module.make_prepared_runner(
            q, k, v, kc, vc, threshold,
            group_size=GROUP_SIZE, block_size=BLOCK_SIZE, scale=scale,
        )
        route_first = module.materialize_route_mask(
            q, kc, threshold,
            group_size=GROUP_SIZE, block_size=BLOCK_SIZE, scale=scale,
        )
        route_second = module.materialize_route_mask(
            q, kc, threshold,
            group_size=GROUP_SIZE, block_size=BLOCK_SIZE, scale=scale,
        )
        torch.cuda.synchronize()
        if not torch.equal(route_first, route_second):
            raise RuntimeError("candidate dense route mask is not repeatable")
        dense = route_second.to(torch.uint8)
        trace_second = _pack_dense_mask(torch, dense)
        trace_evidence = {
            "source": "candidate.materialize_route_mask",
            "trace_sha256": _tensor_sha256(trace_second),
            "dense_sha256": _tensor_sha256(dense),
            "count_mismatch_packets": 0,
            "padding_set_bits": 0,
            "passes": True,
        }
        runner()
        torch.cuda.synchronize()
        first_value = runner()
        torch.cuda.synchronize()
        first_output, first_lse = _snapshot_runner_value(first_value, runner)
        second_value = runner()
        torch.cuda.synchronize()
        second_output, second_lse = _snapshot_runner_value(second_value, runner)
    return {
        "output": second_output,
        "output_repeatable": bool(torch.equal(first_output, second_output)),
        "output_finite": bool(torch.isfinite(second_output).all().item()),
        "lse": second_lse.clone() if second_lse is not None else None,
        "lse_repeatable": bool(torch.equal(first_lse, second_lse)) if first_lse is not None and second_lse is not None else None,
        "route_mask": dense,
        "route_trace": trace_second,
        "trace_evidence": trace_evidence,
        "metadata": metadata,
    }


def _independent_references(torch: Any, prepared_reference: Any, case: CaseSpec, q: Any, k: Any, v: Any, kc: Any, vc: Any, q_scale: Any, k_scale: Any, threshold: Any, scale: float) -> Tuple[Any, Any, Dict[str, Any]]:
    output, lse = prepared_reference(
        q, k, v, q_scale, k_scale, kc, k_scale, vc, threshold,
        scale, BLOCK_SIZE, GROUP_SIZE,
    )
    evidence = {"kind": "prepared_pisa2_fp32_model"}
    if case.threshold_mode == "all_exact":
        logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
        dense = torch.softmax(logits, dim=-1) @ v.float()
        evidence["analytic_kind"] = "dense_attention"
        evidence["prepared_vs_analytic"] = _error_stats(dense, output)
    elif case.threshold_mode == "strict_tie":
        uniform = v.float().mean(dim=2, keepdim=True).expand_as(output)
        evidence["analytic_kind"] = "zero_logit_uniform_mean_with_true_tail_weight"
        evidence["prepared_vs_analytic"] = _error_stats(uniform, output)
    return output, lse, evidence


def _validate_oracle_route_semantics(case: CaseSpec, mask: Any) -> Dict[str, Any]:
    exact_count = int(mask.sum().item())
    expected = None
    if case.threshold_mode == "all_exact":
        expected = case.batch * case.heads * case.num_blocks * case.num_blocks
    elif case.threshold_mode in ("local_exact", "strict_tie"):
        expected = case.batch * case.heads * local_exact_count(case.num_blocks)
    return {
        "expected_exact_count": expected,
        "actual_exact_count": exact_count,
        "passes": expected is None or exact_count == expected,
    }


def _run_case(torch: Any, triton: Any, aligned: Any, prepared_reference: Any, case: CaseSpec, args: argparse.Namespace, runtime: Mapping[str, Any], source: Mapping[str, Any], candidate_module: Optional[Any], candidate_factory: Optional[str], candidate_label: str) -> Dict[str, Any]:
    q, k, v, input_record = _make_or_load_inputs(
        torch, case, args.seed, args.input_mode, args.input_dir
    )
    scale = HEAD_DIM ** -0.5
    fresh_kc, fresh_vc, computed_threshold, q_scale, k_scale = aligned.prepare_qkv(
        q, k, v, tau=args.tau, block_size=BLOCK_SIZE, scale=scale
    )
    torch.cuda.synchronize()
    frozen_payload = None
    oracle_path = None
    if args.oracle_mode == "check":
        frozen_payload, oracle_path = _load_frozen_oracle(
            torch, args.oracle_dir, case, input_record, source
        )
        kc = frozen_payload["kc"].to(device="cuda")
        vc = frozen_payload["vc"].to(device="cuda")
        threshold = frozen_payload["threshold"].to(device="cuda")
    else:
        kc, vc = fresh_kc, fresh_vc
        threshold = _threshold_for_mode(torch, computed_threshold, case)

    prepared_hashes = {
        "kc": _tensor_sha256(kc),
        "vc": _tensor_sha256(vc),
        "threshold": _tensor_sha256(threshold),
    }
    fresh_prepared_hashes = {
        "kc": _tensor_sha256(fresh_kc),
        "vc": _tensor_sha256(fresh_vc),
        "computed_threshold": _tensor_sha256(computed_threshold),
    }
    threshold_hash_before = prepared_hashes["threshold"]
    operand_hashes_before = {
        "q": input_record["hashes"]["q"],
        "k": input_record["hashes"]["k"],
        "v": input_record["hashes"]["v"],
        "kc": prepared_hashes["kc"],
        "vc": prepared_hashes["vc"],
        "threshold": prepared_hashes["threshold"],
    }

    route_g64 = aligned.materialize_route_mask(
        q, kc, threshold, group_size=64, block_size=BLOCK_SIZE, scale=scale
    ).to(torch.uint8)
    route_g32 = aligned.materialize_route_mask(
        q, kc, threshold, group_size=32, block_size=BLOCK_SIZE, scale=scale
    ).to(torch.uint8)
    torch.cuda.synchronize()
    triton_cross_group = _route_comparison(torch, route_g64, route_g32)
    if frozen_payload is not None:
        frozen_mask = frozen_payload["route_mask"].to(device="cuda")
        frozen_replay = _route_comparison(torch, frozen_mask, route_g64)
        if frozen_payload["manifest"].get("prepared_hashes") != prepared_hashes:
            raise RuntimeError("loaded frozen prepared tensor hashes are internally inconsistent")
    else:
        frozen_mask = route_g64
        frozen_replay = _route_comparison(torch, frozen_mask, route_g64)

    oracle_trace = _pack_dense_mask(torch, frozen_mask)
    oracle_census = _route_census(torch, frozen_mask)
    route_semantics = _validate_oracle_route_semantics(case, frozen_mask)

    triton_runner = aligned.make_prepared_runner(
        q, k, v, kc, vc, threshold,
        group_size=GROUP_SIZE, block_size=BLOCK_SIZE, scale=scale,
    )
    triton_runner()
    torch.cuda.synchronize()
    triton_first = triton_runner().clone()
    torch.cuda.synchronize()
    triton_second = triton_runner().clone()
    torch.cuda.synchronize()
    triton_repeatable = bool(torch.equal(triton_first, triton_second))
    triton_finite = bool(torch.isfinite(triton_second).all().item())

    reference_output, reference_lse, reference_evidence = _independent_references(
        torch, prepared_reference, case, q, k, v, kc, vc, q_scale, k_scale,
        threshold, scale,
    )
    triton_reference_error = _error_stats(reference_output, triton_second)
    triton_reference_pass = _stats_pass(triton_reference_error, OUTPUT_LIMITS)

    oracle_manifest = {
        "contract": CONTRACT,
        "case": asdict(case),
        "input_hashes": input_record["hashes"],
        "prepared_hashes": prepared_hashes,
        "route_mask_sha256": _tensor_sha256(frozen_mask),
        "route_trace_sha256": _tensor_sha256(oracle_trace),
        "triton_output_sha256": _tensor_sha256(triton_second),
        "source_mapping_sha256": source["mapping_sha256"],
    }
    oracle_manifest["mapping_sha256"] = _mapping_sha256(oracle_manifest)
    if args.oracle_mode == "write":
        oracle_path = _write_frozen_oracle(
            torch,
            args.oracle_dir,
            case,
            {
                "contract": CONTRACT,
                "case": asdict(case),
                "manifest": oracle_manifest,
                "kc": kc.cpu(),
                "vc": vc.cpu(),
                "threshold": threshold.cpu(),
                "route_mask": frozen_mask.cpu(),
                "route_trace": oracle_trace.cpu(),
            },
        )

    candidate_record = None
    candidate_pass = True
    if candidate_module is not None:
        candidate = _candidate_run_and_route(
            torch, candidate_module, candidate_factory,
            q, k, v, kc, vc, threshold, scale,
        )
        route_comparison = _route_comparison(
            torch, frozen_mask, candidate["route_mask"]
        )
        output_contract = {
            "shape": list(candidate["output"].shape),
            "dtype": str(candidate["output"].dtype),
            "shape_matches": tuple(candidate["output"].shape) == tuple(v.shape),
            "dtype_matches": candidate["output"].dtype == v.dtype,
        }
        output_contract["passes"] = (
            output_contract["shape_matches"]
            and output_contract["dtype_matches"]
        )
        output_vs_triton = (
            _error_stats(triton_second, candidate["output"])
            if output_contract["shape_matches"]
            else None
        )
        output_vs_reference = (
            _error_stats(reference_output, candidate["output"])
            if output_contract["shape_matches"]
            else None
        )
        lse_record = None
        lse_pass = True
        if candidate["lse"] is not None:
            lse_contract = {
                "shape": list(candidate["lse"].shape),
                "dtype": str(candidate["lse"].dtype),
                "shape_matches": tuple(candidate["lse"].shape)
                == tuple(reference_lse.shape),
                "dtype_matches": candidate["lse"].dtype == torch.float32,
            }
            lse_contract["passes"] = (
                lse_contract["shape_matches"]
                and lse_contract["dtype_matches"]
            )
            lse_error = (
                _error_stats(reference_lse, candidate["lse"])
                if lse_contract["shape_matches"]
                else None
            )
            lse_pass = (
                lse_contract["passes"] is True
                and candidate["lse_repeatable"] is True
                and bool(torch.isfinite(candidate["lse"]).all().item())
                and lse_error is not None
                and _stats_pass(lse_error, LSE_LIMITS)
            )
            lse_record = {
                "contract": lse_contract,
                "error_vs_reference": lse_error,
                "passes": lse_pass,
            }
        candidate_pass = (
            route_comparison["passes"] is True
            and candidate["trace_evidence"]["passes"] is True
            and output_contract["passes"] is True
            and candidate["output_repeatable"] is True
            and candidate["output_finite"] is True
            and output_vs_triton is not None
            and _stats_pass(output_vs_triton, OUTPUT_LIMITS)
            and output_vs_reference is not None
            and _stats_pass(output_vs_reference, OUTPUT_LIMITS)
            and lse_pass
        )
        candidate_record = {
            "label": candidate_label,
            "factory": candidate_factory,
            "metadata": candidate["metadata"],
            "route": route_comparison,
            "trace": candidate["trace_evidence"],
            "output_contract": output_contract,
            "output_repeatable": candidate["output_repeatable"],
            "output_finite": candidate["output_finite"],
            "output_sha256": _tensor_sha256(candidate["output"]),
            "output_error_vs_triton": output_vs_triton,
            "output_error_vs_independent_reference": output_vs_reference,
            "lse": lse_record,
            "passes": candidate_pass,
        }

    operand_hashes_after = {
        name: _tensor_sha256(tensor)
        for name, tensor in (
            ("q", q),
            ("k", k),
            ("v", v),
            ("kc", kc),
            ("vc", vc),
            ("threshold", threshold),
        )
    }
    operands_immutable = operand_hashes_after == operand_hashes_before
    threshold_hash_after = operand_hashes_after["threshold"]
    threshold_immutable = threshold_hash_after == threshold_hash_before
    passes = (
        triton_cross_group["passes"] is True
        and frozen_replay["passes"] is True
        and route_semantics["passes"] is True
        and triton_repeatable
        and triton_finite
        and triton_reference_pass
        and threshold_immutable
        and operands_immutable
        and candidate_pass
    )
    return {
        "event": "case",
        "contract": CONTRACT,
        "case": asdict(case),
        "shape_bhtd": [case.batch, case.heads, case.tokens, HEAD_DIM],
        "block_size": BLOCK_SIZE,
        "group_size": GROUP_SIZE,
        "num_blocks": case.num_blocks,
        "route_tiles": case.route_tiles,
        "seed": args.seed,
        "tau": args.tau,
        "runtime": dict(runtime),
        "input": input_record,
        "fresh_prepared_hashes": fresh_prepared_hashes,
        "frozen_prepared_hashes": prepared_hashes,
        "fresh_vs_frozen": {
            "kc_equal": fresh_prepared_hashes["kc"] == prepared_hashes["kc"],
            "vc_equal": fresh_prepared_hashes["vc"] == prepared_hashes["vc"],
            "computed_threshold_equals_frozen": fresh_prepared_hashes["computed_threshold"] == prepared_hashes["threshold"],
            "note": "diagnostic only in oracle check mode; frozen tensors are consumed",
        },
        "threshold": {
            "dtype": str(threshold.dtype),
            "shape": list(threshold.shape),
            "sha256_before": threshold_hash_before,
            "sha256_after": threshold_hash_after,
            "immutable": threshold_immutable,
        },
        "prepared_operands": {
            "hashes_before": operand_hashes_before,
            "hashes_after": operand_hashes_after,
            "immutable": operands_immutable,
        },
        "oracle": {
            "mode": args.oracle_mode,
            "path": str(oracle_path) if oracle_path is not None else None,
            "manifest": oracle_manifest,
            "census": oracle_census,
            "route_semantics": route_semantics,
            "triton_g32_vs_g64": triton_cross_group,
            "frozen_replay": frozen_replay,
            "triton_output_repeatable": triton_repeatable,
            "triton_output_finite": triton_finite,
            "triton_output_error_vs_independent_reference": triton_reference_error,
            "triton_reference_pass": triton_reference_pass,
            "independent_reference": reference_evidence,
        },
        "candidate": candidate_record,
        "passes": passes,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("smoke", "edge"), default="smoke")
    parser.add_argument("--case", action="append", default=[], help="run only named suite case(s)")
    parser.add_argument("--arch", choices=("auto", "sm90", "sm100"), default="auto")
    parser.add_argument("--candidate", default="none", help="none, triton, or module[:factory]")
    parser.add_argument("--require-candidate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau", type=float, default=1.5)
    parser.add_argument("--input-mode", choices=("generate", "write", "load"), default="generate")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--oracle-mode", choices=("off", "write", "check"), default="off")
    parser.add_argument("--oracle-dir", type=Path)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> Tuple[CaseSpec, ...]:
    if args.output.exists():
        raise FileExistsError(args.output)
    if not math.isfinite(args.tau) or args.tau < 1.0:
        raise ValueError("tau must be finite and >= 1")
    cases = suite_cases(args.suite)
    if args.case:
        requested = set(args.case)
        known = {case.name for case in cases}
        unknown = requested - known
        if unknown:
            raise ValueError("unknown case(s) for suite %s: %s" % (args.suite, sorted(unknown)))
        cases = tuple(case for case in cases if case.name in requested)
    if args.input_mode in ("write", "load") and args.input_dir is None:
        raise ValueError("input write/load mode requires --input-dir")
    if args.oracle_mode in ("write", "check") and args.oracle_dir is None:
        raise ValueError("oracle write/check mode requires --oracle-dir")
    module_name, _ = parse_candidate_spec(args.candidate)
    if args.require_candidate and module_name == "none":
        raise ValueError("--require-candidate rejects --candidate none")
    return cases


def main() -> None:
    args = _parse_args()
    cases = _validate_args(args)
    import torch
    import triton
    from experiments.pisa2.prepared_reference import pisa2_prepared_reference
    from kernels import online_piecewise_sparse_attn_bf16_aligned as aligned

    _install_tma_allocator(torch, triton)
    candidate_module, candidate_factory, candidate_label = _resolve_candidate(
        args.candidate, aligned
    )
    runtime = _runtime_record(torch, triton, args.arch)
    source = _source_identity(candidate_module)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.output.open("x", encoding="utf-8") as handle:
        start = {
            "event": "start",
            "contract": CONTRACT,
            "time": time.strftime("%F %T %z"),
            "suite": args.suite,
            "cases": [asdict(case) for case in cases],
            "candidate": candidate_label,
            "candidate_factory": candidate_factory,
            "runtime": runtime,
            "source": source,
        }
        handle.write(json.dumps(start, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        print(json.dumps(start, sort_keys=True), flush=True)
        for case in cases:
            try:
                with torch.no_grad():
                    row = _run_case(
                        torch, triton, aligned, pisa2_prepared_reference,
                        case, args, runtime, source,
                        candidate_module, candidate_factory, candidate_label,
                    )
            except Exception as exc:
                row = {
                    "event": "case",
                    "contract": CONTRACT,
                    "case": asdict(case),
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "passes": False,
                }
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
            torch.cuda.empty_cache()
        complete = {
            "event": "complete",
            "contract": CONTRACT,
            "case_count": len(rows),
            "pass_count": sum(row.get("passes") is True for row in rows),
            "candidate_required": bool(args.require_candidate),
            "passes": bool(rows) and all(row.get("passes") is True for row in rows),
        }
        handle.write(json.dumps(complete, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        print(json.dumps(complete, sort_keys=True), flush=True)
    if not complete["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
