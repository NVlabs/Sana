#!/usr/bin/env python3
"""All-position correctness evidence for the SM100 lean6 route-index kernel.

The audit path intentionally uses only the Python standard library.  CUDA
dependencies are imported only after argument parsing selects device mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from experiments.sol_attn.bf16_triton_sol_attn_contract import (
    BATCH_HEAD_PAIRS,
    CONTRACT_PATH,
    DENSITIES,
    LSE_LIMITS,
    OUTPUT_LIMITS,
    TOKENS,
    load_and_validate_contract,
)
from experiments.sol_attn.full45_candidate_backends import (
    BACKENDS as FULL45_CANDIDATE_BACKENDS,
    DEFAULT_BACKEND,
    get_backend,
    get_backend_for_kernel,
    load_runner_factories,
    source_paths as candidate_source_paths,
)


CORRECTNESS_CONTRACT = "b200_lean6_routeidx_full45_correctness_v1"
BLOCK_SIZE = 64
HEAD_DIM = 128
GROUP_SIZE = 64
CALIBRATION_STEPS = 18
MAX_DENSITY_ERROR = 0.01
ROOT = Path(__file__).resolve().parents[2]
HARNESS_SOURCE = Path(__file__).resolve()
TIMING_HARNESS_SOURCE = (
    ROOT / "experiments/sol_attn/benchmark_b200_lean6_routeidx_vs_triton_sol_attn_full45.py"
)
SOURCE_PATHS = candidate_source_paths(
    ROOT,
    DEFAULT_BACKEND,
    correctness_harness=HARNESS_SOURCE,
    timing_harness=TIMING_HARNESS_SOURCE,
)
KERNEL_SOURCE = SOURCE_PATHS["kernel"]
RUNNER_SOURCE = SOURCE_PATHS["runner"]
SEMANTIC_SOURCE = SOURCE_PATHS["semantic_kernel"]
PARENT_KERNEL_SOURCE = SOURCE_PATHS["parent_kernel"]
PARENT_RUNNER_SOURCE = SOURCE_PATHS["parent_runner"]
REFERENCE_SOURCE = SOURCE_PATHS["prepared_reference"]
SEMANTIC_HELPER_SOURCE = SOURCE_PATHS["semantic_helper"]
LEGACY_SOURCE = SOURCE_PATHS["legacy_kernel"]
ROUTING_SOURCE = SOURCE_PATHS["routing_kernel"]
INPUT_NAMES = ("q", "k", "v", "kc", "vc", "threshold")
EDGE_RECEIPT_CONTRACT = "b200_lean6_routeidx_r0_edge_receipt_v1"
KERNEL_RELATIVE = "kernels/sol_attn_sm100/native_bf16_lean6_routeidx_fwd.py"
RUNTIME_FIELDS = {
    "visible_device_count",
    "device_name",
    "capability",
    "torch",
    "torch_cuda",
    "triton",
    "uuid",
    "cuda_visible_devices",
    "slurm_job_id",
    "slurm_node",
}
SELECTION_CURSOR_POLICY = "word0_then_word1_lowbit_clear"


def case_id(tokens: int, batch: int, heads: int, density: float) -> str:
    """Return the canonical full45 case identifier."""

    return f"T{int(tokens)}-B{int(batch)}-H{int(heads)}-d{float(density):.2f}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label}: expected lowercase SHA256")
    return value


def _canonical_uuid(value: Any) -> str:
    text = str(value).strip().lower()
    if text.startswith("gpu-"):
        text = text[4:]
    return text.replace("-", "")


def _validate_runtime_signature(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != RUNTIME_FIELDS:
        raise ValueError(f"{label}: runtime signature fields mismatch")
    runtime = dict(value)
    if runtime["visible_device_count"] != 1:
        raise ValueError(f"{label}: runtime requires one visible GPU")
    device_name = runtime["device_name"]
    if not isinstance(device_name, str) or not any(
        token in device_name.lower() for token in ("b200", "gb200")
    ):
        raise ValueError(f"{label}: runtime requires B200/GB200")
    if runtime["capability"] != [10, 0]:
        raise ValueError(f"{label}: runtime requires SM100")
    torch_version = runtime["torch"]
    if (
        not isinstance(torch_version, str)
        or not torch_version.startswith("2.11.")
        or not torch_version.endswith("+cu128")
    ):
        raise ValueError(f"{label}: runtime requires torch 2.11 cu128 family")
    if runtime["torch_cuda"] != "12.8":
        raise ValueError(f"{label}: runtime requires CUDA 12.8")
    triton_version = runtime["triton"]
    if not isinstance(triton_version, str) or not triton_version.startswith("3.7."):
        raise ValueError(f"{label}: runtime requires Triton 3.7 family")
    uuid = runtime["uuid"]
    canonical = _canonical_uuid(uuid)
    if len(canonical) != 32 or any(character not in "0123456789abcdef" for character in canonical):
        raise ValueError(f"{label}: runtime GPU UUID is invalid")
    visible = runtime["cuda_visible_devices"]
    if not isinstance(visible, str) or not visible.strip() or "," in visible:
        raise ValueError(f"{label}: runtime must expose exactly one CUDA device")
    for name in ("slurm_job_id", "slurm_node"):
        if not isinstance(runtime[name], str) or not runtime[name].strip():
            raise ValueError(f"{label}: runtime missing {name}")
    return runtime


def _validate_edge_receipt(
    value: Any,
    *,
    expected_kernel_relative: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("edge receipt: expected object")
    if value.get("contract") != EDGE_RECEIPT_CONTRACT:
        raise ValueError("edge receipt contract mismatch")
    if value.get("archive_integrity_pass") is not True:
        raise ValueError("edge receipt archive integrity failed")
    if value.get("device_correctness_pass") is not True:
        raise ValueError("edge receipt device correctness failed")
    passed = value.get("correctness_checks_passed")
    total = value.get("correctness_checks_total")
    if (
        isinstance(passed, bool)
        or isinstance(total, bool)
        or not isinstance(passed, int)
        or not isinstance(total, int)
        or total <= 0
        or passed != total
    ):
        raise ValueError("edge receipt correctness check counts are incomplete")
    kernel = value.get("kernel_identity")
    if kernel is None:
        frozen = value.get("frozen_source")
        if isinstance(frozen, Mapping):
            identities = frozen.get("identities")
            if isinstance(identities, Mapping):
                kernel = identities.get("kernel")
    if not isinstance(kernel, Mapping):
        raise ValueError("edge receipt kernel identity missing")
    kernel_path = kernel.get("path")
    allowed_paths = {
        spec.kernel_relative for spec in FULL45_CANDIDATE_BACKENDS.values()
    }
    if kernel_path not in allowed_paths:
        raise ValueError("edge receipt kernel path mismatch")
    if (
        expected_kernel_relative is not None
        and kernel_path != expected_kernel_relative
    ):
        raise ValueError("edge receipt candidate backend mismatch")
    kernel_sha = _require_sha256(kernel.get("sha256"), "edge receipt kernel SHA256")
    return {
        "contract": EDGE_RECEIPT_CONTRACT,
        "archive_integrity_pass": True,
        "device_correctness_pass": True,
        "correctness_checks_passed": passed,
        "correctness_checks_total": total,
        "kernel_identity": {"path": kernel_path, "sha256": kernel_sha},
    }


def _validate_selection_cursor_source(
    path: Path, selection_cursor_policy: str
) -> None:
    source = path.read_text(encoding="utf-8")
    if (
        selection_cursor_policy
        == "four_ballots_selected_lane_prefix_rank_direct_scatter"
    ):
        required_in_order = (
            "preceding_word_count = Int32(0)",
            "for word in cutlass.range_constexpr(ROUTE_MASK_WORDS):",
            "word_mask = Int32(",
            "cute.arch.vote_ballot_sync(exact_pred)",
            "lane_rank = (",
            "+ preceding_word_count",
            "+ sol_attn_popc_b32(word_mask & lane_mask_lt)",
            "if exact_pred:",
            "route_indices[lane_rank] = route_start + off",
            "preceding_word_count = (",
            "+ sol_attn_popc_b32(word_mask)",
            "exact_count = preceding_word_count",
        )
        forbidden = (
            "while m != Int32(0):",
            "lowbit = m & (Int32(0) - m)",
            "bit = sol_attn_bfind_b32(lowbit)",
            "m = m & (m - Int32(1))",
        )
        for fragment in forbidden:
            if fragment in source:
                raise ValueError(
                    "direct-scatter selection source contains legacy "
                    f"dynamic lowbit walk {fragment!r}"
                )
        exact_counts = {
            "cute.arch.vote_ballot_sync(exact_pred)": 1,
            "route_indices[lane_rank] = route_start + off": 1,
        }
        for fragment, expected in exact_counts.items():
            observed = source.count(fragment)
            if observed != expected:
                raise ValueError(
                    "direct-scatter selection source requires exactly "
                    f"{expected} occurrence of {fragment!r}, got {observed}"
                )
    else:
        # Every existing non-scatter backend retains the frozen word-ordered
        # lowbit cursor contract.  Keeping this branch byte-for-byte in spirit
        # prevents the new source policy from weakening legacy validation.
        required_in_order = (
            "for word in cutlass.range_constexpr(ROUTE_MASK_WORDS):",
            "while m != Int32(0):",
            "lowbit = m & (Int32(0) - m)",
            "bit = sol_attn_bfind_b32(lowbit)",
            "route_indices[route_rank] =",
            "m = m & (m - Int32(1))",
        )
    cursor = -1
    for fragment in required_in_order:
        cursor = source.find(fragment, cursor + 1)
        if cursor < 0:
            raise ValueError(
                f"selection cursor source check missing {fragment!r}"
            )


def _expected_keys() -> set[tuple[int, int, int, float]]:
    return {
        (tokens, batch, heads, density)
        for tokens in TOKENS
        for batch, heads in BATCH_HEAD_PAIRS
        for density in DENSITIES
    }


def _case_key(row: Mapping[str, Any]) -> tuple[int, int, int, float]:
    try:
        return (
            int(row["T"]),
            int(row["B"]),
            int(row["H"]),
            float(row["target_density"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid full45 case key: {exc}") from exc


def _require_stats(
    row: Mapping[str, Any], field: str, limits: Mapping[str, float]
) -> None:
    value = row.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field}: expected statistics object")
    for name, limit in limits.items():
        try:
            measured = float(value[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{field}.{name}: invalid statistic") from exc
        if not math.isfinite(measured) or measured < 0.0 or measured > limit:
            raise ValueError(
                f"{field}.{name}: {measured!r} exceeds approved limit {limit}"
            )


def _require_accumulator(
    row: Mapping[str, Any], field: str, stats: Mapping[str, Any]
) -> None:
    accumulators = row.get("error_accumulators")
    if not isinstance(accumulators, Mapping):
        raise ValueError("error_accumulators: expected object")
    value = accumulators.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"error_accumulators.{field}: expected object")
    count = value.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"error_accumulators.{field}.count: invalid")
    raw: dict[str, float] = {}
    for name in ("max_abs", "sum_abs", "diff_sq", "reference_sq"):
        try:
            measured = float(value[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"error_accumulators.{field}.{name}: invalid"
            ) from exc
        if not math.isfinite(measured) or measured < 0.0:
            raise ValueError(f"error_accumulators.{field}.{name}: invalid")
        raw[name] = measured
    replayed = {
        "max_abs": raw["max_abs"],
        "mean_abs": raw["sum_abs"] / count,
        "rel_l2": math.sqrt(raw["diff_sq"])
        / max(math.sqrt(raw["reference_sq"]), 1.0e-12),
    }
    for name, expected in replayed.items():
        actual = float(stats[name])
        if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-15):
            raise ValueError(
                f"{field}.{name}: does not replay from raw accumulator"
            )


def _require_hash_mapping(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(INPUT_NAMES):
        raise ValueError(f"{label}: expected hashes for {INPUT_NAMES}")
    return {
        name: _require_sha256(value[name], f"{label}.{name}")
        for name in INPUT_NAMES
    }


def _require_source_identity(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(SOURCE_PATHS):
        raise ValueError(f"{label}: source roles must exactly match frozen paths")
    return {
        name: _require_sha256(value[name], f"{label}.{name}")
        for name in SOURCE_PATHS
    }


def _validated_row(
    raw: Mapping[str, Any],
    *,
    edge_receipt_sha256: str,
    candidate_backend: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("correctness row must be an object")
    row = dict(raw)
    stored_row_sha = row.pop("case_row_sha256", None)
    if row.get("contract") != CORRECTNESS_CONTRACT:
        raise ValueError("correctness contract mismatch")
    backend = get_backend(candidate_backend)
    stored_backend = row.get("candidate_backend")
    if stored_backend is not None and stored_backend != backend.name:
        raise ValueError("correctness row candidate backend mismatch")
    tokens, batch, heads, density = _case_key(row)
    expected_id = case_id(tokens, batch, heads, density)
    if row.get("case_id") != expected_id:
        raise ValueError(f"case_id mismatch for {expected_id}")
    if row.get("D") != HEAD_DIM or row.get("seed") != 0:
        raise ValueError(f"{expected_id}: D/seed contract mismatch")
    realized = float(row.get("realized_density", float("nan")))
    if not math.isfinite(realized) or abs(realized - density) > MAX_DENSITY_ERROR:
        raise ValueError(f"{expected_id}: realized density outside tolerance")
    expected_blocks = math.ceil(tokens / BLOCK_SIZE)
    if row.get("query_blocks_checked") != expected_blocks:
        raise ValueError(
            f"{expected_id}: query_blocks_checked must equal {expected_blocks}"
        )

    before = _require_hash_mapping(
        row.get("input_hashes_before"), f"{expected_id}.input_hashes_before"
    )
    after = _require_hash_mapping(
        row.get("input_hashes_after"), f"{expected_id}.input_hashes_after"
    )
    if before != after:
        raise ValueError(f"{expected_id}: immutable input hash drift")
    sources = _require_source_identity(
        row.get("source_identity"), f"{expected_id}.source_identity"
    )
    _validate_runtime_signature(
        row.get("runtime_signature"), f"{expected_id}.runtime_signature"
    )

    route = row.get("route")
    if not isinstance(route, Mapping):
        raise ValueError(f"{expected_id}: missing route evidence")
    exact_route_fields = {
        "group_size": backend.route_group_size,
        "bit_mismatch_count": 0,
        "count_mismatch_packets": 0,
        "padding_set_bits": 0,
        "passes": True,
    }
    for name, expected in exact_route_fields.items():
        if route.get(name) != expected:
            raise ValueError(f"{expected_id}: route.{name} must be {expected!r}")
    expected_packets = batch * heads * expected_blocks * math.ceil(
        expected_blocks / backend.route_group_size
    )
    if route.get("packet_count") != expected_packets:
        raise ValueError(
            f"{expected_id}: route.packet_count must equal {expected_packets}"
        )
    exact_count = route.get("exact_count")
    slot_count = batch * heads * expected_blocks * expected_blocks
    if (
        isinstance(exact_count, bool)
        or not isinstance(exact_count, int)
        or not 0 <= exact_count <= slot_count
    ):
        raise ValueError(f"{expected_id}: route.exact_count is invalid")
    if not math.isclose(
        exact_count / slot_count, realized, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise ValueError(f"{expected_id}: route count/density mismatch")
    if (
        route.get("selection_cursor_policy")
        != backend.selection_cursor_policy
    ):
        raise ValueError(f"{expected_id}: selection cursor policy mismatch")
    if route.get("selection_cursor_source_check") is not True:
        raise ValueError(f"{expected_id}: selection cursor source check failed")
    if route.get("selection_cursor_source_sha256") != sources["kernel"]:
        raise ValueError(f"{expected_id}: selection cursor source SHA256 mismatch")
    _require_sha256(route.get("trace_sha256"), f"{expected_id}.route.trace")

    _require_stats(row, "output_vs_r_sem", OUTPUT_LIMITS)
    _require_stats(row, "output_vs_prepared_reference", OUTPUT_LIMITS)
    _require_stats(row, "lse_vs_prepared_reference", LSE_LIMITS)
    _require_accumulator(row, "output_vs_r_sem", row["output_vs_r_sem"])
    _require_accumulator(
        row, "output_vs_prepared_reference", row["output_vs_prepared_reference"]
    )
    _require_accumulator(
        row, "lse_vs_prepared_reference", row["lse_vs_prepared_reference"]
    )
    for field in (
        "candidate_output_repeatable",
        "candidate_lse_repeatable",
        "candidate_output_finite",
        "candidate_lse_finite",
        "trace_ordinary_output_bitwise",
        "trace_ordinary_lse_bitwise",
        "passes",
    ):
        if row.get(field) is not True:
            raise ValueError(f"{expected_id}: {field} must be true")
    parent = row.get("parent_diagnostic")
    if not isinstance(parent, Mapping):
        raise ValueError(f"{expected_id}: missing parent diagnostic")
    for name in ("output_bitwise", "lse_bitwise"):
        if not isinstance(parent.get(name), bool):
            raise ValueError(f"{expected_id}: parent_diagnostic.{name} invalid")
    if row.get("edge_receipt_sha256") != edge_receipt_sha256:
        raise ValueError(f"{expected_id}: edge receipt identity mismatch")
    _require_sha256(row["edge_receipt_sha256"], "edge receipt SHA256")

    digest = _canonical_sha256(row)
    if stored_row_sha is not None and stored_row_sha != digest:
        raise ValueError(f"{expected_id}: correctness row SHA256 mismatch")
    row["case_row_sha256"] = digest
    return row


def summarize_correctness_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    edge_receipt_sha256: str,
    contract_sha256: str,
    edge_receipt_binding: Mapping[str, Any],
    source_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate and canonicalize the exact all-position 45-case matrix."""

    edge_receipt_sha256 = _require_sha256(
        edge_receipt_sha256, "edge receipt SHA256"
    )
    contract_sha256 = _require_sha256(contract_sha256, "contract SHA256")
    source_manifest_sha256 = _require_sha256(
        source_manifest_sha256, "source manifest SHA256"
    )
    edge_binding = _validate_edge_receipt(edge_receipt_binding)
    backend = get_backend_for_kernel(
        edge_binding["kernel_identity"]["path"]
    )
    materialized = list(rows)
    if len(materialized) != 45:
        raise ValueError(f"correctness matrix must contain exactly 45 rows, got {len(materialized)}")
    validated = [
        _validated_row(
            row,
            edge_receipt_sha256=edge_receipt_sha256,
            candidate_backend=backend.name,
        )
        for row in materialized
    ]
    keys = [_case_key(row) for row in validated]
    if len(set(keys)) != 45 or set(keys) != _expected_keys():
        raise ValueError("correctness matrix must cover the unique exact 45-case grid")
    identities = [row["source_identity"] for row in validated]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("source identity drift across correctness rows")
    if edge_binding["kernel_identity"]["sha256"] != identities[0]["kernel"]:
        raise ValueError("edge receipt kernel identity does not match full45 source")
    runtime_by_json = {
        json.dumps(
            _validate_runtime_signature(
                row["runtime_signature"], f"{row['case_id']}.runtime_signature"
            ),
            sort_keys=True,
            separators=(",", ":"),
        ): row["runtime_signature"]
        for row in validated
    }
    slurm_contexts = {
        (runtime["slurm_job_id"], runtime["slurm_node"])
        for runtime in runtime_by_json.values()
    }
    if len(slurm_contexts) != 1:
        raise ValueError("runtime signatures span multiple Slurm contexts")
    validated.sort(
        key=lambda row: (
            row["T"], row["B"], row["H"], row["target_density"]
        )
    )
    density_errors = [
        abs(float(row["realized_density"]) - float(row["target_density"]))
        for row in validated
    ]
    return {
        "contract": CORRECTNESS_CONTRACT,
        "candidate_backend": backend.name,
        "case_count": len(validated),
        "expected_case_count": 45,
        "passes": True,
        "edge_receipt_sha256": edge_receipt_sha256,
        "edge_receipt_binding": edge_binding,
        "full45_contract_sha256": contract_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "source_identity": identities[0],
        "runtime_signatures": [
            runtime_by_json[key] for key in sorted(runtime_by_json)
        ],
        "max_density_error": max(density_errors),
        "cases": validated,
    }


def _load_jsonl_cases(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if value.get("event") == "case":
                value = {key: item for key, item in value.items() if key not in {"event", "time"}}
                rows.append(value)
            elif value.get("contract") == CORRECTNESS_CONTRACT and "case_id" in value:
                rows.append(value)
    return rows


def _parse_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: malformed source manifest")
        digest = _require_sha256(fields[0], f"manifest line {line_number}")
        relative = fields[1].lstrip("* ")
        if relative.startswith("/") or ".." in Path(relative).parts or relative in result:
            raise ValueError(f"{path}:{line_number}: invalid manifest path")
        result[relative] = digest
    if not result:
        raise ValueError("source manifest is empty")
    return result


def _audit(args: argparse.Namespace) -> None:
    backend = get_backend(args.candidate_backend)
    selected_sources = candidate_source_paths(
        ROOT,
        backend.name,
        correctness_harness=HARNESS_SOURCE,
        timing_harness=TIMING_HARNESS_SOURCE,
    )
    contract = load_and_validate_contract(args.contract)
    del contract
    contract_sha = _sha256_file(args.contract)
    edge_sha = _sha256_file(args.edge_receipt)
    edge = json.loads(args.edge_receipt.read_text(encoding="utf-8"))
    edge_binding = _validate_edge_receipt(
        edge, expected_kernel_relative=backend.kernel_relative
    )
    manifest = _parse_manifest(args.source_manifest)
    manifest_sha = _sha256_file(args.source_manifest)
    summary = summarize_correctness_rows(
        _load_jsonl_cases(args.input),
        edge_receipt_sha256=edge_sha,
        contract_sha256=contract_sha,
        edge_receipt_binding=edge_binding,
        source_manifest_sha256=manifest_sha,
    )
    for role, source_path in selected_sources.items():
        relative = str(source_path.relative_to(ROOT))
        if manifest.get(relative) != summary["source_identity"].get(role):
            raise ValueError(
                f"correctness source identity {role} is not bound by source manifest"
            )
    kernel_source = selected_sources["kernel"]
    if _sha256_file(kernel_source) != summary["source_identity"]["kernel"]:
        raise ValueError("audited route kernel source does not match archived identity")
    _validate_selection_cursor_source(
        kernel_source, backend.selection_cursor_policy
    )
    args.output.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tensor_sha256(torch: Any, tensor: Any) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _source_identity(
    paths: Mapping[str, Path] | None = None,
) -> dict[str, str]:
    selected = SOURCE_PATHS if paths is None else paths
    return {role: _sha256_file(path) for role, path in selected.items()}


def _device_uuid(torch: Any) -> str:
    properties = torch.cuda.get_device_properties(0)
    value = getattr(properties, "uuid", None)
    if value:
        text = str(value)
        return text if text.lower().startswith("gpu-") else f"GPU-{text}"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if visible.lower().startswith("gpu-"):
        return visible
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    uuids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    index = int(visible) if visible.isdigit() else 0
    if not 0 <= index < len(uuids):
        raise RuntimeError("cannot bind visible CUDA device to GPU UUID")
    return uuids[index]


def _runtime_signature(torch: Any, triton: Any) -> dict[str, Any]:
    runtime = {
        "visible_device_count": int(torch.cuda.device_count()),
        "device_name": str(torch.cuda.get_device_name(0)),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "triton": str(triton.__version__),
        "uuid": _device_uuid(torch),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_node": os.environ.get(
            "SLURMD_NODENAME", os.environ.get("SLURM_NODELIST", "")
        ),
    }
    return _validate_runtime_signature(runtime, "runtime_signature")


def _prepare_qkv(torch: Any, aligned: Any, *, tokens: int, batch: int, heads: int, seed: int) -> tuple[Any, ...]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    shape = (batch, heads, tokens, HEAD_DIM)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16).contiguous()
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16).contiguous()
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16).contiguous()
    kc, vc = aligned.legacy.preprocess_kv(k, v, BLOCK_SIZE)
    blocks = math.ceil(tokens / BLOCK_SIZE)
    unit_scale = torch.ones(
        (batch, heads, blocks, 1), device="cuda", dtype=torch.float32
    )
    return q, k, v, kc.contiguous(), vc.contiguous(), unit_scale


def _calibrate_density(
    torch: Any,
    aligned: Any,
    canonical: Any,
    q: Any,
    kc: Any,
    unit_scale: Any,
    target: float,
    *,
    group_size: int = GROUP_SIZE,
) -> tuple[float, Any, int, float]:
    scale = HEAD_DIM**-0.5
    threshold1 = canonical.compute_global_qck_threshold(
        q, unit_scale, kc, unit_scale, scale, BLOCK_SIZE, 1.0
    )
    threshold2 = canonical.compute_global_qck_threshold(
        q, unit_scale, kc, unit_scale, scale, BLOCK_SIZE, 2.0
    )
    slope = threshold2 - threshold1
    slots = q.shape[0] * q.shape[1] * kc.shape[2] * kc.shape[2]
    target_count = int(round(target * slots))
    trials: dict[float, tuple[int, Any]] = {}

    def evaluate(raw_tau: float) -> tuple[int, Any]:
        tau = float(torch.tensor(raw_tau, dtype=torch.float32).item())
        if tau not in trials:
            threshold = (threshold1 + (tau - 1.0) * slope).contiguous()
            route = aligned.materialize_route_mask(
                q, kc, threshold, group_size=group_size,
                block_size=BLOCK_SIZE, scale=scale,
            )
            trials[tau] = (int(route.sum().item()), threshold)
        return trials[tau]

    lower, upper = 1.0, 4.0
    if evaluate(lower)[0] < target_count or evaluate(upper)[0] > target_count:
        raise RuntimeError("density calibration bracket misses target")
    for _ in range(CALIBRATION_STEPS):
        middle = float(torch.tensor((lower + upper) * 0.5).item())
        if evaluate(middle)[0] > target_count:
            lower = middle
        else:
            upper = middle
    tau, (exact_count, threshold) = min(
        trials.items(), key=lambda item: (abs(item[1][0] - target_count), item[0])
    )
    realized = exact_count / slots
    if abs(realized - target) > MAX_DENSITY_ERROR:
        raise RuntimeError("density calibration exceeds tolerance")
    ordered = sorted((value, count) for value, (count, _) in trials.items())
    if any(b[1] > a[1] for a, b in zip(ordered, ordered[1:])):
        raise RuntimeError("density calibration is not monotonic")
    return tau, threshold, exact_count, realized


def _stats(torch: Any, actual: Any, expected: Any) -> tuple[dict[str, float], dict[str, Any]]:
    diff = actual.double() - expected.double()
    accumulator = {
        "max_abs": float(diff.abs().max().item()),
        "sum_abs": float(diff.abs().sum().item()),
        "diff_sq": float(torch.square(diff).sum().item()),
        "reference_sq": float(torch.square(expected.double()).sum().item()),
        "count": int(diff.numel()),
    }
    return _replay_accumulator(accumulator), accumulator


def _replay_accumulator(value: Mapping[str, Any]) -> dict[str, float]:
    return {
        "max_abs": float(value["max_abs"]),
        "mean_abs": float(value["sum_abs"]) / int(value["count"]),
        "rel_l2": math.sqrt(float(value["diff_sq"]))
        / max(math.sqrt(float(value["reference_sq"])), 1.0e-12),
    }


def _accumulator(torch: Any, device: Any) -> dict[str, Any]:
    return {
        "max": torch.zeros((), device=device, dtype=torch.float64),
        "abs": torch.zeros((), device=device, dtype=torch.float64),
        "diff2": torch.zeros((), device=device, dtype=torch.float64),
        "ref2": torch.zeros((), device=device, dtype=torch.float64),
        "count": 0,
    }


def _accumulate(torch: Any, state: dict[str, Any], actual: Any, expected: Any) -> None:
    diff = actual.double() - expected.double()
    state["max"] = torch.maximum(state["max"], diff.abs().max())
    state["abs"] += diff.abs().sum()
    state["diff2"] += torch.square(diff).sum()
    state["ref2"] += torch.square(expected.double()).sum()
    state["count"] += int(diff.numel())


def _finish(torch: Any, state: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    del torch
    accumulator = {
        "max_abs": float(state["max"].item()),
        "sum_abs": float(state["abs"].item()),
        "diff_sq": float(state["diff2"].item()),
        "reference_sq": float(state["ref2"].item()),
        "count": int(state["count"]),
    }
    return _replay_accumulator(accumulator), accumulator


def _pack_route_trace(torch: Any, mask: Any, group_size: int) -> Any:
    """Pack a dense route mask into G64/G128/G256 mask+count packets."""

    if group_size not in (64, 128, 256) or group_size % 32:
        raise ValueError(f"unsupported full45 route group size {group_size}")
    if mask.ndim != 4:
        raise ValueError("dense route mask must have rank four")
    batch, heads, queries, keys = (int(value) for value in mask.shape)
    route_tiles = (keys + group_size - 1) // group_size
    padded = torch.zeros(
        (batch, heads, queries, route_tiles * group_size),
        device=mask.device,
        dtype=torch.bool,
    )
    padded[..., :keys] = mask.to(torch.bool)
    grouped = padded.reshape(
        batch, heads, queries, route_tiles, group_size
    )
    shifts = torch.arange(32, device=mask.device, dtype=torch.int64)
    weights = torch.bitwise_left_shift(torch.ones_like(shifts), shifts)
    words = [
        (
            grouped[..., start : start + 32].to(torch.int64) * weights
        ).sum(dim=-1).to(torch.int32)
        for start in range(0, group_size, 32)
    ]
    words.append(grouped.sum(dim=-1).to(torch.int32))
    return torch.stack(words, dim=-1)


def _normalize_route_trace(
    torch: Any, trace: Any, num_blocks: int, group_size: int
) -> Any:
    """Normalize fused-route rank-6 traces into physical G128 packets.

    The kernel retains a rank-6 ``[..., logical_group, half, 8]`` diagnostic
    packet so its logical G256/G512 control phase remains observable.  The
    semantic reference stays at G128: full45 flattens two or four physical
    halves and compares four mask words plus a recomputed popcount.  Routing
    decisions are block-local; no score, probability, output, or threshold is
    changed.
    """

    if getattr(trace, "ndim", None) != 6:
        return trace
    if group_size != 128:
        raise ValueError("rank-6 fused route traces require physical G128 audit")
    half_count = int(trace.shape[-2])
    if (
        trace.dtype != torch.int32
        or trace.ndim != 6
        or half_count not in (2, 4)
        or int(trace.shape[-1]) != 8
    ):
        raise TypeError(
            "fused route trace must be int32 [B,H,N,R,{2|4},8], got %s %s"
            % (trace.dtype, tuple(trace.shape))
        )
    logical_group_size = half_count * 128
    expected_logical_groups = (
        num_blocks + logical_group_size - 1
    ) // logical_group_size
    if (
        int(trace.shape[2]) != num_blocks
        or int(trace.shape[3]) != expected_logical_groups
    ):
        raise ValueError(
            "fused route trace extent mismatch for logical G%d: %s"
            % (logical_group_size, tuple(trace.shape))
        )
    physical_tiles = (num_blocks + 127) // 128
    mask_words = trace[..., :4].reshape(
        *trace.shape[:3], int(trace.shape[3]) * half_count, 4
    )[..., :physical_tiles, :].clone()
    shifts = torch.arange(32, device=trace.device, dtype=torch.int64)
    unsigned = mask_words.to(torch.int64) & 0xFFFFFFFF
    bits = torch.bitwise_and(
        torch.bitwise_right_shift(unsigned[..., None], shifts), 1
    )
    popcount = bits.sum(dim=(-1, -2)).to(torch.int32)
    return torch.cat((mask_words, popcount[..., None]), dim=-1)


def _unpack_route_trace(
    torch: Any, trace: Any, num_blocks: int, group_size: int
) -> tuple[Any, dict[str, Any]]:
    """Decode and validate G64/G128/G256 mask+count route packets."""

    if group_size not in (64, 128, 256) or group_size % 32:
        raise ValueError(f"unsupported full45 route group size {group_size}")
    mask_words = group_size // 32
    packet_words = mask_words + 1
    if (
        trace.dtype != torch.int32
        or trace.ndim != 5
        or int(trace.shape[-1]) != packet_words
    ):
        raise TypeError(
            "route trace must be int32 [B,H,N,R,%d], got %s %s"
            % (packet_words, trace.dtype, tuple(trace.shape))
        )
    expected_tiles = (num_blocks + group_size - 1) // group_size
    if (
        int(trace.shape[2]) != num_blocks
        or int(trace.shape[3]) != expected_tiles
    ):
        raise ValueError(f"route trace extent mismatch: {tuple(trace.shape)}")
    shifts = torch.arange(32, device=trace.device, dtype=torch.int64)
    decoded_words = []
    for word in range(mask_words):
        unsigned = trace[..., word].to(torch.int64) & 0xFFFFFFFF
        decoded_words.append(
            torch.bitwise_and(
                torch.bitwise_right_shift(unsigned[..., None], shifts), 1
            )
        )
    grouped = torch.cat(decoded_words, dim=-1).to(torch.uint8)
    popcount = grouped.sum(dim=-1).to(torch.int32)
    count_mismatch = trace[..., mask_words] != popcount
    dense_padded = grouped.flatten(start_dim=-2)
    padding = dense_padded[..., num_blocks:]
    dense = dense_padded[..., :num_blocks].contiguous()
    evidence = {
        "count_mismatch_packets": int(count_mismatch.sum().item()),
        "padding_set_bits": int(padding.sum().item()) if padding.numel() else 0,
        "packet_count": int(trace[..., mask_words].numel()),
    }
    evidence["passes"] = bool(
        evidence["count_mismatch_packets"] == 0
        and evidence["padding_set_bits"] == 0
    )
    return dense, evidence


def _run_gpu_case(torch: Any, aligned: Any, canonical: Any, semantic: Any, prepared_query_block: Any, make_candidate: Any, make_parent: Any, *, tokens: int, batch: int, heads: int, density: float, seed: int, edge_sha: str, source_identity: Mapping[str, str], runtime_signature: Mapping[str, Any], candidate_backend: str = DEFAULT_BACKEND) -> dict[str, Any]:
    backend = get_backend(candidate_backend)
    group_size = backend.route_group_size
    q, k, v, kc, vc, unit_scale = _prepare_qkv(
        torch, aligned, tokens=tokens, batch=batch, heads=heads, seed=seed
    )
    tau, threshold, exact_count, realized = _calibrate_density(
        torch, aligned, canonical, q, kc, unit_scale, density,
        group_size=group_size,
    )
    del tau, exact_count
    scale = HEAD_DIM**-0.5
    immutable = {"q": q, "k": k, "v": v, "kc": kc, "vc": vc, "threshold": threshold}
    hashes_before = {name: _tensor_sha256(torch, tensor) for name, tensor in immutable.items()}
    ordinary = make_candidate(tokens, q, k, v, kc, vc, threshold, scale, trace_route_masks=False)
    trace = make_candidate(tokens, q, k, v, kc, vc, threshold, scale, trace_route_masks=True)
    parent = make_parent(tokens, q, k, v, kc, vc, threshold, scale, trace_route_masks=False)
    ordinary()
    torch.cuda.synchronize()
    first_o, first_lse = ordinary.output.clone(), ordinary.lse.clone()
    ordinary()
    torch.cuda.synchronize()
    candidate_o, candidate_lse = ordinary.output.clone(), ordinary.lse.clone()
    trace()
    torch.cuda.synchronize()
    trace_packet = _normalize_route_trace(
        torch,
        trace.route_mask_trace.clone(),
        math.ceil(tokens / BLOCK_SIZE),
        group_size,
    )
    trace_o = trace.output.clone()
    trace_lse = trace.lse.clone()
    expected_route = aligned.materialize_route_mask(
        q, kc, threshold,
        group_size=group_size, block_size=BLOCK_SIZE, scale=scale,
    ).to(torch.uint8)
    if group_size == GROUP_SIZE:
        expected_packet = semantic._pack_dense_mask(torch, expected_route)
        traced_route, trace_evidence = semantic._unpack_trace(
            torch, trace_packet, math.ceil(tokens / BLOCK_SIZE)
        )
    else:
        expected_packet = _pack_route_trace(
            torch, expected_route, group_size
        )
        traced_route, trace_evidence = _unpack_route_trace(
            torch, trace_packet, math.ceil(tokens / BLOCK_SIZE), group_size
        )
    semantic_runner = aligned.make_prepared_runner(
        q, k, v, kc, vc, threshold,
        group_size=group_size, block_size=BLOCK_SIZE, scale=scale,
    )
    semantic_runner()
    torch.cuda.synchronize()
    semantic_o = semantic_runner().clone()
    torch.cuda.synchronize()
    output_vs_semantic, semantic_accumulator = _stats(
        torch, candidate_o, semantic_o
    )
    output_state = _accumulator(torch, q.device)
    lse_state = _accumulator(torch, q.device)
    blocks = math.ceil(tokens / BLOCK_SIZE)
    for query_block in range(blocks):
        reference_o, reference_lse = prepared_query_block(
            q, k, v, unit_scale, unit_scale, kc, unit_scale, vc,
            threshold, scale, query_block, BLOCK_SIZE,
        )
        start = query_block * BLOCK_SIZE
        stop = min(start + BLOCK_SIZE, tokens)
        _accumulate(torch, output_state, candidate_o[:, :, start:stop], reference_o)
        _accumulate(torch, lse_state, candidate_lse[:, :, start:stop], reference_lse)
    output_vs_reference, output_accumulator = _finish(torch, output_state)
    lse_vs_reference, lse_accumulator = _finish(torch, lse_state)
    parent()
    torch.cuda.synchronize()
    hashes_after = {name: _tensor_sha256(torch, tensor) for name, tensor in immutable.items()}
    row = {
        "contract": CORRECTNESS_CONTRACT,
        "candidate_backend": backend.name,
        "case_id": case_id(tokens, batch, heads, density),
        "T": tokens, "B": batch, "H": heads, "D": HEAD_DIM,
        "target_density": density, "realized_density": realized, "seed": seed,
        "query_blocks_checked": blocks,
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
        "source_identity": dict(source_identity),
        "runtime_signature": dict(runtime_signature),
        "route": {
            "group_size": group_size,
            "bit_mismatch_count": int((traced_route != expected_route).sum().item()),
            "count_mismatch_packets": int(trace_evidence["count_mismatch_packets"]),
            "padding_set_bits": int(trace_evidence["padding_set_bits"]),
            "selection_cursor_policy": backend.selection_cursor_policy,
            "selection_cursor_source_sha256": source_identity["kernel"],
            "selection_cursor_source_check": True,
            "exact_count": int(expected_route.sum().item()),
            "packet_count": int(
                trace_packet[..., backend.route_packet_words - 1].numel()
            ),
            "trace_sha256": _tensor_sha256(torch, trace_packet),
            "passes": bool(torch.equal(trace_packet, expected_packet) and trace_evidence["passes"]),
        },
        "output_vs_r_sem": output_vs_semantic,
        "output_vs_prepared_reference": output_vs_reference,
        "lse_vs_prepared_reference": lse_vs_reference,
        "error_accumulators": {
            "output_vs_r_sem": semantic_accumulator,
            "output_vs_prepared_reference": output_accumulator,
            "lse_vs_prepared_reference": lse_accumulator,
        },
        "candidate_output_repeatable": bool(torch.equal(first_o, candidate_o)),
        "candidate_lse_repeatable": bool(torch.equal(first_lse, candidate_lse)),
        "candidate_output_finite": bool(torch.isfinite(candidate_o).all().item()),
        "candidate_lse_finite": bool(torch.isfinite(candidate_lse).all().item()),
        "trace_ordinary_output_bitwise": bool(torch.equal(trace_o, candidate_o)),
        "trace_ordinary_lse_bitwise": bool(torch.equal(trace_lse, candidate_lse)),
        "parent_diagnostic": {
            "output_bitwise": bool(torch.equal(parent.output, candidate_o)),
            "lse_bitwise": bool(torch.equal(parent.lse, candidate_lse)),
        },
        "edge_receipt_sha256": edge_sha,
    }
    row["passes"] = bool(
        row["route"]["passes"]
        and hashes_before == hashes_after
        and row["candidate_output_repeatable"]
        and row["candidate_lse_repeatable"]
        and row["candidate_output_finite"]
        and row["candidate_lse_finite"]
        and row["trace_ordinary_output_bitwise"]
        and row["trace_ordinary_lse_bitwise"]
        and all(output_vs_semantic[name] <= OUTPUT_LIMITS[name] for name in OUTPUT_LIMITS)
        and all(output_vs_reference[name] <= OUTPUT_LIMITS[name] for name in OUTPUT_LIMITS)
        and all(lse_vs_reference[name] <= LSE_LIMITS[name] for name in LSE_LIMITS)
    )
    return row


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split(",") if item)


def _parse_pairs(raw: str) -> tuple[tuple[int, int], ...]:
    return tuple(tuple(map(int, item.lower().split("x"))) for item in raw.split(",") if item)  # type: ignore[return-value]


def _parse_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(item) for item in raw.split(",") if item)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--edge-receipt", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--tokens", default=",".join(map(str, TOKENS)))
    parser.add_argument("--batch-head-pairs", default=",".join(f"{b}x{h}" for b, h in BATCH_HEAD_PAIRS))
    parser.add_argument("--densities", default=",".join(map(str, DENSITIES)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--candidate-backend",
        choices=tuple(FULL45_CANDIDATE_BACKENDS),
        default=DEFAULT_BACKEND,
    )
    args = parser.parse_args(argv)
    if args.audit_only:
        if args.input is None or args.source_manifest is None:
            parser.error("--audit-only requires --input and --source-manifest")
        return args
    if args.input is not None or args.source_manifest is not None:
        parser.error("device mode does not accept audit-only inputs")
    args.tokens = _parse_ints(args.tokens)
    args.batch_head_pairs = _parse_pairs(args.batch_head_pairs)
    args.densities = _parse_floats(args.densities)
    if not set(args.tokens).issubset(TOKENS):
        parser.error("tokens are outside the frozen grid")
    if args.batch_head_pairs != BATCH_HEAD_PAIRS or args.densities != DENSITIES or args.seed != 0:
        parser.error("device shards must preserve frozen B/H, density, and seed")
    return args


def _device_run(args: argparse.Namespace) -> None:
    backend = get_backend(args.candidate_backend)
    selected_sources = candidate_source_paths(
        ROOT,
        backend.name,
        correctness_harness=HARNESS_SOURCE,
        timing_harness=TIMING_HARNESS_SOURCE,
    )
    load_and_validate_contract(args.contract)
    edge_sha = _sha256_file(args.edge_receipt)
    edge = json.loads(args.edge_receipt.read_text(encoding="utf-8"))
    edge_binding = _validate_edge_receipt(
        edge, expected_kernel_relative=backend.kernel_relative
    )
    import torch
    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability(0)) != (10, 0):
        raise RuntimeError("SM100 CUDA device required")
    import triton
    def allocate(size: int, alignment: int, stream: Any) -> Any:
        del alignment, stream
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(allocate)
    from experiments.sol_attn import check_bf16_cutedsl_semantics as semantic
    from experiments.sol_attn.prepared_reference import sol_attn_prepared_reference_query_block
    from kernels import sol_attention_bf16_aligned as aligned
    from kernels import sol_attention as canonical
    make_candidate, make_parent = load_runner_factories(backend.name)
    source = _source_identity(selected_sources)
    if edge_binding["kernel_identity"]["sha256"] != source["kernel"]:
        raise RuntimeError("edge receipt is not bound to the timed route kernel")
    _validate_selection_cursor_source(
        selected_sources["kernel"], backend.selection_cursor_policy
    )
    runtime = _runtime_signature(torch, triton)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for tokens in args.tokens:
            for batch, heads in BATCH_HEAD_PAIRS:
                for density in DENSITIES:
                    row = _run_gpu_case(
                        torch, aligned, canonical, semantic,
                        sol_attn_prepared_reference_query_block,
                        make_candidate,
                        make_parent,
                        tokens=tokens, batch=batch, heads=heads, density=density,
                        seed=args.seed, edge_sha=edge_sha, source_identity=source,
                        runtime_signature=runtime,
                        candidate_backend=backend.name,
                    )
                    handle.write(json.dumps({"event": "case", **row}, sort_keys=True, allow_nan=False) + "\n")
                    handle.flush()


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        _audit(args)
    else:
        _device_run(args)


if __name__ == "__main__":
    main()


__all__ = [
    "CORRECTNESS_CONTRACT",
    "case_id",
    "summarize_correctness_rows",
]
