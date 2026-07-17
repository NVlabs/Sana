#!/usr/bin/env python3
"""Independently verify an executor's DELIVERY.json (anti-fabrication).

This is the trusted check the master runs — it does NOT trust the executor's
numbers. For each frontier point it: (1) confirms the run_dir + out.mp4 +
benchmark.json exist and were produced by a real run (provenance), (2) RE-RUNS
`plan_eval --assess` for quality-gated techniques, and (3) recomputes speedup
directly from the frozen baseline plus durable candidate benchmark. Mismatches,
missing artifacts, or fabricated runs are reported.

For a LOSSLESS technique (for example kernel or topology), correctness is
MATHEMATICAL / ALGORITHMIC — a property of the METHOD, judged by reasoning, NOT
by comparing outputs. So for
those techniques this deterministic check does NOT compare outputs at all (no
bit-identity, no latent/tensor diff, no floating-point tolerance, no LPIPS): two
correct implementations of the same algorithm may diverge numerically and both
are equally correct. It only confirms the STRUCTURAL invariants any
semantics-preserving implementation must keep (denoising-step count and global
logical DiT/model-evaluation count unchanged) and surfaces the recorded
method/semantics argument. Topology techniques must additionally provide durable
rank, process-group, placement, collective, participation, fallback, timing, and
measured-frontier evidence. The lossless path never invokes an output-difference
metric.
The MASTER then independently REASONS about that evidence + the actual code
changes to accept algorithmic-semantic correctness (see master.md); it must never
reject a lossless candidate merely because its output moved.

plan_eval is invoked with $PLAN_EVAL_PYTHON if set (the eval env python), else
this interpreter. Prints JSON: {objective_ok, issues, points}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LITE = ROOT / "orchestration"
SPEEDUP_TOL = 0.05   # 5% relative tolerance on the reported speedup
MIN_FRONTIER_REL_GAIN = 0.01  # reject one-run noise smaller than 1%
MEMORY_REPORT_TOL = 0.05      # trace max vs benchmark max

# Lossless techniques are gated on MATHEMATICAL / ALGORITHMIC correctness (a
# method property), never on any output metric. For these the executor records a
# method/semantics argument + structural counts, re-checked here (structure only)
# and independently REASONED about by the master (see master.md).


def load_technique_modes() -> dict[str, dict[str, str]]:
    path = LITE / "techniques.toml"
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    techniques = raw.get("techniques")
    if not isinstance(techniques, dict) or not techniques:
        raise RuntimeError(f"invalid technique registry: {path}")
    normalized = {
        str(name): {str(key): str(value) for key, value in spec.items()}
        for name, spec in techniques.items()
        if isinstance(spec, dict)
    }
    required = {"workflow_uid", "scope", "correctness"}
    if len(normalized) != len(techniques) or any(
        not required.issubset(spec) or any(not spec[key] for key in required)
        for spec in normalized.values()
    ):
        raise RuntimeError(f"incomplete technique registry: {path}")
    workflow_uids = [spec["workflow_uid"] for spec in normalized.values()]
    if len(workflow_uids) != len(set(workflow_uids)):
        raise RuntimeError(f"technique workflow_uid values must be unique: {path}")
    return normalized


TECHNIQUES = load_technique_modes()
LOSSLESS_TECHS = {
    identifier
    for name, spec in TECHNIQUES.items()
    if spec.get("correctness") == "lossless"
    for identifier in (name, spec.get("workflow_uid", ""))
    if identifier
}
TOPOLOGY_TECHS = {
    identifier
    for name, spec in TECHNIQUES.items()
    if name == "topology"
    for identifier in (name, spec.get("workflow_uid", ""))
    if identifier
}
TECH_IDENTIFIER_TO_COMPONENT = {
    identifier: name
    for name, spec in TECHNIQUES.items()
    for identifier in (name, spec.get("workflow_uid", ""))
    if identifier
}


def load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _num(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if _is_number(v):
            return v
    return None


def find_equivalence(fp: dict, run_dir: Path) -> dict:
    """Locate + MERGE the executor's recorded correctness evidence.

    A delivery may carry an inline summary (status + an `artifact` pointer) while
    the structural counts + method argument live in a JSON file. Merge both — the
    artifact file's fields fill in, the inline dict overlays — so the gate sees the
    actual evidence, not just a pointer.
    """
    inline: dict = {}
    for key in ("equivalence", "correctness", "lossless_evidence"):
        v = fp.get(key)
        if isinstance(v, dict) and v:
            inline = v
            break
    cands = [run_dir / "equivalence.json", run_dir / "outputs" / "equivalence.json",
             run_dir / "correctness.json", run_dir / "outputs" / "correctness.json",
             run_dir / "equivalence_report.json", run_dir / "outputs" / "equivalence_report.json"]
    ptr = inline.get("artifact") if isinstance(inline, dict) else None
    if ptr:
        p = Path(str(ptr))
        if p.is_absolute():
            cands.append(p)
        else:  # pointer may be worktree-relative, not run_dir-relative
            for base in (run_dir, run_dir.parent, run_dir.parent.parent, run_dir.parent.parent.parent):
                cands.append(base / p)
    for art in (fp.get("artifacts") or []):
        ap = Path(str(art))
        if "equival" in ap.name.lower() or "correct" in ap.name.lower() or "lossless" in ap.name.lower():
            cands.append(ap if ap.is_absolute() else run_dir / ap)
    artifact: dict = {}
    for c in cands:
        d = load(c)
        if d:
            artifact = d
            break
    if not artifact and not inline:
        return {}
    merged = dict(artifact)
    for k, v in inline.items():
        merged.setdefault(k, v)
    return merged


def check_correctness(fp: dict, run_dir: Path) -> tuple[list[str], dict]:
    """STRUCTURAL correctness gate for a LOSSLESS point — NO output comparison.

    Correctness is mathematical / algorithmic-semantic (a method property) and is
    REASONED about by the master. This deterministic pass does NOT look at any
    output artifact (no bit / latent / fp-tolerance / LPIPS). It only:
      - confirms the structural invariants a semantics-preserving implementation
        must keep: denoising-step count and DiT/model-call count unchanged (a
        change here means the *work* changed → an algorithmic change, not just a
        different implementation);
      - surfaces whether a method/semantics argument was recorded for the master.
    It NEVER flags a candidate for numeric output divergence.
    """
    ev = find_equivalence(fp, run_dir)
    if not ev:
        return ["correctness_evidence_missing"], {}
    issues: list[str] = []
    cs = _num(ev, "candidate_steps", "on_denoising_steps", "steps")
    bs = _num(ev, "baseline_steps", "off_denoising_steps", "expected_denoising_steps")
    cc = _num(ev, "candidate_dit_calls", "on_dit_calls", "dit_calls")
    bc = _num(ev, "baseline_dit_calls", "off_dit_calls", "expected_dit_calls")
    if cs is None or bs is None:
        issues.append("step_count_evidence_missing")
    elif ev.get("steps_match") is False or cs != bs:
        issues.append("step_count_changed")          # work changed -> algorithmic change
    if cc is None or bc is None:
        issues.append("dit_call_count_evidence_missing")
    elif ev.get("dit_calls_match", ev.get("calls_match")) is False or cc != bc:
        issues.append("dit_call_count_changed")
    argument = next((ev.get(k) for k in ("method_argument", "semantics_argument", "justification",
                                          "rationale", "reference_path", "candidate_path")
                     if isinstance(ev.get(k), str) and ev.get(k).strip()), None)
    if not argument:
        issues.append("method_argument_missing")
    return issues, {"steps": cs, "dit_calls": cc, "method_argument_present": bool(argument),
                    "note": "correctness = algorithmic semantics (master reasons); output NOT compared"}


def expected_world_size(model_id: str, baseline: dict) -> int | None:
    """Resolve the frozen global rank count from durable baseline/profile data."""
    candidates = [baseline.get("world_size"), baseline.get("num_gpus")]
    envelope = baseline.get("resource_envelope")
    if isinstance(envelope, dict):
        candidates.append(envelope.get("world_size"))
    profile_path = ROOT / "models" / f"{model_id}.toml"
    if profile_path.is_file():
        with profile_path.open("rb") as handle:
            profile = tomllib.load(handle)
        official = profile.get("official_config")
        orchestration = profile.get("orchestration")
        if isinstance(orchestration, dict):
            candidates.append(orchestration.get("inference_world_size"))
        if isinstance(official, dict):
            candidates.append(official.get("num_gpus"))
        slurm = profile.get("slurm")
        if isinstance(slurm, dict):
            nodes = slurm.get("nodes")
            gpus_per_node = slurm.get("gpus_per_node")
            if all(isinstance(value, int) and not isinstance(value, bool) and value > 0
                   for value in (nodes, gpus_per_node)):
                candidates.append(nodes * gpus_per_node)
    return next(
        (
            value
            for value in candidates
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        ),
        None,
    )


def _peak_memory_mib(document: dict) -> float | None:
    """Read a durable max-device-memory value from baseline/benchmark shapes."""
    direct = _num(
        document,
        "peak_memory_mib",
        "max_device_memory_used_mib",
        "max_memory_mib",
    )
    if direct is not None:
        return float(direct)
    memory = document.get("memory")
    if isinstance(memory, dict):
        nested = _num(
            memory,
            "peak_memory_mib",
            "max_device_memory_used_mib",
            "max_memory_mib",
        )
        if nested is not None:
            return float(nested)
    return None


def check_performance_evidence(
    fp: dict,
    run_dir: Path,
    baseline: dict,
    *,
    require_complete: bool = False,
    require_improvement: bool = False,
) -> tuple[list[str], dict]:
    """Recompute performance from the frozen baseline + durable benchmark.

    This is authoritative for speedup.  It deliberately does not use model
    profile numbers (which may differ when --baseline-run-dir is supplied) and
    never looks at output similarity.
    """
    issues: list[str] = []
    benchmark = load(run_dir / "outputs" / "benchmark.json")
    performance = fp.get("performance")
    if not isinstance(performance, dict):
        performance = {}

    baseline_total = _num(baseline, "total_s")
    candidate_total = _num(benchmark, "total_s")
    if baseline_total is None or float(baseline_total) <= 0:
        issues.append("frozen_baseline_total_missing_or_invalid")
    if candidate_total is None or float(candidate_total) <= 0:
        issues.append("candidate_total_missing_or_invalid")

    frozen_scope = baseline.get("timing_scope")
    candidate_scope = benchmark.get("timing_scope")
    if require_complete:
        if not isinstance(frozen_scope, str) or not frozen_scope.strip():
            issues.append("frozen_timing_scope_missing")
        if not isinstance(candidate_scope, str) or not candidate_scope.strip():
            issues.append("candidate_timing_scope_missing")
        elif candidate_scope != frozen_scope:
            issues.append("candidate_timing_scope_mismatch")

    recomputed_speedup = None
    if (
        baseline_total is not None
        and candidate_total is not None
        and float(baseline_total) > 0
        and float(candidate_total) > 0
    ):
        recomputed_speedup = float(baseline_total) / float(candidate_total)

    claimed_baseline = _num(performance, "baseline_total_s")
    claimed_candidate = _num(performance, "candidate_total_s")
    claimed_speedup = _num(performance, "speedup")
    if require_complete:
        if claimed_baseline is None:
            issues.append("performance_baseline_total_missing")
        if claimed_candidate is None:
            issues.append("performance_candidate_total_missing")
        if claimed_speedup is None:
            issues.append("performance_speedup_missing")
    if baseline_total is not None and float(baseline_total) > 0 and claimed_baseline is not None:
        if abs(float(claimed_baseline) - float(baseline_total)) / float(baseline_total) > 0.01:
            issues.append(
                f"wrong_baseline claimed={claimed_baseline} frozen={float(baseline_total):.4f}"
            )
    if candidate_total is not None and float(candidate_total) > 0 and claimed_candidate is not None:
        if abs(float(claimed_candidate) - float(candidate_total)) / float(candidate_total) > 0.01:
            issues.append(
                "candidate_total_misreport "
                f"claimed={claimed_candidate} measured={float(candidate_total):.4f}"
            )
    if recomputed_speedup is not None and claimed_speedup is not None:
        if abs(float(claimed_speedup) - recomputed_speedup) / recomputed_speedup > SPEEDUP_TOL:
            issues.append(
                "speedup_misreport "
                f"claimed={claimed_speedup} recomputed={recomputed_speedup:.4f}"
            )

    baseline_peak = _peak_memory_mib(baseline)
    candidate_peak = _peak_memory_mib(benchmark)
    trace = load(run_dir / "outputs" / "topology_trace.json")
    per_rank = trace.get("per_rank") if isinstance(trace, dict) else None
    rank_peaks = [
        float(item["peak_memory_mib"])
        for item in per_rank or []
        if isinstance(item, dict)
        and _is_number(item.get("peak_memory_mib"))
        and float(item["peak_memory_mib"]) > 0
    ]
    trace_peak = max(rank_peaks) if rank_peaks else None
    if candidate_peak is not None and trace_peak is not None:
        denominator = max(candidate_peak, trace_peak)
        if denominator > 0 and abs(candidate_peak - trace_peak) / denominator > MEMORY_REPORT_TOL:
            issues.append("topology_trace_benchmark_peak_memory_mismatch")
    latency_improved = (
        baseline_total is not None
        and candidate_total is not None
        and float(baseline_total) > 0
        and float(candidate_total)
        < float(baseline_total) * (1.0 - MIN_FRONTIER_REL_GAIN)
    )
    memory_improved = (
        baseline_peak is not None
        and baseline_peak > 0
        and candidate_peak is not None
        and candidate_peak < baseline_peak * (1.0 - MIN_FRONTIER_REL_GAIN)
    )
    frontier_axis = performance.get("frontier_axis")
    if require_improvement:
        if frontier_axis not in ("latency", "peak_memory"):
            issues.append("topology_frontier_axis_missing_or_invalid")
        elif frontier_axis == "latency" and not latency_improved:
            issues.append("topology_declared_latency_axis_not_improved")
        elif frontier_axis == "peak_memory" and not memory_improved:
            issues.append("topology_declared_peak_memory_axis_not_improved")

    return issues, {
        "baseline_total_s": baseline_total,
        "candidate_total_s": candidate_total,
        "speedup": recomputed_speedup,
        "timing_scope": candidate_scope,
        "baseline_peak_memory_mib": baseline_peak,
        "candidate_peak_memory_mib": candidate_peak,
        "trace_peak_memory_mib": trace_peak,
        "frontier_axis": frontier_axis,
        "latency_improved": latency_improved,
        "memory_improved": memory_improved,
        "source": "frozen_baseline_and_durable_candidate_benchmark",
    }


def _valid_observed_collectives(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, dict)
        and isinstance(item.get("kind"), str)
        and bool(item["kind"].strip())
        and isinstance(item.get("calls"), int)
        and not isinstance(item["calls"], bool)
        and item["calls"] > 0
        for item in value
    )


def _fallbacks_are_zero(value: object) -> bool:
    if isinstance(value, int) and not isinstance(value, bool):
        return value == 0
    if isinstance(value, dict) and value:
        return all(
            isinstance(count, int) and not isinstance(count, bool) and count == 0
            for count in value.values()
        )
    return False


def _checks_all_pass(value: object) -> bool:
    if isinstance(value, list) and value:
        return all(
            isinstance(item, dict)
            and (item.get("passed") is True or item.get("status") == "pass")
            for item in value
        )
    if isinstance(value, dict) and value:
        return all(
            result is True
            or (
                isinstance(result, dict)
                and (result.get("passed") is True or result.get("status") == "pass")
            )
            for result in value.values()
        )
    return False


def _collective_counts(value: object) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    if not isinstance(value, list):
        return counts
    for item in value:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        calls = item.get("calls")
        if not isinstance(kind, str) or not isinstance(calls, int) or isinstance(calls, bool):
            continue
        group = json.dumps(item.get("group"), sort_keys=True, separators=(",", ":"))
        key = (kind, group)
        counts[key] = counts.get(key, 0) + calls
    return counts


def check_topology_evidence(
    fp: dict,
    run_dir: Path,
    expected_world: int | None = None,
    worktree: Path | None = None,
) -> tuple[list[str], dict]:
    """Require run-local manifests and traces proving the topology actually ran."""
    outputs = run_dir / "outputs"
    equivalence = load(outputs / "equivalence.json")
    preflight = load(outputs / "topology_preflight.json")
    manifest = load(outputs / "topology_manifest.json")
    trace = load(outputs / "topology_trace.json")
    issues: list[str] = []
    if not equivalence:
        issues.append("topology_equivalence_artifact_missing")
    if not preflight:
        issues.append("topology_preflight_missing")
    if not manifest:
        issues.append("topology_manifest_missing")
    if not trace:
        issues.append("topology_trace_missing")

    metadata = load(run_dir / "metadata.json")
    expected_candidate = fp.get("candidate_id")
    expected_run_id = run_dir.name
    if not isinstance(expected_candidate, str) or not expected_candidate.strip():
        issues.append("topology_candidate_id_missing")
    for artifact_name, artifact in (
        ("equivalence", equivalence),
        ("preflight", preflight),
        ("manifest", manifest),
        ("trace", trace),
    ):
        if not artifact:
            continue
        if artifact.get("candidate_id") != expected_candidate:
            issues.append(f"topology_{artifact_name}_candidate_id_mismatch")
        if artifact.get("run_id") != expected_run_id:
            issues.append(f"topology_{artifact_name}_run_id_mismatch")
    if metadata and metadata.get("candidate_id") != expected_candidate:
        issues.append("topology_metadata_candidate_id_mismatch")

    topology = equivalence.get("topology") if isinstance(equivalence, dict) else None
    if not isinstance(topology, dict) or not topology:
        issues.append("topology_evidence_missing")
        topology = {}

    world_size = _num(topology, "world_size")
    active_ranks = topology.get("active_ranks")
    if not isinstance(expected_world, int) or isinstance(expected_world, bool) or expected_world < 2:
        issues.append("topology_expected_world_size_missing")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 2:
        issues.append("topology_world_size_invalid")
    elif isinstance(expected_world, int) and world_size != expected_world:
        issues.append("topology_world_size_changed")
    if not isinstance(active_ranks, list) or not active_ranks:
        issues.append("topology_active_ranks_missing")
    elif any(not isinstance(rank, int) or isinstance(rank, bool) for rank in active_ranks):
        issues.append("topology_active_ranks_invalid")
    elif len(active_ranks) != len(set(active_ranks)):
        issues.append("topology_active_ranks_duplicate")
    elif isinstance(world_size, int) and set(active_ranks) != set(range(world_size)):
        issues.append("topology_active_ranks_incomplete")
    if topology.get("all_ranks_participated") is not True:
        issues.append("topology_rank_participation_unproven")
    if topology.get("no_silent_fallback") is not True:
        issues.append("topology_no_fallback_unproven")

    process_groups = topology.get("process_groups")
    if not isinstance(process_groups, list) or not process_groups:
        issues.append("topology_process_groups_missing")
    else:
        covered_ranks: set[int] = set()
        for group in process_groups:
            ranks = group.get("ranks") if isinstance(group, dict) else None
            identity = None
            if isinstance(group, dict):
                identity = group.get("kind") or group.get("name")
            if not isinstance(identity, str) or not identity.strip():
                issues.append("topology_process_group_identity_missing")
            if not isinstance(ranks, list) or not ranks or any(
                not isinstance(rank, int) or isinstance(rank, bool) for rank in ranks
            ):
                issues.append("topology_process_group_invalid")
                continue
            if len(ranks) != len(set(ranks)):
                issues.append("topology_process_group_duplicate_rank")
            covered_ranks.update(ranks)
        if isinstance(world_size, int) and covered_ranks != set(range(world_size)):
            issues.append("topology_process_group_coverage_incomplete")

    rank_map = topology.get("rank_map")
    if not isinstance(rank_map, list) or not rank_map:
        issues.append("topology_rank_map_missing")
    else:
        mapped_ranks = [item.get("rank") for item in rank_map if isinstance(item, dict)]
        if (
            len(mapped_ranks) != len(rank_map)
            or any(not isinstance(rank, int) or isinstance(rank, bool) for rank in mapped_ranks)
            or len(mapped_ranks) != len(set(mapped_ranks))
            or (
                isinstance(world_size, int)
                and (
                    len(mapped_ranks) != world_size
                    or set(mapped_ranks) != set(range(world_size))
                )
            )
        ):
            issues.append("topology_rank_map_invalid")
    placement = topology.get("placement")
    if not isinstance(placement, dict) or not placement:
        issues.append("topology_placement_missing")
    collectives = topology.get("collectives")
    if not _valid_observed_collectives(collectives):
        issues.append("topology_collectives_invalid")

    for artifact_name, artifact in (
        ("preflight", preflight),
        ("manifest", manifest),
        ("trace", trace),
    ):
        artifact_world = _num(artifact, "world_size")
        if artifact and artifact_world != world_size:
            issues.append(f"topology_{artifact_name}_world_size_mismatch")
    if preflight and preflight.get("status") != "pass":
        issues.append("topology_preflight_not_passed")
    preflight_checks = preflight.get("checks") if preflight else None
    if not _checks_all_pass(preflight_checks):
        issues.append("topology_preflight_checks_missing_or_failed")
    for key in ("process_groups", "rank_map", "placement", "collectives"):
        if manifest and manifest.get(key) != topology.get(key):
            issues.append(f"topology_manifest_{key}_mismatch")
    source_hashes = manifest.get("source_hashes") if manifest else None
    if not isinstance(source_hashes, dict) or not source_hashes:
        issues.append("topology_manifest_source_hashes_missing")
    else:
        source_root = (worktree or run_dir).resolve()
        for relative, digest in source_hashes.items():
            if not isinstance(relative, str) or not isinstance(digest, str):
                issues.append("topology_manifest_source_hash_invalid")
                continue
            rel_path = Path(relative)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                issues.append("topology_manifest_source_path_invalid")
                continue
            source = (source_root / rel_path).resolve()
            try:
                source.relative_to(source_root)
            except ValueError:
                issues.append("topology_manifest_source_path_invalid")
                continue
            normalized_digest = digest.lower()
            if (
                len(normalized_digest) != 64
                or any(char not in "0123456789abcdef" for char in normalized_digest)
            ):
                issues.append("topology_manifest_source_hash_invalid")
            elif not source.is_file():
                issues.append("topology_manifest_source_missing")
            elif hashlib.sha256(source.read_bytes()).hexdigest() != normalized_digest:
                issues.append("topology_manifest_source_hash_mismatch")
    trace_ranks = trace.get("active_ranks") if trace else None
    if trace and trace_ranks != active_ranks:
        issues.append("topology_trace_active_ranks_mismatch")
    trace_collectives = trace.get("collectives") if trace else None
    if trace and not _valid_observed_collectives(trace_collectives):
        issues.append("topology_trace_collectives_invalid")
    elif trace and _collective_counts(trace_collectives) != _collective_counts(collectives):
        issues.append("topology_trace_collectives_mismatch")
    trace_fallbacks = trace.get("fallbacks") if trace else None
    if trace and not _fallbacks_are_zero(trace_fallbacks):
        issues.append("topology_trace_fallbacks_nonzero_or_missing")
    per_rank = trace.get("per_rank") if trace else None
    if not isinstance(per_rank, list) or not per_rank:
        issues.append("topology_trace_per_rank_missing")
    else:
        observed_ranks = [item.get("rank") for item in per_rank if isinstance(item, dict)]
        complete_records = all(
            isinstance(item, dict)
            and item.get("participated") is True
            and _is_number(item.get("total_s"))
            and float(item["total_s"]) > 0
            and _is_number(item.get("peak_memory_mib"))
            and float(item["peak_memory_mib"]) > 0
            for item in per_rank
        )
        if (
            len(observed_ranks) != len(per_rank)
            or any(not isinstance(rank, int) or isinstance(rank, bool) for rank in observed_ranks)
            or len(observed_ranks) != len(set(observed_ranks))
            or (
                isinstance(world_size, int)
                and (
                    len(observed_ranks) != world_size
                    or set(observed_ranks) != set(range(world_size))
                )
            )
        ):
            issues.append("topology_trace_per_rank_invalid")
        if not complete_records:
            issues.append("topology_trace_per_rank_metrics_missing")

    return issues, {
        "world_size": world_size,
        "expected_world_size": expected_world,
        "active_ranks": active_ranks if isinstance(active_ranks, list) else [],
        "process_group_count": len(process_groups) if isinstance(process_groups, list) else 0,
        "collective_count": len(collectives) if isinstance(collectives, list) else 0,
        "preflight": str(outputs / "topology_preflight.json"),
        "manifest": str(outputs / "topology_manifest.json"),
        "trace": str(outputs / "topology_trace.json"),
        "all_ranks_participated": topology.get("all_ranks_participated") is True,
        "no_silent_fallback": topology.get("no_silent_fallback") is True,
    }


def reassess(
    run_dir: Path,
    model_id: str,
    baseline_frames: str,
    *,
    refresh_collection: bool = True,
) -> dict:
    py = os.environ.get("PLAN_EVAL_PYTHON", sys.executable)
    out = run_dir / "reverify_verdict.json"
    # objective only: LPIPS + speedup. Visual is judged by the master's own
    # multimodal vision (no external vision API), so skip Gemini here.
    cmd = [py, "search/plan_eval.py", "--model", model_id, "--no-gemini",
           "--assess", str(run_dir), "--out", str(out)]
    # Lossless correctness must never compute an output-difference metric. Its
    # objective rerun still recomputes speedup from the durable benchmark, while
    # provenance and structural invariants are checked directly below.
    if not refresh_collection:
        cmd.append("--no-refresh-collection")
    if baseline_frames:
        cmd += ["--baseline-frames", baseline_frames]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=3600)
    verdict = load(out)
    verdict["_plan_eval_rc"] = proc.returncode
    if not verdict:
        verdict["_plan_eval_tail"] = (proc.stdout or "")[-500:]
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tech", required=True, choices=sorted(TECH_IDENTIFIER_TO_COMPONENT))
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--lossless", action="store_true",
                    help="Gate on mathematical/algorithmic correctness (structural + method "
                         "argument), NOT any output metric. Auto-enabled for lossless techniques.")
    args = ap.parse_args()

    require_correctness = args.lossless or (args.tech in LOSSLESS_TECHS)
    require_topology = args.tech in TOPOLOGY_TECHS
    expected_component = TECH_IDENTIFIER_TO_COMPONENT[args.tech]

    wt = Path(args.worktree).resolve()
    runs_root = (wt / "runs").resolve()
    baseline = load(Path(args.baseline))
    base_total = baseline.get("total_s")
    base_frames = str(baseline.get("baseline_frames") or "")
    frozen_world_size = expected_world_size(args.model, baseline)

    delivery = load(wt / "DELIVERY.json")
    issues: list[str] = []
    points_out: list[dict] = []

    if not delivery:
        issues.append("delivery_missing_or_unparseable")
    elif delivery.get("schema_version") != 2 or delivery.get("status") != "complete":
        issues.append("delivery_schema_or_status_invalid")
    if delivery.get("component") != expected_component:
        issues.append(f"delivery_component_mismatch:{delivery.get('component')}")
    if delivery.get("model_id") != args.model:
        issues.append(f"delivery_model_id_mismatch:{delivery.get('model_id')}")
    if require_topology:
        delivery_baseline = delivery.get("baseline")
        if not isinstance(delivery_baseline, dict):
            issues.append("delivery_frozen_baseline_missing")
        else:
            delivered_total = _num(delivery_baseline, "total_s")
            if (
                not _is_number(base_total)
                or float(base_total) <= 0
                or delivered_total is None
                or float(delivered_total) <= 0
                or abs(float(delivered_total) - float(base_total)) / float(base_total) > 0.01
            ):
                issues.append("delivery_frozen_baseline_total_mismatch")
            if delivery_baseline.get("timing_scope") != baseline.get("timing_scope"):
                issues.append("delivery_frozen_baseline_timing_scope_mismatch")
            if delivery_baseline.get("run_dir") != baseline.get("run_dir"):
                issues.append("delivery_frozen_baseline_run_dir_mismatch")

    fps = delivery.get("frontier_points") or []
    if not isinstance(fps, list) or not fps:
        issues.append("no_frontier_points")
        fps = []

    for i, fp in enumerate(fps):
        p_issues: list[str] = []
        if not isinstance(fp, dict):
            issues.append(f"point_{i}:frontier_point_invalid")
            continue
        pid = fp.get("candidate_id", f"point_{i}")
        raw_run_dir = fp.get("run_dir")
        run_dir = runs_root / f"__invalid_point_{i}"
        if not isinstance(raw_run_dir, str) or not raw_run_dir.strip():
            p_issues.append("run_dir_missing")
        else:
            raw_path = Path(raw_run_dir)
            if raw_path.is_absolute():
                p_issues.append("run_dir_must_be_worktree_relative")
            else:
                run_dir = (wt / raw_path).resolve()
                try:
                    run_dir.relative_to(runs_root)
                except ValueError:
                    p_issues.append("run_dir_outside_worktree_runs")
        if not run_dir.is_dir():
            if "run_dir_missing" not in p_issues:
                p_issues.append("run_dir_missing")
        else:
            if not (run_dir / "outputs" / "out.mp4").exists():
                p_issues.append("out_mp4_missing")
            if not (run_dir / "outputs" / "benchmark.json").exists():
                p_issues.append("benchmark_missing")
            # provenance: a real launched run leaves metadata + a start sentinel
            meta = load(run_dir / "metadata.json")
            if not (meta.get("slurm_job_id") or (run_dir / "job-started.json").exists()):
                p_issues.append("no_run_provenance")
            if meta.get("candidate_id") != pid:
                p_issues.append("run_candidate_id_mismatch")
        performance: dict = {}
        if run_dir.is_dir():
            perf_issues, performance = check_performance_evidence(
                fp,
                run_dir,
                baseline,
                require_complete=require_topology,
                require_improvement=require_topology,
            )
            p_issues.extend(perf_issues)

        reverify = {}
        if not p_issues:
            if require_correctness:
                # The trusted lossless path never computes/reads an output metric.
                # Its authoritative speedup is already frozen-baseline / durable
                # candidate benchmark above.
                reverify = {
                    "speedup": performance.get("speedup"),
                    "source": performance.get("source"),
                }
            else:
                reverify = reassess(
                    run_dir,
                    args.model,
                    base_frames,
                    refresh_collection=True,
                )
                if not _is_number(reverify.get("lpips_max")):
                    p_issues.append("plan_eval_reverify_no_lpips")
                reverify["plan_eval_speedup"] = reverify.get("speedup")
                reverify["speedup"] = performance.get("speedup")
            if not _is_number(reverify.get("speedup")):
                p_issues.append("reverify_no_frozen_baseline_speedup")

        # LOSSLESS correctness gate: STRUCTURAL + method argument only, NO output compare.
        correctness: dict = {}
        topology: dict = {}
        if require_correctness and run_dir.is_dir():
            c_issues, correctness = check_correctness(fp, run_dir)
            p_issues.extend(c_issues)
        if require_topology and run_dir.is_dir():
            t_issues, topology = check_topology_evidence(
                fp,
                run_dir,
                expected_world=frozen_world_size,
                worktree=wt,
            )
            p_issues.extend(t_issues)

        reverify_summary = {"speedup": reverify.get("speedup")}
        if not require_correctness:
            reverify_summary.update(
                {key: reverify.get(key) for key in ("lpips_max", "tier")}
            )
        points_out.append({
            "candidate_id": pid, "run_dir": str(run_dir), "objective_ok": not p_issues,
            "issues": p_issues,
            "reverify": reverify_summary,
            "lossless_required": require_correctness,
            "correctness": correctness,
            "topology_required": require_topology,
            "topology": topology,
            "performance": performance,
            "candidate_frames": str(run_dir / "outputs" / "frames"),
            "baseline_frames": base_frames,
            "visual_check": (
                "authenticity_only_no_output_comparison"
                if require_correctness
                else "pending_master_multimodal_quality_review"
            ),
        })
        issues.extend(f"{pid}:{x}" for x in p_issues)

    ok = not issues and any(p["objective_ok"] for p in points_out)
    correctness_note = (
        " This is a LOSSLESS technique: correctness is MATHEMATICAL / ALGORITHMIC, "
        "judged by REASONING about the method — NOT by any output comparison. This gate "
        "only checks structure (denoising-step + global logical DiT-evaluation counts "
        "unchanged) and that a "
        "method/semantics argument was recorded. The master MUST independently REASON about "
        "that argument + the actual code changes (same algorithm? no approximation, sparsity, "
        "step-skip, sub-16-bit quant, or reduced work?) and MUST NOT reject a candidate merely "
        "because its numeric output moved." if require_correctness else "")
    topology_note = (
        " Topology evidence additionally proves the declared world size, active ranks, "
        "process groups, placement, collectives, and zero silent fallback; the master must "
        "audit these against the implementation and run trace." if require_topology else ""
    )
    visual_note = (
        "For this lossless technique, the master may inspect run provenance and frames "
        "only to establish authenticity (real claimed run, no baseline resubmission); it "
        "MUST NOT compare output similarity or visual quality."
        if require_correctness
        else
        "The master MUST independently view candidate_frames against baseline_frames and "
        "apply the visual-artifact rubric before accepting."
    )
    print(json.dumps({
        "objective_ok": ok, "issues": issues, "points": points_out,
        "lossless_required": require_correctness,
        "topology_required": require_topology,
        "note": ("Objective checks only (speedup + provenance"
                 + (" + STRUCTURAL correctness" if require_correctness else " + LPIPS")
                 + "). " + visual_note + correctness_note + topology_note),
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
