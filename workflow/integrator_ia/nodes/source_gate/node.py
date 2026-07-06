#!/usr/bin/env python3
"""Validate, hash-pin, and snapshot three unified donor delivery panels."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workflow_types import NodeContext, NodeResult


INVENTORY_NAME = "integration-source-inventory.json"
EXPECTED_TIERS = {
    "kernel": ("exact_fastest",),
    "pisa": ("conservative", "balanced", "aggressive"),
    "cache": ("conservative", "balanced", "aggressive"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def resolve_source(ctx: NodeContext, raw: str) -> tuple[Path | None, Path | None]:
    if not raw:
        return None, None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ctx.root / path
    path = path.resolve()
    if path.name == "DELIVERY.json" and path.is_file():
        return path.parent, path
    if path.is_file() and path.name == "experiment.json":
        metadata = read_json(path)
        root = Path(str(metadata.get("worktree") or path.parent / "worktree")).resolve()
        return root, root / "DELIVERY.json"
    if path.is_dir() and (path / "experiment.json").is_file():
        metadata = read_json(path / "experiment.json")
        root = Path(str(metadata.get("worktree") or path / "worktree")).resolve()
        return root, root / "DELIVERY.json"
    if path.is_dir():
        return path, path / "DELIVERY.json"
    return path.parent, path


def resolve_artifact(root: Path, raw: str) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else root / path).resolve()


def record(root: Path, path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "relative_path": str(path.resolve().relative_to(root.resolve())),
        "sha256": sha256(path),
        "size": path.stat().st_size,
    }


def verified_record(
    root: Path,
    raw: Any,
    role: str,
    issues: list[str],
    expected_hash: str = "",
) -> dict[str, Any]:
    if isinstance(raw, dict):
        expected_hash = expected_hash or str(raw.get("sha256") or "")
        raw = raw.get("path")
    if not isinstance(raw, str) or not raw:
        issues.append(f"{role}_path_missing")
        return {}
    path = resolve_artifact(root, raw)
    if not under(path, root) or not path.is_file():
        issues.append(f"{role}_missing_or_outside_root:{raw}")
        return {}
    item = record(root, path, role)
    if expected_hash and item["sha256"] != expected_hash:
        issues.append(f"{role}_hash_mismatch:{raw}")
    return item


def validate_delivery(
    component: str,
    root: Path | None,
    delivery_path: Path | None,
    integrator_workload_id: str,
    issues: list[str],
) -> dict[str, Any]:
    if root is None or delivery_path is None or not root.is_dir():
        issues.append(f"{component}_delivery_root_missing")
        return {}
    delivery = read_json(delivery_path)
    if not delivery:
        issues.append(f"{component}_delivery_missing_or_invalid")
        return {}
    if delivery.get("schema_version") != 2 or delivery.get("status") != "complete":
        issues.append(f"{component}_delivery_header_invalid")
    if delivery.get("component") != component:
        issues.append(f"{component}_delivery_component_mismatch")
    if delivery.get("delivery_kind") != "component_frontier":
        issues.append(f"{component}_delivery_kind_invalid")
    baseline = delivery.get("baseline") if isinstance(delivery.get("baseline"), dict) else {}
    if str(baseline.get("workload_id") or "") != integrator_workload_id:
        issues.append(f"{component}_baseline_workload_mismatch")
    lock_record = verified_record(
        root,
        baseline.get("lock_path"),
        f"{component}_baseline_lock",
        issues,
        str(baseline.get("lock_sha256") or ""),
    )
    delivery_record = verified_record(root, str(delivery_path), f"{component}_delivery", issues)

    package = delivery.get("implementation_package") if isinstance(delivery.get("implementation_package"), dict) else {}
    files = package.get("files") if isinstance(package.get("files"), list) else []
    implementations = [
        item
        for index, value in enumerate(files)
        if (item := verified_record(root, value, f"{component}_implementation_{index}", issues))
    ]
    if not implementations:
        issues.append(f"{component}_implementation_files_missing")

    points = delivery.get("frontier_points") if isinstance(delivery.get("frontier_points"), list) else []
    expected = EXPECTED_TIERS[component]
    by_tier = {
        str(point.get("tier") or ""): point
        for point in points
        if isinstance(point, dict)
    }
    if len(points) != len(expected) or set(by_tier) != set(expected):
        issues.append(f"{component}_frontier_tiers_invalid")
    normalized_points: list[dict[str, Any]] = []
    point_artifacts: list[dict[str, Any]] = []
    for tier in expected:
        point = by_tier.get(tier)
        if not isinstance(point, dict):
            continue
        manifest = verified_record(
            root,
            point.get("implementation_manifest"),
            f"{component}_{tier}_manifest",
            issues,
        )
        if manifest:
            point_artifacts.append(manifest)
        hashes = point.get("artifact_hashes") if isinstance(point.get("artifact_hashes"), list) else []
        for index, value in enumerate(hashes):
            item = verified_record(root, value, f"{component}_{tier}_artifact_{index}", issues)
            if item:
                point_artifacts.append(item)
        performance = point.get("performance") if isinstance(point.get("performance"), dict) else {}
        quality = point.get("quality") if isinstance(point.get("quality"), dict) else {}
        if not isinstance(performance.get("speedup"), (int, float)):
            issues.append(f"{component}_{tier}_speedup_missing")
        if quality.get("review_status") != "complete":
            issues.append(f"{component}_{tier}_quality_incomplete")
        if component == "kernel" and quality.get("lossless") is not True:
            issues.append("kernel_exact_fastest_not_lossless")
        normalized_points.append(point)

    return {
        "kind": component,
        "root": str(root),
        "source_experiment_uid": str(delivery.get("experiment_uid") or root.parent.name),
        "workflow_uid": delivery.get("workflow_uid"),
        "delivery": delivery_record,
        "baseline": baseline,
        "baseline_lock": lock_record,
        "frontier_points": normalized_points,
        "selection": {
            "candidate_ids": [point.get("candidate_id") for point in normalized_points],
            "points_by_tier": {point.get("tier"): point.get("candidate_id") for point in normalized_points},
        },
        "artifacts": [item for item in [delivery_record, lock_record, *point_artifacts] if item],
        "implementation_files": implementations,
        "build_smoke": package.get("build_smoke"),
    }


def snapshot_sources(ctx: NodeContext, sources: dict[str, Any], issues: list[str]) -> None:
    base = ctx.worktree / "state" / "integration-source-snapshots"
    for component, source in sources.items():
        if not isinstance(source, dict):
            continue
        root = Path(str(source.get("root") or "")).resolve()
        snapshot_root = (base / component).resolve()
        source["snapshot_root"] = ctx.rel_to_worktree(snapshot_root)
        entries: list[dict[str, Any]] = []
        for key in ("artifacts", "implementation_files"):
            values = source.get(key)
            if isinstance(values, list):
                entries.extend(item for item in values if isinstance(item, dict))
        seen: set[str] = set()
        for item in entries:
            relative = str(item.get("relative_path") or "")
            if not relative or relative in seen:
                continue
            seen.add(relative)
            original = Path(str(item.get("path") or "")).resolve()
            destination = (snapshot_root / relative).resolve()
            if not under(original, root) or not under(destination, snapshot_root):
                issues.append(f"{component}_snapshot_path_invalid:{relative}")
                continue
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, destination)
            except OSError as exc:
                issues.append(f"{component}_snapshot_copy_failed:{relative}:{exc}")
                continue
            copied = sha256(destination)
            item["snapshot_path"] = ctx.rel_to_worktree(destination)
            item["snapshot_sha256"] = copied
            if copied != item.get("sha256"):
                issues.append(f"{component}_source_changed_while_pinning:{relative}")


def run(ctx: NodeContext) -> NodeResult:
    issues: list[str] = []
    baseline = read_json(ctx.worktree / "BASELINE-LOCK.json")
    workload_id = str(baseline.get("workload_id") or "")
    if baseline.get("status") != "locked" or not workload_id:
        issues.append("integrator_baseline_lock_invalid")
    roots = {
        component: resolve_source(ctx, str(ctx.config.get(f"{component}_delivery") or ""))
        for component in ("kernel", "pisa", "cache")
    }
    sources = {
        component: validate_delivery(component, root, delivery, workload_id, issues)
        for component, (root, delivery) in roots.items()
    }
    snapshot_sources(ctx, sources, issues)
    payload = {
        "schema_version": 2,
        "workflow_uid": "integrator_ia",
        "experiment_uid": ctx.state.get("experiment_uid"),
        "generated_at_utc": utc_now(),
        "status": "blocked" if issues else "ready",
        "baseline_workload_id": workload_id,
        "sources": sources,
        "issues": issues,
    }
    path = ctx.worktree / "state" / INVENTORY_NAME
    write_json(path, payload)
    return NodeResult(
        "source_blocked" if issues else "ready",
        updates={
            "source_inventory": ctx.rel_to_worktree(path),
            "source_inventory_status": payload["status"],
            "source_issues": issues,
        },
        artifacts=[ctx.rel_to_worktree(path), ctx.rel_to_worktree(ctx.worktree / "state" / "integration-source-snapshots")],
        message=";".join(issues) if issues else "unified_source_deliveries_pinned",
    )
