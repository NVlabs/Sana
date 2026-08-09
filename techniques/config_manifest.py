"""Config-manifest validation and model-agnostic dry-run helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def manifest_id(data: dict[str, Any]) -> str:
    raw = data.get("id")
    if isinstance(raw, dict):
        value = raw.get("name") or raw.get("config") or raw.get("id")
        if value:
            return str(value)
    if raw is not None:
        return str(raw)
    return ""


def manifest_dimension(data: dict[str, Any]) -> str:
    raw = data.get("id")
    if isinstance(raw, dict):
        return str(raw.get("dimension", ""))
    return str(data.get("dimension", ""))


def manifest_family(data: dict[str, Any]) -> str:
    raw = data.get("id")
    if isinstance(raw, dict):
        return str(raw.get("family", ""))
    return str(data.get("family", ""))


def is_efficiency_config(data: dict[str, Any]) -> bool:
    return isinstance(data.get("id"), dict) or "requirements" in data or "efficiency" in data


def load_model_profile(data: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    profile_id = str(data.get("model_profile") or "").strip()
    if not profile_id:
        return {}
    profile_path = repo_root / "models" / f"{profile_id}.toml"
    if not profile_path.exists():
        raise ValueError(f"model_profile {profile_id!r} not found at {profile_path}")
    return load_toml(profile_path)


def schema_errors(data: dict[str, Any], repo_root: Path) -> list[str]:
    if not is_efficiency_config(data):
        return []

    errors: list[str] = []
    raw_id = data.get("id")
    if not isinstance(raw_id, dict):
        errors.append("[id] table is required for efficiency config")
    else:
        for key in ("name", "dimension", "family"):
            if not str(raw_id.get(key, "")).strip():
                errors.append(f"[id].{key} is required")

    refs = data.get("references", {})
    ext = refs.get("external", {}) if isinstance(refs, dict) else {}
    loc = refs.get("local", {}) if isinstance(refs, dict) else {}
    for key in ("paper", "code", "notes"):
        if not str(ext.get(key, "")).strip():
            errors.append(f"[references.external].{key} is required")
    # generic_impl is required. model_adapter_example / runtime_example are
    # OPTIONAL: they used to point into the engine submodule that was removed;
    # when present they must point at an existing file in this repo.
    for key in ("generic_impl", "model_adapter_example", "runtime_example"):
        value = str(loc.get(key, "")).strip()
        if not value:
            if key == "generic_impl":
                errors.append(f"[references.local].{key} is required")
            continue
        path = repo_root / value
        if not path.exists():
            errors.append(f"[references.local].{key} does not exist: {value}")

    req = data.get("requirements", {})
    caps = req.get("capabilities") if isinstance(req, dict) else None
    if not isinstance(caps, list) or not caps:
        errors.append("[requirements].capabilities must be a non-empty list")

    eff = data.get("efficiency", {})
    if not isinstance(eff, dict) or not str(eff.get("name", "")).strip():
        errors.append("[efficiency].name is required")
    if not isinstance(eff, dict) or eff.get("kind") not in {
        "runtime_technique",
        "technique",
        "build_transform",
        "transform",
    }:
        errors.append(
            "[efficiency].kind must be runtime_technique, technique, build_transform, or transform"
        )

    ver = data.get("verification", {})
    if not isinstance(ver, dict) or not str(ver.get("mode", "")).strip():
        errors.append("[verification].mode is required")
    if not isinstance(ver, dict) or not str(ver.get("quality_gate", "")).strip():
        errors.append("[verification].quality_gate is required")

    return errors


def _capability_lookup():
    from techniques.technique import Capability

    lookup = {cap.value: cap for cap in Capability}
    lookup.update({cap.name.lower(): cap for cap in Capability})
    lookup.update(
        {
            "blocks": Capability.BLOCKS,
            "transformer_blocks": Capability.BLOCKS,
            "attention_backend_switch": Capability.HAS_ATTENTION_BACKEND_SWITCH,
            "ffn_linear_modules": Capability.HAS_FFN_LINEAR_MODULES,
            "token_sequence_axis": Capability.HAS_TOKEN_SEQUENCE_AXIS,
            "spatiotemporal_token_layout": Capability.HAS_SPATIOTEMPORAL_TOKEN_LAYOUT,
            "token_gather_scatter": Capability.SUPPORTS_TOKEN_GATHER_SCATTER,
            "step_cache": Capability.SUPPORTS_STEP_CACHE,
            "cuda_graph_probe": Capability.SUPPORTS_CUDA_GRAPH_PROBE,
            "nvfp4_linear": Capability.SUPPORTS_NVFP4_LINEAR,
        }
    )
    return lookup


def resolve_capabilities(names: list[str]):
    lookup = _capability_lookup()
    resolved = []
    unknown = []
    for name in names:
        key = str(name).strip()
        cap = lookup.get(key) or lookup.get(key.lower())
        if cap is None:
            unknown.append(key)
        else:
            resolved.append(cap)
    if unknown:
        raise ValueError(f"unknown capabilities: {unknown}")
    return frozenset(resolved)


def model_spec_key(data: dict[str, Any], repo_root: Path) -> str:
    adapter = data.get("adapter", {})
    if isinstance(adapter, dict) and adapter.get("model_spec"):
        return str(adapter["model_spec"])
    profile = load_model_profile(data, repo_root)
    if profile.get("spec"):
        return str(profile["spec"])
    return "manifest"


def _manifest_model_spec(name: str, capabilities):
    from techniques.spec import ModelSpec
    from techniques.technique import Capability

    caps = set(capabilities)
    if Capability.HAS_TRANSFORMER_BLOCKS in caps:
        caps.add(Capability.BLOCKS)
    if Capability.HAS_ATTENTION_BACKEND_SWITCH in caps:
        caps.add(Capability.SWAPPABLE_ATTENTION)
    if Capability.SUPPORTS_TOKEN_GATHER_SCATTER in caps:
        caps.add(Capability.PRUNABLE_TOKENS)
    return ModelSpec(
        name=name,
        capabilities=frozenset(caps),
        seq_dim=1,
        extra={"source": "manifest_capabilities"},
    )


def _build_efficiency_item(data: dict[str, Any]):
    from techniques.registry import build_technique, build_transform

    eff = data.get("efficiency", {})
    kind = eff.get("kind")
    name = str(eff.get("name", ""))
    params = dict(eff.get("params", {}) or {})
    if kind in {"build_transform", "transform"}:
        return build_transform(name, **params)
    if kind in {"runtime_technique", "technique"}:
        return build_technique(name, **params)
    raise ValueError(f"unsupported efficiency kind: {kind!r}")


def _generic_impl_has_model_path(data: dict[str, Any], repo_root: Path) -> bool:
    refs = data.get("references", {})
    loc = refs.get("local", {}) if isinstance(refs, dict) else {}
    generic = str(loc.get("generic_impl", "")).strip()
    if not generic:
        return False
    path = repo_root / generic
    if not path.exists() or not path.is_file():
        return False
    text = path.read_text(errors="ignore").lower()
    forbidden = (
        "runtime/models/dits/cosmos3video.py",
        "python/sglang/multimodal_gen/runtime/models/dits/cosmos3video.py",
    )
    return any(token in text for token in forbidden)


def dry_run_manifest(
    data: dict[str, Any],
    repo_root: Path,
    *,
    stage: str = "stage2",
) -> dict[str, Any] | None:
    """Validate schema, capabilities, adapter discovery, and transform preview."""

    if not is_efficiency_config(data):
        return None

    errors = schema_errors(data, repo_root)
    if errors:
        raise ValueError("config schema errors:\n  - " + "\n  - ".join(errors))

    from techniques import compose
    from techniques.transform import ModelTransform

    req = data.get("requirements", {})
    required = resolve_capabilities([str(x) for x in req.get("capabilities", [])])
    spec_key = model_spec_key(data, repo_root)
    item = _build_efficiency_item(data)
    item_required = getattr(item, "required_capabilities", frozenset())
    spec = _manifest_model_spec(spec_key, required | item_required)

    missing = spec.missing(required)
    if missing:
        raise ValueError(
            f"{manifest_id(data)!r} requires capabilities "
            f"{[cap.value for cap in missing]} not provided by model {spec.name!r}"
        )

    plan = compose([item], spec)
    env_preview: dict[str, str] = {}
    if isinstance(item, ModelTransform):
        plan.apply_transforms(None, stage=stage, env=env_preview)

    if _generic_impl_has_model_path(data, repo_root):
        raise ValueError("generic implementation hard-codes the Cosmos3 runtime module path")

    return {
        "config_id": manifest_id(data),
        "dimension": manifest_dimension(data),
        "family": manifest_family(data),
        "model_spec": spec.name,
        "model_spec_source": spec.extra.get("source"),
        "required_capabilities": sorted(cap.value for cap in required),
        "effective_capabilities": sorted(cap.value for cap in spec.capabilities),
        "compose": {
            "plan": repr(plan),
            "transforms": [t.name for t in plan.transforms],
            "techniques": [t.name for t in plan.techniques],
        },
        "env_preview": env_preview,
        "runtime_config": data.get("efficiency", {}),
    }


def write_dry_run(path: Path, payload: dict[str, Any] | None) -> None:
    if payload is None:
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
