from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import torch


def _safe_name(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)[:180] or "root"


def _tensor_hash(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().reshape(-1).clone().contiguous()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _tensor_summary(
    tensor: torch.Tensor, *, max_sample: int, hash_tensors: bool
) -> dict[str, Any]:
    detached = tensor.detach()
    summary: dict[str, Any] = {
        "kind": "tensor",
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
    }
    if detached.numel() == 0:
        summary.update(
            {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "l2": None,
                "sample": [],
            }
        )
        if hash_tensors:
            summary["sha256"] = _tensor_hash(detached)
        return summary
    stats_tensor = detached.float()
    summary.update(
        {
            "mean": float(stats_tensor.mean().detach().cpu()),
            "std": float(stats_tensor.std(unbiased=False).detach().cpu()),
            "min": float(stats_tensor.min().detach().cpu()),
            "max": float(stats_tensor.max().detach().cpu()),
            "l2": float(torch.linalg.vector_norm(stats_tensor).detach().cpu()),
            "sample": [
                float(x)
                for x in stats_tensor.flatten()[:max_sample].detach().cpu().tolist()
            ],
        }
    )
    if hash_tensors:
        summary["sha256"] = _tensor_hash(detached)
    return summary


def summarize_value(
    value: Any, *, max_sample: int = 8, hash_tensors: bool = True, depth: int = 0
) -> Any:
    if depth > 6:
        return {"kind": type(value).__name__, "truncated": True}
    if torch.is_tensor(value):
        return _tensor_summary(value, max_sample=max_sample, hash_tensors=hash_tensors)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return {
            "kind": type(value).__name__,
            "len": len(value),
            "items": [
                summarize_value(
                    v, max_sample=max_sample, hash_tensors=hash_tensors, depth=depth + 1
                )
                for v in value
            ],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "len": len(value),
            "items": {
                str(k): summarize_value(
                    v, max_sample=max_sample, hash_tensors=hash_tensors, depth=depth + 1
                )
                for k, v in value.items()
            },
        }
    if isinstance(value, torch.nn.Module):
        return {"kind": type(value).__name__, "module": True}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": type(value).__name__,
            "dataclass": True,
            "items": {
                field.name: summarize_value(
                    getattr(value, field.name),
                    max_sample=max_sample,
                    hash_tensors=hash_tensors,
                    depth=depth + 1,
                )
                for field in fields(value)
                if hasattr(value, field.name)
            },
        }
    if hasattr(value, "_asdict"):
        try:
            return summarize_value(
                value._asdict(),
                max_sample=max_sample,
                hash_tensors=hash_tensors,
                depth=depth + 1,
            )
        except Exception:
            pass
    attrs: dict[str, Any] = {}
    if hasattr(value, "__dict__"):
        try:
            attrs.update(
                {
                    k: v
                    for k, v in vars(value).items()
                    if not k.startswith("_") and not callable(v)
                }
            )
        except Exception:
            attrs = {}
    slots = getattr(type(value), "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    for slot in slots:
        if slot.startswith("_") or slot in attrs:
            continue
        try:
            slot_value = getattr(value, slot)
        except Exception:
            continue
        if not callable(slot_value):
            attrs[slot] = slot_value
    if attrs:
        limited_items = dict(list(attrs.items())[:64])
        return {
            "kind": type(value).__name__,
            "object_attrs": True,
            "len": len(attrs),
            "items": {
                str(k): summarize_value(
                    v, max_sample=max_sample, hash_tensors=hash_tensors, depth=depth + 1
                )
                for k, v in limited_items.items()
            },
        }
    return {"kind": type(value).__name__, "repr": repr(value)[:500]}


def clone_value_to_cpu(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return repr(type(value))
    if torch.is_tensor(value):
        return value.detach().cpu()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: clone_value_to_cpu(getattr(value, field.name), depth=depth + 1)
            for field in fields(value)
            if hasattr(value, field.name)
        }
    if isinstance(value, tuple):
        return tuple(clone_value_to_cpu(v, depth=depth + 1) for v in value)
    if isinstance(value, list):
        return [clone_value_to_cpu(v, depth=depth + 1) for v in value]
    if isinstance(value, dict):
        return {
            str(k): clone_value_to_cpu(v, depth=depth + 1) for k, v in value.items()
        }
    return repr(value)


class ActivationDumpContext:
    """Dump a single DiT forward's module-level activations for alignment debugging.

    The default mode writes JSONL summaries with tensor shape/dtype/stat/hash/sample.
    Full tensor dumps are opt-in and should be scoped with a regex pattern.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        dump_dir: str | os.PathLike[str],
        *,
        prefix: str,
        name_pattern: str = "",
        max_events: int = 1000,
        include_root: bool = False,
        save_tensors: bool = False,
        max_tensor_events: int = 20,
        max_sample: int = 8,
        hash_tensors: bool = True,
    ) -> None:
        self.module = module
        self.dump_dir = Path(dump_dir)
        self.prefix = _safe_name(prefix)
        self.name_re = re.compile(name_pattern) if name_pattern else None
        self.max_events = int(max_events)
        self.include_root = bool(include_root)
        self.save_tensors = bool(save_tensors)
        self.max_tensor_events = int(max_tensor_events)
        self.max_sample = int(max_sample)
        self.hash_tensors = bool(hash_tensors)
        self.handles: list[Any] = []
        self.event_index = 0
        self.forward_index = -1
        self.tensor_event_count = 0
        self.index_path = self.dump_dir / f"{self.prefix}.jsonl"
        self.input_path = self.dump_dir / f"{self.prefix}.inputs.jsonl"
        self._index_file = None
        self._input_file = None

    @classmethod
    def from_env(
        cls,
        module: torch.nn.Module,
        dump_dir_env: str,
        *,
        prefix: str,
        env_prefix: str,
    ) -> "ActivationDumpContext | None":
        dump_dir = os.environ.get(dump_dir_env)
        if not dump_dir:
            return None
        return cls(
            module,
            dump_dir,
            prefix=prefix,
            name_pattern=os.environ.get(f"{env_prefix}_PATTERN", ""),
            max_events=int(os.environ.get(f"{env_prefix}_MAX_EVENTS", "1000") or 1000),
            include_root=os.environ.get(f"{env_prefix}_INCLUDE_ROOT", "0").lower()
            in ("1", "true", "yes", "on"),
            save_tensors=os.environ.get(f"{env_prefix}_SAVE_TENSORS", "0").lower()
            in ("1", "true", "yes", "on"),
            max_tensor_events=int(
                os.environ.get(f"{env_prefix}_MAX_TENSOR_EVENTS", "20") or 20
            ),
            max_sample=int(os.environ.get(f"{env_prefix}_MAX_SAMPLE", "8") or 8),
            hash_tensors=os.environ.get(f"{env_prefix}_HASH_TENSORS", "1").lower()
            not in ("0", "false", "no", "off"),
        )

    def __enter__(self):
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.index_path.open("w", encoding="utf-8")
        self._input_file = self.input_path.open("w", encoding="utf-8")
        self._register_root_pre_hook()
        for name, child in self.module.named_modules():
            if not name and not self.include_root:
                continue
            display_name = name or "__root__"
            if self.name_re is not None and self.name_re.search(display_name) is None:
                continue
            self.handles.append(
                child.register_forward_hook(self._make_hook(display_name))
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles.clear()
        if self._index_file is not None:
            self._index_file.close()
            self._index_file = None
        if self._input_file is not None:
            self._input_file.close()
            self._input_file = None
        return False

    def _register_root_pre_hook(self) -> None:
        def pre_hook(mod, args, kwargs=None):
            self.forward_index += 1
            payload = {
                "forward_index": self.forward_index,
                "module_class": mod.__class__.__name__,
                "args": summarize_value(
                    args, max_sample=self.max_sample, hash_tensors=False
                ),
                "kwargs": summarize_value(
                    kwargs or {}, max_sample=self.max_sample, hash_tensors=False
                ),
            }
            assert self._input_file is not None
            self._input_file.write(json.dumps(payload, sort_keys=True) + "\n")
            self._input_file.flush()

        try:
            self.handles.append(
                self.module.register_forward_pre_hook(pre_hook, with_kwargs=True)
            )
        except TypeError:

            def legacy_pre_hook(mod, args):
                pre_hook(mod, args, {})

            self.handles.append(self.module.register_forward_pre_hook(legacy_pre_hook))

    def _make_hook(self, name: str):
        def hook(mod, args, output):
            if self.max_events >= 0 and self.event_index >= self.max_events:
                return
            event_id = self.event_index
            self.event_index += 1
            payload = {
                "event_index": event_id,
                "forward_index": self.forward_index,
                "name": name,
                "module_class": mod.__class__.__name__,
                "output": summarize_value(
                    output, max_sample=self.max_sample, hash_tensors=self.hash_tensors
                ),
            }
            if self.save_tensors and self.tensor_event_count < self.max_tensor_events:
                tensor_file = (
                    f"{self.prefix}.event_{event_id:06d}.{_safe_name(name)}.pt"
                )
                torch.save(
                    {
                        "event_index": event_id,
                        "forward_index": self.forward_index,
                        "name": name,
                        "module_class": mod.__class__.__name__,
                        "output": clone_value_to_cpu(output),
                    },
                    self.dump_dir / tensor_file,
                )
                payload["tensor_file"] = tensor_file
                self.tensor_event_count += 1
            assert self._index_file is not None
            self._index_file.write(json.dumps(payload, sort_keys=True) + "\n")

        return hook


_ATTENTION_DEBUG_COUNTS: dict[str, int] = {}


def dump_attention_debug_from_env(
    *,
    dump_dir_env: str,
    env_prefix: str,
    name: str,
    payload: dict[str, Any],
) -> None:
    dump_dir = os.environ.get(dump_dir_env)
    if not dump_dir:
        return
    pattern = os.environ.get(f"{env_prefix}_PATTERN", "")
    if pattern and re.search(pattern, name) is None:
        return
    max_events = int(os.environ.get(f"{env_prefix}_MAX_EVENTS", "20") or 20)
    key = f"{os.getpid()}:{env_prefix}"
    event_index = _ATTENTION_DEBUG_COUNTS.get(key, 0)
    if max_events >= 0 and event_index >= max_events:
        return
    _ATTENTION_DEBUG_COUNTS[key] = event_index + 1
    max_sample = int(os.environ.get(f"{env_prefix}_MAX_SAMPLE", "8") or 8)
    hash_tensors = os.environ.get(f"{env_prefix}_HASH_TENSORS", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    prefix = os.environ.get(f"{env_prefix}_FILE_PREFIX", env_prefix.lower())
    out_dir = Path(dump_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "event_index": event_index,
        "name": name,
        "payload": summarize_value(
            payload,
            max_sample=max_sample,
            hash_tensors=hash_tensors,
        ),
    }
    output_path = out_dir / f"{_safe_name(prefix)}.attention.jsonl"
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
