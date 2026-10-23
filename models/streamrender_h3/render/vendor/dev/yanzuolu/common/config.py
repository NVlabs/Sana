"""YAML config node with inherit and override semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class CfgNode(dict):
    def __init__(self, values: dict[str, Any] | None = None):
        values = dict(values or {})
        override_list = values.pop("override_list", [])
        super().__init__({key: self._wrap(value) for key, value in values.items()})
        if override_list:
            self.merge_from_list(override_list)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._wrap(value)

    def __getitem__(self, key: str) -> Any:
        if key == "_config" and key not in self:
            self[key] = CfgNode()
        return super().__getitem__(key)

    @classmethod
    def from_file(cls, path: str | Path) -> "CfgNode":
        path = Path(path).expanduser().resolve()
        return cls(cls._load_yaml(path))

    @classmethod
    def _load_yaml(cls, path: Path) -> dict[str, Any]:
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict), f"Config root must be a mapping: {path}"
        return cls._resolve_inherits(data, path)

    @classmethod
    def _resolve_inherits(cls, node: Any, path: Path) -> Any:
        """Resolve ``__inherit__`` at any node of the tree, not only the root.

        A mapping node carrying ``__inherit__: <file>`` is replaced by that
        file's (recursively resolved) content with the node's remaining keys
        deep-merged on top -- so e.g. ``models.backbone.args`` can inherit a
        whole args file and override single fields in place. Relative paths
        resolve against the containing file's directory first, then against the
        CWD (the repo root, per the launch convention).
        """
        if isinstance(node, list):
            return [cls._resolve_inherits(item, path) for item in node]
        if not isinstance(node, dict):
            return node
        inherit = node.pop("__inherit__", None)
        node = {key: cls._resolve_inherits(value, path) for key, value in node.items()}
        if inherit is None:
            return node
        base_path = Path(inherit).expanduser()
        if not base_path.is_absolute():
            sibling = path.parent / base_path
            base_path = sibling if sibling.exists() else Path.cwd() / base_path
        base = cls._load_yaml(base_path.resolve())
        cls._merge_dict(base, node)
        return base

    @classmethod
    def _merge_dict(cls, base: dict[str, Any], update: dict[str, Any]) -> None:
        for key, value in update.items():
            if (
                isinstance(base.get(key), list)
                and isinstance(value, dict)
                and value
                and all(isinstance(k, int) and not isinstance(k, bool) for k in value)
            ):
                # An integer-keyed mapping written over an inherited list merges
                # into it by index (Python indexing semantics, negatives allowed,
                # out-of-range raises). Any other mapping replaces the list, like
                # any other type change.
                for index, item in value.items():
                    if isinstance(base[key][index], dict) and isinstance(item, dict):
                        cls._merge_dict(base[key][index], item)
                    else:
                        base[key][index] = item
            elif isinstance(base.get(key), dict) and isinstance(value, dict):
                cls._merge_dict(base[key], value)
            else:
                base[key] = value

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, CfgNode):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def merge_from_list(self, opts: list[str]) -> None:
        assert len(opts) % 2 == 0, f"Override list must be KEY VALUE pairs: {opts}"

        for full_key, raw_value in zip(opts[0::2], opts[1::2]):
            node = self
            keys = full_key.split(".")
            for key in keys[:-1]:
                if isinstance(node, list):
                    # A path segment over a list is an index (int() raises on
                    # anything non-numeric, out-of-range raises).
                    node = node[int(key)]
                    continue
                if key not in node:
                    node[key] = CfgNode()
                node = node[key]

            value = yaml.safe_load(raw_value) if isinstance(raw_value, str) else raw_value
            if isinstance(node, list):
                node[int(keys[-1])] = self._wrap(value)
            else:
                node[keys[-1]] = self._wrap(value)
