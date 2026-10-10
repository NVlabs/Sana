"""Autocast / no-grad forward wrapper runtime plugin.

Example::

    runtime:
      plugins:
        - module: common.plugin.autocast
          class_name: AutocastNoGradWrapper
          autocast: { enabled: true, dtype: bfloat16 }
"""

from __future__ import annotations

import functools
from contextlib import ExitStack
from typing import Any

import torch

from ..model.runtime import RuntimePlugin


class AutocastNoGradWrapper(RuntimePlugin):
    def after_runtime(self, state: dict[str, Any]) -> dict[str, Any]:
        for func_name in self.config.get("functions", ["forward"]):
            self._wrap_one(state["model"], func_name, grad_enabled=self.config.get("grad_enabled", True))
        return state

    # A per-call scope is required here: a closure defined directly in the loop
    # would late-bind ``original`` and every wrapper would call the last one.
    def _wrap_one(self, model: Any, func_name: str, *, grad_enabled: bool) -> None:
        original = getattr(model, func_name)
        autocast = self.config.get("autocast", {})

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with ExitStack() as stack:
                if not grad_enabled:
                    stack.enter_context(torch.no_grad())
                if autocast.get("enabled", False):
                    stack.enter_context(
                        torch.autocast(
                            "cuda",
                            dtype=getattr(torch, autocast.get("dtype", "bfloat16")),
                            cache_enabled=autocast.get("cache_enabled", True),
                        )
                    )
                return original(*args, **kwargs)

        setattr(model, func_name, wrapped)
