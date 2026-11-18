"""Text and metric logging with plugin-provided extensions."""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .distributed import ops


def get_logger() -> "Logger":
    """The process-wide Logger; grab once per module: ``logger = get_logger()``.

    Import-time safe -- the singleton exists unconfigured, so text methods
    (``info``/``warning``/``error``, plain stdlib-root forwards) work anywhere,
    including tests that never run launch. Run-scoped outputs touch
    ``self.config`` and naturally fail until ``configure_logging()`` has bound
    the run.
    """
    return _LOGGER


def configure_logging(config: Any) -> Any:
    _LOGGER.configure(config)
    config.logger = _LOGGER
    return config


class LoggingPlugins:
    def __init__(self, plugins: list[Any]):
        self.plugins = plugins

    def before_configuration(self, **payload: Any) -> None:
        self.emit("before_configuration", **payload)

    def after_configuration(self, **payload: Any) -> None:
        self.emit("after_configuration", **payload)

    def before_log_metrics(self, **payload: Any) -> None:
        self.emit("before_log_metrics", **payload)

    def after_log_metrics(self, **payload: Any) -> None:
        self.emit("after_log_metrics", **payload)

    def flush(self) -> None:
        self.emit("flush")

    def close(self, **payload: Any) -> None:
        self.emit("close", **payload)

    def emit(self, hook_name: str, **payload: Any) -> None:
        for plugin in self.plugins:
            hook = getattr(plugin, hook_name, None)
            if hook is not None:
                hook(**payload)


class Logger:
    """Text/metrics logging extended with dynamically injected plugin methods."""

    def __init__(self) -> None:
        # The singleton is created unconfigured at import time; text methods work
        # immediately, everything run-scoped binds in configure().
        self.config: Any = None
        self._plugin_methods: dict[str, list[tuple[str, Callable[..., Any]]]] = {}

    def configure(self, config: Any) -> None:
        for name in self._plugin_methods:
            delattr(self, name)
        self._plugin_methods.clear()
        self.config = config
        logging_config = config.get("logging", {})
        run_dir = Path(config.persistence.run_dir)
        logging_config.logs_dir = str(run_dir / "logs")
        logging_config.metrics_dir = str(run_dir / "metrics")

        for path in [logging_config.logs_dir, logging_config.metrics_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)

        plugins = []
        for plugin_config in config.get("logging", {}).get("plugins", []):
            module = importlib.import_module(plugin_config.module)
            plugin_cls = getattr(module, plugin_config.class_name)
            plugins.append(plugin_cls(plugin_config))
        self.plugins = LoggingPlugins(plugins)

        # Text logging: per-rank file via the root logger, console on local rank 0.
        logs_dir = Path(logging_config.logs_dir)
        self.plugins.before_configuration(config=self.config, logger=self, logs_dir=logs_dir)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        fmt = logging_config.get("format", "[%(asctime)s %(filename)s:%(lineno)s] %(message)s")
        datefmt = logging_config.get("datefmt", "%Y-%m-%d %H:%M:%S")
        rank = int(os.environ.get("RANK", "0"))
        local_rank = ops.get_local_rank()

        logging.basicConfig(
            level=logging.INFO,
            format=fmt,
            datefmt=datefmt,
            filename=logs_dir / f"log_rank{rank}.txt",
            filemode="a",
        )

        if local_rank == 0:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter(fmt, datefmt))
            logging.getLogger().addHandler(console_handler)

        logging.getLogger().info("Run dir: %s", self.config.persistence.run_dir)
        self.plugins.after_configuration(config=self.config, logger=self, logs_dir=logs_dir)

    def info(self, msg: Any, *args: Any) -> None:
        logging.getLogger().info(msg, *args)

    def warning(self, msg: Any, *args: Any) -> None:
        logging.getLogger().warning(msg, *args)

    def error(self, msg: Any, *args: Any) -> None:
        logging.getLogger().error(msg, *args)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None, metadata: dict[str, Any] | None = None) -> Path | None:
        if int(os.environ.get("RANK", "0")) != 0:
            return None
        path = Path(self.config.logging.metrics_dir) / "metrics.jsonl"
        self.plugins.before_log_metrics(config=self.config, path=path, metrics=metrics, step=step, metadata=metadata)
        with path.open("a") as f:
            f.write(json.dumps({"step": step, "metrics": metrics, "metadata": metadata}, default=str) + "\n")
        self.plugins.after_log_metrics(config=self.config, path=path, metrics=metrics, step=step, metadata=metadata)
        return path

    def iter_dir(self, namespace: str, step: int | None = None) -> Path:
        path = Path(self.config.persistence.run_dir) / namespace
        if step is not None:
            path /= f"{step:07d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def inject_methods(self, plugin: Any, methods: Mapping[str, Callable[..., Any]]) -> None:
        owner = f"{plugin.__class__.__module__}.{plugin.__class__.__name__}"

        def make_dispatcher(method_name: str) -> Callable[..., None]:
            def dispatcher(*args: Any, **kwargs: Any) -> None:
                for method_owner, plugin_method in self._plugin_methods[method_name]:
                    self.info("Logger plugin call: %s.%s", method_owner, method_name)
                    plugin_method(*args, **kwargs)

            return dispatcher

        for name, method in methods.items():
            if name not in self._plugin_methods:
                if hasattr(self, name):
                    raise AttributeError(f"Logger plugin method {name!r} conflicts with an existing attribute")
                self._plugin_methods[name] = []
                setattr(self, name, make_dispatcher(name))
            self._plugin_methods[name].append((owner, method))

    def flush(self) -> None:
        self.plugins.flush()

    def close(self, exit_code: int = 0) -> None:
        """``exit_code != 0`` marks tracker runs as failed (e.g. wandb)."""
        self.plugins.close(exit_code=exit_code)

# Created unconfigured at import time; configure_logging() binds the run.
_LOGGER = Logger()
