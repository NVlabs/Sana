"""The runner contract: the base class a pipeline subclasses and the error it raises.
``MGPURunner`` is what a pipeline subclasses (``setup`` + a generator ``__call__``); ``RunnerError``
is the recoverable, symmetric failure a runner raises. The fleet (``fleet.py``) runs runners and
ships them to the spawned workers; the controller (``controller.py``) classifies their results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ltx_pipelines.multigpu.nccl_groups import NCCLGroups


class RunnerError(Exception):
    """Raised by a runner (or synthesized by the controller from a ValueError) for a recoverable,
    SYMMETRIC failure. The worker loop catches it and puts it on the result queue as that rank's
    end; the controller collects it with the other terminals and classifies them -- every rank ->
    SymmetricRunnerError, a mix with clean finishes -> AsymmetricRunnerError. Raise it IDENTICALLY on
    every rank, outside any collective (e.g. validating the broadcast kwargs before the first one);
    a RunnerError on only some ranks is the contract violation AsymmetricRunnerError flags.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class MGPURunner(ABC):
    """Subclass this. The controller builds one per worker, injects groups,
    calls setup() once, then calls the instance per job. Define it anywhere -- the controller
    ships the class to workers by value, so a runner in ``__main__`` or a test module works.
    setup() and __call__() run on EVERY rank. __call__ MUST be a generator (use `yield`, even
    once): each yield is forwarded to the Stream on its own, as it arrives -- yields are NOT gathered
    across ranks. Results are *yielded*; a `return` value, if any, rides the terminal
    StopIteration.value (the rare path), collected per rank once every rank has ended. The framework
    does not wrap them in inference mode -- annotate your setup()/__call__ with @torch.inference_mode()
    if you want it.
    Tensors are transparent: pass them as kwargs (e.g. `stream(latent=t)`) and the relay
    broadcasts them to every rank over NCCL, so `__call__` receives them already on the local GPU.
    Yield tensors back the same way -- as values in a yielded dict -- and they come back to the
    controller without being pickled. See the module docstring.
    Raising an unexpected exception from __call__ is FATAL: it kills the worker, poisons the
    controller, and needs a new one. For a RECOVERABLE failure raise a `RunnerError` (or a `ValueError`,
    which the controller converts) -- identically on every rank, outside any collective. The worker
    loop catches it and the fleet stays alive; iterating the Stream re-raises it as
    SymmetricRunnerError.
    """

    _groups: NCCLGroups

    @property
    def groups(self) -> NCCLGroups:
        return self._groups

    @abstractmethod
    def setup(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        ...

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        ...
