"""Multi-GPU inference controller.
Public API:
- ``MGPUController``: controller-driven persistent multi-GPU fleet (start / stream / drain / shutdown)
- ``MGPURunner``: base class implemented per pipeline (setup + __call__; __call__ is a generator)
- ``Stream``: handle returned by ``controller.stream(...)``; iterate it for each rank's yields as they arrive
- ``RunnerError``: raised by a runner for a recoverable, symmetric failure
- ``SymmetricRunnerError`` / ``AsymmetricRunnerError``: caller-side exceptions raised from a job's result
- ``ControllerBusyError``: ``stream()`` called while a previous job is still uncollected
- ``NCCLGroups``: per-component NCCL process groups passed to a runner's ``setup``
"""

from ltx_pipelines.multigpu.controller import (
    AsymmetricRunnerError,
    ControllerBusyError,
    MGPUController,
    Stream,
    SymmetricRunnerError,
)
from ltx_pipelines.multigpu.nccl_groups import NCCLGroups
from ltx_pipelines.multigpu.runner import MGPURunner, RunnerError

__all__ = [
    "AsymmetricRunnerError",
    "ControllerBusyError",
    "MGPUController",
    "MGPURunner",
    "NCCLGroups",
    "RunnerError",
    "Stream",
    "SymmetricRunnerError",
]
