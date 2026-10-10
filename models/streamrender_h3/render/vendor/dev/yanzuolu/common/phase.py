"""Execution phase context shared by engine mechanics and their callees."""

from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum, auto
from typing import Iterator


class ExecutionPhase(Enum):
    IDLE = auto()
    PREPARE = auto()
    ROLLOUT = auto()
    TRAIN_FORWARD = auto()
    TRAIN_BACKWARD = auto()
    OPTIMIZATION = auto()
    VALIDATION = auto()


_execution_phase: ContextVar[ExecutionPhase] = ContextVar(
    "execution_phase", default=ExecutionPhase.IDLE
)


def get_execution_phase() -> ExecutionPhase:
    return _execution_phase.get()


@contextmanager
def execution_phase(phase: ExecutionPhase) -> Iterator[None]:
    token = _execution_phase.set(phase)
    try:
        yield
    finally:
        _execution_phase.reset(token)
