"""Unified randomness interfaces.

Seeding contract:

* ``seed_everything`` is called once at launch (from ``configure_distributed``)
  with the same value on every rank -- no rank offset. Everything that runs
  before the engine (model construction, LoRA init, from-scratch init) relies
  on rank-identical global RNG so that replicated parameters are born
  bit-identical on every rank; sharding trusts local values and never
  re-synchronizes them.
* Per-rank / per-stream randomness (data sampling, noise, dropout schedules)
  is the engine's business: derive dedicated streams with
  ``RandomState(seed).fork("data", rank)`` and the like. Never reseed or
  advance the global RNGs for per-rank purposes.
* ``local_seed`` is the escape hatch for model code that wants a controlled
  random section without threading generators through call stacks: it seeds
  the global RNGs inside the context and restores the previous states on exit.
"""

from __future__ import annotations

import contextlib
import hashlib
import random

import numpy as np
import torch

from .logging import get_logger

logger = get_logger()


def seed_everything(seed: int) -> None:
    """Seed the python, numpy, torch-CPU and torch-CUDA global RNGs.

    ``manual_seed_all`` covers every device generator of this process, so the
    call order relative to ``torch.cuda.set_device`` does not matter.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Global RNGs seeded: %d", seed)


def combine_seed(*args) -> int:
    """Deterministic, process-stable seed derivation from arbitrary components.

    sha256 of the joined components -- unlike ``hash()`` it does not depend on
    ``PYTHONHASHSEED``, so every rank and every relaunch derives the same value.
    """
    digest = hashlib.sha256("_".join(str(arg) for arg in args).encode()).hexdigest()
    return int(digest, 16) % (2**63)


def yield_seed(seed: int, a: int = 1103515245, c: int = 12345, m: int = 2**31) -> int:
    """Advance a seed to the next one in a deterministic linear-congruential stream.

    The complement to ``combine_seed`` / ``RandomState.fork``: where those derive
    independent child streams order-free, ``yield_seed`` threads a *single* evolving
    seed forward in place -- the ``rng.seed = yield_seed(rng.seed)`` pattern used
    after each draw when a stream must advance sequentially (e.g. per-step diffusion
    timestep sampling). Constants are the glibc LCG parameters; process-stable and
    independent of ``PYTHONHASHSEED``.
    """
    return (a * seed + c) % m


class RandomState:
    """A bundle of per-stream generators, all derived from one seed.

    Generators are the non-global alternative for everything after the build:
    pass ``python_generator`` / ``numpy_generator`` / ``torch_generator`` /
    ``torch_cuda_generator`` explicitly instead of touching global RNG state.
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.python_generator = random.Random(seed)
        self.numpy_generator = np.random.default_rng(seed)
        self.torch_generator = torch.Generator().manual_seed(seed)
        self.torch_cuda_generator = (
            torch.Generator(device=torch.device("cuda", torch.cuda.current_device())).manual_seed(seed)
            if torch.cuda.is_available()
            else None
        )

    def fork(self, *components) -> "RandomState":
        """Derive an independent child stream, O(1) and order-free.

        Components are semantic, e.g. ``fork("data", rank)`` or
        ``fork("noise", rank, iteration)`` -- two forks collide only if their
        component tuples are identical.
        """
        return RandomState(combine_seed(self.seed, *components))


@contextlib.contextmanager
def local_seed(seed: int | None):
    """Seed the global RNGs inside the context, restore previous states on exit.

    ``None`` is a no-op passthrough. CUDA state is saved/restored for the
    current device only (one process drives one device in this framework).
    """
    if seed is None:
        yield
        return
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
