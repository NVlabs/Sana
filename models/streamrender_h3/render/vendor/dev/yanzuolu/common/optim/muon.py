"""Distributed-compatible Muon optimizer (Newton-Schulz orthogonalized momentum).

Usage in a model config -- pair Muon with an AdamW catch-all; Muon claims the
2D weight kernels via ``match``, AdamW takes everything else::

    optimizer:
      - module: dev.yanzuolu.common.optim.muon
        class_name: Muon
        match: ["*attn*", "*mlp*"]   # 2D kernels only
        args:
          lr: 3.0e-4
          momentum: 0.99
          weight_decay: 0.1
      - class_name: AdamW            # no match => catch-all
        args:
          lr: 1.0e-5

Conventions:

- **2D contract**: every parameter routed to Muon must be 2D (asserted at
  step time, on the logical shape). Biases and norm scales must not match
  Muon's patterns -- they belong to the paired AdamW.
- **embedding / LM head**: these are 2D but must NOT be orthogonalized. The
  Muon class makes no special case for them; exclude them with the ``match``
  patterns in yaml (i.e. let them fall to AdamW).
- **weight decay** is decoupled (``p *= 1 - lr * wd``) and applies to the
  orthogonalized parameters only (there is nothing else in Muon).
- **DTensor behavior**: under FSDP2 each parameter is a sharded DTensor whose
  shard is a *flattened-then-sliced* segment, not a row block of the 2D
  matrix. Orthogonalizing the shard would be mathematically wrong, so the
  Newton-Schulz input is always the full logical matrix: ``full_tensor()``
  (a collective) then ``reshape`` to the DTensor's global logical ``p.shape``
  (a DTensor's ``.shape`` is the global shape, not the shard shape). The
  momentum buffer lives in the *sharded* layout (``zeros_like(grad)``), so
  DCP checkpointing, EMA, and memory footprint behave exactly like AdamW's
  ``exp_avg`` (naturally 1/N sharded, no special DCP handling). After
  orthogonalization the full update is re-sharded with placements derived
  from the parameter (``Shard(0)`` per sharded mesh dim, ``Replicate()``
  otherwise), which adapts to both 1-D FSDP and 2-D HSDP meshes, and written
  back as a DTensor -- DTensor and plain tensors must not be mixed in
  arithmetic. This is mathematically exact: orthogonalization on the full
  matrix, not a sharded approximation.
- **Collective safety**: ``full_tensor()`` requires every rank to step the
  same parameter set in the same order. ``build_optimizers`` assigns
  parameters in ``named_parameters`` order on all ranks, so this holds
  naturally. The known hazard is asymmetric ``p.grad is None`` (one rank
  skips a parameter another rank steps) -- that deadlocks; do not freeze
  parameters per-rank under Muon.

Reference: Keller Jordan's Muon (https://github.com/KellerJordan/Muon), with
the quintic Newton-Schulz iteration (bf16, ``X/(X.norm()+1e-7)``, transpose
tall matrices, 5 steps, coefficients 3.4445/-4.7750/2.0315), the lerp-form
momentum, and the ``sqrt(max(1, rows/cols))`` update scaling.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor


@torch.no_grad()
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Newton-Schulz iteration to compute the orthogonalization of ``G``.

    Keller Jordan's quintic variant: bf16, normalized to spectral norm <= 1,
    tall matrices transposed first. Returns an approximation of ``U V^T``
    from the SVD of ``G`` (singular values pushed into roughly [0.7, 1.15] --
    near-orthogonal, not exactly orthogonal; that is the intended tradeoff of
    these coefficients). ``G`` must be a full (non-sharded) 2D plain tensor.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    # Normalize so the spectral norm is <= 1, keeping the iteration stable.
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


def _reshard(full_update: Tensor, ref: Tensor) -> Tensor:
    """Cut a full update matrix back to ``ref``'s sharded layout.

    Placements are derived from the reference parameter: a sharded mesh dim
    becomes ``Shard(0)`` (the update is already in 2D logical form, so dim-0
    sharding of it matches ``ref``'s flattened-shard layout under FSDP2),
    anything else replicates. Plain tensors pass through unchanged. Returns a
    DTensor whenever ``ref`` is one -- DTensor and plain tensors must not be
    mixed in arithmetic.
    """
    if not isinstance(ref, DTensor):
        return full_update
    placements = [Shard(0) if isinstance(pl, Shard) else Replicate() for pl in ref.placements]
    return distribute_tensor(full_update.contiguous(), ref.device_mesh, placements)


class Muon(torch.optim.Optimizer):
    """Muon: momentum + Newton-Schulz orthogonalization for 2D parameters.

    See the module docstring for the 2D contract, the yaml pairing with
    AdamW, and the DTensor (FSDP sharding) behavior.

    Args:
        params: parameters (or param groups) -- each must be 2D; anything
            else belongs to the paired AdamW (yaml ``match`` controls this).
        lr: learning rate applied to the orthogonalized update.
        momentum: momentum coefficient on the gradient (lerp form:
            ``buf.lerp_(grad, 1 - momentum)``).
        nesterov: use the Nesterov-style combination ``grad.lerp(buf, momentum)``.
        ns_steps: number of Newton-Schulz iterations.
        weight_decay: decoupled weight decay on the orthogonalized parameters.
        eps: epsilon guarding the Frobenius-norm normalization before NS.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.99,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.ndim == 2, (
                    "Muon only takes 2D parameters (logical shape); "
                    "route 1D params to the paired AdamW via the yaml match patterns"
                )
                self._step_param(p, group)
        return loss

    def _step_param(self, p: Tensor, group: dict) -> None:
        grad = p.grad
        state = self.state[p]
        if "momentum_buffer" not in state:
            # Sharded layout: same shape/placement as p, so DCP checkpointing
            # and memory match AdamW's exp_avg (1/N sharded under FSDP).
            state["momentum_buffer"] = torch.zeros_like(grad)
        buf = state["momentum_buffer"]

        # Momentum accumulates on the shard (grad has p's sharded layout).
        buf.lerp_(grad, 1 - group["momentum"])
        eff = grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf

        # The shard is a flattened-then-sliced segment, NOT a 2D row block:
        # orthogonalize the full logical matrix only. p.shape is the DTensor's
        # global logical shape (and a plain tensor's own shape otherwise).
        full = eff.full_tensor() if isinstance(eff, DTensor) else eff
        full = full.reshape(p.shape)
        update = zeropower_via_newtonschulz5(full, steps=group["ns_steps"], eps=group["eps"])
        # Muon updates are near-unit-spectral-norm; scale so the effective lr
        # does not drift with the matrix aspect ratio (Keller Jordan variant).
        update *= max(1.0, p.size(-2) / p.size(-1)) ** 0.5
        # Cast back before the write so fp32 master weights stay fp32.
        update = update.to(p.data.dtype)

        if group["weight_decay"] != 0.0:
            p.mul_(1 - group["lr"] * group["weight_decay"])

        # Re-shard to p's layout (no-op for plain tensors) and write back.
        # Under DTensor the update stays a DTensor -- the two must not mix.
        p.data.add_(_reshard(update, p), alpha=-group["lr"])
