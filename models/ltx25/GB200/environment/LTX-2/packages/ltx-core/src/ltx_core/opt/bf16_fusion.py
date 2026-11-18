"""bf16 fusion assembly path for LTX-2.5.

The vendor ships fused Triton kernels but only assembles them behind
``--quantization fp8-*`` (all the fused ops live in
``ltx_core/quantization/blockwise/_impl.py``). In bf16 the model therefore runs
fully eager. Two of those kernels have genuine non-quantizing modes:

* ``blockwise::rms_fma``        -- QUANTIZE=False, allocates a bf16 out
* ``blockwise::gated_attention`` -- quantize=False -> bf16 out, scales=None

This module loads ``triton_ops.py`` BY PATH, bypassing ``ltx_kernels/__init__``
(which imports ``functional.py`` -> the compiled ``ops_cpp`` extension we never
managed to build). triton_ops itself imports only torch + triton, so nothing needs
compiling.

Profile motivation (stage 1, aten-level): glue ~49% of the step, with
``aten::add`` alone at 25.7% and ``aten::mul`` at 14% -- exactly the
``x + y*gate`` then ``rms_norm`` sequence these two kernels collapse.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

_TRITON_OPS = None


def _load_triton_ops():
    global _TRITON_OPS
    if _TRITON_OPS is not None:
        return _TRITON_OPS
    here = Path(__file__).resolve()
    # .../packages/ltx-core/src/ltx_core/opt/bf16_fusion.py -> .../packages/ltx-kernels/...
    root = here.parents[4]
    path = root / "ltx-kernels" / "src" / "ltx_kernels" / "blockwise" / "triton_ops.py"
    if not path.is_file():
        raise FileNotFoundError(f"triton_ops.py not found at {path}")
    spec = importlib.util.spec_from_file_location("_ltx_triton_ops", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _TRITON_OPS = mod
    return mod


class FusedPostSA:
    """bf16 ``blockwise::rms_fma``: fuses ``x + y*gate`` and the following RMSNorm.

    Replaces ``PytorchPostSAFunction`` which does the two as separate ops.
    ``run_rms_fma`` declares ``mutates_args=("x",)`` -- it writes the FMA result
    into ``x`` in place and returns the normed tensor, which is exactly the
    ``(x_fma, rms_norm(x_fma))`` pair the eager version returns.
    """

    def __init__(self) -> None:
        self._f = _load_triton_ops().run_rms_fma

    def __call__(self, x, y, norm_weights, eps, gate):  # noqa: ARG002
        normed = self._f(x, y, gate)
        return x, normed


class FusedGatedAttention:
    """bf16 ``blockwise::gated_attention`` (quantize=False -> bf16 out, scales=None)."""

    def __init__(self) -> None:
        self._f = _load_triton_ops().run_gated_attention

    def __call__(self, x, attn_out, attn_module):
        gate_logits = attn_module.to_gate_logits(x)
        out, _scales = self._f(attn_out, gate_logits, False)
        return out


def install() -> dict:
    """Swap the eager op defaults for fused bf16 ones.

    ``post_sa_function`` / ``gated_attention_function`` are dataclass fields with
    ``field(default_factory=PytorchX)`` -- the factory captured the class object at
    class-definition time, so rebinding the module-level name does nothing. Patch
    the factory on ``__dataclass_fields__`` instead, before the model is built.
    """
    from ltx_core.model.transformer.attention import AttentionOps
    from ltx_core.model.transformer.transformer import TransformerOpsConfig

    applied = {}
    parts = {p.strip() for p in os.environ.get("LTX_FUSE", "").split(",") if p.strip()}

    if parts & {"rmsfma", "all"}:
        TransformerOpsConfig.__dataclass_fields__["post_sa_function"].default_factory = FusedPostSA
        applied["post_sa"] = "blockwise::rms_fma(bf16)"
    if parts & {"gate", "all"}:
        AttentionOps.__dataclass_fields__["gated_attention_function"].default_factory = FusedGatedAttention
        applied["gated_attention"] = "blockwise::gated_attention(bf16)"
    return applied
