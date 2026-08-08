"""Compatibility re-export of the split modules.

The port was one 923-line file before this merge. It is now split along the boundaries the
8xGB200 variant next door already uses — `relayout`, `fusions`, `fusion_install`, `adaln`,
`build` — so the two lines can be read against each other module by module.

This shim exists so the twenty-odd scripts under `bench/` and `checks/` keep importing one
name. They are the evidence for every claim in the README, and rewriting their imports would
have meant re-verifying all of them to prove the rewrite changed nothing. Nothing new should
import from here.
"""

from adaln import AdaLnTableLookup, patch_pruned_adaln  # noqa: F401
from build import build_pruned_fp8_transformer  # noqa: F401
from fp8_linear import Fp8Linear  # noqa: F401
from fusion_install import (  # noqa: F401
    load_sol_attn,
    make_sol_attn_dispatch,
    patch_fused_adaln,
    patch_fused_rope,
    patch_fused_swiglu,
    patch_sol_attn,
)
from fusions import (  # noqa: F401
    QUANTIZERS,
    fused_apply_rotary_emb,
    fused_gate_add,
    fused_modulate,
    fused_swiglu,
)
from relayout import (  # noqa: F401
    FP8_DTYPE,
    FP8_MAX,
    read_pruned_fp8_checkpoint,
    rename_key,
    reorder_interleaved_qkv,
    _swap_swiglu_halves,
)
