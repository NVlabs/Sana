"""Installing the accelerations onto a built model, and taking them off again.

Sibling of `optimized/fusion_install.py`. Everything here is a patch rather than a structural
change, which is what lets one loaded model serve every configuration in
`transfeat/minimax_h3_gb10_*.toml` — and, more importantly, what lets the benchmark interleave
configurations inside a single process. That matters: this machine's clocks drift up to 15%
between processes, while three iterations inside one agree to 0.7%.

The Sol-Attn dispatch is built here rather than imported wholesale because the released kernel
package refuses anything that is not SM90 or SM100, and this is SM121.
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

from fusions import (
    fused_apply_rotary_emb,
    fused_gate_add,
    fused_modulate,
    fused_swiglu,
)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0

def _ensure_sol_attn_on_path() -> None:
    """Put `techniques/sparse_backends` on `sys.path`, as the entrypoint script intends.

    `H3_SOL_ATTN_ROOT` is set by `run_minimax_h3_gpu.sh`, the same variable the
    8xGB200 entrypoint uses; the fallback walks up to the repo root so the module is still
    importable when a benchmark is run by hand.
    """
    root = os.environ.get("H3_SOL_ATTN_ROOT")
    if not root:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.normpath(os.path.join(here, "..", "..", "..",
                                             "techniques", "sparse_backends"))
    if root not in sys.path:
        sys.path.insert(0, root)


H3_PREFIX_TOKENS = 951
"""Text and audio rows at the head of the packed sequence.

951 at *every* resolution — text occupies rows 0-536 and audio 537-950, verified against the
captured inputs at both 832x480 and 1344x768. Only the video tail grows with the canvas.
"""


def load_sol_attn():
    """The released Sol-Attn entry point. Nothing is bypassed and nothing is forked.

    Both were true once. The snapshot this port was written against had no architecture table
    and its dispatcher refused anything that was not SM90 or SM100, so the Triton reference was
    copied out and called directly; that copy also carried a hardcoded 951-row KV sink, because
    the entry point had no way to ask for one.

    Neither is true of the current head. `_backend_for_arch` selects a CuTe backend when the
    capability matches exactly and returns "triton" otherwise, so GB10's (12, 1) reaches the
    Triton path through the public function. And `sink_tokens` / `sink_start` are parameters
    now. Measured on real captured q/k/v across three blocks, the released path with
    `sink_tokens=951` reproduces the fork's output to six decimal places — cos 0.994420,
    0.991519, 0.985544 against dense, identical on both — at the same speed.

    The fork's second policy, handing the prefix's query rows to flash instead of the sparse
    grid, was dropped rather than ported: it costs 7.8% (154.8 ms against 143.6 ms over the
    three blocks) for +0.0005 of cosine. The 56.8-against-61.2 ms that once justified it was a
    comparison against forcing those rows exact *inside* the kernel, which is a third option
    and the worst of them.
    """
    _ensure_sol_attn_on_path()
    import sol_attn

    return sol_attn.sol_attn


def make_sol_attn_dispatch(
    tau: float = 1.0,
    dense_blocks: int = 2,
    dense_first_steps: int = 0,
    dense_last_steps: int = 0,
    num_steps: int | None = None,
    policy: bool = True,
    blocks_per_step: int = 52,
):
    """Build a `dispatch_attention_fn` replacement carrying H3's Sol-Attn policy.

    The tunable surface, all of which the released H3 profile pins:

    * `tau` — routing threshold. Higher keeps fewer key blocks exact.
    * `dense_blocks` — leave the first N transformer blocks dense (released: 2).
    * `dense_first_steps` — leave the first N denoising steps dense (released: 10).
    * `dense_last_steps` — the mirror of the above; not in the released profile, but the
      cheap end of the trajectory to protect if late steps turn out to matter more.
    * `policy` — the 951-row exact KV sink plus dense prefix queries. Verified against the
      captured inputs: text occupies rows 0-536, audio 537-950, video 951 onward, at both
      832x480 and 1344x768.

    Step and block indices are recovered from the call count: each denoising step issues
    `blocks_per_step` attention calls — two token-refiner blocks over the 951-row text stream
    and then the 50 block-stack calls over the packed sequence.
    """
    from diffusers.models.transformers import transformer_minimax_h3 as h3

    sol_attn = load_sol_attn()
    sink = dict(sink_tokens=H3_PREFIX_TOKENS, sink_start=0) if policy else {}

    dense = h3.dispatch_attention_fn
    state = {"call": 0}

    def dispatch(query, key, value, **kwargs):
        index = state["call"]
        state["call"] += 1
        step, position = divmod(index, blocks_per_step)
        block = position - 2                    # the two refiner blocks come first

        late = (
            num_steps is not None
            and dense_last_steps
            and step >= num_steps - dense_last_steps
        )
        if (
            block < dense_blocks
            or step < dense_first_steps
            or late
            or kwargs.get("attn_mask") is not None
        ):
            return dense(query, key, value, **kwargs)
        # Sol-Attn requires contiguous BTHD. The processor's q/k/v are `chunk` + `unflatten`
        # views of the fused QKV output, so they are strided; flash takes them as they are,
        # this path cannot.
        return sol_attn(query.contiguous(), key.contiguous(), value.contiguous(),
                        tau=tau, **sink)

    dispatch.reset = lambda: state.update(call=0)
    return dispatch


def patch_sol_attn(_model: nn.Module, tau: float = 1.0, dense_blocks: int = 2, **kwargs) -> None:
    """Route the block stack's self-attention through Sol-Attn, keeping a dense prefix.

    `dense_blocks` mirrors the released policy's first-two-blocks-dense rule. The two token
    refiner blocks run first and over a different (951-row) shape, so they are counted out
    and always left dense.
    """
    from diffusers.models.transformers import transformer_minimax_h3 as h3

    h3.dispatch_attention_fn = make_sol_attn_dispatch(tau, dense_blocks, **kwargs)


def patch_fused_swiglu(_model: nn.Module) -> None:
    """Replace `SwiGLU.forward`'s chunk/silu/mul with the fused kernel."""
    from diffusers.models.activations import SwiGLU

    def forward(self, hidden_states):
        return fused_swiglu(self.proj(hidden_states))

    SwiGLU.forward = forward


def patch_fused_rope(_model: nn.Module) -> None:
    """Route the attention processor's rotary call through the fused kernel."""
    from diffusers.models.transformers import transformer_minimax_h3 as h3

    h3._apply_rotary_emb = fused_apply_rotary_emb


def patch_fused_adaln(model: nn.Module) -> int:
    """Route every block's modulation and gating through the fused kernels."""
    from diffusers.models.transformers import transformer_minimax_h3 as h3

    def forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)

        residual = hidden_states
        norm_hidden_states = fused_modulate(
            self.norm1(hidden_states), scale_msa, shift_msa, adaln_indices
        )
        attn_output = self.attn(norm_hidden_states, rotary_emb, attention_mask)
        hidden_states = fused_gate_add(residual, gate_msa, attn_output, adaln_indices)

        residual = hidden_states
        norm_hidden_states = fused_modulate(
            self.norm2(hidden_states), scale_mlp, shift_mlp, adaln_indices
        )
        ff_output = self.ff(norm_hidden_states)
        return fused_gate_add(residual, gate_mlp, ff_output, adaln_indices)

    h3.MiniMaxH3TransformerBlock.forward = forward
    return sum(1 for m in model.modules() if isinstance(m, h3.MiniMaxH3TransformerBlock))


