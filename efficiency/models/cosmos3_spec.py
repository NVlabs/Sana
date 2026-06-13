# Cosmos3 ModelSpec -- declares the structural seams the efficiency framework
# plugs into for nvidia/Cosmos3-Super (the acceleration TARGET). Mirrors
# efficiency/models/ltx2_spec.py, the fully-worked LTX-2.3 reference.
#
# PORTED into auto-video from Sol-LTX-Infer @ codex/cosmos3-run-env (29d0d9e),
# python/sglang/multimodal_gen/runtime/efficiency/. The LTX2 spec is the proven
# template; this Cosmos3 spec is the target each acceleration loop wires against.
#
# STATUS: get_blocks is wired to the real generation block list
# (Cosmos3OmniTransformer.gen_layers). Capabilities are declared CONSERVATIVELY
# -- only seams whose accessor is actually set are claimed -- so compose() will
# correctly REFUSE a technique whose seam is not yet wired (the framework's
# capability type-check). Each acceleration loop adds the capability + accessor
# it needs:
#   - loops/token_prune     -> refine prunable_segment to the video-token span;
#                              add prune_gather/prune_scatter if the pruned
#                              forward needs per-token coords/timestep/masks.
#   - loops/sparse_attention -> add Capability.SWAPPABLE_ATTENTION + the attn seam.
#   - loops/step_cache, loops/kwl_fusion, loops/nvfp4_ffn -> step-level techniques
#                              / build-time transforms; see each loop's README.

from __future__ import annotations

from efficiency.registry import register_model_spec
from efficiency.spec import ModelSpec
from efficiency.technique import Capability


def _cosmos3_prunable_segment(hidden, ctx):
    """Prunable video-token span along seq_dim.

    TODO(loop:token_prune): Cosmos3OmniTransformer concatenates
    understanding/text tokens with the generated video patch tokens; restrict
    this to the video patch span once the forward layout is wired. Until then,
    default to the whole sequence handed to the gen-layer loop (correct on the
    single-video generation path)."""
    n = hidden.shape[ctx.spec.seq_dim]
    return (0, n)


@register_model_spec("Cosmos3", "Cosmos3OmniTransformer")
def _cosmos3_spec() -> ModelSpec:
    return ModelSpec(
        name="Cosmos3",
        capabilities=frozenset(
            {
                Capability.BLOCKS,
                Capability.PRUNABLE_TOKENS,
            }
        ),
        get_blocks=lambda tf: getattr(tf, "gen_layers"),
        prunable_segment=_cosmos3_prunable_segment,
        seq_dim=1,
        # The official 4-GPU config runs SP=2; set True when wiring token-prune
        # under sequence-parallel so per-rank-local top-K keeps shards balanced.
        sp_local_prune=False,
    )
