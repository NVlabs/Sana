"""sol_engine_fwd: the SoL-Engine forward measured on one B200 / GB200.

On top of the standard math it applies, in order:
  * lossless block forward: strided cuDNN attention and torch.compile of the non-attention regions (fast_dit.py);
  * Sol-Attn sparse attention: refine steps fully sparse, HyperFlow keeps its first 2 steps and layers 0/1 dense,
    exact thresholds, tau 1.5 (HyperFlow) / 1.75 (refine), the leading [text | audio] rows kept exact;
  * MXFP8 GEMMs for the six big Linears of every block, re-quantized after each stage's LoRA merge, with SwiGLU fused
    into the activation quantizer;
  * resident stages: both stages' merged FP8 weights stay on the GPU, so a stage switch is a pointer swap;
  * one CUDA graph per (stage, step) over the whole transformer forward.
Sparse attention and MXFP8 are lossy; see README for the quality notes.
"""
import types

SPARSE = {
    "dense_steps": {"hyperflow": 2, "refine": 0},
    "dense_layers": {"hyperflow": [0, 1], "refine": []},
    "tau": {"hyperflow": 1.5, "refine": 1.75},
    "thresh_type": "exact",
}
AUDIO_ROWS = 2 * 207           # [text | audio | video] layout: 207 audio latents x 2 rows


def install(model):
    from . import fast_dit
    model.set_attention_backend("_native_cudnn")
    plan = fast_dit.SparsePlan(SPARSE["dense_steps"], SPARSE["dense_layers"], SPARSE["tau"], SPARSE["thresh_type"])
    mx_layers = fast_dit.quantize_mxfp8(model)
    fast_dit.install(model, compile=True, plan=plan, mx=True, fuse_swiglu=True)
    graphs = fast_dit.ForwardGraphs(model)
    return types.SimpleNamespace(name="sol_engine", plan=plan, graphs=graphs, mx_layers=mx_layers, resident=True)


def set_prompt_length(fwd, text_tokens):
    """The exact sink rows depend on the prompt length; captured graphs are only valid for one sequence length."""
    sink = text_tokens + AUDIO_ROWS
    if fwd.plan.sink_tokens != sink:
        fwd.plan.sink_tokens = sink
        fwd.graphs.graphs.clear()
        fwd.graphs.seen.clear()
        fwd.graphs.static.clear()
