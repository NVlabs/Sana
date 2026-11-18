"""sol_engine_fwd: the SoL-Engine forward measured on one B200 / GB200, in two presets (`--fwd`):

  * sol_engine (default): refine step 0 dense, steps 1-2 Sol-Attn sparse with tau 1.0;
  * sol_engine_fast: every refine step Sol-Attn sparse with tau 1.75, the fastest setting.

Upscale + 3-step refine on one GB200 (warm, no VAE): standard 73.1 s, sol_engine 40.6 s, sol_engine_fast 20.5 s.
Speed goes up from standard to sol_engine to sol_engine_fast; the faster presets trade a little fine detail for it.

On top of the standard math it applies, in order:
  * lossless block forward: strided cuDNN attention and torch.compile of the non-attention regions (fast_dit.py);
  * Sol-Attn sparse attention as set by the preset (SPARSE); HyperFlow keeps its first 2 steps and layers 0/1 dense
    with tau 1.5 in both presets, exact thresholds, the leading [text | audio] rows kept exact;
  * MXFP8 GEMMs for the six big Linears of every block, re-quantized after each stage's LoRA merge, with SwiGLU fused
    into the activation quantizer;
  * resident stages: both stages' merged FP8 weights stay on the GPU, so a stage switch is a pointer swap;
  * one CUDA graph per (stage, step) over the whole transformer forward.
Sparse attention and MXFP8 are lossy; see README for the quality notes.
"""
import types

# The presets differ only in the refine stage.
SPARSE = {
    "sol_engine": {
        "dense_steps": {"hyperflow": 2, "refine": 1},
        "dense_layers": {"hyperflow": [0, 1], "refine": []},
        "tau": {"hyperflow": 1.5, "refine": 1.0},
        "thresh_type": "exact",
    },
    "sol_engine_fast": {
        "dense_steps": {"hyperflow": 2, "refine": 0},
        "dense_layers": {"hyperflow": [0, 1], "refine": []},
        "tau": {"hyperflow": 1.5, "refine": 1.75},
        "thresh_type": "exact",
    },
}
AUDIO_ROWS = 2 * 207           # [text | audio | video] layout: 207 audio latents x 2 rows


def install(model, preset="sol_engine"):
    from . import fast_dit
    model.set_attention_backend("_native_cudnn")
    sparse = SPARSE[preset]
    plan = fast_dit.SparsePlan(sparse["dense_steps"], sparse["dense_layers"], sparse["tau"], sparse["thresh_type"])
    mx_layers = fast_dit.quantize_mxfp8(model)
    fast_dit.install(model, compile=True, plan=plan, mx=True, fuse_swiglu=True)
    graphs = fast_dit.ForwardGraphs(model)
    return types.SimpleNamespace(name=preset, plan=plan, graphs=graphs, mx_layers=mx_layers, resident=True)


def set_prompt_length(fwd, text_tokens):
    """The exact sink rows depend on the prompt length; captured graphs are only valid for one sequence length."""
    sink = text_tokens + AUDIO_ROWS
    if fwd.plan.sink_tokens != sink:
        fwd.plan.sink_tokens = sink
        fwd.graphs.graphs.clear()
        fwd.graphs.seen.clear()
        fwd.graphs.static.clear()
