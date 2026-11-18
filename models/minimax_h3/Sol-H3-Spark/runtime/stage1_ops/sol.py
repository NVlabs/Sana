"""One Ref2VA Sol policy around the original post-RoPE attention seam.

No kernel is implemented here. SM121 uses the shared BF16 Triton Sol backend.
The stable visual/text+audio partition changes sparse block grouping explicitly;
Q/K/V receive the same permutation after RoPE and output is restored afterwards.
"""
from pathlib import Path
import sys
from types import MethodType

POLICY = "FA4_Sol_text_audio_sink"
TAUS = (None, 1.0, 1.25, 1.5)


def route(step, layer):
    if not 0 <= step < 4 or not 0 <= layer < 50:
        raise RuntimeError("Sol route requires four updates and 50 body layers")
    return None if step == 0 or layer == 0 else TAUS[step]


def sink_plan(tags, video_indices, condition_video_rows):
    """CPU metadata only; preserve the native relative order within each part."""
    tags = [int(value) for value in tags]
    if not tags or any(tag not in (0, 1, 2) for tag in tags):
        raise RuntimeError("native joint token tags must be visual0/text1/audio2")
    visual = [index for index, tag in enumerate(tags) if tag == 0]
    sinks = [index for index, tag in enumerate(tags) if tag in (1, 2)]
    if not visual or not sinks:
        raise RuntimeError("Ref2VA Sol requires both visual and text/audio rows")
    start = len(visual)
    spill = visual[start // 64 * 64:]
    generated = set(video_indices[condition_video_rows:])
    if not set(spill) <= generated:
        raise RuntimeError("sink KV boundary spill must contain generated video only")
    permutation = visual + sinks
    inverse = [0] * len(tags)
    for destination, source in enumerate(permutation):
        inverse[source] = destination
    receipt = {"joint_tokens": len(tags), "sink_start": start, "sink_tokens": len(sinks),
               "text_sink_tokens": tags.count(1), "audio_sink_tokens": tags.count(2),
               "visual_nonsink_tokens": start, "sink_kv_block_size": 64,
               "boundary_generated_video_kv_tokens": len(spill),
               "boundary_reference_image_kv_tokens": 0, "boundary_qwen_visual_kv_tokens": 0,
               "dense_sink_query_tokens": len(sinks), "partition": "stable_visual_then_text_audio",
               "permutation_placement": "QKV after native QK norm and RoPE; inverse on output",
               "sparse_block_grouping_changed": True}
    return permutation, inverse, receipt


class RequestRoute:
    """Eager request/forward state, read only by real custom-op executions."""
    def __init__(self, state):
        self.state = state
        self.step = -1
        self.permutation = self.inverse = None
        self.gpu_indices = None

    def prepare(self, layout):
        if layout.token_tags.device.type != "cpu" or layout.video_indices.device.type != "cpu":
            raise RuntimeError("Sol routing must use original CPU packed-layout metadata")
        self.permutation, self.inverse, plan = sink_plan(
            layout.token_tags.tolist(), layout.video_indices.tolist(), layout.num_condition_video_rows)
        self.gpu_indices = None
        self.step = -1
        self.state["sol_attention"] = {"policy": POLICY, "backend": "shared_triton_sm121",
            "qkv_dtype": "torch.bfloat16", "softmax_scale": 128 ** -0.5,
            "fallback_calls": 0, "sol_calls": 0, "full_dense_fa4_calls": 0,
            "sink_query_fa4_calls": 0, "tau_calls": {"1.0": 0, "1.25": 0, "1.5": 0},
            "body_routes": [[], [], [], []], "layout": plan}

    def start_forward(self):
        self.step += 1
        if self.step >= 4 or self.permutation is None:
            raise RuntimeError("Sol request was not reset before four native model forwards")

    def record_body(self, layer, tau):
        rows = self.state["sol_attention"]["body_routes"][self.step]
        if layer != len(rows) or tau != route(self.step, layer):
            raise RuntimeError("actual body attention step/layer route drift")
        rows.append("dense" if tau is None else f"sol:{tau}")


def validate_request(receipt):
    expected = [["dense" if route(step, layer) is None else f"sol:{route(step, layer)}"
                 for layer in range(50)] for step in range(4)]
    if (receipt["body_routes"] != expected or receipt["sol_calls"] != 147
            or receipt["full_dense_fa4_calls"] != 55 or receipt["sink_query_fa4_calls"] != 147
            or receipt["tau_calls"] != {"1.0": 49, "1.25": 49, "1.5": 49}
            or receipt["fallback_calls"] != 0):
        raise RuntimeError(f"actual Ref2VA Sol route mismatch: {receipt}")
    return {**receipt, "status": "PASS_RUNTIME_ROUTES", "logical_attention_calls": 202,
            "physical_fa4_calls": 202}


def install(model, state, controller):
    """Keep all 52 original compiled regions; make only body attention opaque."""
    import torch
    from fastvideo.models.loader import fsdp_load
    from .regional import _layer_number

    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("this Sol integration requires the shared SM121 Triton backend")
    repo = str(Path(__file__).resolve().parents[5])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from techniques.sparse_backends.sol_attn_backend import _load_sol_attn
    kernel = _load_sol_attn()
    bodies = [block.attn.distributed_attention.attn_impl for block in model.transformer_blocks]
    attentions = [module for module in model.modules() if type(module).__name__ == "MiniMaxH3Attention"]
    if len(bodies) != 50 or len(attentions) != 52 or any(m.to_gate_compress is not None for m in attentions):
        raise RuntimeError("Ref2VA Sol requires 50 body and two ungated native text attentions")
    if any(type(impl).__name__ != "FlashAttentionImpl" or impl.causal
           or impl.softmax_scale != 128 ** -0.5 for impl in bodies):
        raise RuntimeError("native BF16 noncausal FA4 head128 scale contract changed")
    originals = [impl.forward for impl in bodies]

    @torch.library.custom_op("sol_h3_spark_ref::body", mutates_args=(), device_types="cuda")
    def body_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: torch.Tensor) -> torch.Tensor:
        index = _layer_number(layer, 50)
        tau = route(controller.step, index)
        if tau is None:
            result = originals[index](q, k, v, None)
        else:
            receipt = state["sol_attention"]
            plan = receipt["layout"]
            if (q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype
                    or q.ndim != 4 or q.shape != k.shape or q.shape != v.shape
                    or q.shape[0] != 1 or q.shape[1] != plan["joint_tokens"] or q.shape[-1] != 128):
                raise RuntimeError("Ref2VA Sol actual QKV ABI mismatch; no dense fallback")
            if controller.gpu_indices is None:
                controller.gpu_indices = (
                    torch.tensor(controller.permutation, dtype=torch.long, device=q.device),
                    torch.tensor(controller.inverse, dtype=torch.long, device=q.device))
            permutation, inverse = controller.gpu_indices
            pq, pk, pv = (value.index_select(1, permutation).contiguous() for value in (q, k, v))
            result = kernel(pq, pk, pv, tau=tau, scale=bodies[index].softmax_scale,
                            thresh_type="diag", kv_splits=1,
                            sink_start=plan["sink_start"], sink_tokens=plan["sink_tokens"])
            receipt["sol_calls"] += 1
            receipt["tau_calls"][str(tau)] += 1
            state["attention_calls"] += 1  # One logical layer, separate from the sink subcall.
            state["sol_sink_query_active"] = True
            try:
                dense = originals[index](pq[:, plan["sink_start"]:].contiguous(), pk, pv, None)
            finally:
                state["sol_sink_query_active"] = False
            result[:, plan["sink_start"]:] = dense
            result = result.index_select(1, inverse)
        controller.record_body(index, tau)
        return result.contiguous()

    @body_op.register_fake
    def body_fake(q, k, v, layer):
        return q.new_empty((*q.shape[:-1], v.shape[-1]))

    def body_forward(self, query, key, value, attn_metadata):
        if attn_metadata is not None and getattr(attn_metadata, "attn_mask", None) is not None:
            raise RuntimeError("Ref2VA Sol does not accept masked native attention")
        return body_op(query, key, value, self._sol_h3_cpu_layer)

    for index, impl in enumerate(bodies):
        impl._sol_h3_cpu_layer = torch.tensor(index, dtype=torch.int64, device="cpu")
        impl.forward = MethodType(body_forward, impl)
    model.prepare_for_compile()
    enabled = fsdp_load._enable_regional_attention_compile(model)
    compiled = fsdp_load._compile_model_regions(model, {"fullgraph": True,
                                                       "options": {"emulate_precision_casts": True}})
    if (enabled, compiled) != (52, 52):
        raise RuntimeError("Ref2VA Sol must retain all 52 native compiled regions")
    state["sol_regional_compile"] = {"regions_wrapped": compiled, "attention_modules_enabled": enabled,
        "fullgraph": True, "boundary": "body post-RoPE attention only; native text attention unchanged"}
