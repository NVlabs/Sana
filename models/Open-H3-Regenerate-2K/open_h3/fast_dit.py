"""Lossless-intent speedups for the single-GPU MiniMax-H3 DiT forward (no change of weights, precision or schedule).

install(model, compile=True, plan=None, mx=False)
  * replaces every MiniMaxH3TransformerBlock.forward with the same math split into three torch.compile'd regions
    (pre-attention: adaln + norm1 + modulate + q/k/v + qk-norm + rope; post-attention: to_out + gated residual +
    norm2 + modulate + SwiGLU FFN + gated residual), with the attention kernel called eagerly in between;
  * attention: cuDNN SDPA on the strided (B, H, S, D) view (no permute+contiguous copies); Sol-Attn on sparse calls.
The AdaLN table, curve branch (Sharp) and LoRA-merged weights are read from the original modules, so stage switching
(StageMerger, time_embedder passthrough, sharp strength) keeps working.
"""
import types

import torch
import torch.nn.functional as F


def _rope(x, cos, sin):
    # same as transformer_minimax_h3._apply_rotary_emb
    d = cos.shape[-1]
    xr, xp = x[..., :d], x[..., d:]
    cos = cos.to(x.dtype)[None, :, None, :]
    sin = sin.to(x.dtype)[None, :, None, :]
    x1, x2 = xr.chunk(2, dim=-1)
    xr = xr * cos + torch.cat((-x2, x1), dim=-1) * sin
    return torch.cat((xr, xp), dim=-1)


# ---------------------------------------------------------------- MXFP8 (E4M3 data, E8M0 scale per 32 along K)
E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0


def _mx_quant(x):
    """x (..., K) bf16 -> (data (rows, K) e4m3, scale (rows, K/32) e8m0). Ceil exponent: the block max never clips."""
    k = x.shape[-1]
    blocks = x.reshape(-1, k // 32, 32).float()
    amax = blocks.abs().amax(-1).clamp(min=2.0 ** -120)
    exponent = torch.ceil(torch.log2(amax / E4M3_MAX))
    data = (blocks * torch.exp2(-exponent).unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3)
    scale = (exponent + 127).clamp(1, 254).to(torch.uint8).view(torch.float8_e8m0fnu)
    return data.reshape(-1, k), scale


def _to_blocked(scales):
    """cuBLAS 128x4 tile swizzle of a (rows, cols) e8m0 scale matrix."""
    rows, cols = scales.shape
    nrb, ncb = -(-rows // 128), -(-cols // 4)
    if rows % 128 or cols % 4:
        padded = scales.new_zeros(nrb * 128, ncb * 4)
        padded[:rows, :cols] = scales
        scales = padded
    blocks = scales.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


class MXLinear(torch.nn.Module):
    """Bias-free Linear whose GEMM runs in MXFP8, outside torch.compile (activation quantization is the one-pass
    Triton kernel in mx_quant_kernel.py). The BF16 weight stays the source of truth: the stage merger rewrites it,
    refresh() re-quantizes. With refresh(stage) the quantized weight is kept per stage and later switches are pointer
    swaps (activate), so resident stages hold exactly one FP8 copy per stage."""

    def __init__(self, base):
        super().__init__()
        assert base.bias is None
        self.base = base
        self._cache = {}
        self.wq, self.ws = _mx_weight(base.weight.detach())

    @property
    def weight(self):
        return self.base.weight

    @torch.no_grad()
    def refresh(self, stage=None):
        wq, ws = _mx_weight(self.base.weight)
        if stage is not None:
            self._cache[stage] = (wq, ws)
        self.wq, self.ws = wq, ws

    def cached(self, stage):
        return stage in self._cache

    def activate(self, stage):
        self.wq, self.ws = self._cache[stage]

    @torch.compiler.disable
    def forward(self, x):
        return self.matmul(*_mx_act(x), x.shape)

    @torch.compiler.disable
    def matmul(self, xq, xs, shape):
        out = torch._scaled_mm(xq, self.wq.t(), scale_a=xs, scale_b=self.ws, out_dtype=torch.bfloat16)
        return out.view(*shape[:-1], out.shape[-1])


def _mx_act(x):
    from .mx_quant_kernel import mx_quant_blocked
    return mx_quant_blocked(x.contiguous())


def _swiglu_mx(h):
    from .mx_quant_kernel import swiglu_mx_quant_blocked
    return swiglu_mx_quant_blocked(h.contiguous())


def _mx_weight(w):
    from .mx_quant_kernel import mx_quant_blocked
    return mx_quant_blocked(w.contiguous())


def _mx_targets(block):
    attn, ff = block.attn, block.ff
    return [(attn, "to_q"), (attn, "to_k"), (attn, "to_v"), (attn.to_out, 0), (ff.net[0], "proj"), (ff.net, 2)]


def quantize_mxfp8(model):
    """Swap the six big Linears of every transformer block for MXLinear; returns them (call .refresh() after merges)."""
    layers = []
    for block in model.transformer_blocks:
        for parent, key in _mx_targets(block):
            base = parent[key] if isinstance(key, int) else getattr(parent, key)
            mx = MXLinear(base)
            if isinstance(key, int):
                parent[key] = mx
            else:
                setattr(parent, key, mx)
            layers.append(mx)
    return layers


def _pre(block, hidden_states, temb, adaln_indices, cos, sin):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(temb)
    attn = block.attn
    x = block.norm1(hidden_states)
    x = x * (1.0 + scale_msa.index_select(0, adaln_indices)) + shift_msa.index_select(0, adaln_indices)
    q = attn.to_q(x).unflatten(-1, (attn.heads, -1))
    k = attn.to_k(x).unflatten(-1, (attn.heads, -1))
    v = attn.to_v(x).unflatten(-1, (attn.heads, -1))
    q = _rope(attn.norm_q(q), cos, sin)
    k = _rope(attn.norm_k(k), cos, sin)
    return q, k, v, (gate_msa, shift_mlp, scale_mlp, gate_mlp)


def _post(block, hidden_states, attn_out, mods, adaln_indices):
    gate_msa, shift_mlp, scale_mlp, gate_mlp = mods
    attn_out = block.attn.to_out[0](attn_out.flatten(2, 3))
    hidden_states = hidden_states + gate_msa.index_select(0, adaln_indices) * attn_out
    x = block.norm2(hidden_states)
    x = x * (1.0 + scale_mlp.index_select(0, adaln_indices)) + shift_mlp.index_select(0, adaln_indices)
    return hidden_states + gate_mlp.index_select(0, adaln_indices) * block.ff(x)


def _mx_norm1(block, hidden_states, temb, adaln_indices):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(temb)
    x = block.norm1(hidden_states)
    x = x * (1.0 + scale_msa.index_select(0, adaln_indices)) + shift_msa.index_select(0, adaln_indices)
    return x, (gate_msa, shift_mlp, scale_mlp, gate_mlp)


def _mx_qk(block, q, k, v, cos, sin):
    attn = block.attn
    q = _rope(attn.norm_q(q.unflatten(-1, (attn.heads, -1))), cos, sin)
    k = _rope(attn.norm_k(k.unflatten(-1, (attn.heads, -1))), cos, sin)
    return q, k, v.unflatten(-1, (attn.heads, -1))


def _mx_norm2(block, hidden_states, attn_out, mods, adaln_indices):
    gate_msa, shift_mlp, scale_mlp, gate_mlp = mods
    hidden_states = hidden_states + gate_msa.index_select(0, adaln_indices) * attn_out
    x = block.norm2(hidden_states)
    x = x * (1.0 + scale_mlp.index_select(0, adaln_indices)) + shift_mlp.index_select(0, adaln_indices)
    return hidden_states, x


def _mx_swiglu(h):
    # diffusers SwiGLU: hidden, gate = proj(x).chunk(2); hidden * silu(gate)
    hidden, gate = h.chunk(2, dim=-1)
    return hidden * torch.nn.functional.silu(gate)


def _mx_residual(hidden_states, ff_out, gate_mlp, adaln_indices):
    return hidden_states + gate_mlp.index_select(0, adaln_indices) * ff_out


def _attn_cudnn(q, k, v):
    # q, k, v: (B, S, H, D) -> SDPA wants (B, H, S, D); cuDNN accepts the strided transpose directly.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2)


class SparsePlan:
    """Which attention calls run Sol-Attn. Set .stage ("hyperflow"/"refine") and .step (0-based) before each
    forward; layers in dense_layers and steps < dense_steps[stage] stay on the dense kernel."""

    def __init__(self, dense_steps=None, dense_layers=(0, 1), tau=1.0, thresh_type="diag", sink_tokens=0):
        self.dense_steps = dict(dense_steps or {"hyperflow": 2, "refine": 1})
        # dense_layers: one list for every stage, or {stage: list}
        self.dense_layers = ({k: frozenset(v) for k, v in dense_layers.items()} if isinstance(dense_layers, dict)
                             else {stage: frozenset(dense_layers) for stage in self.dense_steps})
        # tau / thresh_type: a value for every stage, or {stage: value}
        self.tau = tau if isinstance(tau, dict) else {stage: float(tau) for stage in self.dense_steps}
        self.thresh_type = thresh_type if isinstance(thresh_type, dict) else {s: thresh_type for s in self.dense_steps}
        self.sink_tokens = int(sink_tokens)     # leading rows ([text | audio]) kept exact; 0 = off
        self.stage, self.step = None, None
        self.calls = {"sparse": 0, "dense": 0}

    def kernel_kwargs(self):
        kw = dict(tau=self.tau[self.stage], thresh_type=self.thresh_type[self.stage])
        if self.sink_tokens:
            kw.update(sink_start=0, sink_tokens=self.sink_tokens)
        return kw

    def sparse(self, layer):
        if self.stage is None or self.step is None or self.stage not in self.dense_steps:
            return False
        return self.step >= self.dense_steps[self.stage] and layer not in self.dense_layers[self.stage]


def _attn_sol(q, k, v, plan, layer=None):
    from sol_attn import sol_attn
    return sol_attn(q.contiguous(), k.contiguous(), v.contiguous(), **plan.kernel_kwargs())


def install(model, compile=True, mode=None, plan=None, mx=False, fuse_swiglu=True):
    """plan: optional SparsePlan; its sparse calls use Sol-Attn, everything else dense cuDNN attention.
    mx: the six big Linears are MXLinear (quantize_mxfp8 already applied) -> use the split forward."""
    attend = _attn_cudnn
    # 50 blocks x (pre, post) share one graph each (module params are graph inputs); stage switches change the
    # Sharp strength scalar, so allow a few recompiles before dynamo would fall back to eager.
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 512)
    pre, post = _pre, _post
    if compile:
        pre = torch.compile(_pre, dynamic=False, mode=mode)
        post = torch.compile(_post, dynamic=False, mode=mode)

    if mx:
        wrap = (lambda f: torch.compile(f, dynamic=False, mode=mode)) if compile else (lambda f: f)
        norm1, qk, norm2, swiglu, residual = (wrap(f) for f in (_mx_norm1, _mx_qk, _mx_norm2, _mx_swiglu, _mx_residual))

        def attend_any(self, q, k, v):
            if plan is not None and plan.sparse(self._layer_index):
                plan.calls["sparse"] += 1
                return _attn_sol(q, k, v, plan, self._layer_index)
            if plan is not None:
                plan.calls["dense"] += 1
            return attend(q, k, v)

        def forward_mx(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
            assert attention_mask is None
            attn_mod, ff = self.attn, self.ff
            x, mods = norm1(self, hidden_states, temb, adaln_indices)
            xq, xs = _mx_act(x)                                              # one quantization for q, k and v
            q = attn_mod.to_q.matmul(xq, xs, x.shape)
            k = attn_mod.to_k.matmul(xq, xs, x.shape)
            v = attn_mod.to_v.matmul(xq, xs, x.shape)
            q, k, v = qk(self, q, k, v, *rotary_emb)
            out = attend_any(self, q, k, v).flatten(2, 3)
            out = attn_mod.to_out[0](out)
            hidden_states, x = norm2(self, hidden_states, out, mods, adaln_indices)
            h = ff.net[0].proj(x)                                             # (.., 2 * ffn) bf16 = [hidden | gate]
            if fuse_swiglu:
                hq, hs = _swiglu_mx(h)                                        # SwiGLU fused into the MX quantizer
                down = ff.net[2].matmul(hq, hs, h.shape[:-1] + (h.shape[-1] // 2,))
            else:
                down = ff.net[2](swiglu(h))
            return residual(hidden_states, down, mods[3], adaln_indices)

        for index, block in enumerate(model.transformer_blocks):
            block._layer_index = index
            block.forward = types.MethodType(forward_mx, block)
        return model

    def forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
        assert attention_mask is None
        q, k, v, mods = pre(self, hidden_states, temb, adaln_indices, *rotary_emb)
        if plan is not None and plan.sparse(self._layer_index):
            plan.calls["sparse"] += 1
            out = _attn_sol(q, k, v, plan, self._layer_index)
        else:
            if plan is not None:
                plan.calls["dense"] += 1
            out = attend(q, k, v)
        return post(self, hidden_states, out, mods, adaln_indices)

    for index, block in enumerate(model.transformer_blocks):
        block._layer_index = index
        block.forward = types.MethodType(forward, block)
    return model


class ForwardGraphs:
    """One CUDA graph per (stage, step) over the WHOLE transformer forward (all 50 blocks + embeddings + heads).

    Inputs are copied into per-stage static buffers only at the forward boundary (a few small tensors plus the latent
    rows), outputs (video/audio velocity) are cloned once per forward. All graphs share one private pool. First sight
    of a key runs eagerly (lazy compiles), second captures, later calls replay. The caller sets .key = (stage, step)
    before each forward (None = eager)."""

    def __init__(self, model):
        self.model = model
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs, self.seen, self.static = {}, set(), {}
        self.key = None
        self.stats = {"replay": 0, "capture": 0, "eager": 0}
        inner = model.forward

        def forward(*args, **kwargs):
            return self._call(inner, args, kwargs)
        model.forward = forward

    @staticmethod
    def _flatten(args, kwargs):
        names = [f"a{i}" for i in range(len(args))] + sorted(kwargs)
        values = list(args) + [kwargs[k] for k in sorted(kwargs)]
        return names, values

    def _side_inputs(self):
        """Tensors the forward reads that are not arguments: HyperFlow's endpoint timesteps (set per step)."""
        emb = getattr(self.model, "time_embedder", None)
        endpoint = getattr(emb, "_endpoint_timesteps", None)
        return {"endpoint": endpoint} if isinstance(endpoint, torch.Tensor) else {}

    def _call(self, inner, args, kwargs):
        if self.key is None or torch.cuda.is_current_stream_capturing():
            return inner(*args, **kwargs)
        stage, key = self.key[0], tuple(self.key)
        side = self._side_inputs()
        if side:
            kwargs = dict(kwargs, __endpoint=side["endpoint"])
        names, values = self._flatten(args, kwargs)
        tensors = [(n, v) for n, v in zip(names, values) if isinstance(v, torch.Tensor)]
        call_kwargs = {k: v for k, v in kwargs.items() if k != "__endpoint"}
        if key not in self.graphs:
            if key not in self.seen:
                self.seen.add(key)
                self.stats["eager"] += 1
                return inner(*args, **call_kwargs)
            buf = self.static.get(stage)
            sig = [(n, tuple(v.shape), v.dtype) for n, v in tensors]
            if buf is None or buf["sig"] != sig:
                buf = self.static[stage] = {"sig": sig, "in": {n: v.clone() for n, v in tensors}, "out": None}
            for n, v in tensors:
                buf["in"][n].copy_(v)
            static_args = [buf["in"].get(f"a{i}", a) if isinstance(a, torch.Tensor) else a for i, a in enumerate(args)]
            static_kwargs = {k: (buf["in"][k] if isinstance(v, torch.Tensor) else v)
                             for k, v in kwargs.items() if k != "__endpoint"}
            if "__endpoint" in buf["in"]:                 # the graph must read the endpoint from the static buffer
                self.model.time_embedder.set_endpoint(buf["in"]["__endpoint"])
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                out = inner(*static_args, **static_kwargs)
                if buf["out"] is None:
                    buf["out"] = tuple(torch.empty_like(o) for o in out)
                for dst, src in zip(buf["out"], out):
                    dst.copy_(src)
                del out
            self.graphs[key] = graph
            self.stats["capture"] += 1
        buf = self.static[stage]
        for n, v in tensors:
            dst = buf["in"][n]
            if dst.data_ptr() != v.data_ptr():
                dst.copy_(v, non_blocking=True)
        self.graphs[key].replay()
        self.stats["replay"] += 1
        return tuple(o.clone() for o in buf["out"])
