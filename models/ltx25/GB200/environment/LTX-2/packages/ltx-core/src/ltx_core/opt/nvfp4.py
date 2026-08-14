"""NVFP4 Linear for LTX-2.5 -- real 4-bit COMPUTE, not storage.

torch 2.11 ships the dtype (float4_e2m1fn_x2) and the GEMM (torch._scaled_mm)
but no bf16->fp4 cast, so the quantizer is here. Three things this file gets
right that a naive port does not:

1. SCALE LAYOUT.  _scaled_mm validates only numel and contiguity of the block
   scales, but Blackwell wants them SWIZZLED into 128x4 tiles. Row-major scales
   pass validation and silently compute garbage -- measured relerr 0.51 vs
   0.14 for the correct layout, with identical timing. The swizzle is written
   straight from the quantizer kernel as index math, not as a second pass.

2. ENCODE COST.  A tl.where ladder over the E2M1 midpoints costs ~20 fp32 ops
   per element and made the quantizer compute bound at 0.7-2.2 TB/s. Blackwell
   has cvt.rn.satfinite.e2m1x2.f32, one instruction per PAIR; with it the
   quantizer sits on the pure load/store floor (measured: ptx 0.220 ms vs
   no-encode 0.211 vs copy 0.208 at 24576x16384).

3. SHARED INPUTS.  to_q/to_k/to_v of a self-attention receive the SAME tensor.
   Quantizing once and reusing turns 3 quantizations into 1. The cache holds a
   reference to x and hits only on `x is cached_x`, so a freed tensor cannot
   have its address reused underneath us (the id()-reuse trap that already cost
   us once in the ada-cache).

Measured per-Linear vs bf16 at M=24576: 2.57x (16384x4096), 2.69x (4096x16384),
1.28x (4096x4096). The 4096x4096 case is quantizer-bound, which is what the
shared-input cache and epilogue fusion address.
"""

from __future__ import annotations

import os
import re

import torch
import triton
import triton.language as tl

BS = 16                     # NVFP4 block size
E2M1_MAX = 6.0


# ---------------------------------------------------------------- quantizer
@triton.jit
def _e2m1x2_ptx(hi, lo):
    """cvt.rn.satfinite.e2m1x2.f32: two fp32 -> one packed byte, hi in the
    upper nibble. Round-to-nearest-even, which is also more correct than a
    midpoint ladder."""
    return tl.inline_asm_elementwise(
        asm=("{ .reg .b8 t; "
             "cvt.rn.satfinite.e2m1x2.f32 t, $1, $2; "
             "cvt.u32.u8 $0, t; }"),
        constraints="=r,f,f",
        args=[hi, lo],
        dtype=tl.uint8,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _quant_kernel(X, OUT, SC, K, KP, KS, NCB, BSZ: tl.constexpr, CH: tl.constexpr):
    """CH = output bytes per program. Reads 2*CH contiguous elements."""
    row = tl.program_id(0)
    c0 = tl.program_id(1) * CH
    off = c0 * 2 + tl.arange(0, 2 * CH)
    v = tl.load(X + row * K + off, mask=off < K, other=0.0).to(tl.float32)

    nb: tl.constexpr = 2 * CH // BSZ
    amax = tl.max(tl.reshape(tl.abs(v), (nb, BSZ)), axis=1)
    s8 = tl.maximum(amax / 6.0, 1e-8).to(tl.float8e4nv)

    # to_blocked() swizzle as index math:
    #   (nrb,128,ncb,4) -permute-> (t,4,32,4) -transpose(1,2)-> (t,32,4,4)
    # flat = (((rb*ncb + cb)*32 + rin%32)*4 + rin//32)*4 + cin
    sblk = c0 // (BSZ // 2) + tl.arange(0, nb)
    rb = row // 128
    rin = row % 128
    so = (((rb * NCB + sblk // 4) * 32 + (rin % 32)) * 4 + (rin // 32)) * 4 + (sblk % 4)
    tl.store(SC + so, s8, mask=sblk < KS)

    # divide by the ROUNDED scale so stored scale and codes agree exactly
    s = tl.reshape(tl.broadcast_to(s8.to(tl.float32)[:, None], (nb, BSZ)), (2 * CH,))
    lo, hi = tl.split(tl.reshape(v / s, (CH, 2)))
    pair = c0 + tl.arange(0, CH)
    tl.store(OUT + row * KP + pair, _e2m1x2_ptx(hi, lo), mask=pair < KP)


# use_fast_accum is rejected outright for Float4_e2m1fn_x2 -- fp8 only.
# Step-wise dense guard. total is explicit rather than inferred: getting it
# wrong would silently guard the wrong steps, and "last N" needs it.
GUARD = {
    "step": -1,
    "total": int(os.environ.get("LTX_FP4_TOTAL_STEPS", "40")),
    "first": int(os.environ.get("LTX_FP4_KEEP_FIRST_STEPS", "0")),
    "last": int(os.environ.get("LTX_FP4_KEEP_LAST_STEPS", "0")),
    "n_bf16": 0,
    "n_fp4": 0,
}


# Set during graph capture so a whole set is captured at one precision; None
# means "follow the step guard", which is what happens at replay time.
FORCE = {"mode": None}


def use_bf16() -> bool:
    if FORCE["mode"] is not None:
        return FORCE["mode"] == "bf16"
    return step_is_dense()


def step_is_dense(i=None) -> bool:
    if not (GUARD["first"] or GUARD["last"]):
        return False
    i = GUARD["step"] if i is None else i
    return i >= 0 and (i < GUARD["first"] or i >= GUARD["total"] - GUARD["last"])


def guard_summary() -> str:
    n = GUARD["n_bf16"] + GUARD["n_fp4"]
    pct = 100.0 * GUARD["n_bf16"] / max(n, 1)
    return (f"blocks(first={_KEEP_FIRST},last={_KEEP_LAST}) "
            f"steps(first={GUARD['first']},last={GUARD['last']},total={GUARD['total']}) "
            f"linear_calls_bf16={GUARD['n_bf16']} fp4={GUARD['n_fp4']} ({pct:.1f}% bf16)")


_QUANT_BACKEND = os.environ.get("LTX_FP4_QUANT", "flashinfer")
_FI = {"fn": None, "checked": False}


def _flashinfer_quant():
    """-> callable(x) or None. Resolved once; absence is not an error."""
    if _FI["fn"] is None and not _FI["checked"]:
        _FI["checked"] = True
        try:
            from flashinfer.quantization import nvfp4_quantize_cute_dsl
            one = torch.ones((), device="cuda", dtype=torch.float32)

            def _q(x):
                k = x.shape[-1]
                xc = x.reshape(x.numel() // k, k)
                if not xc.is_contiguous():
                    xc = xc.contiguous()
                q, sf = nvfp4_quantize_cute_dsl(xc, one, sf_layout=0)
                # flashinfer hands back the block scales as raw uint8 in an
                # (M, K/16) view; _scaled_mm wants them typed e4m3 and flat.
                # sf_layout=0 is already the 128x4 swizzle it needs.
                if sf.dtype == torch.uint8:
                    sf = sf.view(torch.float8_e4m3fn)
                return q.view(torch.float4_e2m1fn_x2), sf.reshape(-1)

            _FI["fn"] = _q
        except Exception as e:
            print(f"[fp4] flashinfer quantizer unavailable ({type(e).__name__}: "
                  f"{str(e)[:120]}), using ours", flush=True)
    return _FI["fn"]


def quantize(x: torch.Tensor):
    """Whichever backend is selected and actually importable."""
    if _QUANT_BACKEND == "flashinfer":
        fn = _flashinfer_quant()
        if fn is not None:
            return fn(x)
    return quantize_nvfp4(x)


_CH = int(os.environ.get("LTX_FP4_CH", "1024"))
_NW = int(os.environ.get("LTX_FP4_NW", "2"))


def quantize_nvfp4(x: torch.Tensor):
    """(rows, K) bf16 -> (packed fp4 (rows, K//2), flat swizzled e4m3 scales).

    Rows are padded to 128 and scale columns to 4 for the tile; the pad entries
    are never read by the GEMM. torch accepts the larger numel (verified at
    M=126 and M=8208, both non-multiples of 128)."""
    k = x.shape[-1]
    rows = x.numel() // k
    xc = x.reshape(rows, k)
    if not xc.is_contiguous():
        xc = xc.contiguous()
    ks = k // BS
    nrb = -(-rows // 128)
    ncb = -(-ks // 4)
    packed = torch.empty(rows, k // 2, dtype=torch.uint8, device=x.device)
    # The GEMM never reads scale slots for rows >= rows, so the pad is don't-care
    # when it exists at all -- but e4m3 garbage can be NaN and the tile load is
    # 128 rows wide, so only skip the memset when there IS no pad.
    scales = (torch.empty if rows % 128 == 0 else torch.zeros)(
        nrb * 128 * ncb * 4, dtype=torch.float8_e4m3fn, device=x.device)
    ch = min(_CH, triton.next_power_of_2(max(k // 2, 1)))
    _quant_kernel[(rows, triton.cdiv(k // 2, ch))](
        xc, packed, scales, k, k // 2, ks, ncb, BSZ=BS, CH=ch, num_warps=_NW)
    return packed.view(torch.float4_e2m1fn_x2), scales


class _InputCache:
    """One entry. Holds a reference to x so its address cannot be recycled, and
    hits only on identity -- never on data_ptr alone."""

    __slots__ = ("x", "q", "s", "hits", "misses")

    def __init__(self):
        self.x = None
        self.q = self.s = None
        self.hits = self.misses = 0

    def get(self, x):
        if self.x is x:
            self.hits += 1
            return self.q, self.s
        self.misses += 1
        q, s = quantize(x)
        self.x, self.q, self.s = x, q, s
        return q, s

    def put(self, x, q, s):
        """Producer-side fill: a fused epilogue already emitted the fp4 form, so
        the consuming Linear must not quantize again."""
        self.x, self.q, self.s = x, q, s

    def clear(self):
        self.x = self.q = self.s = None


CACHE = _InputCache()


# ------------------------------------------------------------------- module
class NVFP4CastLinear(torch.nn.Linear):
    """Installed by __class__ reassignment onto an existing Linear, NOT by
    replacing the module -- the same trick Fp8CastLinear uses, and for the same
    reason: `weight` stays in the module tree where the loader put it.

    Replacing the module instead broke stage 2, which disposes the transformer
    between stages and rebuilds it: the state dict no longer had a `.weight` to
    materialize into, and the rebuild died on a meta buffer. The fp4 copy is
    derived state, so it is (re)built lazily and invalidated in _apply, which is
    what every .to() / .to_empty() / dispose goes through.
    """

    _COMPUTE = (torch.bfloat16, torch.float16)

    def _prepare(self):
        w = self.weight.data
        if w.dtype not in self._COMPUTE:
            # fp8-cast is a plain cast with no scale, so dequantizing is a cast.
            # Doing it once here also deletes Fp8CastLinear's per-forward upcast.
            w = w.to(torch.bfloat16)
        b = self.bias
        if b is not None and b.dtype not in self._COMPUTE:
            b = b.data.to(torch.bfloat16)
        wq, ws = quantize(w)
        self._nvfp4 = (wq, ws, b)
        return self._nvfp4

    def _apply(self, fn, recurse=True):
        self._nvfp4 = None          # weight storage may be about to change
        return super()._apply(fn, recurse)

    def forward(self, x):
        q = getattr(self, "_nvfp4", None) or self._prepare()
        wq, ws, bias = q
        if use_bf16():
            GUARD["n_bf16"] += 1
            w = self.weight
            if w.dtype not in self._COMPUTE:
                w = w.to(torch.bfloat16)   # same upcast the fp8-cast baseline pays
            return torch.nn.functional.linear(x, w, bias)
        GUARD["n_fp4"] += 1
        xq, xs = CACHE.get(x)
        out = torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws, bias=bias,
                               out_dtype=torch.bfloat16)
        return out.view(*x.shape[:-1], self.out_features)


# Only the video-side linears see long sequences. audio_* run on 126 tokens and
# attn2.to_k/to_v run on the text context, where the GEMM saving is smaller than
# the quantizer launch -- swapping everything measured 368 ms vs 298 ms eager.
_VIDEO = re.compile(r"\.(attn1|attn2|ff)\.")


_BLK = re.compile(r"transformer_blocks\.(\d+)\.")
_KEEP_FIRST = int(os.environ.get("LTX_FP4_KEEP_FIRST_BLOCKS", "0"))
_KEEP_LAST = int(os.environ.get("LTX_FP4_KEEP_LAST_BLOCKS", "0"))
_N_BLOCKS = int(os.environ.get("LTX_FP4_N_BLOCKS", "48"))


def video_only(name: str) -> bool:
    if not (bool(_VIDEO.search(name)) and ".attn2.to_k" not in name
            and ".attn2.to_v" not in name):
        return False
    # Mixed precision: the first and last blocks stay bf16. Early blocks set the
    # structure the rest of the stack refines and late blocks write the output,
    # so those are where 4-bit rounding is most visible.
    m = _BLK.search(name)
    if m and (_KEEP_FIRST or _KEEP_LAST):
        i = int(m.group(1))
        if i < _KEEP_FIRST or i >= _N_BLOCKS - _KEEP_LAST:
            return False
    return True


def swap_linears(model, predicate=None) -> int:
    """Collect targets first, then mutate -- swapping during the walk revisits
    the replacements."""
    if predicate is None and os.environ.get("LTX_FP4_SCOPE", "video") == "video":
        predicate = video_only
    targets = []
    for name, mod in model.named_modules():
        if isinstance(mod, NVFP4CastLinear):
            continue
        for cname, child in mod.named_children():
            full = f"{name}.{cname}" if name else cname
            if not isinstance(child, torch.nn.Linear) or isinstance(child, NVFP4CastLinear):
                continue
            if predicate is not None and not predicate(full):
                continue
            # K must tile the 16-element block and the 4-wide scale tile; tiny
            # out_features (to_gate_logits has 32) are not worth the padding.
            if child.in_features % 128 or child.out_features % 128:
                continue
            targets.append((mod, cname, child))
    dtypes = {(str(c.weight.dtype), str(c.bias.dtype) if c.bias is not None else "none")
              for _, _, c in targets}
    print(f"[fp4] swapping {len(targets)} linears "
          f"(keep_first={_KEEP_FIRST} keep_last={_KEEP_LAST}), dtypes: {dtypes}",
          flush=True)
    for _, _, child in targets:
        child.__class__ = NVFP4CastLinear
        child._nvfp4 = None
    return len(targets)


def swap_report(model) -> dict:
    from collections import Counter
    c = Counter()
    for name, m in model.named_modules():
        if isinstance(m, NVFP4CastLinear):
            c[re.sub(r"\.\d+\.", ".N.", name).split("transformer_blocks.N.")[-1]] += 1
    return dict(c)
