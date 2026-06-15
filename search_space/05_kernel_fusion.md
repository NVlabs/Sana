# Search Space: Kernel Fusion and Lossless Operator Optimization

**Scope**: Identify kernel-level fusion and optimization opportunities that are **algorithmically lossless** (bit-exact or numerically equivalent up to floating-point accumulation order). Results may not be bit-exact due to reordering of additions or fused multiply-adds, which is acceptable.

---

## Background

The `KWLFusions` transform applies a bundle of operator fusions at model-build time. Each fusion is gated by an environment flag (`SGLANG_HQ_KWL_*`) and can be ablated independently.

---

## 1. Example Fusion Directions

The following are example patterns worth exploring. Each is a direction, not a prescription — whether it helps depends on profiling the specific model and hardware setup.

---

### FFN Residual as GEMM Epilogue

**Pattern**: Fuse the residual add and gate multiply into the GEMM epilogue of FFN proj_out, using the cuBLAS / cuDNN epilogue API (`output = alpha * A @ B + beta * C` where C is the residual tensor). This avoids a separate elementwise kernel for the residual add.

- **Numerical character**: Accumulation order changes slightly; not bit-exact, within normal FP rounding.
- **Example forms**: `torch.addmm` with residual as bias term; custom CUTLASS `LinearCombinationResidual` epilogue; `out = proj_out(ffn_hidden) * gate + residual` mapped to scaled addmm.

---

### QK Norm + RoPE Extension to Other Attention Types

**Pattern**: `FUSED_QKNORM_ROPE` currently handles the main self-attention path. If cross-attention Q/K also go through QK norm before RoPE, the same fusion pattern applies.

- **Numerical character**: Same as existing fusion (max_diff ~0.125 in BF16).
- **Example**: Check cross-attention forward; if `rms_norm(Q) → rope(Q)` appears, wrap with the same fused kernel.

---

### Attention Score Bias Fusion

**Pattern**: If an additive attention bias (positional bias, learnable bias, sliding-window mask) is applied before softmax, fuse it into the FlashAttention kernel via `attn_bias` argument instead of a separate elementwise add.

- **Numerical character**: Exact same result (fewer kernel dispatches only).
- **Example**: `flash_attn_func(q, k, v, attn_bias=bias_tensor)`.

---

### AdaLN Norm Reuse Across Multiple Modulation Passes

**Pattern**: Each transformer block typically applies modulation (scale/shift) multiple times (e.g., pre-attention, post-attention, pre-FFN). Each recomputes the RMSNorm factor. Fusing all passes into one kernel computes the norm factor once and applies multiple scale/shift pairs.

- **Numerical character**: Bit-exact (same norm factor reused; no accumulation reordering).
- **Example**: Single kernel takes `x` and three `(scale, shift)` pairs; outputs three modulated tensors.

---

### QKV Projection Batching

**Pattern**: Q, K, V are currently three separate linear projections (three GEMM calls, each reading `x` from memory). Concatenating the weight matrices and issuing one GEMM reduces memory reads of `x`.

- **Numerical character**: Identical to separate GEMMs.
- **Example**: `W_qkv = cat([Wq, Wk, Wv], dim=0)`; `qkv = x @ W_qkv.T`; split output.

---

### Compile-Based Fusion for Elementwise Chains

**Pattern**: Elementwise sequences (e.g., `x = x + residual; x = rms_norm(x); x = x * scale + shift`) not covered by existing fusions can be handed to `torch.compile` for automatic kernel fusion via Inductor.

- **Numerical character**: Equivalent semantics; minor FP reordering in reduction ops.
- **Example**: Wrap norm + modulate chains not yet patched with `torch.compile(mode="reduce-overhead")`.

---

### Parallel Execution of Independent Sub-Sequences

**Pattern**: When a block contains two independent computation paths (e.g., processing different token types or modalities), launching them on separate CUDA streams allows overlap when the GPU has spare SM capacity.

- **Numerical character**: Exact (stream ordering does not affect arithmetic for independent GEMMs).
- **Example**: `with torch.cuda.stream(stream_a): out_a = module_a(x_a)` alongside `with torch.cuda.stream(stream_b): out_b = module_b(x_b)`.

---

## 2. Profiling Setup

```bash
# nsys for timeline:
nsys profile --trace cuda,nvtx -o /tmp/profile \
    python -m sglang.multimodal_gen ...

# ncu for kernel-level throughput:
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --target-processes all python ...
```

Key metrics:
- Time per kernel type (GEMM, elementwise, softmax, norm)
- Memory bandwidth vs. compute utilization
- Number of kernel launches per transformer block
