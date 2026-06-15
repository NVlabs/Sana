# Search Space: NVFP4 Linear Quantization

**Scope**: Apply NVIDIA FP4 quantization to linear layers in the DiT using the **Transformer Engine FP4 linear backend**. The search space covers which modules to quantize, and how to use dense guards on layers and steps to trade quality against speedup.

---

## Background

**Backend**: Transformer Engine FP4 linear (single backend option)

---

## Which Modules to Quantize

Profile each linear layer to identify which ones show meaningful end-to-end speedup when quantized. Apply FP4 to the subset of modules where the speedup is real, and leave the rest in BF16.

---

## Dense-Guard Layers

Keeping some transformer blocks in BF16 can reduce quality loss. The search space is over how many and which blocks to keep dense.

Example configurations (L = total number of transformer blocks):

| Example | Dense Blocks |
|---|---|
| No guard | — |
| Guard first N | 0 – (N-1) |
| Guard last N | (L-N) – (L-1) |
| Guard both ends | 0 – (N-1), (L-M) – (L-1) |

---

## Dense-Guard Steps

Keeping some denoising steps in BF16 can reduce quality loss. The search space is over how many and which steps to keep dense.

Example configurations (T = total number of denoising steps):

| Example | Dense Steps |
|---|---|
| No guard | — |
| Guard head only | 0 – (N-1) |
| Guard tail only | (T-N) – (T-1) |
| Guard both ends | 0 – (N-1), (T-M) – (T-1) |
