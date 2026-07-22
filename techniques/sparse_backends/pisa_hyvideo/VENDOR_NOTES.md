# Vendored: Sparse-VideoGen PISA hyvideo kernels

Source: https://github.com/hp-l33/Sparse-VideoGen @ branch `pisa-bidirectional`
Path: `pisa_kernels/kernels/` — fetched 2026-07-18 (private repo, MIT license
headers preserved).

Files (under `pisa_hyvideo_kernels/`):
- `piecewise_sparse_attn_hyvideo.py` — HunyuanVideo bidirectional piecewise
  attention (top-k routing + centroid contribution for non-selected blocks +
  exact text-suffix sink).
- `piecewise_sparse_attn_0th.py` — `chunk_reduce_qkv` / `piecewise_attn_fwd`.
- `sol_attention.py` — GROUP_SIZE / int8 preprocess / global
  threshold (used by the online tau variant; the top-k path uses bf16 directly).
- `utils.py` — density calibration helpers.

**Only modification:** the package was renamed `kernels` -> `pisa_hyvideo_kernels`
(3 import lines in `piecewise_sparse_attn_hyvideo.py`) to avoid a `sys.modules`
collision with the top-level `kernels` packages already vendored under
`sol_attn/` and `sol_attn_colmask/`. No functional changes.

Consumed by `techniques/sparse_backends/sol_attn_hunyuan_v3.py`
(`HUNYUAN_SOL_V3=1`).
