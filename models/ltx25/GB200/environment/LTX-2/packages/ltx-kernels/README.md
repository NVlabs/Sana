# ltx-kernels

Custom CUDA/C++ kernels for `ltx-core`. Three compiled extensions:

- **`all2all_cpp`** -- All2All communication kernels for multi-GPU tensor
  parallelism, used by the sequence-parallel inference path.
- **`ops_cpp`** -- Fused element ops for blockwise quantization: `rms_norm_rope`,
  `rms_norm_split_rope`, and FP6 pack/unpack.
- **`blockwise_cpp`** -- Blockwise FP8 GEMM. SM89 (GeForce/Ada) kernel always;
  the SM90 (Hopper, `deep_gemm`) kernel is added when a `9.0` architecture is
  requested.

The Python surface for blockwise quantization lives in
`ltx_kernels.blockwise` (`functional`, `linear`, `triton_ops`).

## Requirements

- CUDA toolkit (nvcc) matching your GPU architecture
- PyTorch with CUDA support
- Linux

## Building

`ltx-kernels` is excluded from the uv workspace, so a plain `uv sync` does not
build it. From the repository root, build it via the opt-in `kernels` group
(editable, no build isolation -- torch must already be installed):

```bash
uv sync --group kernels
```

Equivalently, install it directly:

```bash
uv pip install -e packages/ltx-kernels --no-build-isolation
```

Set `TORCH_CUDA_ARCH_LIST` to target specific architectures (speeds up compilation):

```bash
# H100 only
TORCH_CUDA_ARCH_LIST="9.0" uv pip install -e packages/ltx-kernels --no-build-isolation

# Multiple architectures
TORCH_CUDA_ARCH_LIST="9.0 9.0a 10.0 12.0" uv pip install -e packages/ltx-kernels --no-build-isolation
```

When `TORCH_CUDA_ARCH_LIST` is unset the build targets every supported
architecture (so `uv pip install` "just works" on a dev box); pin it on build
hosts to cut compile time. Any `9.0` entry enables the SM90 GEMM kernel, which
is compiled for `sm_90a` (the deep_gemm kernel uses wgmma/TMA).

### cutlass headers

`blockwise_cpp` includes cute/cutlass headers (header-only; compiled into the
extension, with no runtime dependency). The build fetches them automatically on
first use: a blobless, `include/`-only sparse clone of cutlass pinned to commit
`afa17722` (v3.8.0), cached under `~/.cache/ltx-kernels/` (~25 MB) and reused
across builds.

- Set `CUTLASS_DIR=/path/to/cutlass` to use an existing checkout (uses
  `$CUTLASS_DIR/include` and skips the fetch).
- Set `LTX_KERNELS_CACHE_DIR` to override the cache location.

To bump cutlass, change `CUTLASS_REF` in `setup.py`.

## Testing

Tests require a CUDA GPU:

```bash
uv run pytest packages/ltx-kernels/tests/ -v
```

## Operations

`all2all_cpp`:

- **send_recv_heads** -- Redistributes attention heads across GPUs (All2All)
- **gather_heads** -- Inverse of send_recv_heads
- **allgather** -- Gathers sequence tokens from all ranks

All operations support BFloat16 and Float8 (e4m3fn) data types.
