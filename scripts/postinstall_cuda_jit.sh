#!/usr/bin/env bash
# Post-install fixups so the runtime CUDA JIT (tvm-ffi / sgl_kernel) can compile
# and link on a fresh env. `pip install -e python[diffusion]` only pulls the
# *runtime* CUDA libs (torch's deps); the JIT additionally needs the compiler
# toolchain + CCCL headers + dev symlinks, none of which are hard deps of any
# wheel. Run this once after the editable install, inside the activated env.
#
# Usage:  bash scripts/postinstall_cuda_jit.sh [--with-te]
#   --with-te : also install NVIDIA transformer_engine (needed for the NVFP4
#               'fullopt' path of Cosmos3-Super and LTX-2.3; Blackwell-only at run).
#
# Idempotent. Pins match the known-good aarch64 / CUDA-13 resolution; override
# any version via the env vars below if your stack differs.
set -euo pipefail

PY="${PYTHON_BIN:-python}"
NVCC_VER="${NVCC_VER:-13.2.78}"          # nvcc/crt/nvvm — JIT "nvcc: not found"
CCCL_VER="${CCCL_VER:-13.2.75}"           # cuda-cccl headers must match the nvcc minor series
CUBLAS_VER="${CUBLAS_VER:-13.2.2.2}"      # TE needs cublasLtGroupedMatrixLayoutInit_internal
TE_VER="${TE_VER:-2.16.0}"
WITH_TE=0
[[ "${1:-}" == "--with-te" ]] && WITH_TE=1

echo "[postinstall] python: $("$PY" -c 'import sys;print(sys.executable)')"

# 1. CUDA compiler toolchain (JIT compile): nvcc + crt + nvvm
echo "[postinstall] installing CUDA compiler toolchain (nvcc/crt/nvvm $NVCC_VER)"
"$PY" -m pip install --no-deps \
  "nvidia-cuda-nvcc==${NVCC_VER}" "nvidia-cuda-crt==${NVCC_VER}" "nvidia-nvvm==${NVCC_VER}"

# 2. CCCL headers (JIT needs <nv/target>, <cuda/...>, cub, thrust)
echo "[postinstall] installing CCCL headers (nvidia-cuda-cccl)"
if [[ -n "$CCCL_VER" ]]; then
  "$PY" -m pip install --no-deps "nvidia-cuda-cccl==${CCCL_VER}"
else
  "$PY" -m pip install --no-deps nvidia-cuda-cccl
fi

# 3. cublas with the symbol TransformerEngine links against (NVFP4 path)
echo "[postinstall] aligning nvidia-cublas to $CUBLAS_VER (TE / NVFP4)"
"$PY" -m pip install --no-deps "nvidia-cublas==${CUBLAS_VER}" || \
  echo "[postinstall] WARN: could not pin cublas==$CUBLAS_VER (only needed for NVFP4 fullopt)"

# 4. JIT linker expects lib64/ + unversioned dev symlinks (pip wheels ship only
#    versioned libs in lib/), else: `ld: cannot find -lcudart`.
NV_DIR="$("$PY" -c 'import importlib.util,os; s=importlib.util.find_spec("nvidia"); print(os.path.dirname(s.submodule_search_locations[0]) if s else "")')"
NV_DIR="${NV_DIR}/nvidia"
if [[ -d "$NV_DIR" ]]; then
  echo "[postinstall] creating lib64/ + unversioned .so dev symlinks under $NV_DIR"
  for libdir in "$NV_DIR"/*/lib; do
    [[ -d "$libdir" ]] || continue
    [[ -e "$(dirname "$libdir")/lib64" ]] || ln -s lib "$(dirname "$libdir")/lib64"
    for so in "$libdir"/lib*.so.*; do
      [[ -e "$so" ]] || continue
      base="$(basename "$so")"; unver="${base%%.so.*}.so"
      [[ -e "$libdir/$unver" ]] || ln -s "$base" "$libdir/$unver"
    done
  done

  # Make CUDA wheel headers/libs visible to source builds. TransformerEngine's
  # torch extension includes cudnn.h directly, which is not under CUDA_HOME.
  CUDA_HOME="${CUDA_HOME:-$NV_DIR/cu13}"
  CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
  export CUDA_HOME CUDACXX
  export PATH="$CUDA_HOME/bin:$PATH"
  include_paths="$(find "$NV_DIR" -maxdepth 3 -type d -name include | paste -sd: -)"
  lib_paths="$(find "$NV_DIR" -maxdepth 3 -type d \( -name lib -o -name lib64 \) | paste -sd: -)"
  export CPATH="${include_paths}:${CPATH:-}"
  export C_INCLUDE_PATH="${include_paths}:${C_INCLUDE_PATH:-}"
  export CPLUS_INCLUDE_PATH="${include_paths}:${CPLUS_INCLUDE_PATH:-}"
  export LIBRARY_PATH="${lib_paths}:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="${lib_paths}:${LD_LIBRARY_PATH:-}"
else
  echo "[postinstall] WARN: nvidia/ site-packages dir not found; skipped symlink step"
fi

# 5. transformer_engine (optional; NVFP4 fullopt). No aarch64 prebuilt wheel as of
#    cu13 — may build from source (needs nvcc + cmake) or be copied from a known-good env.
if [[ "$WITH_TE" == "1" ]]; then
  if [[ -z "${TMPDIR:-}" || "$TMPDIR" == /tmp* ]]; then
    export TMPDIR="$PWD/.tmp"
    export TMP="$TMPDIR"
    export TEMP="$TMPDIR"
  fi
  mkdir -p "$TMPDIR"

  echo "[postinstall] installing transformer_engine $TE_VER (source build may take a while)"
  "$PY" -m pip install --no-deps \
    "nvidia-cublas==${CUBLAS_VER}" \
    "transformer_engine==${TE_VER}" \
    "transformer_engine_cu13==${TE_VER}" \
    "nvdlfw-inspect==0.2.2" \
    "onnx==1.22.0" \
    "onnx_ir==0.2.1" \
    "onnxscript==0.7.0" && \
  "$PY" -m pip install --no-build-isolation --no-deps "transformer_engine_torch==${TE_VER}" && \
  "$PY" -m pip install --no-deps "nvidia-cublas==${CUBLAS_VER}" || \
    echo "[postinstall] WARN: TE install failed — NVFP4 fullopt will gracefully fall back to BF16"
fi

echo "[postinstall] done. Verify JIT: run any model's baseline once on a GPU node."
