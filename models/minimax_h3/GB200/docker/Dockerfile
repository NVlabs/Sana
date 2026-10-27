# MiniMax-H3 on one GB200: SGLang-compatible HTTP video server (Sol-Engine single-GPU stack).
# Build from the repository root (see build.sh, which also sets the version tag and git labels):
#   docker build -f models/minimax_h3/GB200/docker/Dockerfile -t sol-engine-h3-gb200:<ver> .
ARG BASE=nvidia/cuda:13.0.2-runtime-ubuntu24.04
FROM ${BASE}

COPY models/minimax_h3/GB200 /src/models/minimax_h3/GB200
COPY techniques/sparse_backends /src/techniques/sparse_backends
RUN bash /src/models/minimax_h3/GB200/docker/install.sh /src /opt/sol-engine && rm -rf /src /root/.cache

ARG GIT_SHA=unknown
ARG VERSION=dev
LABEL org.opencontainers.image.title="sol-engine-h3-gb200" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.description="MiniMax-H3 T2VA on one GB200, SGLang-compatible /v1/videos API"

ENV PATH=/opt/sol-engine/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
    SOL_ENGINE_VERSION=${VERSION} SOL_ENGINE_GIT_SHA=${GIT_SHA} \
    PYTHONPATH=/opt/sol-engine/techniques/sparse_backends:/opt/sol-engine/models/minimax_h3/GB200 \
    H3_SOL_ATTN_ROOT=/opt/sol-engine/techniques/sparse_backends SOL_ATTN_STRICT=1 \
    H3_MODEL_PATH=/models/MiniMax-H3-diffusers H3_OUTPUT_DIR=/outputs H3_SCRATCH=/cache/scratch \
    TRITON_CACHE_DIR=/cache/triton TORCHINDUCTOR_CACHE_DIR=/cache/inductor CUTE_DSL_CACHE_DIR=/cache/cute \
    HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

EXPOSE 30000
VOLUME ["/models", "/outputs", "/cache"]
HEALTHCHECK --start-period=20m --interval=30s CMD curl -sf http://127.0.0.1:${PORT:-30000}/health || exit 1
ENTRYPOINT ["python", "/opt/sol-engine/models/minimax_h3/GB200/serve.py"]
CMD ["--precision", "fp8", "--attention", "sol", "--port", "30000"]
