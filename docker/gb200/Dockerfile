# Sol-Engine on GB200: SGLang-compatible HTTP video servers for MiniMax-H3 (1 GPU) and Wan2.2 TI2V-5B /
# T2V-A14B (1-8 GPUs).  One image; the model, GPU count and attention are chosen at `docker run`.
# Build from the repository root (docker/gb200/build.sh sets the git-revision tag and labels):
#   docker build -f docker/gb200/Dockerfile -t sol-engine-gb200:<ver> .
ARG BASE=nvidia/cuda:13.0.2-runtime-ubuntu24.04
FROM ${BASE}

ENV PATH=/opt/sol-engine/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
    SOL_ENGINE_ROOT=/opt/sol-engine \
    PYTHONPATH=/opt/sol-engine/techniques/sparse_backends \
    H3_SOL_ATTN_ROOT=/opt/sol-engine/techniques/sparse_backends SOL_ATTN_STRICT=1 \
    H3_OUTPUT_DIR=/outputs WAN_OUTPUT_DIR=/outputs H3_SCRATCH=/cache/scratch \
    TRITON_CACHE_DIR=/cache/triton TORCHINDUCTOR_CACHE_DIR=/cache/inductor CUTE_DSL_CACHE_DIR=/cache/cute \
    HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# dependencies first, so a code change rebuilds only the cheap layers below.  Optional mirrors for
# networks where the upstream ones are slow (e.g. APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports
# PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple); empty = upstream.
ARG APT_MIRROR=
ARG PIP_INDEX=
COPY docker/gb200/install.sh docker/gb200/requirements.txt /src/
RUN APT_MIRROR=${APT_MIRROR} PIP_INDEX=${PIP_INDEX} bash /src/install.sh /src /opt/sol-engine && rm -rf /src /root/.cache
COPY models/minimax_h3/GB200 /opt/sol-engine/models/minimax_h3/GB200
COPY models/wan22/GB200 /opt/sol-engine/models/wan22/GB200
COPY techniques/sparse_backends /opt/sol-engine/techniques/sparse_backends
COPY docker/gb200 /opt/sol-engine/docker/gb200
RUN python -c "import sys; sys.path[:0] = ['/opt/sol-engine/techniques/sparse_backends', \
    '/opt/sol-engine/models/wan22/GB200']; import wan_fast, sol_fa4, sol_attn; print('sol-engine code imports ok')"

ARG GIT_SHA=unknown
ARG VERSION=dev
ENV SOL_ENGINE_VERSION=${VERSION} SOL_ENGINE_GIT_SHA=${GIT_SHA}
LABEL org.opencontainers.image.title="sol-engine-gb200" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.description="MiniMax-H3 and Wan2.2 on GB200, SGLang-compatible /v1/videos API"

EXPOSE 30000
VOLUME ["/models", "/outputs", "/cache"]
HEALTHCHECK --start-period=30m --interval=30s CMD curl -sf http://127.0.0.1:${PORT:-30000}/health || exit 1
ENTRYPOINT ["python", "/opt/sol-engine/docker/gb200/serve.py"]
CMD ["--help"]
