# Sol-Engine on GB200 / H100 (MiniMax-H3, Wan2.2)

One image per architecture serves MiniMax-H3 (1 GPU, GB200 only) and Wan2.2 TI2V-5B / T2V-A14B (1-8 GPUs) behind an
SGLang-compatible HTTP video API (`POST /v1/videos`, `GET /v1/videos/{id}`, `GET /v1/videos/{id}/content`,
`/health`, `/v1/models`, `/model_info`). An unmodified SGLang video client works against it.

| image | arch | GPUs | models |
|---|---|---|---|
| `ghcr.io/lyttttt3333/sol-engine-gb200:0.6.0-gd6d1c426f` | linux/arm64 | GB200 | MiniMax-H3, Wan2.2 5B / A14B |
| `ghcr.io/lyttttt3333/sol-engine-h100:0.6.0-gd6d1c426f` | linux/amd64 | H100 80 GB | Wan2.2 5B / A14B (H3 needs ~121 GB) |

Both are private; ask for read access. Both are built from the same Dockerfile and revision.
The tag names the exact sol-engine revision; the same revision is in the image label
`org.opencontainers.image.revision` and in `GET /model_info` (`git_sha`).

## Pull

```bash
echo <your GitHub classic token with read:packages> | docker login ghcr.io -u <your GitHub user> --password-stdin
docker pull ghcr.io/lyttttt3333/sol-engine-gb200:0.6.0-gd6d1c426f     # or sol-engine-h100 on H100
```

Requirements: NVIDIA Container Toolkit; a driver with CUDA 13.0 support, or a data-center driver >= 535 using
the CUDA forward-compatibility libraries shipped in the image (validated on H100 with driver 535.216).
Weights are not in the image; mount the diffusers checkpoint read-only.

## Run

```bash
# Wan2.2-T2V-A14B on 4 GPUs (defaults: bf16 + fa4sol sparse attention)
docker run --rm --gpus '"device=0,1,2,3"' --ipc=host -p 30000:30000 \
  -v /data/Wan2.2-T2V-A14B-Diffusers:/models/Wan2.2-T2V-A14B-Diffusers:ro -v /data/sol-cache:/cache \
  ghcr.io/lyttttt3333/sol-engine-gb200:0.6.0-gd6d1c426f \
  --model /models/Wan2.2-T2V-A14B-Diffusers --num-gpus 4 --port 30000

# Wan2.2-TI2V-5B on 1 GPU
docker run --rm --gpus '"device=0"' --ipc=host -p 30001:30001 \
  -v /data/Wan2.2-TI2V-5B-Diffusers:/models/Wan2.2-TI2V-5B-Diffusers:ro \
  ghcr.io/lyttttt3333/sol-engine-gb200:0.6.0-gd6d1c426f --model /models/Wan2.2-TI2V-5B-Diffusers --port 30001

# MiniMax-H3 on 1 GPU (the converted diffusers checkpoint; mount the folder its symlinks point to as well)
docker run --rm --gpus '"device=0"' --ipc=host -p 30002:30002 \
  -v /data/MiniMax-H3-diffusers:/models/MiniMax-H3-diffusers:ro -v /data/MiniMax-H3:/data/MiniMax-H3:ro \
  ghcr.io/lyttttt3333/sol-engine-gb200:0.6.0-gd6d1c426f --model /models/MiniMax-H3-diffusers --port 30002
```

The entry point picks the server from the checkpoint (`modular_model_index.json` -> H3, `WanPipeline` -> Wan);
every other option is passed through. `/health` answers after the model is loaded and warmed up
(Wan: two standard-workload requests; H3: one). Mount `/cache` to keep Triton / Inductor / CuTe compile caches
across restarts, `/outputs` to keep the mp4s.

| option | values | default | meaning |
|---|---|---|---|
| `--model` | path | — | diffusers checkpoint directory inside the container |
| `--num-gpus` | 1, 2, 4, 8 | 1 | Wan only. CFG parallel 2 x Ulysses N/2; the server relaunches itself under torchrun |
| `--attention` | `fa4sol`, `sol`, `dense` | `fa4sol` | sparse attention: Sol routing on FlashAttention-4 block-sparse tiles; `sol` = Sol-Attn kernel; `dense` = cuDNN |
| `--rho` | 0-1 | 0.35 | fa4sol: fraction of (256 query x 128 key) blocks computed exactly |
| `--expert-swap` | `auto`, `on`, `off` | `auto` | A14B: keep only the expert in use on the GPU (the idle one in pinned host memory); `auto` = on GPUs under 120 GB, so A14B fits a single 80 GB H100 |
| `--precision` | `bf16`, `fp8` | `bf16` | `fp8` = MXFP8 linears, Blackwell only (lower quality, faster) |
| `--port` | int | 30000 | HTTP port |

8 GPUs = two GB200 nodes in one NVL72 rack: start one container per node with `--network host`, access to the
IMEX channels, and `--num-gpus 8 --nnodes 2 --node-rank {0,1} --dist-init-addr <node0>:<port>`; HTTP is served by node 0.

## Request

```bash
curl -s http://127.0.0.1:30000/v1/videos -H 'Content-Type: application/json' -d '{
  "prompt": "A lone hiker stands atop a towering cliff ...", "size": "1280x720", "num_frames": 81,
  "num_inference_steps": 40, "guidance_scale": 4.0, "guidance_scale_2": 3.0, "flow_shift": 12.0, "fps": 16, "seed": 1024}'
# -> {"id": "video_...", "status": "queued", ...}; poll GET /v1/videos/<id>, then GET /v1/videos/<id>/content
```

Fields left out take the model's standard workload (Wan2.2-5B 1280x704x121, 50 steps, CFG 5, shift 5;
A14B 1280x720x81, 40 steps, CFG 4/3, shift 12; official negative prompt). Text-to-video only.

## Reference numbers (warm, bf16 + sol sparse attention, standard workload, seconds)

Client wall time of the same SGLang-style client for every backend, same node per column.

GB200 (`--attention fa4sol`):

| model | 1 GPU | 2 GPU | 4 GPU | 8 GPU | SGLang 0.5.22 (1/2/4/8) |
|---|---|---|---|---|---|
| Wan2.2-TI2V-5B | 33.9 | 21.5 | 10.3 | 5.8 | 43.5 / 23.6 / 12.5 / 7.5 |
| Wan2.2-T2V-A14B | 304.8* | 138.9 | 74.9 | 37.4 | 371.4 / 190.6 / 102.8 / 50.6 |
| MiniMax-H3 (768p, 5 s) | 111.9 | — | — | — | 131.6 |

\* measured on a slower node; relative to dense on the same node fa4sol is 1.37x.

H100 80 GB (`--attention sol`; dense attention on Hopper uses FlashAttention-4 sm90):

| model | 1 GPU | 2 GPU | 4 GPU | 8 GPU | SGLang 0.5.22 (1/2/4/8) |
|---|---|---|---|---|---|
| Wan2.2-TI2V-5B | 69.2 | 35.1 | 19.1 | 10.5 | 89.7 / 46.6 / 25.1 / 13.0 |
| Wan2.2-T2V-A14B | 552.7 | 277.7 | 157.7 | 78.2 | 775.9† / 405.5† / 210.8† / 103.8 |

† SGLang's default runs out of memory on 80 GB there; measured with `--dit-cpu-offload --text-encoder-cpu-offload`.
Quality (PSNR of the sparse output vs this engine's dense output, same seed): 18–20 dB, at the level of the
numerical-noise floor between two exact attention kernels on this model (19–22 dB).

## Build

`docker/gb200/build.sh` on an arm64 Docker host, or with kaniko where no Docker daemon exists (pass
`--build-arg APT_NO_SANDBOX=1` when the builder runs in a single-uid user namespace). Optional
`APT_MIRROR` / `PIP_INDEX` build args for slow networks.
