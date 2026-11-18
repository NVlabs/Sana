# LTX-2.5 on RTX 5090

Single-GPU Sol-Engine acceleration for the official LTX-2.5 distilled BF16
and NVFP4 pipelines with the Conv VAE.

## Performance

| Workload | Pipeline | Dense Stage 2 | Sol Stage 2 | Stage 2 speedup | Dense E2E | Sol E2E | E2E speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| 4K / 5s | Distilled BF16 | 270.78 s | 130.32 s | **2.08x** | 413.20 s | 273.63 s | **1.51x** |
| 4K / 5s | Distilled NVFP4 | 217.63 s | 72.82 s | **2.99x** | 316.34 s | 171.82 s | **1.84x** |
| 1080p / 20s | Distilled BF16 | 253.48 s | 122.27 s | **2.07x** | 393.42 s | 261.65 s | **1.50x** |
| 1080p / 20s | Distilled NVFP4 | 200.14 s | 66.78 s | **3.00x** | 297.59 s | 164.40 s | **1.81x** |

Sol-Attn is enabled for Stage 2 video self-attention. The three Stage 2
forwards use `tau=1.0`, `1.25`, and `1.5`; layer 0 remains dense.

## Setup

Checkout the official LTX-2 source and create its environment:

```bash
git clone https://github.com/Lightricks/LTX-2.git
git -C LTX-2 checkout fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca
cd LTX-2 && uv sync && cd -
uv pip install --python LTX-2/.venv/bin/python "nvidia-cutlass-dsl>=4.5" cuda-python
LTX-2/.venv/bin/python -m pip install -e techniques/sparse_backends
```

Set the source and model paths:

```bash
export LTX25_LTX_ROOT=/path/to/LTX-2
export LTX25_WEIGHTS_ROOT=/path/to/LTX-2.5
```

## Run

```bash
python3 scripts/run.py models/ltx25/RTX5090/ltx25_rtx5090_distill_bf16.toml
python3 scripts/run.py models/ltx25/RTX5090/ltx25_rtx5090_distill_nvfp4.toml
```

Both configs default to 4K / 5s. Select the 1080p / 20s workload with:

```bash
python3 scripts/run.py models/ltx25/RTX5090/ltx25_rtx5090_distill_nvfp4.toml \
  --set LTX25_WORKLOAD=1080p20s
```

Each run writes `out.mp4` and `benchmark.json` under its output directory.
