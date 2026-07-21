# QUICKSTART — run SANA-Video / Cosmos3-Super / LTX-2.3 (baseline + full-opt)

Human, copy-paste quickstart. (For the agent-oriented, machine-portable rules
see `AGENTS.md` — this file is the short human version.)

**Goal.** Run inference for the three models in both `baseline` and `fullopt`
settings, with:
- **Environment**: create a *brand-new* conda env (do not reuse an existing one).
- **Weights**: reuse local weights if present, otherwise download.
- **Adapt**: the setup/download/run scripts may have machine-specific errors —
  expect to troubleshoot and adapt to your box.
- **Log**: keep a detailed execution log (progress, problems, the exact fixes).

GPU: CUDA-13 class (Blackwell/GB200 tested). Cosmos3-Super needs **4 GPUs**;
SANA and LTX run on **1**.

---

## 1. Set up a fresh environment (one time)

```bash
git clone https://github.com/NVlabs/Sol-Video-Inference-Engine.git Sol-Video-Inference-Engine
cd Sol-Video-Inference-Engine

PYTHON_VERSION=3.12 bash scripts/create_code_conda_env.sh   # -> ./.conda/ltx23 (new env)
conda activate "$PWD/.conda/ltx23"

uv pip install -e "$PWD/python[diffusion]" --prerelease=allow

# CUDA JIT fixups (nvcc/CCCL/dev-symlinks the editable install doesn't pull).
# Add --with-te for the NVFP4 full-opt path (Cosmos/LTX); without it those fall back to BF16.
PYTHON_BIN=.conda/ltx23/bin/python bash scripts/postinstall_cuda_jit.sh   # [--with-te]
```

Sanity check:
```bash
.conda/ltx23/bin/python -c "import torch,diffusers,sglang; print(torch.__version__, diffusers.__version__)"
# expect 2.11.0+cu130 0.38.0
```

## 2. Get the weights (reuse if local, else download)

| Model | HF repo |
|---|---|
| SANA-Video 2B 480p | `Efficient-Large-Model/SANA-Video_2B_480p_diffusers` |
| Cosmos3-Super 64B | `nvidia/Cosmos3-Super` |
| LTX-2.3 | `Lightricks/LTX-2.3` |

If they are already in your HF cache, just point `HF_HOME`/`HF_HUB_CACHE` at it.
Otherwise download (network node only):
```bash
export HF_HOME="$PWD/.hf_cache"        # or your existing cache
huggingface-cli download Efficient-Large-Model/SANA-Video_2B_480p_diffusers
huggingface-cli download nvidia/Cosmos3-Super
huggingface-cli download Lightricks/LTX-2.3
# (or the per-model helpers: scripts/{sana,cosmos,ltx}/*download*.sh)
```

## 3. Run — each model, baseline + full-opt

### SANA-Video (1 GPU)
```bash
PY=.conda/ltx23/bin/python
# baseline (dense)
$PY scripts/sana/sana_video_sglang_run.py --output sana_baseline
# full-opt = EasyCache + fusion + compile  (~2.1x, safe, runs cold anywhere)
$PY scripts/sana/sana_video_sglang_run.py --output sana_fullopt \
    --easycache 0.1 --linattn-bf16 --qkv-merge --compile
# (optional) peak speed ~2.56x: add --max-autotune. First run is slow (warms a
#  persistent inductor cache); every run after is fast.
```

### LTX-2.3 1080p/10s (1 GPU)
```bash
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh baseline   # dense reference
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh fullopt    # ~2.5x, self-contained
```

### Cosmos3-Super 64B (4 GPUs)
```bash
MODEL_REPO=nvidia/Cosmos3-Super ROOT=outputs/cosmos3 \
PROMPT_FILE=prompts/cosmos/robot_plate.json PROMPT_TAG=robot_plate \
bash scripts/cosmos/slurm_cosmos3_super.sh baseline   # or: fullopt  (TeaCache + NVFP4)
```
The entry has `#SBATCH` headers for SLURM (`sbatch ...`); on a single box adjust
the few site paths near the top and run with `bash`.

### Prompts

Prompts are versioned under `prompts/<model>/` (each model uses only its own):
- `prompts/sana/default.txt` — SANA default (read automatically; override with `--prompt` / `--prompt-file`)
- `prompts/ltx/default.txt` + `prompts/ltx/negative.txt` — LTX defaults (override with `PROMPT=` / `NEGATIVE_PROMPT=` or `PROMPT_FILE=` / `NEGATIVE_PROMPT_FILE=`)
- `prompts/cosmos/robot_plate.json` — official structured-JSON prompt; Cosmos has **no default**, you must pass `PROMPT_FILE=...`

## 4. Read the result + keep a log

- Output videos land under `outputs/` (look for the `.mp4` printed as
  `output_file_path`).
- Quote speed from the **`... warmup excluded`** line (cold runs are
  compile-dominated; warmup is on by default).
- Keep an execution log (e.g. `EXECUTION_LOG.md`): what you ran, what broke, and
  the exact workaround — the scripts can need machine-specific fixes.

## Measured reference (480p / 1080p / 720p, warmup-excluded, GB200)

| Model | baseline | full-opt | speedup |
|---|---|---|---|
| SANA-Video 480p | 28.5 s | 13.5 s (`--compile`) / 11.0 s (`--max-autotune`, warm) | ~2.1x / **~2.56x** |
| LTX-2.3 1080p/10s | 95.7 s | 39.2 s | ~2.4x |
| Cosmos3-Super 64B | 97.2 s | 43.1 s | ~2.26x |

## Known gotchas (already handled in the current main branch; see AGENTS.md for detail)
- SANA `--compile` defaults to a **safe** inductor mode; `--max-autotune`
  (peak speed) needs the persistent-cache warm-up — a cold in-process
  max-autotune run deadlocks on a grouped-conv autotune.
- NVFP4 full-opt needs Blackwell + `transformer_engine`; otherwise it auto-falls
  back to BF16 (no crash).
- LTX audio may be silent if the bundled ffmpeg can't AAC-mux — frames are fine.
