# hunyuan_diffusers_baseline — reconstructed HunyuanVideo runtime (+ TeaCache seam)

The canonical `Hunyuan-Diffusers` submodule (haozhel-local, permission-denied, no
remote) is unobtainable on this machine, so this directory is a from-scratch
reconstruction: run the stock `diffusers` `HunyuanVideoPipeline` and emit the
artifacts the auto-video harness consumes. It also carries the TeaCache
acceleration **seam** (the model-specific glue; the algorithm itself is the
generic controller in `efficiency/techniques/teacache.py`).

## Contents
- `gpu_infer.py` — loads `hunyuanvideo-community/HunyuanVideo` and runs one
  generation at the official config (1280×720, 129f, 50 steps, guidance 6.0,
  true_cfg 1.0, seed 42), writing `out.mp4`, `frames/f_*.png`, `benchmark.json`
  (nested `timings`+`memory`), `run_config.json`.
- `scripts/run_hunyuan_diffusers_gpu.sh` — the `run_script` the launcher executes.
- `step_cache_runtime.py` — **the TeaCache seam.** Feeds HunyuanVideo's
  `time_text_embed` output (`temb`) into the generic `efficiency` TeaCache
  controller and wraps `transformer.forward` so block compute is reused on
  controller-approved steps. No-op unless `SGLANG_HQ_TEACACHE_*` is set.

## Wiring
`config/hunyuan_diffusers_baseline.toml` points here via `[runtime] root =
"runtime/hunyuan_diffusers_baseline"`, so the unobtainable submodule gitlink is
left untouched. Launch:
```
python3 scripts/launch_config.py config/hunyuan_diffusers_baseline.toml --mode sbatch --confirm-submit
```

## Environments
- **Generation**: `ltx23` (profile `PYTHON_BIN`) — diffusers 0.38 + a working
  HunyuanVideoPipeline + imageio/ffmpeg + torch cu130 (verified on the GB200 node).
  (`hunyuanvideo15` is NOT usable here — its transformers can't import `CLIPTextModel`.)
- **Quality gate** (LPIPS): `hunyuanvideo15` (`SANA_PYTHON`) — has `lpips` + weights.

## One-time fixes required (else the baseline run fails fast)
These were needed because ltx23 ships a *nightly* `transformers` (5.8.1) and the
HF repo has a junk root config:
1. **`local_files_only=True`** in `from_pretrained` (in `gpu_infer.py`) —
   `HF_HUB_OFFLINE=1` alone still triggers a `model_info` API call → crash.
2. **`tokenizers==0.22.1`** in ltx23 (was a buggy `0.23.0-rc0` pre-release that
   broke CLIP `RobertaProcessing`). Installed 2026-06-22.
3. **Fixed the malformed root `config.json`** in the HF cache snapshot — the repo
   ships `{"Name":["HunyuanVideo"],}` (illegal trailing comma); transformers
   5.8.1's mistral-patch `json.load`s it and crashes. Rewrote the cached snapshot
   `config.json` to valid JSON (symlink replaced with a real file; blob intact).
   Re-apply this if the model cache is re-downloaded.

## Status / not-yet-done
- **Baseline**: written + dry-run/import-validated; first full GPU run in progress.
  Re-measure timing on THIS env (do not assume haozhel's 881.85 s); re-freeze
  `models/hunyuan_diffusers.toml [baseline]` from the first clean run.
- **Seam (`step_cache_runtime.py`)**: written + CPU-validated (controller engages,
  skips `orig_forward` on reuse steps; OFF == baseline byte-identical). NOT yet
  GPU-validated, no speed/quality claim. First cut is the WHOLE-STEP output cache;
  the block-residual variant (`TeaCacheResidual`) is the planned accuracy
  refinement. The nightly transformers also warns `fix_mistral_regex` on the llama
  tokenizer — benign for normal prompts, but note it if tokenization looks off.
