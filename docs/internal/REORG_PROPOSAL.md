# Sol-Video Reorg Proposal

Target: reorganize the codebase around **our six models × five techniques**, dropping
other people's models and the SGLang LLM-serving legacy. This is a *move/rename* refactor —
it must not change any numerics. Correctness = post-refactor outputs reproduce the
pre-refactor golden benchmarks bit-for-bit (or within accepted quality-gate tolerance for the
non-bit-exact methods).

## 1. Diagnosis (why reorganize)

The six models are split across **two repos and two runtime paradigms** by historical
accident, not by design:

- **SANA-Video / Cosmos3-Super / LTX-2.3** run through the SGLang `multimodal_gen` diffusion
  runtime (they came from the framework fork `cosmos3/Sol-LTX-Infer`).
- **Wan-5B / Wan-14B / LingBot** run through standalone diffusers drivers in the `autovideo`
  workspace (`runtime/<model>/gpu_infer.py`, bolted on later).

Symptoms: two repos + an empty submodule; byte-identical drivers copy-pasted 3× (a14b≡ti2v-5b,
helios≡vace≡skyreels); the whole SGLang LLM serving stack (~40–55 MB, ~1,500+ py) dragged along
unused; acceleration logic scattered across `efficiency/`, `workflow/`, and each `gpu_infer.py`.

Root cause: **organized by lineage, not by our needs.**

## 2. Target structure

```
sol-video/
├─ models/                    # one THIN adapter per OUR model (only the six)
│   ├─ wan5b/     {pipeline.py, config.toml, prompts/}
│   ├─ wan14b/    {pipeline.py, config.toml, prompts/}
│   ├─ lingbot/   {pipeline.py, lingbot_src/ (vendored + moe shim), config.toml, prompts/}
│   ├─ sana/      {pipeline.py, config.toml, prompts/}
│   ├─ cosmos3/   {pipeline.py, config.toml, prompts/}
│   └─ ltx/       {pipeline.py, config.toml, prompts/}
├─ techniques/                # the 5 reusable methods — SINGLE implementation
│   ├─ cache/      (easycache, teacache, step/payload cache)
│   ├─ sparse/     (pisa)
│   ├─ kernel/     (kwl fusions, fused qkv, compiled glue)
│   ├─ quant/      (nvfp4)
│   ├─ token_prune/
│   └─ _core/      (registry, compose, spec, transform, presets, schedule)
├─ runtime/                   # ONE minimal inference core
│   ├─ harness.py             # warmup + median-over-prompts + benchmark.json + video export
│   └─ diffusion_core/        # extracted from multimodal_gen (SANA/Cosmos/LTX) + sgl-kernel
├─ orchestration/             # = current workflow_lite (master + executors + scopes + gates)
├─ scripts/                   # launch_transfeat, collect_run, create_model_experiment, per-model run_*.sh
├─ transfeat/                # declarative manifests (pruned to live ones)
├─ docs/                      # site_docs + pipelines + techniques + agent-workflow
└─ evals/                     # eval profiles, quality-gate rubrics, golden snapshots
```

## 3. Organizing principles

1. **Models are thin, techniques are shared.** Each model adapter is ~30 lines: pick the
   pipeline class, set default env, declare `techniques = [kernel, cache, pisa]`. Every
   acceleration method has exactly one implementation under `techniques/`. This kills the
   a14b≡ti2v-5b / helios≡vace≡skyreels copies and the scattered per-driver technique code.
2. **One harness, one benchmark schema.** The warmup + median + `benchmark.json` (schema v2)
   + frame/video export logic — copied across 6+ `gpu_infer.py` today — collapses into
   `runtime/harness.py`. Model adapters only supply the pipeline + env.
3. **Keep only the diffusion core; drop the serving stack and other-model zoo.** `multimodal_gen`
   is extracted into `runtime/diffusion_core` with just what SANA/Cosmos/LTX need + `sgl-kernel`.
   SGLang's `srt` server / LLM model zoo / gateway / gRPC never enter. We become *our inference
   library that happens to reuse an SGLang diffusion kernel*, not a fork we must maintain.

## 4. Migration map (current → target)

| Current | Target | Note |
|---|---|---|
| `runtime/wan22_ti2v_5b_baseline/gpu_infer.py` | `models/wan5b/pipeline.py` + `runtime/harness.py` | split shared harness out |
| `runtime/wan22_t2v_a14b_baseline/` (symlink → ti2v-5b) | `models/wan14b/` (config only; shares harness) | already deduped |
| `runtime/lingbot_video_{baseline,optimized}/` | `models/lingbot/` (keep `lingbot_src/` + `sglang_moe_shim.py`) | real baseline/opt divergence stays |
| `models/<m>/model.toml` + `models/<m>/prompts/` | `models/<m>/config.toml` + `models/<m>/prompts/` | merge manifest into adapter dir |
| `efficiency/techniques/{payload_cache,step_cache,teacache,token_prune}.py` | `techniques/{cache,token_prune}/` | one impl per method |
| `efficiency/transforms/{kwl_fusions,nvfp4_ffn,sparse_attention}.py` | `techniques/{kernel,quant,sparse}/` | |
| `efficiency/{registry,compose,spec,transform,presets,schedule,technique}.py` | `techniques/_core/` | shared plumbing |
| framework `python/sglang/multimodal_gen/` (SANA/Cosmos/LTX slice) | `runtime/diffusion_core/` | drop non-shipped dits/pipelines |
| framework `sgl-kernel/` | `runtime/diffusion_core/sgl-kernel/` | KEEP (build dep) |
| framework `scripts/{sana,cosmos,ltx}/*.sh` | `scripts/<model>/run_*.sh` | |
| `workflow_lite/` | `orchestration/` | rename only |
| `workflow/<uid>/nodes/codex_executor/*_scope.md` | `orchestration/scopes/<uid>/` | repoint `techniques.toml` + `spawn_executor` prefix check |
| `scripts/{launch_transfeat,collect_run,create_model_experiment}.py` | `scripts/` | KEEP as-is |
| `transfeat/*.toml` (live set) | `transfeat/` | prune ~24 `*_fused_invariant` sweep residue |
| 7 onboarding runtime dirs (bernini, hunyuan*, cosmos_predict2, helios/vace/skyreels) | `models/_onboarding/` or drop | NOT among the six |
| SGLang `srt` serving / LLM zoo / `test/` / `benchmark/` / `sgl-model-gateway/` / `rust/` / `proto/` / `3rdparty/` | **deleted** | see framework slim table |

## 5. Phased plan (low → high risk)

- **P0 (done):** retire heavy `workflow/` engine, dedup drivers, junk cleanup.
- **P1:** consolidate `efficiency/` + per-driver technique code → `techniques/`; extract
  `runtime/harness.py`; convert Wan/LingBot drivers to thin `models/<m>/pipeline.py`.
- **P2:** rename `workflow_lite/` → `orchestration/`; move scopes; repoint `techniques.toml`.
- **P3:** framework slim — strip SGLang serving stack; extract `runtime/diffusion_core`.
- **P4:** merge the two repos into one `sol-video/`; converge SANA/Cosmos/LTX and Wan/LingBot
  on the same `runtime/harness.py` (diffusers models get a thin pipeline wrapper — not forced
  through mm_gen's scheduler).

## 6. Correctness / reproduction eval (confirm before any code change)

The refactor **moves code, it must not change numerics.** So the acceptance bar is: for the
same transfeat/script + same seed, post-refactor output equals the pre-refactor **golden**
snapshot. Tiered checklist:

### Tier 0 — static (CPU, seconds; run on every commit)
- [ ] `import` smoke: every `models/<m>/pipeline.py`, `techniques/*`, `runtime/harness.py`,
      `orchestration/bin/*` import cleanly (`python -c "import ..."`).
- [ ] `benchmark.json` schema unchanged: `schema_version==2`, same top-level keys and
      `config{}` keys as golden.
- [ ] every live `transfeat/*.toml` still validates: `launch_transfeat.py <c> --mode dry-run`.
- [ ] orchestration intact: `spawn_executor.py ... --no-launch` assembles a prompt and reads
      all 4 scopes.

### Tier 1 — golden capture (BEFORE refactor; one run each)
Snapshot per model × {baseline, optimized}: `benchmark.json` (`total_s`, `denoise_s`), the
output frames (store a per-frame hash + the mp4), and the speedup. Save under
`evals/_golden/<model>/<setting>/`.

### Tier 2 — post-refactor equivalence (GPU; the real proof)
Re-run the SAME unit with the SAME seed through the new structure and assert:
- [ ] **Numerics:** frames **bit-identical** to golden (the refactor changes no math) — hash match.
      (Fallback for any unavoidable nondeterminism: SSIM ≥ 0.999 vs golden.)
- [ ] **Timing:** `total_s` within ±5 % of golden (no perf regression from the harness change).
- [ ] **Artifacts:** `benchmark.json` same schema/keys; `out.mp4` present; speedup reproduced.

### Tier 3 — headline reproduction runs (what demonstrates "改对了")
One baseline+optimized pair per model; the ratio must reproduce the published speedup:

| Model | Baseline unit | Optimized unit | Target |
|---|---|---|---|
| Wan-5B (1 GPU) | `transfeat/wan22_ti2v_5b/baseline.toml` | `transfeat/wan22_ti2v_5b/wan5b_kernel_easycache_pisa.toml` | ~2.885× |
| Wan-14B (1 GPU) | `transfeat/wan22_t2v_a14b/baseline.toml` | `transfeat/wan22_t2v_a14b/singlegpu_opt.toml` | ~2.17× |
| LingBot (4 GPU) | `transfeat/lingbot_video/baseline.toml` | `transfeat/lingbot_video_cudnn_pisa_full.toml` | ~2.60× |
| SANA-Video (1 GPU) | `scripts/sana/run_sana_video_t2v.sh` (baseline env) | same (fullopt env) | ~2.77× |
| Cosmos3-Super (4 GPU) | `scripts/cosmos/slurm_cosmos3_super.sh` (baseline) | same (fullopt) | ~2.27× |
| LTX-2.3 (1 GPU) | `scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh` (baseline) | same (fullopt) | ~2.38× |

Run via `launch_transfeat.py <c> --mode local` (or `--mode sbatch`), then `collect_run.py runs/<id>`.

### Cost note / recommended minimal subset
Full Tier-3 is 12 GPU runs (Wan-14B baseline ~450 s, LingBot 4-GPU, Cosmos 4×B200 — expensive).
Since P1–P2 only touch the Wan/LingBot plane, the **minimal correctness gate** is the three
`/lustre` models (Wan-5B, Wan-14B single-GPU, LingBot) in baseline+optimized = 6 runs; SANA/
Cosmos/LTX get Tier-0 import + one smoke until P3 touches them.

## 7. Rollback
All P0–P2 changes are git-tracked moves → `git restore` / branch revert. P3 (framework slim)
is done on a branch with the Tier-2 equivalence gate before merge.
