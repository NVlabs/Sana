# Mechanism Issues — eval/collect pipeline (HunyuanVideo onboarding)

Mechanism issues found during HunyuanVideo diffusers ONBOARD/FREEZE (2026-06-20),
all caused by Cosmos3-specific assumptions baked into the model-agnostic eval
plane. Each broke the authoritative gate for the Hunyuan target. All fixed and
verified by a baseline-vs-baseline self-gate (see bottom). Recorded per
orchestrator request.

Speedup convention (decision): the gate measures **generation time**, EXCLUDING
one-time model load/placement, matching the Cosmos3 `[baseline]` precedent
(`total_s == denoise_s + decode_s`). For the HunyuanVideo diffusers `pipe()` the
single generate call (denoise + VAE decode) is `generate_s = 881.85s`; that is
the baseline `total_s`. The end-to-end wall time (`1007.47s`, incl. ~79s load) is
preserved as evidence only. Including fixed load in the denominator would
systematically under-credit every generation technique (a true 2x gen → 1.78x
wall).

## MI-1 — collect_run nulled the runner's benchmark timings (data loss)
- Symptom: collecting the canonical baseline overwrote `outputs/benchmark.json`
  with `{total_s:null, denoise_s:null, decode_s:null}`, destroying the runner's
  timing/memory evidence.
- Cause: the HunyuanVideo runner (`Hunyuan-Diffusers/hunyuan_diffusers/gpu_infer.py`)
  writes timings NESTED under `"timings"` + `"memory"`. `collect_run.parse_existing_benchmark`
  only read FLAT top-level `total_s/denoise_s/decode_s` and Cosmos-shaped run.log
  regex, found nothing, wrote nulls, and overwrote the file.
- Fix (`scripts/collect_run.py`): `parse_existing_benchmark` now also reads nested
  `timings` (mapping `generate_s` → `total_s`/`denoise_s`, generation convention)
  and `memory`; `build_benchmark` passes the raw nested `timings`+`memory` through
  so a collect pass never discards them.
- Recovery: restored `outputs/benchmark.json` from the in-session capture and
  saved an immutable `outputs/benchmark.runner.json` backup.

## MI-2 — plan_eval base_total was null (no profile [baseline])
- Symptom: `speedup=None` for every candidate.
- Cause: `models/hunyuan_diffusers.toml` had no `[baseline]` table;
  `plan_eval.assess` reads `base_total = profile["baseline"]["total_s"]`.
- Fix: added `[baseline]` (generation `total_s=881.85`, plus `generate_s`,
  `load_s`, `placement_s`, `export_s`, `wall_total_s=1007.47`, peak memory).

## MI-3 — frame extraction used the Cosmos 189-frame profile (mis-aligned)
- Symptom: baseline extracted 188 frames; Hunyuan canonical default is 129
  (ffprobe `nb_frames=129`, 5.375s, 24fps). Mis-aligned LPIPS/Gemini pairing.
- Cause: `DEFAULT_FRAME_COUNT=189` hardcoded; per-timestamp input-seek also
  dropped the trailing frame (189→188).
- Fix (`scripts/collect_run.py`): `--frame-count` now defaults to the run's own
  `run_config.json` `num_frames` (model-agnostic: Hunyuan=129, Cosmos=189), CLI
  override wins, `DEFAULT_FRAME_COUNT` only as last resort. Extraction replaced
  with a single-pass `-vsync 0` passthrough decode that yields exactly the native
  frame count (129), with even subsample only when a smaller cap is requested.
  Baseline and every candidate use the identical policy, keeping pairs aligned.

## MI-4 — Cosmos3DenoisingStage stage label drift
- Symptom: Hunyuan benchmarks labeled `stage_seconds["Cosmos3DenoisingStage"]`.
- Cause: `build_benchmark` injected the Cosmos label for all models.
- Fix: model-aware label — generic `generation`/`decode` for non-Cosmos models;
  `Cosmos3DenoisingStage`/`Cosmos3DecodingStage` preserved only for Cosmos so the
  Cosmos-only `scripts/audit_public_reference_alignment.py` keeps working.

## MI-5 — eval_profile pointed at the Cosmos profile
- Symptom: `models/hunyuan_diffusers.toml` referenced
  `evals/profiles/official_video_t2v.toml` (Cosmos3-Super, frames=189, steps=35,
  seqlen=4096). Currently inert (plan_eval passes the path through; collect_run
  never loads it), but a latent footgun if eval_profile consumption is wired.
- Fix: created `evals/profiles/official_video_t2v_hunyuan.toml` (frames=129,
  steps=50, guidance=6.0, true_cfg=1.0, seqlen=256, 1280x720, 1 GPU,
  `frame_count_expected=129`) and repointed the model profile.

## Verification — baseline-vs-baseline self-gate (129 frames)
`search/plan_eval.py --assess <baseline_run> --baseline-frames <baseline frames>
--model hunyuan_diffusers`:
`baseline_total_s=881.85, candidate_total_s=881.85, speedup=1.0,
gemini_overall=pass, max_artifact_severity=none, lpips_max=0.0,
quality_blockers=[]`. Non-null timing, identity LPIPS, reachable Gemini pass.
