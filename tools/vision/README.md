# Vision Judge Tools

This directory contains project-level wrappers for visual artifact review.

## NVIDIA-Hosted Gemini Provider

The local Codex skill is:

```text
~/.codex/skills/nvidia-vision-api
```

The relevant non-secret API shape extracted from that skill:

- Base URL: `https://inference-api.nvidia.com/v1`
- Endpoint: OpenAI-compatible `/chat/completions`
- Default model: `gcp/google/gemini-3.5-flash`
- API key env var: `NVIDIA_API_KEY`
- Video mode: sample frames and send image parts by default

Do not write API keys into this repo, reports, shell history, or committed code.

## Dry Run

```bash
python3 tools/vision/nvidia_gemini_judge.py \
  --baseline-frame /abs/baseline.png \
  --config-frame /abs/config.png \
  --out /tmp/visual_judge.json \
  --dry-run
```

## Real Call

```bash
export NVIDIA_API_KEY=...
python3 tools/vision/nvidia_gemini_judge.py \
  --side-by-side-frame /abs/side_by_side.png \
  --out runs/<run-id>/outputs/visual_judge.json
```

The wrapper delegates to the local skill helper when available:

```text
~/.codex/skills/nvidia-vision-api/scripts/nvidia_multimodal_chat.py
```

Use normal sampled video frames for smoke tests. Extremely small placeholder
images can trigger provider-side 500s even when the API key and endpoint are
working.

## LPIPS Judge

`lpips_judge.py` computes a learned perceptual distance between baseline and
config frames. Lower scores are better. The CLI is self-contained and imports
the optional `lpips` and `torch` dependencies only when scoring, so `--help` and
offline collector probes keep working without those packages installed.

Stable frame contract:

```bash
python3 tools/vision/lpips_judge.py \
  --baseline-frame A.png \
  --config-frame B.png \
  --out OUT.json
```

Frame arguments are repeatable and paired by order:

```bash
python3 tools/vision/lpips_judge.py \
  --baseline-frame baseline_0001.png \
  --config-frame config_0001.png \
  --baseline-frame baseline_0002.png \
  --config-frame config_0002.png
```

The tool can also sample a baseline/config video pair before scoring:

```bash
python3 tools/vision/lpips_judge.py \
  --baseline-video baseline.mp4 \
  --config-video config.mp4 \
  --sample-fps 1 \
  --out lpips.json
```

Video sampling uses `ffmpeg` from `PATH`, falling back to
`~/lustre/bin/ffmpeg`. Missing `ffmpeg`, `lpips`, or `torch` produces an
`unavailable` JSON payload and exits 0. Bad arguments, such as unmatched frame
counts or missing input paths, exit nonzero.

Successful output schema:

```json
{
  "metric": "lpips",
  "status": "ok",
  "per_frame": [0.0123],
  "mean": 0.0123,
  "median": 0.0123,
  "max": 0.0123,
  "n": 1,
  "notes": ["lower_is_better", "frames_paired_by_order"]
}
```

## Promotion Gate Notes

`scripts/collect_run.py` now treats visual quality as structured gate data:

- baseline frames are required for promotion;
- frame extraction defaults to the official 189-frame Cosmos3 profile so
  PSNR/MSE/mean absolute diff are audited across the whole sampled output;
- pixel metrics include PSNR/MSE/mean absolute diff, sharpness, temporal delta
  error, temporal jitter ratio, and multi-scale patch-boundary discontinuity
  ratios when image dependencies are available;
- LPIPS is required for promotion, and unavailable dependencies are recorded as
- blocked quality rather than silently deferred. LPIPS receives stratified
  chronological pairs plus worst-case pixel-drift pairs;
- Gemini verdict JSON must be present in `quality.json` or the assessment
  verdict, not only in prose logs. Gemini receives stratified + worst-case
  frame pairs and, when available, baseline/config/side-by-side video inputs
  to catch flicker, patch-level discontinuity, motion breakage, blur, ghosting,
  snow/static, and severe temporal degradation.

Unavailable output schema:

```json
{
  "metric": "lpips",
  "status": "unavailable",
  "reason": "lpips is not importable: ...",
  "n": 0
}
```
