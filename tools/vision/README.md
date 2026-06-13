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
  --candidate-frame /abs/candidate.png \
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
