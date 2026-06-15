# Eval And Sub-Loop Design

## What Was Missing

Before M1.6, the repo could launch and collect a baseline, but lacked:

- a canonical eval profile
- promotion/rejection vocabulary
- a visual judge rubric
- a canonical search-space contract for open-ended agent exploration
- a mature folder shape for independent sub-loops

## Eval Model

Every candidate should point to an eval profile, initially:

```toml
eval_profile = "evals/profiles/official_video_t2v.toml"
```

Promotion requires:

```text
artifact pass
official_config pass
performance pass
off_identity pass or not_applicable
quantitative_quality pass or explicitly_deferred
visual_artifact pass
```

The visual gate is provider-neutral but currently wired for NVIDIA-hosted Gemini
multimodal review:

```text
evals/rubrics/gemini_visual_artifact_gate.md
tools/vision/nvidia_gemini_judge.py
```

The local skill source is `~/.codex/skills/nvidia-vision-api`; this repo records
only the non-secret API shape. API keys stay in environment variables.

## Sub-Loop Model

Each acceleration family gets an independent folder:

```text
loops/<loop-id>/
  README.md
  goal.md
  acceptance.md
  candidate.toml
  eval.toml
  runs/
  scratch/
```

This lets each Codex goal own a bounded task and acceptance criteria.

## Search-Space Strategy

For every implementation loop, start from:

- `search_space/` for method families and broad axes;
- `loops/<dim>/exploration.md` for the natural-language dimension brief;
- the live Cosmos3 inference code under `Sol-LTX-Infer/`.

Do not add per-dimension reference archives. Subagents should discover
model-specific layer, step, signal, routing, and fallback choices from code,
traces, and reproduction artifacts.
