# Eval And Sub-Loop Design

## What Was Missing

Before M1.6, the repo could launch and collect a baseline, but lacked:

- a canonical eval profile
- promotion/rejection vocabulary
- a visual judge rubric
- a place to record reference snippets from successful branches
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
  references.md
  runs/
  scratch/
```

This lets each Codex goal own a bounded task and acceptance criteria.

## Reference Strategy

Do not rebuild from scratch. For every loop, start from:

- `snippets/sol-ltx-infer-reference.md`
- relevant `Sol-LTX-Infer/scripts/slurm_*.sh`
- relevant `Sol-LTX-Infer/docs/*.md`
- remote branch names recorded in `snippets/README.md`

Then move only the smallest useful snippet into the candidate manifest or goal.
