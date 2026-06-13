# Gemini Visual Artifact Gate

Use this rubric for a multimodal judge such as Gemini when comparing a candidate
video against the official baseline.

## Inputs

- baseline sampled frames
- candidate sampled frames
- optional side-by-side video or side-by-side sampled frames
- candidate manifest
- performance summary

Frame indices must match between baseline and candidate.

## Task

Compare candidate frames against baseline frames. Decide whether the candidate
introduces any new visible artifacts that are absent or materially weaker in the
baseline.

## Artifact Categories

Return a decision for each category:

- `snow_static_speckle`
- `blur_detail_loss`
- `mosaic_blocking_patch_artifacts`
- `banding_posterization`
- `oversaturation_color_shift`
- `ghosting_smearing`
- `melting_morphing_structure`
- `temporal_flicker_popping`
- `degraded_text_faces_hands`
- `composition_or_motion_regression`

## Required JSON Output

```json
{
  "overall": "pass | fail | inconclusive",
  "new_artifacts": [
    {
      "category": "blur_detail_loss",
      "severity": "low | medium | high",
      "frame_indices": [12, 24],
      "evidence": "Short visual evidence."
    }
  ],
  "baseline_notes": "Short description.",
  "candidate_notes": "Short description.",
  "recommendation": "promote | tune | reject | rerun"
}
```

## Passing Rule

`overall=pass` only if no new artifact is medium/high severity. Low-severity
differences may pass only when the candidate has a meaningful speedup and the
artifact is not temporal flicker, mosaic, snow, or major blur.
