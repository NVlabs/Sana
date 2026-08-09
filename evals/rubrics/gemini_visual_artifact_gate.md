# Gemini Visual Artifact Gate

Use this rubric for a multimodal judge such as Gemini when comparing a transfeat
video against the official baseline.

## Inputs

- baseline sampled frames
- transfeat sampled frames
- optional side-by-side video or side-by-side sampled frames
- transfeat manifest
- performance summary

Frame indices must match between baseline and transfeat.

## Task

Compare transfeat frames against baseline frames. Decide whether the transfeat
introduces any new visible artifacts that are absent or materially weaker in the
baseline.

Use the inputs as a temporal sequence, not independent screenshots. Inspect
consecutive frames and any provided video clips for:

- frame-to-frame flicker, shimmer, popping, or unstable lighting/detail;
- patch-level discontinuity, visible grid/block boundaries, local patch texture
  mismatch, or patch boundary popping;
- broken temporal movement, including motion that stutters, melts, smears,
  ghosts, snaps, or becomes inconsistent with the baseline;
- severe degradation, including blur/detail loss, snow/static speckle, mosaic
  blocking, posterization, or corrupted local structure.

If a single frame looks acceptable but the same region changes incoherently
across neighboring frames, classify it as a temporal artifact. If patch
boundaries are only visible during movement, still report
`patch_boundary_discontinuity` and `temporal_flicker_popping`.

## Artifact Categories

Return a decision for each category:

- `snow_static_speckle`
- `blur_detail_loss`
- `mosaic_blocking_patch_artifacts`
- `patch_boundary_discontinuity`
- `banding_posterization`
- `oversaturation_color_shift`
- `ghosting_smearing`
- `melting_morphing_structure`
- `temporal_flicker_popping`
- `loss_of_temporal_coherence`
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
  "temporal_checks": {
    "flicker_or_popping": "pass | fail | uncertain",
    "patch_boundary_stability": "pass | fail | uncertain",
    "motion_coherence": "pass | fail | uncertain",
    "detail_degradation": "pass | fail | uncertain"
  },
  "baseline_notes": "Short description.",
  "transfeat_notes": "Short description.",
  "recommendation": "promote | tune | reject | rerun"
}
```

## Passing Rule

`overall=pass` only if no new artifact is medium/high severity. Low-severity
differences may pass only when the transfeat has a meaningful speedup and the
artifact is not temporal flicker/popping, loss of temporal coherence,
patch-boundary discontinuity, mosaic/blocking, snow/static, ghosting/smearing,
broken motion, or major blur. Treat frame-to-frame shimmer, block boundary
popping, local patch misalignment, and inconsistent patch texture as temporal
artifacts even when a single sampled frame looks acceptable.
