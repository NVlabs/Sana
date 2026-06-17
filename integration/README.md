# Fan-In Integration

This directory is the home for composed delivery-profile planning and artifacts.
Fan-out dimension winners are not final delivery profiles until an integration
loop stacks them, launches a composed run, and re-gates the merged output.

The integration loop must produce one of these outcomes for each risk tier:

- a composed manifest/profile plus benchmark, collection, aligned LPIPS, aligned
  pairwise Gemini, and output video;
- or an explicit tier blocker such as `no_eligible_profile`, merge conflict that
  needs human design input, unavailable GPU/runtime dependency, or exhausted
  interaction search budget.

Required integration close artifacts:

- `INTEGRATION-STATUS.json`
- `INTEGRATION-JOURNAL.md`
- composed low/medium/high manifests or per-tier blockers
- run artifacts for every launched composed profile
- a release matrix that distinguishes per-dimension winners from gated composed
  delivery profiles
