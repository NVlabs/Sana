# Runtime integration validation

## Completed

- CPU protocol tests: 5 tests passed (ordering, bounded queue, unchanged
  frame sequence on backpressure, close, one active session, configuration).
- Frontend build: Vite 6.4.3, Node 22.20.0 ARM64, passed.
- Package-local H3 pipeline import: passed in the existing production environment.
- Deployment asset preflight: all configured local assets present.

## Actual browser/H3 loop

Job **7848438**, NVL72 rack `nvl72032`, nodes T01/T04, eight GB200 GPUs,
latest 1008b step224 merged EMA, 50 DiT layers, two NFEs.

- `BROWSER_LOOP_PASS`: actual engine input and 1344x768 JPEG feedback verified.
- 31 generated chunks, 285 decoded output frames, approximately 12 seconds
  of simulated driving. 288 input frames were accepted; 285 were consumed
  by complete render rounds before the producer stopped.
- Every recorded chunk has exactly two denoising evaluations.
- Static prefix prefill occurs during initialization/first continuation;
  recorded steady rounds use cached banks.
- After dropping the first four rounds, 27 measured steady chunks:

| Metric | Median |
| --- | ---: |
| GPU-resident reference RGB to RGB-ready, wall time | 295.58 ms |
| H3 render interval: two NFEs plus cache/sampler work | 238.25 ms |
| Two-latent chunk RGB-ready wall time, 14 samples | 284.68 ms |
| Three-latent chunk RGB-ready wall time, 13 samples | 304.98 ms |
| Actual RGB publication interval in the headless browser setup | 1026.07 ms |

The last row includes engine production/waiting, PNG/CPU transport, H2D,
feedback encoding and other between-chunk work. This test demonstrates
the functional loop, **not 24-fps end-to-end browser performance**.
The shorter GPU interval must not be presented as user-visible feedback latency.

Artifacts are under the local ignored `outputs/` directory:

- `browser-validation/result.json` and `browser.png`
- `3002998e930d42bba46d3c3800163743/controls.jsonl`
- `3002998e930d42bba46d3c3800163743/timings.jsonl`
- `slurm/runtime-7848438.out` / `.err`

Although the browser loop passed, job 7848438 exited nonzero because the
default communication group was destroyed before the unified-parallel
context's exit barrier. Teardown is now moved after that context exits.
Final clean-exit rerun **7848537** completed successfully: Slurm `COMPLETED`,
exit `0:0`, elapsed 5:04, rack `nvl72078`, nodes T08/T17. The actual browser
received 1344x768 feedback across nine render rounds (bootstrap plus eight
continuations), with exactly two NFEs per round. This shorter run validates
startup, input/render/feedback and clean distributed shutdown; the longer
run above remains the timing reference.

Final artifacts: `browser-validation-final/result.json`,
`browser-validation-final/browser.png`,
`c41583cc1afb45f59698d314730db619/controls.jsonl` and `timings.jsonl`, and
`slurm/runtime-7848537.out` / `.err`.
The separate two-rank CPU distributed teardown regression also passed.

The preceding job 7848184 passed H3 warmup but stopped before browser input
because Chromium's Unix-socket path exceeded its path limit. The browser
now uses a short temporary directory for ephemeral metadata; model/compiler
caches remain on code storage.

The new runtime does not overwrite any historical run or weight.
