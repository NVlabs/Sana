# Validation status

The default T2VA release extracts the fixed 384p-to-768p, three-update Spark
route. Its image/reference task extensions retain the same Stage2 recipe;
Ref2VA necessarily selects its own model partition, adapter and dense draft
attention instead of borrowing the T2VA VSA checkpoint.

## Completed source and CPU checks

- The public joint audio/video state preparation, denoising and Conv streaming
  methods match the reference methods structurally. The Sol attention policy
  is reused without changing its thresholds or layer selection.
- CPU contracts exercise four-update scheduling, VSA selection and reference
  counters, same-request latent ownership, prompt-cache shapes/precision,
  native layout restoration, singleton copies and owned-model cleanup.
- Orchestration tests verify warmup exclusion, worker reuse and continuous
  MP4 endpoint timing. CPU substitutes do not validate video quality or CUDA.
- CLI help and test imports do not initialize GPU frameworks.

## Single-Spark T2VA integration

The release entry completed one full warmup and three consecutive formal
requests, retaining the same Stage1, Stage2 and Qwen model sessions. Each
request produced a final video with audio. Validation confirms:

- The offline builder's video and audio BF16 contexts are bitwise equal to
  the fixed-prompt reference cache, with no video models loaded during cache
  generation.
- The three Stage1 H3 latent and PCM payloads match the reference bytes for
  the same prompts and seeds.
- Stage1 executes four updates, 200 BSA calls and 200 fused VSA merges, with
  no attention fallback. The transformer materializes once.
- Each Stage2 request executes three updates and 141 Triton Sol calls, with
  the original tiles, 16 direct-NHWC operations and 121 output frames.
- Gemma and the connector are absent from online inference.

## Image/reference extensions

The 52 CPU checks cover task/input selection, ordered references, native
conditioning timestep plans, generated-versus-reference latent rows and
task-aware model paths. CPU substitutes do not establish CUDA execution or
image fidelity.

FL2VA completed one full warmup followed by three consecutive requests on a
single DGX Spark: first-frame only, last-frame only and both endpoints. All
three produced a final video with audio using the same resident model
sessions. Native input conditioning reached Stage1; capture excluded the
reference-video prefix and retained only generated H3 latents and original
PCM. No intermediate H3 video decode was used and no offload or encoder
precision change was added. This is a functional integration check on one
source scene, not a broad quality benchmark or a memory guarantee for every
input.

Ref2VA GPU inference, real checkpoint loading and memory fit remain unvalidated.
Its task-specific checkpoints must be available before this mode can be
promoted as release-validated.

The native FL2VA first/last inputs guide generated latent tokens; neither H3
nor the unchanged three-step Stage2 path promises pixel-exact endpoint
replacement. The final timeline remains the first 121 frames of the converted
124-frame draft. Ref2VA uses the dedicated LightX2V four-step adapter and dense
FA4 draft attention, not VSA. Input encoders and reference activations use
additional memory that is not covered by the text-only residency result.

## Required before release promotion

- Installation from public dependency pins in a clean runtime. Observed
  package-version inventories alone are not a tested environment lockfile.

The integration validation above reused the existing runtime environments.
It does not establish that a fresh container/environment build is equivalent.
