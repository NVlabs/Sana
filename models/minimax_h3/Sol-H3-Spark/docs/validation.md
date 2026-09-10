# Validation status

The default T2VA release extracts the fixed 384p-to-768p, three-update Spark
route. Its image/reference task extensions retain the same Stage2 recipe;
Ref2VA selects its own model partition and adapter instead of borrowing the
T2VA VSA checkpoint. Its default is Stage1-sized reference images and dense
FA4 draft attention; the supported alternatives are the Stage2-sized image
budget and Stage1 Sol. See the [Ref2VA interface](../README.md#reference-generation-ref2va).

## Completed source and CPU checks

- The public joint audio/video state preparation, denoising and Conv streaming
  methods match the reference methods structurally. The Stage2 Sol attention
  policy is reused without changing its thresholds or layer selection.
- CPU contracts exercise four-update scheduling, VSA selection and reference
  counters, same-request latent ownership, prompt-cache shapes/precision,
  native layout restoration, singleton copies and owned-model cleanup.
- Orchestration tests verify warmup exclusion, worker reuse and continuous
  MP4 endpoint timing. Warmup covers the first case's shape, not every later
  reference layout. The permanent Qwen worker's first encoding is included
  in the first formal request; that request is not assumed fully warm.
  CPU substitutes do not validate video quality or CUDA.
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

CPU checks cover task/input selection, ordered references, native
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

Ref2VA first completed one full warmup followed by two consecutive requests
on a single DGX Spark: one image reference, then the same image plus an audio
reference. These initial dense-FA4 checks used the earlier native image size,
before the bounded reference-budget interface. Both produced final video
with audio while the model sessions remained resident. Real execution confirmed:

- The dedicated Ref2VA checkpoint and rank-128, alpha-8 four-step LoRA load;
  all 312 adapter projections merge before W8A8 FP8 conversion.
- Each draft executes four updates and 202 BF16 FA4 calls with no fallback.
- Capture excludes 7,168 image-reference rows in both requests and 406
  audio-reference rows in the image-plus-audio request. Only generated H3
  latents and generated original H3 audio pass to the unchanged Stage2.
- H3 video decoding remains bypassed; Stage2 completes its three-update
  Sol refinement and official Conv decode, producing 1344 × 768, 121-frame
  H.264 video at 24 FPS with stereo AAC audio.

Subsequent full-pipeline executions also covered the following image-budget
and Stage1-attention combinations. The single-image cases use one source
scene; the three-image cases use two distinct sets of character and scene
references. Every listed request produced its final video with audio.

| Reference inputs | Reference area budget | Stage1 attention exercised |
| --- | --- | --- |
| One image; the same image plus audio | Stage1 (672 × 384) | Sol |
| One image; the same image plus audio | Stage2 (1344 × 768) | Sol |
| Two cases with three images each | Stage1 (672 × 384) | Dense FA4 and Sol |
| The same two three-image cases | Stage2 (1344 × 768) | Sol |

The dimensions above describe **per-image area budgets**, not a forced image
shape. Qwen and native H3 receipts agree on each prepared image and its order.
Images retain their own aspect ratios up to nearest-32 alignment, without
cropping; alignment may slightly exceed the nominal budget.

Dense drafts execute four updates and 202 FA4 calls. The Stage1 Sol route
executes 147 Sol calls, 55 full-layer dense calls and 147 dense text/audio-query
calls. The first update and layer 0 of later updates stay dense, with Sol
thresholds 1 / 1.25 / 1.5. Only text/audio are forced sinks; reference-image
and Qwen-visual tokens are not. The sink range uses the existing 64-token
block granularity. Stage2 remains unchanged at three updates and 141 Triton
Sol calls, with the original Conv decoder and output/audio contract.

These runs validate the underlying execution paths and selected inputs, not
every combination exposed by the new CLI/Python selectors. In particular,
the new selectable release entry has not been exhaustively GPU-tested, and
Stage2-sized references with dense Stage1 attention are not covered by the
table. Video references and arbitrary mixtures/counts remain GPU-unvalidated.
No identity, accessory or voice fidelity guarantee follows from a successful
run, and reducing reference detail is not established as quality-preserving.

The native FL2VA first/last inputs guide generated latent tokens; neither H3
nor the unchanged three-step Stage2 path promises pixel-exact endpoint
replacement. The final timeline remains the first 121 frames of the converted
124-frame draft. Ref2VA uses the dedicated LightX2V four-step adapter with
dense FA4 or the supported Sol draft policy, not VSA. Both Ref2VA selectors
apply to the whole batch and leave Stage2 unchanged. Input encoders and
reference activations use additional memory; fit for untested inputs cannot
be inferred from the text-only residency result or the bounded cases above.

## Required before release promotion

- Installation from public dependency pins in a clean runtime. Observed
  package-version inventories alone are not a tested environment lockfile.

The integration validation above reused the existing runtime environments.
It does not establish that a fresh container/environment build is equivalent.
