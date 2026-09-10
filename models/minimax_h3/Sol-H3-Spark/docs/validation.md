# Validation status

The release extracts the fixed 384p-to-768p, three-update Spark route. It does
not promote an alternative attention backend, precision or sampler.

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

## Single-Spark integration

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

## Required before release promotion

- Installation from public dependency pins in a clean runtime. Observed
  package-version inventories alone are not a tested environment lockfile.

The integration validation above reused the existing runtime environments.
It does not establish that a fresh container/environment build is equivalent.
