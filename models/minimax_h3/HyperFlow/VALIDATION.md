# Validation record

## Source-package checks — 2026-09-19

**54 CPU tests passed**: 41 tests in this directory and 13 existing Sol-H3 LoRA
tests. The tests ran with `CUDA_VISIBLE_DEVICES=''`, one CPU thread, no model
weights, and no changes to running inference services.

```bash
pip install -r requirements-test.txt
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests ../Sol-H3/tests/test_lora_branches.py ../Sol-H3/tests/test_lora_fusion_state.py
```

The executed environment used PyTorch 2.10.0+cu130, Diffusers 0.41.0.dev0 at
`c419dac0152186060246c93a095bc1bfaea342b3`, Transformers 5.12.1, PEFT 0.20.0,
Safetensors 0.8.0, Triton 3.6.0, and pytest 9.1.1. See
[cpu-tests.json](validation/cpu-tests.json) for the result and tested source hashes.

Coverage includes:

- Fixed eight-step request contracts, frame presets, invalid inputs, and
  rejection of aliased base-transformer partitions.
- Actual HyperFlow row-time planning for all four conditioning variants;
  synthetic CPU equality between direct AdaLN computation and cached values,
  including distinct endpoint values at the same timestep.
- Synthetic text/image encoder equality when pruning to 51 layers; the selected
  feature remains pre-normalization and the unused output head is removed.
- Mocked resident loading: two distinct DiTs, shared common components,
  initialization staging, and fused/separate mode selection.
- Request-policy switching and Ulysses state reset, reference-resize patch
  compatibility, process-group ownership, and warmup input forwarding.
- Existing Sol-H3 BF16 LoRA rounding, weight preservation, fusion installation,
  and rejection of unsupported adapter states.

Both CLI `--help` commands work without importing CUDA dependencies. All 22
Python files parse successfully; original HyperFlow source hashes match, and
the patch passes whitespace checks. These checks do not execute CUDA kernels.

## Official public release — 2026-09-19

The six vendored modules match
[`Video-Rebirth/hyperflow` at `1dd2f342`](https://github.com/Video-Rebirth/hyperflow/tree/1dd2f342aba5ab51da02b62885939655e8e268da/src/hyperflow_h3).
Compared with the previously tested source snapshot, only the module docstring
in `__init__.py` changed; the executable Python code is unchanged.

The official adapter is now available from
[`videorebirth/hyperflow`](https://huggingface.co/videorebirth/hyperflow).
Its complete-file SHA256 is
`9297f4505bfdef59c3014d11274411809c19b0abfe26161cab2b425a696df447`.
The earlier validation snapshot has complete-file SHA256
`4d7dec1363ebcb9fd63117621b65f8bd19fecacf7ba41f38dd098be363d3972d`.

Full-file streaming checks verified that all **632 tensor names, dtypes, shapes,
offsets, and tensor-payload bytes match**. Their tensor payloads share SHA256
`8759644bc34309d0d322294876598a6be2867968b09c087b5220a03197b3b9a1`.
The public file omits an unused provenance metadata entry; inference metadata
is unchanged. See [public-checkpoint.json](validation/public-checkpoint.json).
This establishes checkpoint equivalence for the historical checks below;
it is not a new GPU benchmark. The README uses the official download repository,
an immutable revision, and the public checksum.

## B300 performance benchmark — 2026-09-19

The [README performance table](README.md#performance) compares the packaged
Sol-Engine runtime from [PR #507](https://github.com/NVlabs/Sana/pull/507), commit
`d0e6fe4745f06ee5761116ebf2c0fc22ec4b031e`, with the original HyperFlow Python
runtime distributed through Drive. The two configurations ran on the same
eight-GPU NVIDIA B300 SXM6 AC host. The base revision is
`MiniMaxAI/MiniMax-H3@83db0c0efe6ef9824e0e194be110346c0a9542ed`; the official
HyperFlow adapter was verified against its published SHA256. Its tensor payload
matches the earlier Drive adapter, as documented above.

Both configurations use resident models, BF16 DiT linears, unmerged LoRA, eight
steps, 1344 x 768 output, 24 FPS, and synchronized audio. T2V and single-image
Ref2VA each cover the 5/10/15-second presets. Every case has two warmups followed
by five measured calls; the table reports medians. The
[machine-readable record](validation/b300-performance.json) preserves all
60 measured latencies, their ranges, and configuration details.

Timing starts at the pipeline call and ends after CUDA completion on all ranks,
including conditioning, denoising, and video/audio VAE decoding. Both return
tensor outputs. Loading, initialization, warmup, prompt rewriting, RGB8/PIL
conversion, MP4 encoding, file saving, and noise-fixture I/O are excluded.

The original runtime uses dense SDPA, native PEFT branches, its original rank-0
conditioning broadcast, and 2048-short-edge reference preprocessing. Its rank-0
conditioner remains resident for the comparison. The Sol-Engine configuration
uses fused LoRA consumers, paired AdaLN tables, conditioner pruning, SOL/BSA,
compressed Ulysses transport, parallel VAE paths, and its default reference
sizing. MXFP8 and reference KV caching are disabled. The original runtime's
duration limit is extended only to accept the existing 362-frame preset.

The actual first DiT inputs verified identical target video/audio noise on all
ranks for all six pairs, preserving the original random draw order. Each
configuration retains its own reference preprocessing. All 84 calls, including
warmups, used eight steps and unmerged LoRA. All 42 Sol-Engine calls matched the
expected fusion counters: 2400 split-linear calls and 400 each for QKV packing,
attention output, SwiGLU, and FFN output. T2V used 336 sparse/64 dense attention
calls; Ref2VA used 400 sparse calls per rank.

This benchmark uses one prompt per task, one seed, and one reference image.
Its speedups include the approximate paths described in the README; it does
not establish output equivalence or a broad perceptual-quality result. I2V,
multiple references, and reference audio/video were not benchmarked here.
The measurements were already complete; adding this record did not rerun GPU
experiments.

## Historical B200 checks — 2026-09-16

[historical-b200.json](validation/historical-b200.json) is a reduced record of
checks on the deployed runtime from which this integration was extracted.
Private paths, input media, prompts, service details, and compile-inclusive
latencies have been removed. **These are historical checks, not a new GPU run
of the reorganized package.**

| Check | Recorded result |
|---|---|
| T2V / I2V / image-reference Ref2VA, each at 5 / 10 / 15 seconds | All nine completed on eight B200 GPUs at eight steps |
| Encoded media | All nine had 1344 x 768 video, 124 / 243 / 362 decoded frames, and an audio stream |
| Native versus fused BF16 LoRA, one 5-second case per task | Eight DiT outputs and decoded video/audio were bitwise equal on all eight ranks |
| Attention-mode switches | Dense T2V, dense Ref2VA, and a subsequent sparse T2V completed |
| Invalid four-step request | Rejected |
| Conditioner pruning | Text and image features were bitwise equal; 14,233,381,376 bytes freed per rank |

The native/fused comparison changes only the LoRA consumer-fusion switch.
Other acceleration settings are held fixed. It does not measure equality to
the unoptimized upstream checkpoint, subjective quality improvement, or the
quality effect of sparse attention, compressed transport, and reference sizing.
It is evidence for the tested inputs and software/hardware configuration, not
a universal bitwise-equality guarantee.

No new GPU inference, warm latency benchmark, PSNR/SSIM sweep, or perceptual
evaluation was run while preparing this source package. Smoke times with first
shape compilation are not used as performance claims. There is no new validation
for one/two/four GPUs, multiple references, or reference audio/video inputs.

## Optional GPU verification

After installing the environment and supplying the matching local model,
adapter, and a reference image, the following command exercises nine generation
cases and compares native/fused LoRA on three additional pairs of 5-second
requests. It loads both DiTs and uses all eight GPUs; run it only when those
resources are available.

```bash
torchrun --standalone --nproc_per_node=8 validate_runtime.py \
  --model "$H3_MODEL" --adapter "$H3_ADAPTER" \
  --reference-image /path/to/reference.png \
  --durations 5 10 15 --compare-native \
  --report outputs/validation.json
```

The verifier records tensor hashes and pipeline times, without encoding MP4.
Parity checks hash every DiT forward and decoded media on every rank, then
reduce equality flags across ranks. Native and fused runs use the same prompt,
seed, conditioning, and attention policy. Functional times can contain kernel
compilation and are not a warmed benchmark. The verifier is provided for future
GPU validation and was not run on GPUs during this packaging task.
