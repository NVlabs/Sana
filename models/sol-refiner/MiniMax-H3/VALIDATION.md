# Validation scope

## Offline weight fusion

The H3 transformer was exported from the step-835 EMA with the ReFL step-400
adapter fused once. The EMA already contained the official distilled adapter.
All 240 target layers were independently re-merged with PEFT 0.19.1 using FP32
weights and adapters, then cast to BF16. All 240 stored results matched exactly.
The runtime contains zero adapter modules.

The input files were hash-checked before fusion, and all 1,361 output tensors
were checked after serialization. The merged native transformer is 26,246,884,384
bytes, SHA256:

```text
c94faea2df6cec4d4c6fb03b17d271c076438486eb4a370a8af155078c594e39
```

## Native BF16 output comparison

On H100 80GB, the fixed H3 sample used seed **303000**. Input latents, prompt
embeddings and the actual noisy latents were bitwise identical between the
adapter and merged paths. The merged path's output latent had relative L2
`0.03822786` and cosine similarity `0.99916041` against the online-adapter path.
The originally selected 1% relative-L2 check did not pass. This result is recorded
as a low-precision behavioral difference, not bitwise-equivalent inference.

Online adapter branches and premerged BF16 weights have different intermediate
rounding. The independent PEFT weight comparison verifies the merge calculation;
it does not certify visual quality for every input.

## Diffusers conversion

The converted transformer loaded strictly with upstream
`LTX2VideoTransformer3DModel`, using Diffusers commit
`e0abab83b5df05de9e7abd788643c1a7c1e42e28`. Frozen upstream audio parameters were
retained to match that class. Audio/video cross-attention was disabled.

A forward comparison on the same captured inputs, after matching prompt timestep
scaling, yielded relative L2 `0.02430939` and cosine similarity `0.99970460`
against the native merged path. The original 1% check did not pass; this is a
measured cross-backend difference, not a claim of exact parity.

## Release implementation checks

The four CPU contract tests passed: output geometry, exactly one Euler step and
scaled prompt timestep, invalid input handling, and rejection of shifted schedules.
Ruff fatal/error checks and whitespace checks passed.

The actual `SoLRefinerH3Pipeline.denoise_latents` method was run on H100 80GB with
the converted transformer and captured H3 inputs. Output shape was
`[1, 128, 16, 34, 60]`, all values were finite, and runtime LoRA module count was
zero. Relative L2 against the independently exercised Diffusers forward probe was
`0.00335560`, below the test's 1% bound. This tests the released denoising method;
it does not replace the remaining pixel-input/output integration check.

## Remaining release validation

- Package and reload all encoder, text, upsampler and diffusion-decoder components.
- Run the complete MP4 → refiner → MP4 entry point from a clean model package.
- Check video dimensions, frame rate, frame count and visual output.
- Publish the model package and replace local-path examples with its confirmed ID.
- Validate any combined upstream H3 generator example separately.
