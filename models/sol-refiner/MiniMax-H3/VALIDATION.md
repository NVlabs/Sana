# One-step inference validation

## Step-count contract

The Refiner uses a single conditional Transformer forward followed by one Euler
update. `set_timesteps(num_inference_steps=1, sigmas=[sigma])` creates the schedule
`[0.9093750119, 0]`. Runtime checks require one timestep, two sigma endpoints and
a terminal sigma of zero. There is no multi-step Refiner loop or CFG branch.

The contract test counts both the Transformer calls and scheduler updates. It
also verifies the scaled prompt timestep and checks that a second invocation
resets the schedule correctly.

## Checks performed

- Five CPU contract tests cover geometry, one-step execution, invalid inputs and
  rejection of shifted schedules, and encoder-only save/reload without missing
  or fabricated decoder parameters.
- Ruff error/fatal checks and whitespace checks passed.
- The actual `SoLRefinerH3Pipeline.denoise_latents` method ran on H100 80GB with
  the converted checkpoint and captured H3 inputs. It produced finite
  `[1, 128, 16, 34, 60]` output. Relative L2 against an independent Diffusers
  forward probe was `0.00335560`, below that test's 1% bound.

Video encoding, spatial upsampling and final diffusion-VAE decoding are separate
pipeline stages; they do not add Refiner denoising steps. The independent-forward comparison covers
the Refiner denoising method. The complete-pipeline checks below exercise video I/O.

## Video checks

A complete local package, including text encoder, connectors, video encoder,
upsampler and diffusion decoder, was reloaded with `from_pretrained` on H100.
A 17-frame MP4-to-MP4 smoke test produced exactly 1920×1080 output with one
Refiner Transformer call.

Three 121-frame reference clips were replayed with identical captured noisy
latents and prompt embeddings, then decoded using the same reference codec and
per-sample decoder seed on both sides:

| Sample | Refiner seed | Latent relative L2 | Video SSIM |
| --- | ---: | ---: | ---: |
| Snow leopard | 303000 | 0.038562 | 0.992212 |
| Potter | 303008 | 0.039995 | 0.992236 |
| Container port | 303018 | 0.072831 | 0.985203 |

Each replay made exactly one Transformer call. These are similarity measurements,
not bitwise equivalence or a universal quality threshold. Reference-codec rendering
used 768-pixel tiles and 256-pixel overlap on both sides; this overlap is below the
native decoder's 384-pixel recommendation and can differ from an untiled reference.

The same three sources also completed the full Diffusers MP4-to-MP4 pipeline,
including fresh prompt encoding, video encoding, upsampling and diffusion decode.
All outputs contain 121 frames at 24 fps and 1920×1080, with one Refiner call.
The full comparison supplies captured reference denoising noise and explicit
decoder seeds (20260826, 20260834, 20260844). Identical seed numbers alone do not
align RNG streams when implementations consume random numbers in different orders.
Full-pipeline video comparisons also include preprocessing, codec, tiling and
lossy MP4 export differences; they are not isolated Transformer comparisons.

## Remaining release validation

- Select and publish the official model repository, then verify public downloads.
- Validate any combined upstream H3 generator example separately.
