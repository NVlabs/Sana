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

- Four CPU contract tests cover geometry, one-step execution, invalid inputs and
  rejection of shifted schedules.
- Ruff error/fatal checks and whitespace checks passed.
- The actual `SoLRefinerH3Pipeline.denoise_latents` method ran on H100 80GB with
  the converted checkpoint and captured H3 inputs. It produced finite
  `[1, 128, 16, 34, 60]` output. Relative L2 against an independent Diffusers
  forward probe was `0.00335560`, below that test's 1% bound.

Video encoding, spatial upsampling and final diffusion-VAE decoding are separate
pipeline stages; they do not add Refiner denoising steps. The H100 check covers
the Refiner denoising method, not the complete video I/O pipeline.

## Remaining release validation

- Package and reload all encoder, text, upsampler and diffusion-decoder components.
- Run the complete MP4 → Refiner → MP4 entry point from a clean model package.
- Check video dimensions, frame rate, frame count and visual output.
- Publish the model package and replace local-path examples with its confirmed ID.
- Validate any combined upstream H3 generator example separately.
