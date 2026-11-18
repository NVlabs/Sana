# Stage-2 contract: H3-to-LTX-2.5 v2

This document freezes the Stage-2 behavior used by the MiniMax-H3 Super
Acceleration v2 end-to-end run. It distinguishes the algorithmic contract from
two deliberate delivery changes introduced after the earlier
`FINAL_REFINER_HANDOFF` document.

## Contract lineage

The 2026-08-17 handoff remains the algorithmic reference for the LTX-2.5
Refiner. The v2 end-to-end implementation keeps its model components,
conditioning, denoising schedule, attention policy, compile boundary, decoder,
and output-audio policy. It deliberately changes two boundaries:

| Boundary | Earlier frozen handoff | v2 contract validated by job 6304303 |
| --- | --- | --- |
| Stage-2 input transport | Existing H3 MP4, then H.264/AAC decode | Direct BF16 pixels plus FP32 PCM; no intermediate MP4 |
| Input Video VAE temporal tiling | Default tiling policy | One full temporal tile for all 121 consumed frames |

These are real semantic differences, not documentation aliases. Direct tensor
transport avoids lossy H.264/AAC round trips, and the full temporal tile changes
the input encoder's tiling schedule. The successful v2 run validates the new
contract operationally, but it does not prove perceptual or bitwise equivalence
to the earlier MP4/default-tile arm.

The frozen diagnostic source hash was
`69c3f1c1b45386a00eaea71ac03b57aeea1d1a2d8dc8b28bdc7b9e699ef2a591`.
The source executed by the v2 lineage hashes to
`226dfcaab2d3fd90a2f3034a256be2933c51deb157e1c5db964546a45cebbbf9`.
The checked-in integration then adds safe relative first-frame resolution and
SHA validation, producing
`2498594958bf67d5600ff5df6fe709898100b05259f62ea4a14f32681232f660`.
The base Refiner and Sol policy files did not drift. This last hardening step has
not rerun the GPU formal. Full provenance is in
[`SOURCE_SNAPSHOT.json`](SOURCE_SNAPSHOT.json).

## Input and transport

Each Stage-2 service process sees exactly one GB200 and accepts requests from
its paired Stage-1 process over authenticated loopback TCP.

The fixed binary payload is:

| Tensor | Dtype | Shape | Bytes | Meaning |
| --- | --- | --- | ---: | --- |
| Video | BF16 | `[1, 3, 121, 384, 672]` | 187342848 | NCTHW pixels, contiguous, finite, in `[-1, 1]` |
| Audio | FP32 | `[1, 2, 161333]` | 1290664 | Contiguous stereo PCM at 32000 Hz |

Stage 1 obtains 124 decoded 896x512 frames from TAEH3, retains the first 121,
resizes them on CUDA to 672x384, and converts the pixel range for the LTX Video
VAE. Audio is normalized and conformed to the fixed PCM shape. The transport
uses pinned CPU staging buffers and a destination-CUDA-complete acknowledgement.
The control request may proceed only when its pair ID, sequence number, token,
prompt identity, seed, and tensor token all match.

The default production handoff is `direct_tensor`. The MP4 path is a diagnostic
fallback and is not the profile represented by job 6304303 or its latency.

## Frozen Stage-2 algorithm

The service keeps the complete fixed-shape model fleet resident and performs:

1. Encode the direct 672x384/121-frame video with the original LTX-2.5 Video
   VAE input encoder.
2. Apply the official learned x2 latent spatial upsampler, producing the
   1344x768 Stage-2 latent canvas.
3. Encode the pinned 1344x768 opening frame with the original LTX-2.5 Video VAE
   and use it as hard first-frame conditioning.
4. Encode the direct H3 PCM with the original LTX-2.5 Audio VAE and jointly
   denoise video and audio latents.
5. Run exactly three updates with sigma endpoints
   `[0.909375, 0.725, 0.421875, 0.0]`.
6. Decode the final video with the wide LTX TAEHV decoder.
7. Discard the jointly denoised Stage-2 audio output and mux the original H3
   PCM into the final MP4.

The output contract is one H.264 video stream at 1344x768, 121 frames, 24 FPS,
plus one stereo AAC stream at 32 kHz.

## Full-temporal input VAE policy

All 121 input frames fit in one temporal tile:

| Dimension | Tile size | Overlap |
| --- | ---: | ---: |
| Frames | 128 | 24 |
| Height | 768 | 64 |
| Width | 768 | 64 |

The resulting original-VAE latent must be contiguous and have shape
`[1, 128, 16, 24, 42]`. Selecting `default` temporal tiling creates a different
diagnostic arm; it is not permitted when reproducing the v2 benchmark.

## Attention and compile invariants

- The Transformer has 48 layers.
- Video self-attention in layer 0 is dense.
- Video self-attention in layers 1-47 uses strict Sol-Attn with taus
  `1.0, 1.25, 1.5`, diagonal thresholding, and automatic KV splits.
- Cross-attention and audio attention remain dense.
- A complete request records exactly 3 dense layer-0 calls, 141 Sol calls, and
  141 actual `cute_sm100` kernel calls.
- Only the 48 Transformer blocks are compiled with TorchInductor mode
  `max-autotune-no-cudagraphs`, no full graph, no sequence-dynamic shape, and no
  CUDA graph capture. The stateful Sol inner callable remains an eager graph
  boundary.
- The Video VAE encoders, learned upsampler, Audio VAE encoder, TAEHV decoder,
  transport, and media mux remain eager.
- Model offload, quantization, and cache-based denoising reuse are disabled.

Any request with a different visible GPU count, tensor contract, schedule,
attention count, backend, output media shape, or warmup ordering must fail
closed rather than silently run another profile.

## Timing and evidence boundary

Formal job `6304303` ran two independent services and ten hot requests per
service. Model load, compile prime, and one complete warmup were excluded.
Complete end-to-end timing begins immediately before MiniMax-H3 generation and
ends after Stage 2 returns from the final MP4 mux and its in-process validation.
The stricter client-side `ffprobe` is performed after that endpoint and is
therefore correctness evidence, not timed work.

The reported Stage-2 service median, 2.446938 seconds, starts after the direct
payload has been staged to the destination GPU. Payload receive and H2D time are
accounted in the enclosing handoff/Stage-1 side of the end-to-end interval. It
must not be compared directly with a Refiner timing that begins at MP4 decode.

Hardened-integration smoke job `6449281` later completed one hot request on
each independent pair. Its fail-closed aggregation verified `3` dense, `141`
Sol, and `141` actual `cute_sm100` calls per request, plus 1344x768, 121-frame,
24-FPS output media. The two hot end-to-end values were `6.675905103` and
`6.701609667` seconds. This is runtime-contract evidence only; it does not
replace the 20-hot formal result or add a speedup/quality claim.

## Change control

Changing either of the two v2 deltas, any source hash, model/checkpoint identity,
tensor shape, tiling rule, sigma, attention routing, compile boundary, or output
contract creates a new contract version. Update `SOURCE_SNAPSHOT.json`, run a
fresh smoke and formal benchmark, and keep speed and perceptual-quality claims
disabled until matched gates exist.
