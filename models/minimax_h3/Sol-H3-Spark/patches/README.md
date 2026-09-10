# FastVideo loader compatibility

The Apache-2.0 patch applies to the exact FastVideo revision recorded in
fastvideo-spark-loader.json. It is required by this release's resident,
constructor-merged VSA DataFree LoRA route:

- Move dense additive LoRA deltas to the destination tensor device before
  the original FP32 multiply/add. Shapes, strength and arithmetic are unchanged.
- Use the in-flight concrete transformer for parameter-name mapping during
  lazy constructor loading, avoiding reentrant materialization.
- Do not retain a CPU master copy for inference-only constructor dense LoRA.
  Such a merged model cannot be unmerged; training and other routes retain
  the existing default. A new session is required for a different adapter.

The patch changes only these loader/lifecycle behaviors. It does not change
attention, diffusion schedules, quantization, conditioning, video or audio
decoding. The release capture hook owns normalized-latent extraction; no
external latent-decoding patch is required.

The manifest records pristine and patched SHA-256 hashes for all three files.
Applying the patch to pristine pinned files was verified on CPU to reproduce
the three loader files used by the successful three-request Spark validation
byte for byte. No fresh environment build is implied by that check.
