# MiniMax-H3 on one GB200 — Docker server

One container per GPU. It serves MiniMax-H3 text-to-video+audio (T2VA) with the same HTTP API as
`sglang serve --model-type diffusion`, so an SGLang client works unchanged.

## Build

```bash
models/minimax_h3/GB200/docker/build.sh          # -> sol-engine-h3-gb200:<version>-g<sha>
```

The tag carries the git revision; `org.opencontainers.image.revision` holds the full SHA. Weights are
not in the image.

## Weights (once)

The server loads the diffusers-format checkpoint. Convert the official FL2VA partition with the
converter that ships in the image:

```bash
docker run --rm --gpus all -v /data/models:/models --entrypoint python sol-engine-h3-gb200:<tag> \
  /opt/sol-engine/diffusers/scripts/convert_minimax_h3_to_diffusers.py \
  --checkpoint_path /models/MiniMax-H3/FL2VA --output_path /models/MiniMax-H3-diffusers
```

## Run

```bash
docker run -d --gpus '"device=0"' --ipc=host -p 30000:30000 \
  -v /data/models:/models -v /data/h3-cache:/cache -v /data/h3-out:/outputs \
  sol-engine-h3-gb200:<tag>                                  # fp8 + Sol-Attn (default)
# ... --precision bf16 --attention sol    lossless kernels, BF16 GEMMs
# ... --precision bf16 --attention dense  no sparse attention
```

Startup loads the model and runs one warmup request (JIT kernels compile into `/cache`, so later
starts are faster); `/health` answers only after that.

## API (SGLang-compatible)

```bash
curl -s localhost:30000/v1/videos -H 'Content-Type: application/json' -d '{
  "prompt": "...", "seconds": 5, "size": "1344x768", "num_inference_steps": 50, "seed": 0}'
# -> {"id": "video_...", "status": "queued", ...}
curl -s localhost:30000/v1/videos/<id>             # status, progress, inference_time_s, peak_memory_mb
curl -s localhost:30000/v1/videos/<id>/content -o out.mp4
```

Also accepted: SGLang's H3 fields `task: "t2va"` and `target: {short_edge, aspect_ratio, duration_seconds}`,
`width`/`height`, `num_frames`. Frame counts snap up to 17n+5 at 24 fps (5 s -> 124 frames). Only T2VA
is served; conditioned requests (FL2VA/Ref2VA inputs) are rejected with 400.

## Measured on one GB200 (1344x768, 5 s, 50 steps, warm)

| config | request latency |
|---|---:|
| `--precision fp8 --attention sol` | 90.7 s |
| `--precision bf16 --attention sol` | 111.2 s |
| SGLang 0.5.22, BF16 dense (reference) | 131.6 s |
