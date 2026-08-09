# MiniMax-H3 on GB200 — SGLang runtime

The second implementation of MiniMax-H3 on GB200. `../gb200/` runs the vendored
Diffusers PR #14355; this one runs `sglang.multimodal_gen` out of the pinned
container, the same runtime `../h100/` and `../a100/` use.

```bash
python3 scripts/run.py models/minimax_h3/gb200_sglang/dense.toml       # control
python3 scripts/run.py models/minimax_h3/gb200_sglang/aggressive.toml  # full stack
```

## Measured

4x GB200, 1344x768, 124 frames @ 24 fps, 50 steps, seed 0, `t2va_example_1`.
Warmup request excluded; `inference_time_s` from the runtime's own timer.

| Arm | Profile | Time | Speedup | Peak alloc |
| --- | --- | ---: | ---: | ---: |
| `dense.toml` | Sol-Attn off, cache none | **39.896 s** | 1.00x | 40096 MB |
| `aggressive.toml` | Sol-Attn + FirstBlockCache 0.08 | **13.795 s** | **2.89x** | 40779 MB |

Jobs 5996891 and 5997282 on hsg, Sol-Attn backend `cute_sm100` in both.

**These numbers do not compare to `../gb200/`.** That runtime is a different
framework with different scheduling, kernels and memory policy, and its recorded
baseline is a single resident GB200 rather than four with Ulysses-4. Compare
dense-to-dense within a runtime, or profile-to-profile across cards.

## What made the port small

The acceleration in this runtime is algorithm-level — sparse attention and step
reuse — and neither depends on the framework underneath. So `profiles.py` keeps
the H100 `RuntimeProfile` table verbatim and only the hardware row changes:

```python
HARDWARE = HardwareProfile(
    name="gb200", display_name="4x NVIDIA GB200",
    capability=(10, 0), sol_backend="cute_sm100",
)
```

Everything else is the module path in `gpu_infer.py`, the in-container re-exec
path in the shim, and four error strings that named H100.
`test_gb200_sglang_differs_from_h100_only_in_hardware` compares the two profile
tables field by field, so a future edit to one has to be made to both or fail.

## Gates checked before writing it

Each of these could have made the port impossible, so all three were verified on
a GB200 node first:

| Gate | Result |
| --- | --- |
| Does the pinned digest publish `linux/arm64`? GB200 is aarch64 | yes — sglang `0.0.0.dev1+g12eadf86f`, torch 2.11.0+cu130, CUDA available on nvl72150-T03 |
| Does `registration.py`'s sha256 pin still match inside that image? | exact match on `sglang/multimodal_gen/runtime/models/dits/minimax_h3.py` |
| Is there a Sol-Attn kernel for this card? | capability `(10, 0)` maps to `cute_sm100`, which ships |

## Weights

The SGLang path loads the released repository directly, so `H3_MODEL_PATH` is
`MiniMaxAI/MiniMax-H3` — not the `-diffusers` conversion `../gb200/` needs. The
first run downloads ~135 GB into `H3_CACHE_ROOT/huggingface`; later runs reuse
it, which is why job 5996891 took 20 minutes and 5997282 took 7.

Everything the container reads must sit under `H3_STORAGE_ROOT`: the shim mounts
that one directory at `/h3` and refuses any path outside it.

## Not a silent fallback

`profiles.py` sets `SOL_ATTN_STRICT=1` and pins `H3_EXPECTED_SOL_BACKEND` to
`cute_sm100`, and `gpu_infer.py` compares the backend actually selected against
it. A sparse configuration that quietly fell back to Triton would be a dense
measurement wearing a sparse label; here it fails the run instead.
