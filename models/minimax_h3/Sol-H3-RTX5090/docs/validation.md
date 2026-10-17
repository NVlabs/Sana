# RTX 5090 validation

## Measured T2VA integration

One full warmup and one formal `mountain-lake` request completed on one RTX
5090 on September 23, 2026, using the shipped example prompt, warmup seed 999
and formal seed 42. The entry used CPU offload and the released 384p/four-update
H3 draft, learned ×2 upscale and three-update LTX refinement recipe.

| Measurement | Result |
| --- | --- |
| Continuous request entry to completed MP4 | 39.796751168 s |
| Video-model initialization plus full warmup | 375.143728772 s |
| H3 formal generation/capture | 11.224092651 s |
| LTX formal stage | 12.000252239 s |
| Whole-run memory sampled every 2 s | Selected GPU maximum 21,583 MiB; unexposed second GPU 0 MiB |
| Fully decoded output | H.264, 1344 × 768, 121 frames, 24 FPS |
| Audio | Original H3 source audio; AAC, stereo, 32 kHz |

The continuous clock includes temporary Qwen loading and closure, CPU/GPU
transfers, serial video stages and final MP4 muxing. Initialization and warmup
are excluded. Summing the two component times does not reproduce E2E.

The H3 formal request recorded four transformer forwards, 200 cuDNN BSA calls
and two text-attention calls. All 200 selected BSA outputs were compared with
the original Triton backend during warmup and passed; maximum relative L2 was
`6.378843681886792e-05`. The formal request ran no reference comparisons.
H3 retained at most one active body block and zero active blocks on return,
including all 300 nonpersistent FP8 weight buffers. LTX completed three
updates and 141 Triton Sol kernel calls. The three dense-layer calls are
inferred from entry into the following layer.

Full CPU media decoding verified all 121 video frames and stereo AAC audio.
The final MP4 has 2,711,675 bytes and SHA256
`c6644e0f374f0be932061f641be97ebbdc046d0b34ba9f99dc621c71ddc244de`.

## Package provenance and local checks

The measured implementation was first validated in a Spark-derived package.
This independent RTX 5090 entry reuses that implementation; packaging
does not constitute a new GPU benchmark. Its CLI and Python API now select CPU
offload by default. Profile names, Qwen placement metadata and the matching
recipe hash describe that default. Source and checkpoint pins are unchanged.

The [sanitized receipt](../validation/rtx5090-t2va.json) distinguishes historical
measured file hashes from the few packaging changes. It retains exact timing
endpoints, observed environment versions, call counts and output verification.
The runtime comparison accounts for entrypoint defaults, docstrings and the
recipe-hash constant; inference operations and numerical settings are unchanged.

The CPU suite passed 66 tests in the independent package. It covers the new
CPU-offload default, serial stages, temporary Qwen lifetime, frozen recipe
and original media/conditioning contracts.
Run it from the package directory:

```bash
python -B -m unittest discover -s tests -v
python -B infer.py --help
python -B prepare.py --plan
```

A separate small GPU check already passed Triton JIT, exact eager and
matching-compile-scope offload parity, FP8 buffer transfer, construction-time
mutation and exception cleanup. To run it explicitly with the Stage1 environment:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python - <<'PYTHON'
import subprocess
from runtime.config import PACKAGE, load_paths
from runtime.pipeline import worker_environment

paths = load_paths("paths-5090.json")
subprocess.run(
    [paths["stage1_python"], str(PACKAGE / "tests/check_offload_gpu.py")],
    cwd=PACKAGE, env=worker_environment("stage1", paths), check=True,
)
PYTHON
```

## Limits

This is one formal request after one full warmup, not a multi-run latency
statistic or a cross-device speedup. Memory sampling may miss peaks between
the two-second samples. Host RAM of 188 GiB is the observed system capacity,
not a tested minimum.

Fresh-environment installation, FL2VA/Ref2VA with CPU offload and perceptual
parity to Spark remain unvalidated. HTTP transport, external queueing and
service cold start were not benchmarked. Generated media and model weights are
not redistributed with this package. The separate Spark entry is unchanged.
