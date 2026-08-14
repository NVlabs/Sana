# LTX-2.5

Hardware-specific inference implementations live below this directory. They
share prompts and model identity, but each hardware target owns its driver,
runtime snapshot, launch configs, and tuning decisions.

| Hardware | GPUs | Runtime | Delivered stack |
|---|---:|---|---|
| [GB200](GB200/README.md) | 4 | Lightricks native two-stage MGPU pipeline | Stage-1 CFG parallelism + FBCache 0.08 + cache-aware compile; Stage-2 2x2 TDP |

The GB200 delivery vendors its runtime and environment workspace, defaults to
the repository-local `.venv`, and has no code, environment, or Git-metadata
dependency on another checkout beside this repository.

The existing `models/ltx23` implementation is a separate LTX-2.3 SGLang
pipeline and is intentionally not reused here.
