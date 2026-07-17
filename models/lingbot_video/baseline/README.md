# LingBot-Video 4-GPU baseline runtime

This directory is the clean baseline registration for LingBot-Video MoE
30B-A3B plus refiner. The source under `lingbot_src/` was copied from the
working filesystem snapshot, with the two locally modified core files replaced
by their pristine blobs from upstream commit
`a2bb04b78edd848500dc27a26e035a95442ae186`.

The registered baseline is CP4 Ulysses + FSDP + batched CFG with the original
FA2 attention path. `gpu_infer.py` only fixes the workload, starts torchrun, and
normalizes outputs. A timing-only phase-marker patch is applied to the pristine
runner so hot latency can be compared without including model load.

The 122 GiB model checkpoint and the Conda environment remain external,
read-only prerequisites. The source tree contains no Git metadata, model
weights, run outputs, logs, caches, or optimized cuDNN implementation.

For provenance, `lingbot_src/slurm/` retains the original FSDP4 and CP4 scripts.
Those scripts are not the registered entrypoint because their original
environment file hard-coded the old checkout path.
