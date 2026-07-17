# LingBot-Video 4-GPU optimized runtime

This directory preserves the optimized LingBot-Video filesystem snapshot from
`~/code/lingbot`. Unlike the baseline runtime, its `runner.py` and
`transformer_lingbot_video.py` include the locally implemented, environment-
gated optimization paths.

The registered winner keeps the same CP4 Ulysses + FSDP + batched-CFG workload
and selects `LINGBOT_ATTN_KERNEL=cudnn`. The packed sequence is split using the
same cumulative sequence boundaries as FA2, then dispatched to the prioritized
cuDNN/Flash/efficient SDPA backends. The original FA2 path remains available
when the switch is off.

The copied provenance under `lingbot_src/agent_opt/` and
`lingbot_src/slurm/RESULTS.md` records the source-reported phase subset:
375.6 s to 207.9 s (1.81x). Recomputing the contiguous load-excluded request
interval from the raw phase log gives 210.07 s, or 1.788x versus the 375.55 s
baseline. Generated videos, Slurm logs, caches, model weights, HunyuanImage-3
experiments, CODA, and the 1.3 GiB third-party vLLM tree were intentionally not
copied.

The 122 GiB model checkpoint and the Conda environment remain external,
read-only prerequisites. The registered entrypoint is
`scripts/run_lingbot_video_gpu.sh`; copied sbatch files are provenance only.
