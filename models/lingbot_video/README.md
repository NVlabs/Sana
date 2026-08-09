# LingBot-Video model contract

This contract materializes only the clean CP4+FSDP+batched-CFG FA2 baseline.

Model weights and the GB200 Python environment are external read-only assets.
The experiment materializer also adds the repository's generic search-space
and Symposium support closure.

Prepare the registered controls without submitting GPU work:

```bash
python3 scripts/launch_transfeat.py transfeat/lingbot_video/baseline.toml --mode dry-run
```

Create a clean experiment from the baseline contract:

```bash
python3 scripts/create_model_experiment.py \
  --model lingbot_video \
  --workflow-uid kernel_aw \
  --experiment-uid lingbot-kernel_aw-0001
```
