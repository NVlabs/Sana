# Goals

Generated interactive-goal bundles live here.

Use `tools/symposium/prepare_goal.py` to create a bundle:

```bash
python3 tools/symposium/prepare_goal.py \
  --goal-id token-prune \
  --candidate candidates/baseline.toml \
  --objective "Use Symposium to close the candidate into a precise Codex goal."
```

Everything under `goals/` is ignored except this README.
