# Open-Ended Exploration: Token Pruning

Start from `search_space/02_token_pruning.md`, then inspect the Cosmos3
inference code directly. If `search_space/` is missing, stop and ask the main
agent to repair the search-space contract.

Explore token pruning as a model-specific token-layout and salience problem:

- Inspect Cosmos3 token layout, sequence parallelism, RoPE/frequency alignment,
  and cross-attention sensitivity.
- Derive prunable spans, scoring signals, compensation policy, layer windows, and
  step windows from traces/code.
- Do not use predefined keep ratios or step windows. Discover them from Cosmos3
  code, traces, and artifacts.
- Prefer conservative probes that identify the quality cliff before aggressive
  speed targets.

Required output:

- Candidate manifest or structured negative result.
- Evidence that gathered tokens, positional tensors, and scatter restoration stay
  aligned.
- OFF identity and structured quality-gate status.
