# orchestration — master-agent-orchestrated optimization (lightweight)

A lightweight alternative to the heavy `workflow/` state machines. It reuses
workflow-owned technique scopes but has its own orchestration runner.

## Model

One **master orchestrator agent** schedules the model-selected executor
sub-agents. The registry supports kernel / cache / pisa / topology; the global
default remains the original first three, while multi-rank models such as LingBot
may opt into topology by default in their model profile. Almost no Python: the
scheduling logic lives in the master agent's prompt, not a state machine.

```
run_orchestrated_experiment.py            (only deterministic python)
  ├─ freeze baseline ONCE  -> BASELINE.json (read-only, shared to all sub-agents)
  ├─ launch 1 master agent  (prompts/master.md)
  └─ heartbeat watchdog     (restart master if it dies, until INTEGRATED-DELIVERY.json)

master agent  (prompts/master.md, runs in the coordinator checkout)
  ├─ spawn_executor.py   xN   -> model-selected technique sub-agents
  ├─ poll_executor.py         -> wait for each DELIVERY.json (scope-owned round budget)
  ├─ verify_delivery.py       -> INDEPENDENT performance/provenance/correctness gate
  ├─ resume_executor.py       -> on bad/fabricated delivery, inject a correction + restart
  └─ integrates itself        -> compose recipes, gate them, write INTEGRATED-DELIVERY.json

executor sub-agent  (prompts/loop_and_gate_contract.md + the technique scope)
  └─ bounded loop: implement -> launch_config -> collect_run -> mode-specific gate -> DELIVERY.json
```

## Design decisions (as agreed)

- **Baseline** is measured once at the start, frozen to `BASELINE.json`, and
  referenced by every sub-agent + the master. No one re-runs it.
- **Round limits are technique-owned.** The shared loop contract requires a hard
  budget; each workflow scope supplies its exact number.
- **No separate reviewer agent.** Independent verification comes from (a) the
  master being a *different* agent, (b) `verify_delivery.py` recomputing
  performance from the frozen baseline plus durable benchmark and checking
  provenance, and (c) a mode-specific correctness gate. Lossy cache/PISA points
  re-run LPIPS and receive the master's own multimodal visual-quality review.
  Lossless kernel/topology points never compare outputs; they receive structural,
  method, and (for topology) distributed-evidence audits, with frames used only
  to establish run authenticity. No external NVIDIA/Gemini API is used.
- **Master does the integration itself.**
- **Heartbeat watchdog** restarts the master if it dies.

## Run

```bash
python orchestration/run_orchestrated_experiment.py --model bernini --dry-run   # preview
python orchestration/run_orchestrated_experiment.py --model bernini             # launch
python orchestration/run_orchestrated_experiment.py --model lingbot_video --techs topology --dry-run
```

The technique registry is `orchestration/techniques.toml`. Each registered scope
is owned by its workflow directory:
`workflow/{kernel_aw,cache_ca,attention_pa,topology_ta}/nodes/codex_executor/*_scope.md`.
The topology scope is lossless and owns CP/SP/TP/EP/FSDP, process groups,
collectives, placement, and distributed scheduling; local kernels remain owned
by the kernel scope. A model enables it by default with
`[orchestration].default_techniques`; users may always select it explicitly with
`--techs topology`. A real topology run requires a frozen baseline with at least
two active ranks.

## Caveats (honest)

- Agent-driven orchestration is **less deterministic** than a Python state
  machine; the watchdog + the thin reliable primitives mitigate, but this needs
  a live shakedown.
- For lossy techniques, `verify_delivery.py` re-runs
  `plan_eval --no-gemini`; set `PLAN_EVAL_PYTHON` to the eval-env Python. Visual
  quality is judged by the agents' own multimodal vision. For lossless techniques,
  verification reads no LPIPS/output-difference metric and instead recomputes
  speed from the frozen baseline plus run benchmark. LPIPS on a non-GPU
  coordinator node can still be slow for lossy frontiers.
- Nested agent sessions (master spawning sub-agents via `codex_goal_session`)
  depend on that infra working in the coordinator checkout (tmux + autorun).
