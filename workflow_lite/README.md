# workflow_lite — master-agent-orchestrated optimization (lightweight)

A lightweight alternative to the heavy `workflow/` state machines. **The heavy
`workflow/` is untouched;** this is a separate subsystem.

## Model

One **master orchestrator agent** schedules three **executor sub-agents**
(kernel / cache / pisa). Almost no Python: the scheduling logic lives in the
master agent's prompt, not a state machine.

```
run_orchestrated_experiment.py            (only deterministic python)
  ├─ freeze baseline ONCE  -> BASELINE.json (read-only, shared to all sub-agents)
  ├─ launch 1 master agent  (prompts/master.md)
  └─ heartbeat watchdog     (restart master if it dies, until INTEGRATED-DELIVERY.json)

master agent  (prompts/master.md, runs in the coordinator checkout)
  ├─ spawn_executor.py   x3   -> kernel / cache / pisa sub-agents
  ├─ poll_executor.py         -> wait for each DELIVERY.json (each self-runs <=20 rounds)
  ├─ verify_delivery.py       -> INDEPENDENTLY re-run plan_eval + provenance (anti-fabrication)
  ├─ resume_executor.py       -> on bad/fabricated delivery, inject a correction + restart
  └─ integrates itself        -> compose recipes, gate them, write INTEGRATED-DELIVERY.json

executor sub-agent  (prompts/loop_and_gate_contract.md + the technique scope)
  └─ 20-round loop: implement -> launch_candidate -> collect_run -> plan_eval -> retain -> DELIVERY.json
```

## Design decisions (as agreed)

- **Baseline** is measured once at the start, frozen to `BASELINE.json`, and
  referenced by every sub-agent + the master. No one re-runs it.
- **Round limit = 20** per executor: stated in the executor prompt
  (`loop_and_gate_contract.md`) as a hard execution-round budget.
- **No separate reviewer agent.** Independent verification comes from (a) the
  master being a *different* agent, (b) `verify_delivery.py` objectively
  **re-running** the LPIPS + speedup gate (`plan_eval --no-gemini`) on each
  candidate vs the frozen baseline + provenance checks, and (c) the master
  **viewing the candidate vs baseline frames with its OWN built-in multimodal
  vision** (per `evals/rubrics/gemini_visual_artifact_gate.md`). Visual judgment
  uses codex's multimodal ability — **no external NVIDIA/Gemini API** (the old
  API key returned HTTP 401). A format-only check can't catch fabrication;
  re-eval + an independent visual re-view can.
- **Master does the integration itself.**
- **Heartbeat watchdog** restarts the master if it dies.

## Run

```bash
python workflow_lite/run_orchestrated_experiment.py --model bernini --dry-run   # preview
python workflow_lite/run_orchestrated_experiment.py --model bernini             # launch
```

The technique scopes are reused (read-only) from the de-sana'd heavy prompts:
`workflow/{kernel_aw,cache_ca,attention_pa}/nodes/codex_executor/*_scope.md`.

## Caveats (honest)

- Agent-driven orchestration is **less deterministic** than a Python state
  machine; the watchdog + the thin reliable primitives mitigate, but this needs
  a live shakedown.
- `verify_delivery.py` re-runs `plan_eval --no-gemini` (LPIPS + speedup); set
  `PLAN_EVAL_PYTHON` to the eval-env python. Visual quality is judged by the
  agents' own multimodal vision (no external API / no `NVIDIA_API_KEY`). Quality
  is independently re-checkable (LPIPS + the master's own visual re-view); the
  speed claim is provenance-verified (real Slurm run). NOTE: LPIPS on a non-GPU
  coordinator node is slow (minutes) — run the master's verify on a GPU node, or
  accept the per-final-frontier latency.
- Nested agent sessions (master spawning sub-agents via `codex_goal_session`)
  depend on that infra working in the coordinator checkout (tmux + autorun).
