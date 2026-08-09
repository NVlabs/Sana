# Agent workflow

Sol-Engine is *agent-native*: a coding agent (Claude Code / OpenAI Codex) drives model onboarding and optimization, not just fixed scripts. This page documents the orchestration behind the "hand an agent a goal" quick start.

## Orchestration modes

**Lightweight (master + executors).** A single master orchestrator supervises up to three executor sub-agents, one per acceleration technique (`kernel`, `cache`, `sparse`/PISA). Each executor explores its technique's search space (detached, watchdog-guarded), returns validated transfeat, and the master gates, dedupes, and integrates them. Preferred for a fast, bounded optimization sweep on a new model.

**Full framework.** Per-technique executor nodes with richer fan-out for deep multi-round searches.

## Transfeat contract

Every optimization is a transfeat manifest: an id, kind (baseline / patch / control), the runtime it uses, the environment flags that turn the technique on, and its eval profile. A single launcher renders a reproducible run bundle and submits it. Baselines and same-topology controls are first-class, so every speedup has an explicit, stated denominator (single-GPU / same-topology / vs-naive).

## Quality gates

No speedup is accepted on wall-clock alone:

- **LPIPS** against frozen baseline frames.
- **Multimodal visual gate** — a hosted VLM reviews baseline-vs-transfeat side-by-side against a rubric (snow/speckle, blur, mosaic / patch-boundary, banding, ghosting, melting, temporal flicker, coherence & motion regressions).
- **Authenticity check** — confirms the technique actually engaged (PISA wrote its stats, cache actually reused steps) so a run cannot report a "fake" speedup from a no-op optimization.

## Conventions

- Report speedups against a stated baseline; never mix denominators silently.
- Timing excludes model load; medians over official validation prompts; same seed.
- Optimizations that change floating-point reduction or attention sparsity are marked non–bit-exact.
