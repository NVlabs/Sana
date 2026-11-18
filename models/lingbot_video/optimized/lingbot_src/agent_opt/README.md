# agent_opt — autonomous lossless-parallelism optimizer

Adapts the SGLang runtime multi-agent search pattern (goal.md + search-space +
bounded fan-out loop + correctness-gate-first + journal) to LingBot-Video, driven
by a Claude Code subagent instead of Codex goal sessions.

## Contents
- `GOAL.md` — the agent's system prompt (mission, constraints, loop contract, gate, deliverable).
- `search_space.md` — technique families (Ulysses / Ring / EP / TP / FSDP / replication /
  offload scheduling), their status (implemented / needs-impl), exact knobs, and the
  4-GPU / lossless constraints.
- `verify_lossless.py` — the lossless gate (PSNR of config refined video vs golden).
- `JOURNAL.md` — append-only per-config log + frontier + rejected signatures.
- `STATUS.json` — machine-readable loop state (iter, frontier, blockers).
- `baseline/golden_refined.mp4` — the golden output (produced at iter 0).
- `config/` — per-config sbatch/config the agent generates.
- `REPORT.md` — final deliverable (frontier + best config + reproduction + rejects).

## Scope (fixed by the orchestrator)
Exactly 4×GB200, one NVLink node. Optimize the **combination of MoE + Attention
parallelism / communication + weight placement + stage scheduling** to minimize
end-to-end latency while staying **lossless** (PSNR ≥ 45 dB vs golden). Steps,
resolution, scheduler, seed, dtype-for-quality are FIXED (changing them = lossy = out of scope).

## Launch (orchestrator = monitor only)
Launched as a background Claude Code subagent told to follow `GOAL.md`. The
orchestrator monitors `JOURNAL.md` / `STATUS.json` and relays progress; it does
not implement config itself.

## Monitor
- `cat agent_opt/STATUS.json` — iter / frontier / blockers
- `tail -40 agent_opt/JOURNAL.md` — latest config entries
- `squeue -u $USER` — in-flight GB200 config jobs
