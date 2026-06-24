SANA-Video 2B 480p (832x480, 81f, 50 steps), 64-prompt set (10 official demo + 54 authored), sglang pipeline.
 baseline: no optimizations (29.4s/clip)
 fullopt:  EasyCache 0.1 + linattn-bf16 + qkv-merge, NO torch.compile (17.2s/clip, 1.71x)
(note: torch.compile was dropped — it deadlocked in Inductor autotune; with compile the target is ~2.77x.)
