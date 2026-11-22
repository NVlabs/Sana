// Numbers entered by hand. Every block names the place in the paper it comes from
// (SANA-Robot-Paper at commit 9641318). The benchmark tables live in bench_data.js,
// which tools/build_bench_data.py generates from chapters/experiments.tex.
window.SITE = {
  // §2.1: 32 action tokens per chunk. §8: the controller executes each action for about 33 ms (30 Hz).
  chunk: { actions: 32, hz: 30 },

  // §7.1 Memory Designs, Fig. "Memory designs on RoboDojo" (b): success on the memory tasks (%). These three are
  // our own evaluation of the three designs under one protocol; the paper keeps them when the RoboDojo table moved
  // to the official result (52.33% for the hybrid memory), so they stay as they are here too.
  memoryDesigns: [
    { id: "hybrid", name: "Hybrid memory (ours)", success: 48.8 },
    { id: "interndelta", name: "InternW0-Δ memory", success: 31.4 },
    { id: "vpp2", name: "VPP2 memory", success: 28.3 },
  ],

  // §8 Table "Real-world comparison with baselines (%)". 20 rollouts per task.
  real: {
    hanoi: [["SANA-Robot-AR", 83.6, true], ["OpenWAM-α", 45.7], ["π0.5", 44.3]],
    coffee: [["SANA-Robot-AR", 87.1, true], ["OpenWAM-α", 65.7], ["π0.5", 57.1]],
    cube: [["SANA-Robot-AR", 75.0, true], ["π0.5", 70.0], ["OpenWAM-α", 60.0]],
    // §8 "Long Memory": SANA-Robot-AR with N = 3 against SANA-Robot-base; chance is 1 in 3.
    shell: [["SANA-Robot-AR", 75.0, true], ["SANA-Robot-base", 30.0]],
    // §8 "Fast Reaction": SANA-Robot-AR on Air Hockey, asynchronous execution (60% with BF16 and with FP8 GEMMs)
    // against synchronous execution with BF16 (45%). The same numbers as `hockey` below.
    hockey: [["Asynchronous", 60, true], ["Synchronous", 45]],
  },

  // Three-Cup Shell Game clip used in the memory timeline (three-cups/clip3.mov). Times in seconds, read off
  // the footage by eye: the ball is in view until a cup covers it, a person shuffles the cups, the robot picks.
  shellPhases: [
    { from: 0, to: 1.8, label: "ball visible" },
    { from: 2.0, to: 7.4, label: "cups shuffled" },
    { from: 8.8, to: 16.0, label: "robot picks a cup" },
  ],

  // §5.1, Fig. "Asynchronous action error": open-loop end-effector position error relative to synchronous
  // execution (C = 32) on 70 RoboDojo episodes, as plotted. Values from the record kept as a comment in
  // chapters/inference.tex ("EE ratio to sync").
  asyncError: { d: [8, 16, 24], memory: [0.989, 0.981, 0.993], direct: [2.663, 4.785, 7.098] },

  // §8 "Fast Reaction" and "Low-Bit Inference" (Air Hockey and Tower of Hanoi on RTX PRO 6000).
  hockey: {
    sync: { hit: 45, stallMs: 181.3, hz: 25.6 },
    async: { hit: 60, stallMs: 4, hz: 30 },
  },
  lowbit: {
    hanoi: [["BF16", 83.6], ["FP8", 82.9], ["NVFP4", 81.4]],
    hockey: [["BF16", 60], ["FP8", 60], ["NVFP4", 50]],
  },

  // §5.3 Table "Per-request latency (ms) of SANA-Robot-AR (N = 1) in synchronous execution with K = 4".
  latency: [
    { gpu: "RTX 5090", eager: 607.8, opt: 173.3, fp8: 135.1, nvfp4: 115.0 },
    { gpu: "RTX PRO 6000", eager: 573.3, opt: 175.0, fp8: 140.6, nvfp4: 123.1 },
    { gpu: "Jetson Thor", eager: 1390.1, opt: 420.2, fp8: 333.6, nvfp4: 288.0 },
  ],

  // Teaser panel (c): latency per action chunk on one RTX 5090 against the RoboDojo memory score.
  // Latencies: drafts/teaser/teaser_data.json (baselines measured on the same RTX 5090; ours from the
  // latency table, FP8). Scores: the Memory column of the RoboDojo table in chapters/experiments.tex; ours is the
  // official evaluation (52.73, paper commit 0f245b3), as in bench_data.js.
  tradeoff: [
    { name: "SANA-Robot-AR", ms: 135.1, score: 52.73, ours: true },
    { name: "InternW0-Δ", ms: 127.1, score: 34.67 },
    { name: "OpenWAM-α", ms: 166.9, score: 10.41 },
    { name: "Fast-WAM", ms: 70.9, score: 3.55 },
    { name: "X-WAM", ms: 641.8, score: 6.32 },
  ],

  // §6.3.2 DROID policy evaluation: droid_success_rate.csv (real, Ctrl-World, ours; % over 20 rollouts).
  droid: [
    ["Pick and place", "π0", 70, 55, 100], ["Pick and place", "π0-FAST", 85, 70, 85], ["Pick and place", "π0.5", 85, 70, 85],
    ["Towel folding", "π0", 40, 20, 20], ["Towel folding", "π0-FAST", 55, 40, 65], ["Towel folding", "π0.5", 75, 65, 85],
    ["Drawer", "π0", 5, 0, 0], ["Drawer", "π0-FAST", 15, 5, 5], ["Drawer", "π0.5", 80, 30, 50],
    ["Table wiping", "π0", 20, 0, 55], ["Table wiping", "π0-FAST", 50, 35, 65], ["Table wiping", "π0.5", 80, 65, 95],
    ["Closing a laptop", "π0", 15, 0, 5], ["Closing a laptop", "π0-FAST", 50, 0, 40], ["Closing a laptop", "π0.5", 70, 5, 60],
    ["Tissue", "π0", 30, 20, 20], ["Tissue", "π0-FAST", 10, 0, 65], ["Tissue", "π0.5", 75, 50, 100],
    ["Stacking", "π0", 60, 40, 80], ["Stacking", "π0-FAST", 70, 55, 70], ["Stacking", "π0.5", 65, 55, 75],
  ],
};
