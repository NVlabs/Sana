# LingBot Video Full Optimization: Kernel + Topology + Cache + PISA

日期：2026-07-13

## 最终结果

最终采用 phase-specific PISA：base 阶段关闭 PISA，保留已验证的 cuDNN attention；refiner 阶段启用 PISA。这样避免了 PISA 在 base 小 attention shape 上的额外开销，同时保留它在 refiner 大 attention shape 上的收益。

- run：`runs/20260713-173238-lingbot_video_cudnn_pisa_full-phase_specific`
- Slurm job：`5009222`，`COMPLETED`，`gres/gpu=4`
- seed：`42`
- authoritative load-excluded request：`182.82s`
- source phase subset：`180.81s`
- base pipeline：`61.61s`
- refiner input preparation：`19.02s`
- refiner pipeline：`98.63s`

相对此前 c5 cuDNN 优化的 `210.07s` 为 **1.149x**，时间下降 **12.97%**；相对 baseline `375.55s` 为 **2.054x**，时间下降 **51.32%**。

## 四个优化方向

- 拓扑通信：4-GPU 单节点 CP4/Ulysses，保持 FSDP inference sharding；未启用经实测回退的 MoE EP。
- Kernel：base/refiner 使用 cuDNN attention；MoE 使用 `grouped_mm + vectorized padding + sort reorder + scatter restore`。
- Cache：沿用 condition-feature Cache，`reuse_condition_features=1`；没有把 LingBot 的 128-expert 结构误当成需要 EP 的稀疏显存卸载问题。
- Attention：refiner 使用 PISA density `0.10`，并保留 head/tail/layer Dense Guard。

## PISA 配置与 Dense Guard

`outputs/pisa_stats_rank0.json` 记录：

- base PISA：关闭（`pisa_base_enabled=false`）
- refiner PISA：开启（`pisa_refiner_enabled=true`）
- density：`0.10`，sparsity：`0.90`，block size：`64`
- Dense layers：`0-3`
- Dense steps：`0,1,7`（refiner 8 steps 的前 2 步和最后 1 步）
- refiner rank-0 dispatch：440 PISA、328 dense policy
- PISA 只作用于 video self-attention，dense fallback 保留

## 质量与可复现性

base 输出为 `121×480×832`，refiner 输出为 `121×1088×1920×3`，生成视频已成功写出且通过 runner 的非空 artifact gate。配置、日志、benchmark、四个 rank 的 PISA stats 和峰值显存均在 run 目录中；模型请求使用与 c5 对照相同的 prompt/seed/采样配置。由于 PISA 是近似 sparse attention，允许 bf16/attention reduction 的数值漂移，不宣称逐像素等价。

## 选择依据

此前把 PISA 同时用于 base 和 refiner 时，base pipeline 为约 `149s`，总请求为 `279.36s`，反而慢于 c5；phase-specific 版本把 base 恢复到 `61.61s`，并将 refiner 降到 `98.63s`，最终得到 `182.82s`。因此当前交付配置明确禁止 base PISA，避免回退。
