# Wan 2.2 A14B Full Optimization: Kernel + Topology + Cache + PISA

日期：2026-07-13

## 最终结论

已将 PISA Attention 叠加到此前已验证的 Wan Full OPT 栈，并用正确的全局 CFG forward clock 与 EasyCache 组合。有效的 4-GPU measured run 为：

- run：`runs/20260713-172727-wan22_t2v_a14b_pisa_density010_easycache030_full-torchrun_retry`
- Slurm job：`5008130`，`COMPLETED`，`gres/gpu=4`
- 5 prompts median，2 warmup，`seed=1024`
- total：`58.89s`；denoise：`52.40s`

相对之前同配置的 Full OPT（`75.60s`）为 **1.284x**，时间下降 **22.10%**；相对公平的 CP4 dense/no-cache control（`115.06s`）为 **1.954x**，时间下降 **48.82%**。

## 生效的优化栈

```text
context_parallel_ulysses4
fused_qkv_projections
compiled_block_glue
compiled_qk_rope
async_qkv_ulysses_a2a
direct_ulysses_output_a2a
reusable_ulysses_a2a_buffers
invariant_rope_cache
invariant_conditioning_cache
easycache threshold=0.30
pisa_attention density=0.10
```

拓扑为单节点 CP4/Ulysses4，4 张 GB200 全部参与；cross-attention 保持 dense，PISA 只作用于 video self-attention。

## PISA Dense Guard

配置已被记录在 `outputs/pisa_stats_rank0.json`：

- density：`0.10`，sparsity：`0.90`，block size：`64`
- Dense layers：`0-3,40-43`
- Dense steps：`0-3,37-39`
- kernel stages：`2`
- measured prompt 的全局 forward calls：`80`
- PISA dispatch、dense policy 和 step/layer guard 统计均已落盘

此前与 Cache 组合时误用了 expert-local call count，不能正确表达真实 timestep；本版本改为全局 CFG forward pair clock，并用 `global_forward_calls=80` 验证 40 个 denoising steps 的语义。

## Cache

EasyCache 作用于 blocks 1–39，block 0 保持 fresh CP-sharding。5 个 measured prompts 的 median reuse 为 `14/40` steps，hit rate `0.35`；不同 prompt 的 reuse 为 `13–15/40`。

## 质量与可复现性

输出视频由相同 prompt 列表、相同 `seed=1024`、相同 720×1280/81 帧/40 steps 配置生成。优化改变了浮点 reduction 和 attention pattern，因此不要求逐像素一致；seed 保证初始噪声一致。视频和完整 benchmark、PISA、Cache、call ledger 均保存在上述 run 目录。

## 失败尝试说明

同一候选曾用 `srun` task-local GPU masking 启动，在 direct output A2A 触发 NCCL error；该 job 不计入 benchmark。最终候选恢复到此前已验证的单节点 `torchrun` launcher，并由 job 5008130 完整跑通。
