# 加速实验报告（Optimization Reports）

本目录汇总各视频模型在本 repo 下的**加速实验全过程记录 + 可复现产物**。所有 orchestration 原始产物都在 gitignore 的 `output/`、`runs/` 下,易随清理丢失 —— 故此处**把定稿结果、可复现 recipe、整合 runtime 代码、证据 JSON 一并落到受版本控制的目录**保存。

## 报告一览

| 报告 | 模型 | 基线 | 最终加速（定稿,含 cache,有损） | 纯无损最好 | 编排 |
|---|---|---|---|---|---|
| [bernini_t2v.md](bernini_t2v.md) | Bernini T2V | 128.989s（4-GPU Ulysses）| **2.257×** / 57.14s | 1.594× / 80.9s | workflow_lite（master + kernel/cache/pisa）|
| [wan22_ti2v_5b.md](wan22_ti2v_5b.md) | Wan2.2 TI2V-5B | 70.25s（**1-GPU**）| **2.885×** / 24.35s | 1.519× / ~46.3s | workflow_lite（master + kernel/cache）|
| [wan22_t2v_a14b_status_20260713.md](wan22_t2v_a14b_status_20260713.md)（历史：[wan22_t2v_a14b.md](wan22_t2v_a14b.md)） | Wan2.2 T2V-A14B | 129.01s（**4-GPU CP4** 公平基线）| **1.707×** / 75.60s（当前 Full OPT）· 1.603× / 80.47s（历史均衡）| 1.136× / ~113.6s | workflow_lite（master + kernel/cache）|

> 加速倍数口径:均为**同 GPU 数**下 vs pristine 基线,warmup 后热态,5-prompt 中位。**"最终加速"含 EasyCache（有损近似,视觉门 pass）;"纯无损最好"仅无损 kernel,不含 cache。**

## 共同套路（三次实验的一致方法）

1. **诚实基线**:vanilla diffusers 官方 pipeline 作基线(非优化器 fork),固定分辨率/步数/seed,warmup 后热态计时。
2. **无损定义**:kernel = **数学/算法层面正确性**,只从方法/规则论证,**不做输出比较**(不看 bit-exact / fp 容差 / LPIPS)。同一算法两种正确实现可数值发散但都正确。
3. **有损分层**:cache(EasyCache 跨步复用)、pisa 等按 metric + 视觉门控,本就有损,单独计。
4. **组合门控**:integrator 按"先无损 kernel,后有损 cache 外裹"顺序整合,复跑独立验证 + master 内置多模态视觉门 + authenticity + provenance,取 Pareto 非支配点。
5. **质量门**:LPIPS + master 内置多模态看帧;external gemini 本批故意关闭(`--no-gemini`),`api_key_missing` 为预期,非失败。

## 产物目录 `artifacts/`

每个模型一子目录,含:
- `INTEGRATED-DELIVERY.json` —— 定稿权威记录（所有数字来源）
- `BASELINE.json` —— 冻结基线
- `*.toml` —— **可复现 candidate**（env + config 的完整 recipe）
- `integrated_runtime/` —— **实际整合后的加速代码**（gpu_infer + kernel runtime + cache controller）
- `*_evidence/` · `winning_run_evidence/` —— benchmark / quality / cache_stats / ledger 证据 JSON + 样例视频

参考视频(基线 + 部分加速)另存 HF 私有 dataset:`yitongl/wan22-t2v-baseline-videos`。

## 一句话

- **Bernini**:2.257×（57.14s）—— 无损 kernel 1.594× ∘ easycache;曾试"砍第 4 分支"经自证有损已撤回。
- **Wan 5B**:2.885×（24.35s，单卡）—— 无损 kernel 1.519× ∘ easycache。
- **Wan 14B**:1.707×（75.60s,4-GPU CP4 公平基线)—— 无损 kernel/通信栈 1.136× ∘ EasyCache;后续拓扑、Kernel、Attention、Cache 候选均已实测筛选。
