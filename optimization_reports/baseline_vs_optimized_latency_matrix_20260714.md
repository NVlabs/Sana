# Baseline vs Optimized — Latency Matrix (fixed)

日期：2026-07-14
硬件：NVIDIA **GB200**（aarch64），单 NVLink 节点。计时口径均**排除模型加载**（load-excluded request wall），多 prompt 取 **median**，同 seed / 同采样配置。

每个模型固化 **两种 baseline**（naive 通用低效实现 + 官方推荐/同拓扑实现）和 **一种优化方法**。

---

## Wan2.2-T2V-A14B

分辨率 720×1280，81 帧，40 steps，seed 1024，5-prompt median（官方验证集 `t2v_val5.json`）。

| 技法 | 类型 | GPU | Latency (median) | 说明 |
|------|------|-----|------------------|------|
| **Naive baseline** | baseline① | 1 | **449.67 s** | 单卡 vanilla diffusers `WanPipeline`，无 context-parallel、无 cache、无 kernel 优化。最朴素"能跑起来"的实现。 |
| **CP4 dense control** | baseline② | 4 | **115.06 s** | 4-GPU CP4 Ulysses，dense attention，无 cache。纯并行化（作者式多卡跑法），无我方 kernel 优化。 |
| **Full OPT** | optimized | 4 | **59.26 s** | CP4 Ulysses4 + fused_qkv + compiled block/qk_rope + async/direct/reusable a2a + invariant caches + **EasyCache(0.30)** + **PISA(density 0.10)**。 |

**加速比**
- vs naive baseline①：**7.59×**（含 4 卡并行化）
- vs CP4 dense control②：**1.94×**（同拓扑，纯 kernel + PISA + EasyCache 优化）

---

## LingBot-Video (MoE 30B-A3B)

base 480×832 → refiner 1088×1920，121 帧，base 40 steps / refiner 8 steps，seed 42，3-prompt median（官方验证集 `t2v_val3.json`，官方仅 3 个 t2v prompt）。

| 技法 | 类型 | GPU | Latency (median) | 说明 |
|------|------|-----|------------------|------|
| **Naive baseline (FSDP-only)** | baseline① | 4 | **~1409 s** | `context_parallel_degree=1`（关 Ulysses 计算并行）+ 串行 CFG（`batch_cfg=0`）。FSDP 仅为放下 30B+refiner（显存必要，非提速）；4 卡各算冗余相同计算。`lingbot_video_fsdp4_reference`。〔n=2 完成 prompt 均值 1408.9 s；第 3 prompt 同轨迹后取消〕 |
| **官方推荐 baseline** | baseline② | 4 | **375.53 s** | CP4 Ulysses + FSDP + batched CFG（base+refiner）+ fa2 + grouped_mm MoE。作者官方 `dit_inference.md` 推荐的多卡跑法。 |
| **Full OPT (phase-specific PISA)** | optimized | 4 | **187.88 s** | 在官方推荐拓扑上叠加：**cuDNN attention** + condition-feature cache + **refiner-only PISA(density 0.10)**（base 阶段关 PISA 避免回退）。 |

**加速比**
- vs naive baseline①：**7.50×**（含 context-parallel + batched CFG + kernel + PISA 全栈）
- vs 官方推荐 baseline②：**2.00×**（同拓扑，纯 cuDNN + refiner-PISA 优化）

---

## 汇总

| 模型 | naive baseline | 官方/同拓扑 baseline | optimized | ×naive | ×官方 |
|------|----------------|----------------------|-----------|--------|-------|
| Wan2.2-A14B | 449.67 s (1-GPU) | 115.06 s (CP4 dense) | **59.26 s** | 7.59× | 1.94× |
| LingBot-Video | ~1409 s (FSDP-only) | 375.53 s (CP4+batchCFG) | **187.88 s** | 7.50× | 2.00× |

**读法**
- **×naive** = 完整栈（并行化 + kernel + cache/PISA）相对"什么都不做"的价值；两模型均 ~7.5×。
- **×官方** = 在作者推荐/同拓扑实现之上，我方纯 kernel/attention 优化的净贡献；两模型 1.9–2.0×。

**方法论说明**：优化改变浮点 reduction 顺序与 attention 稀疏（PISA/EasyCache），输出非逐像素等价；同 seed 固定初始噪声。计时排除模型加载。LingBot naive baseline 的 cp1+FSDP 使 4 卡冗余计算，refiner @1088p 单卡约 18 min/prompt 是主要开销。

## 对应 run / candidate

| 技法 | candidate | run_dir |
|------|-----------|---------|
| Wan naive | `wan22_t2v_a14b_baseline` | `runs/20260714-031656-wan22_t2v_a14b_baseline-valset` |
| Wan CP4 dense | `wan22_t2v_a14b_dense_control_no_cache` | `runs/20260713-151518-wan22_t2v_a14b_dense_control_no_cache` |
| Wan OPT | `wan22_t2v_a14b_pisa_density010_easycache030_full` | `runs/20260714-031658-...-valset` |
| LingBot naive | `lingbot_video_fsdp4_reference` | `runs/20260714-093354-lingbot_video_fsdp4_reference-naive_baseline`（cancelled after 2/3 prompts）|
| LingBot 官方 | `lingbot_video_baseline` | `runs/20260714-031659-lingbot_video_baseline-valset` |
| LingBot OPT | `lingbot_video_cudnn_pisa_full` | `runs/20260714-031659-lingbot_video_cudnn_pisa_full-valset` |
