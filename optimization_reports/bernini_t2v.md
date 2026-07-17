# Bernini T2V 加速总结

实验目录：`output/orchestrated/bernini-20260705-083900/`
定稿时间：2026-07-05 17:11Z
编排方式：`workflow_lite`（1 个 master orchestrator agent + 3 个 executor sub-agent：kernel / cache / pisa），codex 驱动、detached、watchdog 守护。

---

## 一、最终结果（以 integrator 定稿的 `INTEGRATED-DELIVERY.json` 为准）

**Bernini 最终加速 = 2.257×（128.989s → 57.14s）。**

integrator 最后给出两个 Pareto 交付点，两个用的 kernel 都是同一个**无损、4 分支完整**的版本，再叠加 easycache：

| 交付点 | 总时延 | **加速倍数** | 组成 | LPIPS(mean / max) | 无损? |
|---|---|---|---|---|---|
| **e010（最快）** | **57.14s** | **2.257×** | 无损 kernel(80.9s) + easycache r06/e010 | 0.108 / 0.050 | ❌ 含有损 cache |
| e008 | 59.04s | 2.185× | 无损 kernel(80.9s) + easycache r05/e008 | 0.096 / 0.045 | ❌ 含有损 cache |

> 视频产物：`integration/runs/20260705-162318-bernini_integrated_kernel_easycache_e010/outputs/`
> （以及对应的 e008 run 目录）

### 这 2.257× 是怎么组成的
- **无损 kernel（不砍任何分支）**：128.989s → 80.9s = **1.594×**（这部分是真无损）
- **+ easycache（跨步缓存/跳步，有损近似）**：80.9s → 57.14s，再叠约 **1.42×**
- 合计：1.594 × 1.42 ≈ **2.257×**

**诚实说明**：最终 57.14s 因为含 easycache（近似方法）**整体是有损的**（LPIPS mean ≈ 0.108，视觉可用但非逐位/逐帧一致）。若只要**纯无损**（不含 easycache），最好成绩是 **80.9s / 1.594×**。

---

## 二、基线（口径）

| 项 | 值 |
|---|---|
| 模型 | Bernini T2V |
| 硬件 | GB200 × 4 |
| 分辨率 / 帧数 | 480×848 / 81 帧 |
| 采样 | 50 steps, seed 42 |
| 并行 | Ulysses-4 |
| 注意力 | cuDNN flash SDPA |
| 计时范围 | `text_to_vae_decode`（热身后热态） |
| **基线时延** | **128.989s** |

---

## 三、经历了怎样的变化（演进过程）

### 1) 编排框架落地
搭建 `workflow_lite`（轻量：master + 3 executor），全流程首次跑通（2026-07-04，约 7.5h）：master 自动派生 3 个 executor → 各自诚实交付 → master 独立复核（objective plan_eval 复跑 + 自己的多模态看帧 + provenance）→ 组合并门控。首版整合结果 1.87×（kernel+pisa+TeaCache，有损）。

### 2) "无损"定义逐步收敛（用户主导）
- 起初用 **bit-exact** → 太严，误杀了 fp 等价的更快候选（编译器 fp-contraction 舍入变化），kernel 被卡在 1.314×。
- 放宽到 **fp 容差** → 用户进一步指出也不要容差。
- 最终定义：**无损 = 数学/算法层面的正确性，只从方法/规则/道理去论证，不做任何输出比较**（不看 bit-exact、不看 fp 容差、不看 LPIPS）。理由：同一算法的两种正确实现可以数值发散，但同样正确，不能因为和参考实现有差异就判错。
- 门控相应改造：`kernel_scope.md` / `verify_delivery.py`（结构不变量 + 方法论证，零输出比较）/ `master.md`（master 用推理判断方法，不重跑 latent）。cache/pisa 仍按 metric 门控（它们本就是有损）。

### 3) kernel 换新 prompt + 组合大杠杆 → 1.594×
用新的数学-正确性 prompt 重跑，kernel 组合了全部主要无损杠杆（权重驻留、通信、launch/sync 开销、fastest-equivalent 原语分派等），做到 **1.594×（80.9s，5-prompt 终态验证）**，随后其杠杆家族用尽、自然收敛。

### 4) 差距分析 + 一次"砍分支"的尝试与撤回（重要教训）
- gap 分析怀疑：我们每步分派 **4 个 guidance 分支**，而参考 Bernini 是 **3 个**；第 4 个（"img"）分支疑似 base 的冗余复制。
- kernel 据此实现"砍第 4 分支"（候选 `bernini_redundant_t2v_img_cse_v17`，60.5s / 2.13×）。
- **但 kernel 自己的等价性证明推翻了这个前提**：`latent_equivalence.json` 显示 3 分支 vs 4 分支的最终潜变量在**计时那一遍逐 prompt 差 max_abs 2.25–4.0**（`all_equal:false`, `measured_max_abs:4.003`）——这不是浮点噪声（噪声是 1e-3 量级），说明第 4 分支**确有实质贡献,不是零贡献冗余**。
- 结论：**v17 不是无损，已撤回**。kernel 正确地没把它当无损交付，integrator 也正确地用了真无损的 80.9s 版本。

---

## 四、和参考 Bernini 的对比（诚实口径）

| | 我们 | 参考 Bernini |
|---|---|---|
| 纯无损最好 | 80.9s（1.594×，4 分支） | 60.9s(all-HBM) / 65.9s(stage-offload) |

两者约 20s 的无损差距**几乎全部 = 那第 4 个 img 分支**（4 分支 denoise 74.85s → 3 分支 55.09s；我们 3 分支的 60.5s ≈ 参考 60.9s）。即**参考本质上是 3 分支基线**。所以这不是"我们优化不到位"，而是**基线定义不同口径**：想在纯 T2V 合法拿到 ~60s，前提是确认"纯 T2V（无源图/源视频，`has_source_vae=False`)本就不该跑第 4 个 img 分支"，那属于**改基线**（需拍板），不是对现有 4 分支基线的无损优化。此项待核实：`bernini/models/wan_diffusion.py:1160` 的 `has_source_vae` 门控 vs 我们的制导循环。

---

## 五、一句话结论

- **Bernini 最终加速：2.257×（128.989s → 57.14s）**（integrator 定稿、有损组合：无损 kernel + easycache）。
- **纯无损最好：1.594×（80.9s）**，未砍任何分支。
- v17"砍分支"路线（2.13×）经其自身证明为有损，已撤回。
- 追平参考 ~60s 无损属于"改基线"决策，非无损优化范畴，待定。
