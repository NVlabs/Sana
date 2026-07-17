# Wan2.2 TI2V-5B 加速总结

- 实验目录：`output/orchestrated/wan5b-20260706-130525/`（seq 0001）
- 定稿：2026-07-06（watchdog `bz8a0h40k` 触发 integrator 写盘）
- 编排方式：`workflow_lite`（1 个 master orchestrator agent + 2 个 executor sub-agent：`kernel_aw` / `cache_ca`），codex 驱动、detached、watchdog 守护。**只开 kernel + cache 两个 executor，未开 pisa。**
- 交付原件：`INTEGRATED-DELIVERY.json`（本目录 `artifacts/wan22_ti2v_5b/` 已保存副本 + 可复现 candidate + 整合 runtime 代码 + 证据 JSON + 视频）。

---

## 一、最终结果（以 integrator 定稿的 `INTEGRATED-DELIVERY.json` 为准）

**Wan2.2 5B 最终加速 = 2.885×（70.25s → 24.35s），单卡 GB200。**

唯一 Pareto 交付点（nondominated）：

| 交付点 | 总时延 | **加速** | denoise | decode | 组成 | LPIPS(mean / max) | 门控 |
|---|---|---|---|---|---|---|---|
| **wan5b_integrated_kernel_easycache_t0036** | **24.35s** | **2.885×** | 18.12s | 5.33s | 无损 kernel(1.519×) + EasyCache@0.036 | 0.242 / 0.270 | 视觉 pass · authenticity pass |

runner-up（被支配、未入 frontier）：`t0031` = 2.646× / 26.5s，LPIPS max 0.271 —— 在**速度和 LPIPS 两个维度都被 t0036 支配**，故排除。

### 这 2.885× 是怎么组成的
- **无损 kernel**：70.25s → ~46.3s = **1.519×**（真无损，见 §三）
- **+ EasyCache（跨步复用，有损近似）**：46.3s → 24.35s，再叠 **≈1.90×**
- 合计：1.519 × 1.90 ≈ **2.885×**

**诚实口径**：最终 24.35s 含 EasyCache（近似方法）→ **整体有损**（LPIPS mean ≈ 0.242，视觉门通过但非逐位/逐帧一致）。**纯无损最好 = 1.519× / ~46.3s**（kernel-only，不含 cache）。

---

## 二、基线口径

| 项 | 值 |
|---|---|
| 模型 | Wan2.2 **TI2V-5B**（vanilla diffusers `WanPipeline`，官方集成，非 FastVideo）|
| 硬件 | GB200 × **1**（官方即单卡模型，不做 CP）|
| 分辨率 / 帧数 | 704×1280 / 121 帧 |
| 采样 | 50 steps，guidance 5.0，fps 24，seed 42 |
| 计时范围 | `text_to_video`（warmup 后热态），5-prompt 中位数 |
| **基线时延** | **70.25s**（denoise 63.6 / decode 5.3）|
| run_dir | `runs/20260706-102948-wan22_ti2v_5b_baseline` |
| frozen_at | 2026-07-06T13:05:25Z（`BASELINE.json`，sha256 `5835f008…`）|

> 5B 保持 1-GPU 口径（见 §四"option B"）：它的候选是真单卡 kernel 工作，若强逼 4-GPU 基线会误判其单卡合法收益。

---

## 三、方法细节

### kernel（无损，1.519×）— 候选 `wan5b_kernel_terminal_exact_fastest`
杠杆栈：`regional_compile, qkv_fusion, invariant_cache_v2, cross_kv_cache, bf16_block_glue`
- **regional compilation + autotuning**：保持编译图方程不变（`max-autotune-no-cudagraphs`，fullgraph）
- **Q/K/V 拼接仿射 + 精确 chunk**：等价于原始投影
- **identity/version-guarded caches**：只复用严格 step-/branch- 不变量
- **cross-attention K/V 复用**：受 live conditioning identity/version 守护
- **BF16 block glue**：维持 16-bit 精度
- 全 50 步、cond+uncond 两路、30 个 dense block、self/cross-attn、FFN **全部保留**；无近似 / 跳步 / 稀疏 / 降秩 / 低于 16-bit 量化。**数值输出差异不作为正确性判据**（无损 = 算法/规则层面论证）。

### cache（有损）— EasyCache
- 采纳阈值：**0.036**（frontier）与 0.031（runner-up），retain 7 步 / cooldown 1
- 复用统计（t0036）：700 次 wrapper 调用 → 370 compute / 330 reuse = **47.1% 步复用**；每 prompt 中位 54 reuse / 46 compute
- **被拒候选（诚实记录）**：`r03_t005` / `r12_t008` / `r13_t012` 出现 ghosting/smearing；`r04_t002` 未认证（对齐片段零复用）。→ 更激进阈值被**整合期视觉门**挡下,只有保守的 t0036/t0031 通过。

### 质量门
- **LPIPS**：max 0.270 / mean 0.242 / median 0.250（43 对）
- **视觉门**：master 内置多模态看帧,51 帧 / 5 clip 对齐参考,artifact-free coherent,max severity = none
- **authenticity**：pass（确认 kernel 栈真装载 + cache 真复用,非"假加速"）
- **external gemini**：本次**故意关闭**（`--no-gemini`）→ `nvidia_gemini:api_key_missing` 是预期状态,非失败；视觉门由 master 内置多模态直接做。

---

## 四、经历了怎样的变化（演进过程）

1. **基线搭建**：用 vanilla diffusers `WanPipeline` 作诚实基线（非 FastVideo 优化器），5-prompt 中位 70.25s,注册为一等 target(`models/wan22_ti2v_5b`)。
2. **并行编排启动**：与 14B 同时起两个 parallel master（`wan5b-master-0001` + `wan14b-master-0001`),各只派生 kernel+cache executor。为并行不撞,给 `run_orchestrated_experiment.py` 的 MODEL_PREFIX 加了 `wan5b`/`wan14b` 独立前缀(否则都退化成 `wan22` → 撞 exp_root/session)。
3. **option B（多卡允许 + 公平基线）非对称应用**：14B 重设 4-GPU CP4 基线；**5B 保持 1-GPU 70.25s**,因为它的 kernel 候选(如 qkv_fusion 46.7s)是真单卡工作。规则：每个候选按**其 GPU 数**对比 pristine；纯并行 = 同卡基线 = 不算赢。
4. **kernel 收敛**：kernel executor 用尽无损杠杆家族,终态 `wan5b_kernel_terminal_exact_fastest` = 1.519×。
5. **cache 探索 + 整合门控**:cache executor 扫多个 EasyCache 阈值;整合期 master 逐一复核,拒掉有 ghosting/零复用的激进点,保留 t0036/t0031。
6. **integrator 组合 + 定稿**:按"先装 verified 精确 kernel,再在 kernel 后的 DiT forward 外裹 verified EasyCache"顺序整合,复跑两点,t0036 单点非支配 → 定稿。(该实验的 provenance 组装一度极慢,直到 watchdog `bz8a0h40k` 触发才写盘,结果早已知。)

---

## 五、复现方式

激活 env（完整见 `artifacts/wan22_ti2v_5b/wan5b_integrated_kernel_easycache_t0036.toml`）：
```
WAN22_KERNEL_STACK=regional_compile,qkv_fusion,invariant_cache_v2,cross_kv_cache,bf16_block_glue
WAN22_COMPILE_MODE=max-autotune-no-cudagraphs
WAN22_COMPILE_FULLGRAPH=1
WAN22_CACHE_FAMILY=easycache
WAN22_EASYCACHE_THRESHOLD=0.036
WAN22_EASYCACHE_RETAIN_STEPS=7
WAN22_EASYCACHE_COOLDOWN_STEPS=1
WAN22_NUM_PROMPTS=5
WAN22_WARMUP_PASSES=2
```
- 整合 runtime 代码：`artifacts/wan22_ti2v_5b/integrated_runtime/{gpu_infer.py, wan_kernel_runtime.py, cache_runtime.py, scripts/}`
- 起跑：`python scripts/launch_candidate.py <candidate.toml> --mode sbatch --confirm-submit`

---

## 六、provenance（可信度）

- slurm **4202859** COMPLETED `0:0`，node nvl72098-T12，**121 帧 / 5 视频**，`independently_verified=true`
- 唯一瑕疵：`collect_run` 因忽略的 TorchInductor autotune traceback 误标 metadata failed → 但 slurm 独立报 COMPLETED + DONE sentinel + 全部产物齐全,不影响结果可信。

---

## 七、一句话结论

**Wan2.2 5B：最终 2.885× / 24.35s**（无损 kernel 1.519× ∘ EasyCache,视觉门 pass,整体有损）；**纯无损最好 1.519× / ~46.3s**。单卡口径,不含 CP。
