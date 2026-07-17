# Wan2.2 T2V-A14B 加速总结

- 实验目录：`output/orchestrated/wan14b-20260706-155110/`（seq **0002** —— 见 §四"守护进程事故",0001 被我手动 force-restart 撞死后干净重启）
- 定稿：2026-07-06
- 编排方式：`workflow_lite`（1 个 master orchestrator agent + 2 个 executor sub-agent：`kernel_aw` / `cache_ca`），codex 驱动、detached、watchdog 守护。**只开 kernel + cache,未开 pisa。**
- 交付原件：`INTEGRATED-DELIVERY.json`（本目录 `artifacts/wan22_t2v_a14b/` 已保存副本 + 3 个可复现 candidate + 整合 runtime 代码 + 证据 JSON + balanced 视频）。

---

## 一、最终结果（以 integrator 定稿的 `INTEGRATED-DELIVERY.json` 为准）

**Wan2.2 14B 相对 4-GPU CP4 公平基线（129.01s）加速 1.457×–1.670×。** 三个 Pareto 交付点全部非支配：

| 取向 | 交付点 | 总时延 | **加速** | denoise | EasyCache 阈值 | 步复用(中位) | LPIPS(mean / max) | 门控 |
|---|---|---|---|---|---|---|---|---|
| **最高保真** | `wan14b_integrated_kc_t018_0002` | **88.57s** | **1.457×** | 82.13s | 0.18 | 10 步 / 命中 0.25 | 0.166 / 0.191 | 视觉 pass |
| **均衡(推荐)** | `wan14b_integrated_kc_t021_0002` | **80.47s** | **1.603×** | 74.18s | 0.21 | 12 步 / 命中 0.30 | 0.176 / 0.205 | 视觉 pass |
| **最快** | `wan14b_integrated_kc_t030_0002` | **77.24s** | **1.670×** | 70.79s | 0.30 | 14 步 / 命中 0.35 | 0.319 / 0.356 | 视觉 pass |

三点均 authenticity pass、artifact-free（t030 为"较大轨迹/外观偏移但无 rubric 级瑕疵"）,共复核 243 帧。

### 加速是怎么组成的（以均衡点 t021 为例）
- **无损 kernel**（在 CP4 之上）：129.01s → ~113.6s = **1.136×**（真无损,见 §三）
- **+ EasyCache（有损）**：再叠到 80.47s → 合计 **1.603×**
- **纯无损最好 = 1.136× / ~113.6s**（kernel-only）；含 EasyCache 的 1.46–1.67× 整体有损但视觉门 pass。

> 注意口径：**基线本身已是 4-GPU CP4**（纯无损并行,见 §二/§四）。这里的 1.46–1.67× 是**在公平 4-GPU 基线之上的真实优化**,不是并行红利。相对 1-GPU pristine(450.8s)则是 450.8/77.24 ≈ 5.8×,但那大部分是并行,不作为"加速成果"口径。

---

## 二、基线口径（option B 公平基线）

| 项 | 值 |
|---|---|
| 模型 | Wan2.2 **T2V-A14B**（MoE:transformer + transformer_2, boundary_ratio 0.875）|
| 硬件 | GB200 × **4** |
| 并行 | **Ulysses 4-way context-parallel**（diffusers 原生 `WanTransformer3DModel._cp_plan`,`WAN22_KERNEL_PROFILE=0`,精确 dense attn）|
| 分辨率 / 帧数 | 720×1280 / 81 帧 |
| 采样 | 40 steps，guidance 4.0 + guidance_2 3.0，flow_shift 12.0，fps 16，seed 42 |
| 计时范围 | `text_to_video`（warmup 后热态），5-prompt 中位数 |
| **基线时延** | **129.01s**（denoise 122.33 / decode ~3.6）|
| run_dir | `runs/wan14b_4gpu_cp4_baseline`（`wan14b_cp4_ulysses_v1`，纯无损并行）|
| frozen_at | 2026-07-06T15:51:10Z（`BASELINE.json`，sha256 `5ad90fe6…`）|

> 为什么是 4-GPU 基线:A14B 官方即多卡模型,1-GPU pristine=450.8s。若拿 1-GPU 当基线,任何候选只要开 CP4 就"白得"3.5×,不公平。故把**纯无损 CP4 并行(129.01s)本身设为基线**,强制后续候选必须"CP4 + 真优化"才算赢。

---

## 三、方法细节

### kernel（无损，1.136× on top of CP4）— 候选 `wan14b_cp4_async_qkv_a2a_v1`
杠杆栈：`context_parallel_ulysses4, compiled_block_glue, compiled_qk_rope, async_qkv_ulysses_a2a`
- **compiled block modulation/residual glue + compiled pairwise Q/K RoPE**：保持原 BF16 Wan block 方程 + 全 dense attention
- **asynchronous Ulysses Q/K/V all-to-all**：保留原张量、rank 映射、message 大小、process group、重构顺序、每个 block 的全部 4 次 collective;**只改 enqueue-before-wait 调度**(通信/计算 overlap)
- 去噪步数 + DiT 调用数不变,无近似 / 低于 16-bit 量化 / 稀疏 / 降秩 / model-work 改动。数值差异不作正确性判据。
- ⚠️ 头绪有限:`.venv` **未装 flash_attn** → 注意力走 Torch SDPA;async-a2a 的 overlap 是本 kernel 的主杠杆,SP 扩展效率本就中等(all-to-all + MoE expert 切换)。

### cache（有损）— EasyCache
- 采纳 3 个阈值:0.18 / 0.21 / 0.30;`start_step 5`,`tail_steps 3`,`max_reuse 1`,probe 64 tokens / 128 channels
- 块执行量(每 prompt 满算 3200):t018 观测 2420 / t021 2264 / t030 2108(阈值越大复用越多、块执行越少)
- 命中时 bypass block 1–39,ledger 用 async-Ulysses dispatch 计数器如实记录实际执行块数(`async_qkv_calls_match_observed_blocks=true`)

### 质量门
- **LPIPS**(每点 42–43 对):t018 max 0.191 / t021 max 0.205 / t030 max 0.356
- **视觉门**:master 内置多模态,每点看 81 帧(prompt-0 全帧)对齐冻结基线,共 243 帧;t018/t021 = artifact-free coherent,t030 = 较大轨迹偏移但无瑕疵
- **authenticity**:三点全 pass
- **external gemini**:故意关闭(`--no-gemini`),`api_key_missing` 为预期,视觉门由 master 内置多模态做。

---

## 四、经历了怎样的变化（演进过程）

1. **1-GPU 基线 → 多卡再思考**：A14B 1-GPU pristine=450.8s(慢,因单卡无并行)。用户采纳 **option B**:允许多卡但**重设公平基线**。
2. **重设 4-GPU CP4 基线**：`wan14b_cp4_ulysses_v1` = pristine A14B + 4-way Ulysses(diffusers 原生,精确 dense) = 纯无损并行 = **129.01s**,存 `runs/wan14b_4gpu_cp4_baseline`。
3. **⚠️ 守护进程事故（重要教训）**：seq 0001 跑起后,我在 master 拥有的 executor 上**外部 `resume_executor --force`**,撞上**共享单例 codex 审批守护进程**(`~/codex_auto_run.py`,`approver-daemon.lock` 指纹锁),**撞死了 14B kernel session + 卡住 cache**。
   - **修复**:直接改冻结的 `BASELINE.json`(master 每轮复核会重读) + 用 `run_orchestrated_experiment` 干净重启为 **seq 0002**(复用同一共享守护 = 同指纹,无冲突)。
   - **教训**:绝不对 master 拥有的 executor 外部 force-restart;改基线就改 `BASELINE.json`,换策略就整体干净重启。
4. **seq 0002 全绿跑通**：kernel 收敛到 async-a2a 1.136×;cache 扫 3 阈值;integrator 按"无损 kernel patch → diffusers CP4 hooks → verified EasyCache tail proxy"顺序整合,3 点全非支配 → 定稿。

---

## 五、复现方式

激活 env（均衡点 t021,完整见 `artifacts/wan22_t2v_a14b/wan14b_integrated_kc_t021_0002.toml`）：
```
WAN22_CONTEXT_PARALLEL=1
WAN22_CP_DEGREE=4
WAN22_KERNEL_PROFILE=0
WAN22_KERNEL_STACK=context_parallel_ulysses4,compiled_block_glue,compiled_qk_rope,async_qkv_ulysses_a2a
WAN22_CACHE_METHOD=easycache
WAN22_EASYCACHE_THRESHOLD=0.21      # 0.18=最高保真 / 0.30=最快
WAN22_CACHE_START_STEP=5
WAN22_CACHE_TAIL_STEPS=3
WAN22_CACHE_MAX_REUSE=1
WAN22_CACHE_PROBE_TOKENS=64
WAN22_CACHE_PROBE_CHANNELS=128
WAN22_NUM_PROMPTS=5
WAN22_WARMUP_PASSES=2
```
- 整合 runtime 代码：`artifacts/wan22_t2v_a14b/integrated_runtime/{gpu_infer.py, cache_controller.py, wan_kernel_runtime.py, wan_kernel_optimizations.py}`
- 起跑：`python scripts/launch_candidate.py <candidate.toml> --mode sbatch --confirm-submit`（需 4-GPU 整节点 `--gpus-per-node=4 --exclusive`,QOSMinGRES 要求）

---

## 六、provenance（可信度）

- slurm 三点全 COMPLETED `0:0`：t018=4189604(nvl72083-T10) / t021=4190445(nvl72137-T01) / t030=4191627(nvl72150-T13),各 81 帧,`independently_verified=true`,`full_run_config_match=true`。

---

## 七、一句话结论

**Wan2.2 14B：相对 4-GPU CP4 公平基线(129.01s),1.457× / 1.603× / 1.670× 三个 Pareto 点**(kernel 无损 1.136× ∘ EasyCache,视觉门 pass,整体有损),推荐均衡点 **1.603× / 80.47s**;纯无损最好 1.136× / ~113.6s。事故教训:勿外部 force-restart master 拥有的 executor。
