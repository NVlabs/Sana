# Sol-Engine 发布更新清单 (NVlabs/Sana @ sol-engine)

## 已发布现状 (sol-engine branch)
- **模型 (3)**: Cosmos3-Super 64B (~2.27×) · LTX-2.3 22B (~2.38×) · SANA-Video 2B (~2.77×)
- **五方法**: Cache (TeaCache/EasyCache) · Quantization (NVFP4) · Kernel fusion · Sparse attention (PISA) · Token pruning
- **结构**: SGLang-based (python/, rust/sglang-grpc, sgl-kernel, technique/, examples/, prompts/, site_docs/, docker/)
- **AGENTS.md**: 纯部署指南 (per-model 脚本 scripts/sana|ltx|cosmos/…) —— **不含任何 agent 编排语义**
- README 章节: News · Models & speedups · 五方法 · Quick start (agent-native) · Getting started · To-do · Ack · Citation · License

---

## 需要更新的清单

### ① 三个新模型  ⚠️待确认是哪三个 (推测 = 本轮做完整的 Wan-5B / Wan-14B / LingBot)
本轮实测口径:
| 模型 | 参数 | 加速线 | 加速 (诚实口径) |
|------|------|--------|------|
| **Wan2.2 TI2V-5B** | 5B dense | kernel-fusion + EasyCache | **2.885×** (单卡; PISA 实测回退,不用) |
| **Wan2.2-A14B** | 14B MoE(2-expert) | kernel + EasyCache + PISA | **单卡 2.172×**(kernel1.13×·cache1.42×·PISA1.35×) / 同拓扑CP4 1.95× / vs单卡naive 7.6× |
| **LingBot-Video** | MoE 30B-A3B | cuDNN-attn-backend + refiner-PISA + base+refiner-EasyCache | **2.6×** (同拓扑) |

每个模型要补:
- [ ] README「Models & speedups」表加一行
- [ ] `scripts/<model>/` baseline + optimized 运行脚本(对齐现有 3 模型的 per-model 脚本模式)
- [ ] `prompts/` 官方验证集(Wan `t2v_val5.json`、LingBot `t2v_val3.json`)
- [ ] `technique/` 或 `site_docs/` 单模型文档:用了哪几个方法 + 配置 + 分解系数
- [ ] 模型卡 / 权重引用(HF repo 路径)

### ② 五方法文档补充(本轮引入的新变体)
- [ ] **Attention-backend swap (fa2→cuDNN)** —— LingBot 主 win(1.79×),归到 kernel 还是单列?
- [ ] **Phase-specific PISA**(refiner 开、base 关;避免小 attention 上回退)
- [ ] **2nd-stage (refiner) EasyCache**(base+refiner 两阶段各自 cache)
- [ ] **单卡 PISA dispatch**(拦 `dispatch_attention_fn`,单卡走 PISA、多卡走 CP 路径)—— 让 PISA 不再 CP-only
- [ ] **NVFP4 诚实注记**:高分辨率视频是 attention-bound,GEMM 微基准 2-3× 但端到端仅 ~1.1×(SSIM 0.89);是弱杠杆,不是万能。别把 microbench 数当端到端。

### ③ Agent Workflow —— 最大缺口 ⚠️
published AGENTS.md 只有部署脚本,**我们这套编排完全没进去**:
- [ ] **workflow_lite**(1-master + 3-executor sub-agent)—— 无文档
- [ ] **symposium** skills / goal-mode.env / socrates·evolve·ontology 等交互收敛工具 —— 未提
- [ ] **质量门**(LPIPS + NVIDIA-Gemini vision gate,side-by-side rubric)—— 未提
- [ ] **fanout search / candidate manifest / run_orchestrated_experiment.py** —— 未提
- [ ] **per-technique executors**(kernel_aw / cache_ca / pisa)—— 未提
- 建议:要么写一份真正描述编排的 AGENTS.md(或单独 `docs/agent-workflow.md`),要么至少在「Quick start (agent-native)」里链到这套工具 + 说明 master/executor + 质量门的用法

### ④ Assets / 可视化
- [ ] baseline-vs-fullopt **对比视频**(已在 HF `yitongl/video-opt-baseline-vs-fullopt`)—— 链接或镜像到 `assets/` / `site_docs/`
- [ ] **Ablation 站点**(HF Space `yitongl/video-opt-ablation`)—— README 链接
- [ ] 各模型 method-contribution 分解表(single-GPU: 5B kernel×cache; 14B kernel×cache×PISA)

### ⑤ README 其它
- [ ] **News** 加新模型条目
- [ ] **Models 表**:现有 3 + 新 3 = 6(或按 tier/family 分组)
- [ ] **To-do** 更新(哪些做完了)
- [ ] 加速口径统一说明(单卡 vs 同拓扑 vs vs-naive —— 避免像内部那样混口径)

---

## 决策已锁定 (2026-07-15)
1. **落地方式 = 方案B**:走现有 per-model 部署脚本 + 文档路径(和 sana/ltx/cosmos 一致,不做 SGLang 集成)。
2. **三个新模型 = Wan-5B / Wan-14B / LingBot**。
3. **对外加速口径(最终对外表)**:

| Model | Params | GPUs | Baseline → Optimized | 对外加速 | 加速线 |
|-------|--------|------|----------------------|---------|--------|
| **Wan2.2 TI2V-5B** | 5B dense | **单卡** | 70.25s → 24.35s | **2.885×** | kernel-fusion + EasyCache |
| **Wan2.2-A14B** | 14B MoE | **单卡** | 449.67s → 207.01s | **2.172×** | kernel-fusion + EasyCache + PISA |
| **LingBot-Video** | MoE 30B-A3B | **4 卡** | 375.53s → 144.36s | **2.6×** | cuDNN-attn + refiner-PISA + base/refiner-EasyCache |

- Wan 两个都是**单卡推理**(A14B 双 expert 都放进单张 192GB HBM),不涉及多卡。
- LingBot 是 **4 卡 baseline vs 4 卡优化**(同拓扑)。
- 对外只用上表这三个数,不出现 vs-naive 的 7.6×/9.8×(那含并行化,内部用)。

## 每个新模型要落的文件 (方案B)
per model (Wan-5B / Wan-14B / LingBot):
- [ ] `scripts/<model>/run_<model>_baseline.sh` + `run_<model>_optimized.sh`
- [ ] `prompts/<model>/…`(Wan `t2v_val5.json` / LingBot `t2v_val3.json`)
- [ ] `site_docs/models/<model>.md`(方法 + 配置 + 分解系数 + baseline/opt 数)
- [ ] README 表加一行 + News 一条
- [ ] AGENTS.md 加该模型的 env/download/run 段(对齐现有 sana/ltx/cosmos 格式)

## Agent workflow 文档(单独一份)
- [ ] `docs/agent-workflow.md`:workflow_lite (master+executor) · symposium skills · LPIPS+Gemini 质量门 · fanout/candidate-manifest · per-technique executors
