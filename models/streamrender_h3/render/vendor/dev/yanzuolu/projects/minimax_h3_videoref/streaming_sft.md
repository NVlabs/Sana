# VideoRef streaming SFT

`MiniMaxH3VideoRefStreamingSFT` 使用完整双向窗口训练音视频生成，窗口包含文本、配对 reference、已生成历史和当前 noisy target。默认文本只走语言编码，可选择为每个窗口编码 Qwen visual context。reference 按配对的全局视频 latent 索引与 target 对齐，允许各自使用不同空间分辨率。

窗口计划、双向 forward、SFT 目标、流式状态和音视频输出复用 `minimax_h3` 的共享 streaming 实现。VideoRef 保留配对媒体编码及 reference validation 入口，原配置与实时接口继续可用。纯文本条件入口见 [`minimax_h3/streaming_sft.md`](../minimax_h3/streaming_sft.md)。

## 配置入口

以下字段接入现有 `DiffusionFinetuning` 配置。数据源、分辨率、帧数、模型权重、优化器及 diffusion 节点仍按项目已有接口配置。

```yaml
data:
  module: dev.yanzuolu.projects.minimax_h3_videoref.data.video_ref_streaming
  class_name: VideoRefStreamingRawT2AVDataset
  args:
    sink_size: 1
    sink_switch_at: 1
    sink_size_after_switch: null
    window_size: 3
    chunk_size: 1
    merge_single_frame_units: false
    audio_lookahead_latents: 17

meta_model:
  module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_sft
  class_name: MiniMaxH3VideoRefStreamingSFT
  bootstrap_probability: 0.5
  guidance_scale: 1.0
  negative_loss_weight: 0.0
  separate_reference_rope: true
  fixed_window_rope: false
  keep_sink_reference: true
  keep_negative_reference: true
  qwen_visual_context: false
  selective_video_encoding: false
  loss_prediction_type: x0
  loss_weighting: uniform
  audio_loss_weight: 1.0
  bootstrap_loss_weight: 1.0
  first_frame_loss_weight: null
  continuation_loss_weight: 1.0

models:
  backbone:
    module: dev.yanzuolu.projects.minimax_h3.modeling.build
    class_name: MiniMaxH3X0DiTSP
```

S、W、C 分别为 `sink_size`、`window_size`、`chunk_size`，单位为窗口 unit。默认一个 unit 对应一个原生视频 latent，保持原行为。S、W 可以为零，C 必须为正整数。首块生成最多 W+C 个 unit，后续每次生成最多 C 个。续块保留已经生成的前 S 个和最近 W 个历史 unit，按全局原生 latent 索引合并去重，再接当前新帧。历史不足 S 时只取已有帧，不读取未来 GT。首块或末块可以短于指定长度，不复制历史补齐。

`data.args.merge_single_frame_units` 默认 `false`。设为 `true` 时，原生 H3 每周期的 1 RGB latent 与其后 4 RGB latent 组成一个 unit，unit 的原生 latent 跨度依次为 `(2,1,1,1)`，沿完整时间轴重复。unit 边界为 `0,2,3,4,5,7,8,9,10…`，不会每轮重新开始周期。S、W、C 和 `sink_size_after_switch` 按 unit 计，音频 L、R 仍按 40 Hz audio latent 计。原生 VAE 和普通 TAE 的 H3 时序描述符支持此模式，连续 `MiniMaxH3StreamingTAE` 开启时明确报错。

例如 W6/C1 的首块为 7 个 unit，即 9 个原生 latent、30 RGB。后续 C1 的原生结束边界依次为 `10,12,13,14,15…`，本轮新增 RGB 分别为 `4,5,4,4,4…`。unit 只改变选窗分组，reference 和 target 仍按各自完整的原生 latent 编码，不平均或合并其通道，也不修改 codec 时间映射。

`data.args.sink_switch_at` 为从 1 开始的续写块编号，不含首块，默认 1。`sink_size_after_switch` 默认 `null`，表示沿用原 S，整数值要求 `0 <= sink_size_after_switch <= sink_size`。例如 `sink_switch_at: 8`、`sink_size_after_switch: 1` 表示前 7 个续块保留原 S，第 8 个起保留 S=1，也可以缩减到 0。首块长度和续块起点不受影响。每次当前块去噪完成后，状态按下一块需要的 sink 裁剪 target、音频和保留 sink 的 reference 历史。

L 为 `data.args.audio_lookahead_latents`，控制音频 decoder 的连续已提交左缓存，默认 17。R 为可选的 `data.args.audio_right_lookahead_latents`，控制 DiT 右侧预测范围、保留的未来音频、noisy mask、窗口 token 预算和 decoder 右上下文。两者单位都是 40 Hz 音频 latent，接受非负整数。R 省略或为 `null` 时使用同一有效 policy 的 L，旧配置仍得到 L=R 的行为。L 不改变 DiT 的 S/W 历史选择。

以下配置保留 17 个 decoder 左历史 latent，同时关闭右侧预测和右上下文。

```yaml
data:
  args:
    audio_lookahead_latents: 17
    audio_right_lookahead_latents: 0
```

每个窗口在 S/W/C 对应音频之后带最多 R 个右侧 latent，已知语料末尾截断到真实音频长度。音频的生成进度领先于当前视频的发布进度，已生成右侧音频保留原值，在后续窗口作为 clean 条件使用。只有尚未生成的右端新增部分加噪并参与音频 loss。R 不改变视频窗口大小。

首块从零生成当前音频及右侧 R 个 latent。续写训练用 GT 模拟此前已生成的缓存，旧生成边界为 `min(audio_boundary(video_start) + R, total_audio_latents)`，在线 step 则使用实际缓存边界。输入仍仅选定 S/W/C/R，不随累计生成时长增长。当前待发布音频通常已经生成，因此提交 mask 与 noisy mask 独立。允许当前块有音频输出但没有新增 noisy audio。

例如连续 TAE、S=2、W=2、C=1、R=17，且音频总长足够。

| 窗口 | 当前提交音频区间 | 输入音频数（每声道） | 保留 clean 数 | 新生成区间 |
| --- | --- | --- | --- | --- |
| 首块 3 个视频 latent | `[0,15)` | 32 | 0 | `[0,32)` |
| 下一块 | `[15,22)` | 39 | 32 | `[32,39)` |
| 再下一块 | `[22,29)` | 46 | 39 | `[39,46)` |

数据集读取完整配对 24 FPS 媒体，按所有合法窗口的最大实际 token 数打包，预算包含 R，不包含 decoder 左缓存 L。SP 同步后，meta 才按 `bootstrap_probability` 抽首块或续写窗口。没有续写窗口的短 clip 总是使用首块。各 sample 独立抽一对配对 video/audio timestep，同一个 sample 的 noisy 视频和新增 noisy audio 使用各自模态的 timestep。保留的音频输入原值作为 clean 条件。

## RoPE

`meta_model.separate_reference_rope` 在 streaming SFT 中默认为 `true`。`fixed_window_rope` 默认为 `false`，此时采用全局时间位置。令 T 为当前分支的文本长度，p(i) 为 codec 声明的视频全局时间位置。reference 使用 `T + p(i)`，target video 使用 `T + p(stop) + p(i)`，target audio 使用 `T + p(stop) + j`，其中 j 是全局 40 Hz 音频索引，stop 是当前窗口生成到的全局 video latent 结束索引，不含该索引。

设为 `false` 时，reference 与 target video 共用 `T + p(i)`，target audio 同步使用 `T + j`。全局索引及 sink/window 之间的时间间隙保留。只有 reference 和 target 空间几何相同时，完整 `(t,h,w)` 坐标才共位。空间分辨率不同时仅时间共位，空间坐标仍按各自网格计算。

上述 `separate_reference_rope` 只改变坐标，不改变 reference 选择、打包预算、双向 attention、噪声、timestep 或监督选择。

`meta_model.keep_sink_reference` 独立控制 reference 内容选择，默认为 `true`。首块始终保留全部 reference。续写为 `true` 时 reference 与去重后的 target 帧相同，为 `false` 时只保留最近 W 个 unit 和当前新帧，重叠于 sink 的 recent 帧仍保留。target 最多包含 S_eff+W+C 个 unit，展开为实际原生 latent 行，历史不足、重叠和短尾会减少实际数量，其中 S_eff 为当前有效 sink 大小。此开关适用于三种 RoPE 模式，不改变音频选择。实际 plan 和 SP 行数按 reference 的独立选择计算，数据集按所有合法窗口计算预算。

`meta_model.fixed_window_rope: true` 启用固定窗口模式，要求 VideoRef 同时保持 `separate_reference_rope: true`，两者冲突时抛出 `ValueError`。首块按真实 stop 保留分离位置，短首块也按此处理。续写使用容量 K=S_eff+W+C 的模板，K 与首块长度 W+C 含义不同。即使设置 `keep_sink_reference: false` 删除 reference sink tokens，也不把剩余 reference 从零重新编号。

令 B(q) 为 q 个 unit 对应的原生 latent 边界，未合并时 B(q)=q。令 u 为当前窗口起点的 unit 索引，s=B(S_eff)，b=B(max(0,u-W))，r=max(s,b)，K=S_eff+W+C，Δ=p(B(K))。续写的位置为

```text
reference sink   T + p(i)
reference WC     T + p(s) + p(i) - p(r)
target sink      T + Δ + p(i)
target WC        T + Δ + p(s) + p(i) - p(r)
audio sink       T + Δ + j
audio WC + R     T + Δ + p(s) + j - p(r)
```

sink 只包括 `i < min(s,start)` 的已有视频历史，audio sink 对应 `j < ceil(p(min(s,start)))`。sink 与 recent 重叠的帧只出现一次，此时 WC 平移量为 0，视频、音频和 reference 的联合窗口保留原生位置及内部时间间距。recent 与 sink 分离后使用 `p(s)-p(b)` 的平移。本轮新帧即使索引小于 s 也走 WC 坐标，不充当已有 sink。reference 与 target 的对应行使用相同角色，保持 Δ 偏移。

连续 TAE、S2/W2/C1 且 sink 不缩减时，sink 与 recent 不再重叠后的 reference sink 为 `(0, 1⅔)`，reference WC 为 `(8⅓, 15, 21⅔)`，target sink 为 `(28⅓, 30)`，target WC 为 `(36⅔, 43⅓, 50)`，以上省略共同文本偏移。历史不足或重叠时只使用实际已有行，不补齐模板。sink 作为固定 anchor，仅在它与 WC 分离时压缩两者间的时间距离。原生 H3 的周期 cadence，以及 S=0 时首块特殊首帧与续写的差异，仍按实际 p 计算，不强行改成均匀间隔。

音频保留全局 j 与视频边界间的 40 Hz 分数相位。固定窗口只改变模型位置，不改变真实音频 source、生成缓存、提交范围或连续 decoder 输入。sink 切换时 p(B(S_eff)) 和 Δ 一起更新。短尾沿用当前有效 sink 对应的完整 K 模板，不因本次 C 变短而缩小 Δ。

三个模式均作用于训练、validation、实时 `start_stream`/`step` 和 CFG 正负分支。默认正负分支共享同一 reference 内容及其噪声，只有文本条件不同。`meta_model.keep_negative_reference: false` 时负分支不含 reference 行，见「Loss 与 CFG」。

## 每窗口 Qwen visual context

`meta_model.qwen_visual_context` 默认 `false`，关闭时保持原有文本编码、选窗、RNG 和 CFG 行为。开启时必须显式提供本地 `meta_model.qwen_processor_path`。meta 将这两个字段及 `fixed_window_rope`、`keep_sink_reference` 注入 dataset，用同一个 presentation 规则计算准确 token 预算。

```yaml
meta_model:
  qwen_visual_context: true
  qwen_processor_path: /path/to/local/qwen_processor
```

开启后，训练在 prepare 内先按 shape 和 policy 选定窗口，再编码视频和完整音频，随后只用本 rank 原始 reference RGB 中 `plan.reference_video_indices` 对应的帧构造 presentation。每个 rank 做一次合批 Qwen forward，将实际 prefix embeddings、tags、完整媒体负条件及更新长度的计划一并放入 encoded payload。SP 只同步这些准备好的结果，随后复用同一计划并采样配对 timestep 和噪声，不再调用 Qwen 或 VAE。每个 source rank 和 sample 使用独立的窗口 RNG，相同 prepare seed 可复现。

`keep_sink_reference: true` 时 VAE 与 Qwen 都读取去重后的 S/W/C reference，为 `false` 时两者都只读取 W/C，target sink 保留。RGB 范围由同一 codec 的 decode timeline 从实际 reference latent 索引展开。带缺口的连续片段分别按 12 帧步长抽帧，各自补齐偶数个采样槽。最后一个连续片段的末槽固定使用其最后一帧，确保当前 C 至少有一帧进入 Qwen，不增加采样槽或 vision token 数，也不把缺口两侧混成一个 temporal patch。只有一帧的片段自然重复该帧。首块、训练和推理使用同一规则，presentation 与预算共用抽帧函数。例如 W6/C1 合并 unit 的首块采样 `0,12,24,29`，S0 第一续块从 RGB `[5,34)` 采样 `5,17,29,33`，其中末帧属于当前 C。

Qwen 时间戳跟随 reference 的 DiT 时间坐标。sink 不平移，WC 使用同一个 40 Hz 窗口平移量后换算成秒，保留片段内部的真实时间差。每个 temporal pair 的时间戳由最终选中的两个真实 RGB 帧时间取平均。`fixed_window_rope` 关闭时使用原始时间，开启时使用固定窗口位置。`keep_sink_reference: false` 不会让 WC 从零重新编号。Qwen 不加入文本行数或 target 域的 Δ 偏移。

Qwen visual tokens 使用 presentation 给出的 VIDEO tags，但仍作为条件 embedding 放在 `text_pos`，不作为额外 VAE 图像输入。完整条件 prefix 的每行 timestep 都跟随当前 video timestep。VAE reference 的 near-clean 噪声约定保持不变。

原生 Qwen 的媒体 labels、timestamps 和 vision 都在 caption 之前。固定 causal RoPE 下，正分支编码后的完整媒体前缀与空语言编码相同，因此 CFG 直接按 `media_prefix_length` 复用这段 hidden。默认负分支保留全部媒体前缀及同一 VAE reference，不按 VIDEO tag 筛行。`keep_negative_reference: false` 时负分支的前缀为空，也不含 VAE reference。两支各用自己的实际 prefix 长度重建位置，共享采样 grid、AV noise 和 reference 扰动。

推理在每个 denoise 窗口进入 solver 前重新准备一次 Qwen context，同一窗口所有 solver steps 复用。不同 sample 独立保留原始 RGB 历史，裁剪范围与各自 reference latent 历史一致。提前结束的分布式组仍用真实 bootstrap RGB 执行 dummy Qwen，保持视觉塔与 DiT 的 collective 次序。

## Loss 与 CFG

backbone 与 diffusion schedule 均保持 x0 接口。`loss_prediction_type: x0` 直接拟合 clean latent，`flow_v` 在正噪声时间步转换为原生 data-ward velocity 后拟合，要求线性插值 schedule。每个 sample 在各模态内先取均值，再对 sample 取均值。`audio_loss_weight` 控制两模态组合。

`meta_model.bootstrap_loss_weight` 和 `continuation_loss_weight` 默认 1.0，接受有限非负值。`first_frame_loss_weight` 省略或设为 `null` 时继承 bootstrap 权重，不单独改变首个 latent 的权重。显式数值同样必须有限非负，包括 0 和 1，分别覆盖首块视频和音频各自全局索引 0 的权重，与 bootstrap 不相乘。续写块实际监督的 noisy 音视频 latent 使用 continuation 权重，保留的第 0 个 sink 不触发 first 加权。历史、reference 和保留音频不参与 loss。音频在上述加权后统一乘 `audio_loss_weight`。

例如 bootstrap=4、audio=0.1 且未配置 first 或设为 `null` 时，首块视频权重为 `[4, 4, 4, ...]`，音频有效权重为 `[0.4, 0.4, 0.4, ...]`。显式设置 first=10 后，分别为 `[10, 4, 4, ...]` 和 `[1, 0.4, 0.4, ...]`。显式设置 first=1 后，分别为 `[1, 4, 4, ...]` 和 `[0.1, 0.4, 0.4, ...]`。continuation 默认 1 时，续写视频和音频有效权重分别为 1 和 0.1。设 continuation=2.5 后，分别为 2.5 和 0.25。

加权在 x0 或 flow_v 转换后进行，仍以各 sample 原始监督元素数取均值，不用权重和重新归一化。权重 0 会屏蔽对应误差，continuation=0 时续块仍计入原样本分母。视频合并 unit 不改变全局 latent 索引，显式 first 仅覆盖首块的原生视频 latent 0，partner 仍用 bootstrap。音频按自身的全局 latent 索引 0 应用同样规则。

训练的 `meta_model.guidance_scale = w` 采用下式，负分支停止梯度。

```text
prediction = (conditional + (w - 1) * stopgrad(captionless)) / w
```

`meta_model.keep_negative_reference` 默认 `true`，负分支仅移除语言，开启 Qwen visual context 时仍保留完整媒体前缀和同一 VAE reference。设为 `false` 时，负分支同时移除 reference 行和媒体前缀，只保留 target 音视频窗口，packing rows 相应减少。target 与音频行之间的相对位置不变，只是整体不再带文本长度和 reference 域的 Δ 偏移。

w 为 1 时不运行负分支。采样 CFG 独立使用 `guidance_scale` 参数或 `validation.guidance_scale`，按通常的 `unconditional + scale * (conditional - unconditional)` 组合，其负分支遵循同一个 `keep_negative_reference`。采样以空语言识别负分支，同一分支内所有 sample 必须一致，否则抛出 `ValueError`。

`meta_model.negative_loss_weight` 默认 `0.0`，负分支只作为停止梯度的锚点，w 为 1 时不运行。设为正数 λ 后，负分支带梯度前向，并按与主项相同的 x0 或 flow_v 转换和窗口加权计算自身预测的 MSE，以 λ 倍加入总 loss，记录为 `train/negative_video_loss` 与 `train/negative_audio_loss`，音频同样乘 `audio_loss_weight`。fitting 项仍使用停止梯度的负分支，λ 只负责把负分支校准到其残缺条件下的后验均值。λ 为正时即使 w 为 1 也运行负分支，此时等价于带条件 dropout 的普通训练。

### 多条件分支与顺序 fitting

共享 SFT 支持分别移除语言、reference、音视频历史及其组合，并对每个分支独立监督。配置和归约规则见 [共享 SFT 的多分支说明](../minimax_h3/streaming_sft.md#多分支监督与顺序-fitting)。以下配置的 fitting 路径是「无条件 → reference → reference + text → 完整条件」，同时独立训练仅缺 reference 或仅缺文字的分支。

```yaml
meta_model:
  guidance_branches:
    - name: unconditional
      drop: [text, reference, history]
      loss_weight: 0.1
    - name: reference_only
      drop: [text, history]
      loss_weight: 0.1
    - name: no_history
      drop: [history]
      loss_weight: 0.1
    - name: no_reference
      drop: [reference]
      loss_weight: 0.1
    - name: no_text
      drop: [text]
      loss_weight: 0.1
  guidance_fitting:
    stages:
      - branch: unconditional
        scale: 1.5  # Reference guidance.
      - branch: reference_only
        scale: 2.0  # Text guidance conditioned on reference.
      - branch: no_history
        scale: 1.5  # History guidance conditioned on reference and text.
```

`reference` 包含 VAE reference 和 Qwen 媒体条件。丢 reference 而保留 text 时，使用原 caption 单独编码的纯文本 hidden，不能截取已经读取视觉上下文的 caption hidden。额外纯文本 presentation 仅在需要该类分支时准备，与视觉 presentation 合批进行一次 Qwen 前向，结果在 PREPARE 阶段完成并随 SP 输入一起同步。丢 text 而保留 reference 时继续复用完整的 Qwen 媒体前缀。两者均丢时使用零行前缀，空 caption 也受支持。

`meta_model.reference_drop_keeps_picture: true` 让 picture 留在每个分支中，就像 t2va 中的 keyframe。丢 reference 只去掉 reference 行，picture 行及其 Qwen 上下文保留。丢 text 只去掉 caption。该选项要求配置 `guidance_branches`，且 Qwen 不读 reference 视频（`qwen_reference_video: false`）。四个状态的 Qwen 输入因此如下，∅ 与只有 reference 时是 picture 媒体前缀，只有 text 与完整条件时是 picture 加 caption，都来自同一次 Qwen 前向，不再需要纯文本 presentation。packing 行数按实际去掉的 reference 行与 caption 行扣减。默认 false，行为与之前逐位相同。

Ref2VA SFT 的分支还可以丢弃 `picture`，使 picture 成为独立的 guidance 条件。任一分支丢弃 `picture` 时自动开启 `reference_drop_keeps_picture`，显式设为 false 会报错。丢 picture 时去掉 picture 行和 Qwen 中的 Picture 1。保留 text 时改读 caption 的纯文本 presentation，tags 全为 1，text 也丢时使用零行前缀。没有 picture 的样本不计入该条件，按其余条件编译 fitting。像素条件与 semantic keyframe host 不接受 `picture`。以下配置沿 `∅ --1.5--> T --2.0--> IT --3.0--> ITS` fitting，I 为 picture，S 为 semantic reference。

```yaml
meta_model:
  reference_drop_keeps_picture: true
  guidance_branches:
    - {name: "null", drop: [text, reference, picture], loss_weight: 0.1}
    - {name: text_only, drop: [reference, picture], loss_weight: 0.5}
    - {name: image_text, drop: [reference], loss_weight: 0.5}
  guidance_fitting:
    anchors: guided
    stages:
      - {branch: "null", scale: 1.5}
      - {branch: text_only, scale: 2.0}
      - {branch: image_text, scale: 3.0}
```

`history` 只影响已经生成的 target AV 条件，不改变 reference 选择。视频的 S/W 和对应已生成音频一起裁去，包含保留的 audio lookahead，当前 noisy 行及其监督保持不变。首块的 history-only 分支自动无效。若只需历史 fitting，可只声明 `no_history`，并让 `stages` 仅引用这一项，text 和 reference 全程保留。

分支定义与拟合路径分开，因此未参加 fitting 的分支也可通过独立的 `loss_weight` 训练。省略新配置时，旧 trial 的条件、梯度、随机数和编码调用路径保持不变。新配置不会增加 `validation.guidance_scale: 1.0` 下的推理分支。

## Codec

`meta_model.selective_video_encoding` 默认 `false`。在 raw VideoRef SFT 中开启后，各 rank 在 prepare 内按 shape 和 policy 预选一次窗口，然后并行编码所需的 target/reference latent。窗口 RNG 从 prepare RNG 按 source rank 和 sample 索引独立派生，SP payload 携带最终计划，训练阶段直接复用。VAE posterior 使用 prepare RNG 派生的独立 target/reference seeds，编码处于 PREPARE phase。此开关和 `qwen_visual_context` 均关闭时保持原有选窗及随机数路径。

原生 `MiniMaxH3VideoVAE` 只执行覆盖选中 latent 的必要 temporal blocks，同时保持原边界处理和完整 posterior 噪声规则。普通 TAE 和连续 TAE 使用完整编码后按索引选择，保持通用语义，但不因此减少其 encoder 计算。音频始终整段编码。视频编码结果写入原生完整形状的 carrier，未编码位置为零，后续仅可读取已编码的计划索引。

该开关仅优化训练时不会使用的 VAE 块，不改变 validation、live 输出或 decoder 的接口。Qwen visual context 仍在已选窗口上编码一次。已有 latent 的 T2AV/TSCD 输入不能启用该选项。FSDP 分片的 video VAE 也不支持此选项，因为各 rank 所需 temporal block 数可能不同，不能保证 encoder collective 次序。

meta 从 `models.video_vae` 的类属性取得 `VideoTemporalMapping` 并传给数据集，运行时 codec 必须声明相同映射。S/W/C 与 codec 的时间压缩周期无关。

可选的 `models.video_decoder` 是独立的媒体解码模型节点，未配置时继续使用 `video_vae`。它按标准模型节点配置类、权重和运行精度。训练和 validation 的 target/reference 编码始终使用 `video_vae`，实时输出及 validation 回放则统一使用选中的 decoder，包括最终 flush。两者必须使用同一 normalized latent 空间、时间映射、latent 通道数和空间缩放比例。配置阶段检查时间映射，运行时还检查通道与空间比例。

原生 `MiniMaxH3VideoVAE` 可以负责编码，普通 `MiniMaxH3TAE` 负责媒体解码。传入适配器的模型输出保持 normalized H3 latent，只应用 decoder 自己的 `latents_mean` 与 `latents_std` 一次。普通 TAE 使用零均值和单位标准差，不先套用原生 encoder 的反归一化。

| 所选视频 decoder | 时间映射 | 媒体输出能力 |
| --- | --- | --- |
| `MiniMaxH3VideoVAE` | 原生 H3 周期 | 当前兼容路径在流结束时解码完整视频 |
| `MiniMaxH3TAE` | 原生 H3 周期 | 当前兼容路径在流结束时解码完整视频 |
| `MiniMaxH3StreamingTAE` | 编码每 4 RGB 一个 latent，解码首 latent 1 RGB、随后各 4 RGB | `create_stream()` 保存 codec 状态并逐步解码 |

完整训练 clip 的帧数同时受时间描述符和实际编码器配置约束。H3 时间描述符允许 `17n` 或 `17n+5`，普通 TAE 支持这两类。当前原生 VAE 的 `token_drop: 3`、`frame_overlap: 5` 配置在正常 `encode_videos` 路径中要求 `17n+5`，不能仅凭描述符就使用 `17n` 帧。107 RGB 对应 32 个原生 latent，即 `5n+2`，开启合并后为 25 个 unit。连续 TAE 的 target 使用 `1+4n`，例如 105 或 121 帧，使输入帧数与最终解码帧数一致。

前两类的兼容支持不代表逐 latent 发布画面。连续 TAE 也不能把 decoder 的首帧节奏当成 encoder 的输入可用性，其首 latent 需要收齐 4 RGB。

音频通过 BigVGAN 对最多 L 个连续已提交左历史 latent、当前音频和最多 R 个已生成右侧 latent 一起解码，只保留当前 PCM。首块没有左历史，后续使用 `min(实际已有左历史数, L)`，不为凑足 L 重复 latent 或等待。decoder 的连续缓存独立于 DiT 的带间隔 S/W 选择，不把带间隔的历史拼成连续声音。相邻调用重合的音频 latent 必须保持原值，只追加新生成的尾部。默认 L=R=17 是可配置的经验值。当前 32 kHz BigVGAN 按架构推导并向外取整的完整上下文覆盖界为左右各 29 个 audio latent。

感受野由架构决定，L、R 分别控制实际使用的左右上下文长度。R=0 时仍保留配置的左历史，L=0 时不保留 decoder 左历史。`StreamingAVDecoder.audio_lookahead_seconds` 返回 R 对应的秒数，即 R/40。分块与整段解码的数值差异还受 cuDNN TF32、卷积输入形状和运行精度影响，覆盖完整感受野也不构成位级等价保证。

音频发布按累计 RGB 帧数的 `floor(frames * 32000 / 24)` 边界限发。音频 latent 使用整数 ceil 边界，因此一个对应视频区间可能解出少量额外 PCM，这部分顺接到下一块。最终 flush 保留真实完整音频长度，不补静音。原生 H3 VAE 和普通 TAE 仅在最终 flush 解码完整视频，音频仍在每次 push 时分块解码，PCM 暂存到最终 flush 与视频一起发布。

已知总长度时，生成右边界提前截断到真实音频末尾。未知总长度时，最后一步通过 `is_last` 通知结束，并根据真实视频终点确定音频长度。终点外尚未提交的右侧缓存直接裁掉，终点内已经生成的音频保持原值并全部发布，不重新生成。flush 不会把终点外的预测当成额外声音输出。

实时输出与 validation 使用同一生成序列和 codec 调用方式，并按每个 sample 的 policy 传入相同的 L、R。分段音频解码与整段解码的数值误差仍取决于 L、R、权重和运行精度，保留未来 latent 本身不构成任意上下文长度下的整段等价保证。

## 使用接口

`start_stream(...)` 创建每个 sample 独立的 `StreamingState`，无需观察未来 reference。默认传入预编码文本、target/reference 空间几何、音频通道数、独立 RNG 和窗口配置。几何格式是 `(channels, latent_height, latent_width)`。可通过 `audio_lengths` 提供已知的总音频长度。

```python
states = meta.start_stream(
    prompt_embeds=prompt_embeds,
    video_geometries=video_geometries,
    reference_geometries=reference_geometries,
    audio_channels=[32] * len(prompt_embeds),
    rngs=sample_rngs,
    streaming_configs=window_configs,
    audio_lengths=known_audio_lengths,
    guidance_scale=3.0,
)

event = meta.step(
    models["backbone"],
    active_states,
    new_reference_latents,
    is_last=last_flags,
)
```

`step(...)` 只接收本次新到达的 reference latents，首块对应 W+C units，随后对应 C units，实际传入数量按当前位置的原生 latent 边界计算，仅终止块允许缩短。`video_counts`、`video_lengths`、shape、事件 indices 和 start/stop 始终以原生 latent 计，不改成 unit 数。state 保留下一块所需的去重 target sink/recent 历史和连续的尚未提交音频缓存。reference 历史单独记录索引，`keep_sink_reference: true` 时保留下一块所需的 sink/recent，为 `false` 时只保留最近 W 个 unit，因此 reference 和 target 历史长度可以不同。返回的 `sample_indices` 指回最初的 sample 顺序，`video` 和 `audio` 是本次提交的 latents，`audio_lookahead` 是保留到后续调用的已生成右侧音频，`audio_new_indices` 标出本轮新增预测。对应 indices 均为全局时间轴索引。

开启 Qwen visual context 后，`start_stream` 必须接收原始 `prompts`，可以省略 `prompt_embeds`。每次 `step` 还要提供 `models` 和本次新增的原始 `reference_pixels`，形状为 `[3, RGB_frames, H, W]`、取值为 `[0,1]`。若本次原生 latent 区间为 `[start,stop)`，其 RGB 帧数必须恰为 `decode_timeline.boundary(stop) - decode_timeline.boundary(start)`。不接受由生成 latent 或 TAE decode 重建的替代参考图像。

```python
states = meta.start_stream(
    prompts=original_prompts,
    video_geometries=video_geometries,
    reference_geometries=reference_geometries,
    audio_channels=[32] * len(original_prompts),
    rngs=sample_rngs,
    streaming_configs=window_configs,
    guidance_scale=3.0,
)
event = meta.step(
    models["backbone"], active_states, new_reference_latents,
    reference_pixels=new_reference_rgb, models=models, is_last=last_flags,
)
```

完整时间轴的 `StreamingBatch` 在此模式下还需携带 `prompts` 与 CPU `reference_pixels`。`stream(models, ...)` 自动传递模型并切出每次新增 RGB，直接调用 `stream_latents` 时显式传 `models=models`。validation 采用同一路窗口准备和输出回放。

validation 配置、单个 validation request 和 `start_stream` 的 `streaming_configs` 可以只覆盖部分窗口参数，未指定项继承训练 policy。`sink_size_after_switch: null` 按合并后的 `sink_size` 解释。`audio_right_lookahead_latents` 在这些覆盖合并后解析，值为 `null` 时跟随有效 L。例如训练未显式设 R，validation 只把 L 改为 5 时，验证使用 L=R=5。训练显式设 R=0 时，仅修改 L 不会开启右侧预测。每个 sample 根据自己的全局视频起点计算续块编号。

调用者可以把单个 sample 的输出交给独立的 `StreamingAVDecoder`，同时传入该 sample 的 L、R。未指定的 R 可以保留为 `None`。

```python
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.streaming_output import StreamingAVDecoder

decoder = StreamingAVDecoder(
    models.get("video_decoder", models["video_vae"]), models["audio_vae"],
    audio_lookahead_latents=window_config.get("audio_lookahead_latents", 17),
    audio_right_lookahead_latents=window_config.get("audio_right_lookahead_latents"),
)
media = decoder.push(
    video=event.video[local_index],
    audio=event.audio[local_index],
    audio_lookahead=event.audio_lookahead[local_index],
)
if event.is_last[local_index]:
    tail = decoder.flush()
```

`StreamingAVChunk.video` 为 `[1,3,F,H,W]` RGB，`audio` 为 `[2,samples]` PCM，没有新增输出时为 `None`。`video_start` 和 `audio_start` 是已经发布的帧、采样点偏移。`audio_generated_stop` 和 `audio_committed_stop` 是音频 latent 单位的预测、提交边界。不要把同一个事件中两种单位的长度直接对应。

已准备完整 reference 时间轴时，`stream(models, inputs, rngs, guidance_scale=...)` 自动循环 step 并维护每个 sample 的 codec 状态。每个事件的 `decoded` 保留 push 输出，终止时还保留 flush 输出。`stream_latents(...)` 只返回 latents。不同 sample 可以有不同几何和长度，完成后退出 active 集合。

分布式 `step` 必须由参与进程以一致的模型调用顺序执行。完整时间轴的 `stream` 路径会为已结束的组补齐独立 dummy forward，使不同组的长度不会改变 FSDP collective 次序。每个窗口重算双向 DiT，不复用跨窗口 transformer KV。

validation 保存每步的当前音频及保留的右侧音频，以及每个 sample 的 L、R，并通过同一个输出适配器回放。它不会把最终提交的 audio latents 改成另一次整段解码。

每个 validation entry 设置 `save_latent: true` 可改为导出采样结束后的完整音视频 latent，配置和基本文件格式见 [共享导出接口](../minimax_h3/streaming_sft.md#validation-latent-export)。可选的 `latent_dir` 指定固定 corpus 目录。导出的是 normalized CPU BF16 `video`、`audio`，同时保存 `prompt` 和 `prompt_idx`，不经过媒体解码。VideoRef streaming 还保存完整的 clean `reference_video` latent、原始 `reference_video_path`、reference 几何、24 FPS 时间原点及 `video_temporal_mapping`，供下述 latent 数据入口使用。reference 与 Qwen 条件仍按每个窗口准备。默认关闭，不改变原有视频输出。

## Latent corpus

`VideoRefStreamingLatentT2AVDataset` 读取生成时直接导出的归一化音视频及 reference latent。SFT meta 继续使用 `MiniMaxH3VideoRefStreamingSFT`，其 prepare 根据输入字段选择读取缓存或编码 raw 媒体，后续训练路径共享。

替换数据入口时删除原来的 `sources`，设置 `latent_dir`，其余几何、tokenizer、S/W/C、sink 切换、音频 L/R、packing 和 text dropout 配置继续沿用。

```yaml
data:
  module: dev.yanzuolu.projects.minimax_h3_videoref.data.video_ref_streaming_latent
  class_name: VideoRefStreamingLatentT2AVDataset
  args:
    latent_dir: runs/ref2va_latent_corpus/validation/backbone
    # 保留原有 num_frames、height、width、reference_height、reference_width 等参数。
meta_model:
  module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_sft
  class_name: MiniMaxH3VideoRefStreamingSFT
  selective_video_encoding: false
```

目录递归读取 `prompt*.pt`，不再使用原索引的 ratio 筛选。应指向一个 checkpoint、一个 validation variant 的 corpus，避免把 backbone、EMA 或不同 CFG 输出混在一起。恢复 dataloader 时 corpus 文件集合应保持不变。该入口有独立的 worker 状态格式，不能直接续用 raw dataset 的 dataloader checkpoint。

每个文件必须包含 `video`、`audio`、`reference_video` 和 `prompt`。三路 latent 都在 diffusion 的归一化空间，不再次过 VAE，也不重复做 mean/std 变换。loader 核对形状及导出时的帧数、FPS 和时间映射。完整 clips 留到 meta 中随机抽取首块或续写窗口，合并 unit、短尾和 reference sink 策略继续共用同一个 planner。packing 预算仍包含实际 reference 和 Qwen 前缀，支持每个 pack 多样本及 worker 断点恢复。

开启 `qwen_visual_context` 时还需 `reference_video_path`。路径可以是绝对路径，也可以相对 `.pt` 文件所在目录。loader 只读取 reference 图像，不读取 target RGB 或音频波形。streaming 导出的 `reference_start_frame` 使用先统一 FPS 再裁剪的帧网格，原生双向导出的 `reference_start_time_seconds` 复用生成时先 seek 再解码的规则。选好随机窗口后才编码 Qwen，条件和窗口在 SP 输入同步前准备完毕。关闭 visual context 时不读取 reference 媒体。

原生 `MiniMaxH3Ref2VABase` 和 game factory 的 `save_latent` 对单个视频 reference、无参考音轨、reference 与 target 帧数一致的请求也导出上述配对字段。缓存的是 VAE posterior 编码后、0.001 条件加噪前的 clean reference，训练时仍按现有方式进行条件加噪。其它原生多参考、图片或音频条件保持基础 AV latent 导出，它们不属于此同步单视频训练入口的输入格式。

模型的 `video_vae` 配置仍声明 codec 时间映射，并供 validation 编码 reference 或解码媒体使用。使用 latent 数据不会改动验证生成流程、优化器、时间步采样、首块/续写比例、CFG fitting 或 loss 权重。

## Resampling forcing 链式训练

`MiniMaxH3VideoRefStreamingRF` 继承 `MiniMaxH3VideoRefStreamingSFT`，把窗口的 GT 历史换成模型自己的一步重建。teacher forcing 每次迭代为每个 clip 随机抽一个窗口，sink 和 recent 历史行读取语料 latent。resampling forcing 把一个 pack 的所有 clip 按窗口顺序连续训练，第 w 次迭代训练每个 clip 的第 w 个窗口，历史行读取此前窗口写回的重建值，监督目标仍是语料 GT。loss、窗口权重、`flow_v` 转换、负分支和 fitting 与父类相同，validation 和流式接口也沿用父类。

每次迭代对同一窗口执行两次 DiT 前向。第一次是 resample 前向，在 `eval()` 和 `no_grad` 下用该 clip 固定的 timestep 对 `t_s` 给当前块加噪，噪声与 reference 扰动来自独立派生的 RNG，clean 与 noisy 行的划分与训练前向一致，输出的 x0 就是当前块的一步重建。第二次是父类原有的训练前向，含 captionless 负分支和 fitting。训练前向结束后才把重建写回工作副本，因此本次迭代的 target 和 noisy 输入都不会读到新值，下一窗口把它们当作历史读取。reference latents 不参与重建，始终是 GT。最后一个窗口的 resample 前向照常执行，结果丢弃。

每个 rank 在进程内存中按 `chain_id` 维护 pool。head 迭代初始化条目，内容包括设备上 float32 的 video/audio 工作副本、reference latents、`chain_length` 个窗口的 plan、每个 clip 的一对 `t_s`、packing 预算、shape、policy 和条件状态。工作副本初值为 GT。`t_s` 由 head 的 prepare RNG 按 clip 序号派生 seed 抽取，整条 chain 复用。开启 `qwen_visual_context` 时，head 用 `reference_video_pixels` 一次构造所有窗口的 presentation 并保留在 CPU，之后不再保留像素，每次迭代只编码当前窗口的 presentation，负前缀、`media_prefix_lengths` 和纯文本 hidden 的处理与父类相同。关闭时 head 只保留 prompt 的 token ids，每次迭代重新调用一次 `_encode_prompts`。text encoder 可能是 FSDP 分片的，各 rank 处于不同的 chain 相位，只在 head 编码会让它们的 collective 错开。chain 训练完最后一个窗口后删除条目，相同 `chain_id` 的 head 覆盖已有条目，light batch 找不到条目时抛出 `RuntimeError`。

checkpoint 保存未完成的 chain。meta 构造时把自己注册为 persistence plugin，每次保存 checkpoint 之前，各 rank 把 `resampling_chains/rank_<rank>.pt` 写进该 checkpoint 目录，逐条记录 chain 的进度、head 的 prepare seed 和工作副本。plan、`t_s`、picture latent、Qwen presentation 等其余状态不保存。meta 在代码中把 `data.args.resume_mid_chain` 设为 true，不需要 trial 配置。续跑时数据入口把每条中断 chain 的完整 pack 作为 head 重新发出，保留中断时的 `chain_index` 和 `chain_length`。meta 用保存的 seed 从这个 head 重建条目，得到相同的 plan、`t_s`、picture 和条件，再载入保存的工作副本，从断点窗口接着训练。文件按内存映射读取，head 不再出现的 chain 不占内存。checkpoint 中没有该文件，或保存时的 world size、训练 UP 或 dataloader worker 数与当前不同，就记录 warning，续跑 head 对应的 chain 从窗口 0 重新开始并截短到剩余窗口数，与此前的续跑方式一致。旧 checkpoint 的 worker 状态不含 chain 起点，数据入口本身就截短重启，从这类 checkpoint 续跑的那一段结束时保存的 checkpoint 已带 chain 状态，之后的续跑从断点接上。

配套数据入口 `VideoRefStreamingLatentChainDataset` 把每个 pack 展开成一条 chain，按窗口顺序连续输出。head batch 是普通 collated pack 加四个标量字段 `chain_id`、`chain_index`、`chain_length` 和 `chain_windows`。head 的 `chain_index` 为 0，从 checkpoint 续跑的 head 除外，`chain_windows` 是 clip 的窗口总数 L，`chain_length` 是本条 chain 训练的窗口数，取 clip 窗口的前缀。之后的 light batch 只含这四个字段，`chain_index` 取 `1..chain_length-1`。meta 核对 `chain_windows` 等于每个 clip 的窗口数、`chain_index` 等于 pool 中的下一个窗口、`chain_length` 与 head 一致。`chain_phase_stagger` 默认开启，按 rank 所属的 SP sync group 和 worker 决定相位，把每个流的第一条 chain 截短为 `L - phase` 个窗口的前缀，使各 rank 落在不同的 chain 相位，截掉的尾部这次不训练，之后的 chain 训练完整的 L 个窗口。

每次迭代每个 rank 都执行同样的 collective，即一次 text encoder 调用，开启 visual context 时编码当前窗口的 presentation，关闭时编码 prompt，加一次 no_grad resample 前向、一次训练前向、按配置的负分支前向，以及 engine 的一次 backward，与当前窗口序号无关，因此 rank 之间的 chain 相位可以不同。

该 meta 只接受 latent 语料，`selective_video_encoding` 抛出 `ValueError`。工作副本只保存在加载该 chain 的 rank 进程内。`engine.up_size.train` 大于 1 时，SP 组每次迭代训练一个源 rank 的窗口，SP 输入广播携带该窗口的工作副本、plan、条件和 `t_s`，组内各 rank 算出相同的重建结果，只有源 rank 写回。每个窗口在一条 chain 中恰好训练一次，`meta_model` 出现 `bootstrap_probability` 时抛出 `ValueError`。

pool、head 规划、持久化、SP 广播和写回由 `StreamingChainPoolMixin` 实现，resample 前向和 `t_s` 由其上的 `StreamingResamplingChainMixin` 提供，videoref 的窗口条件由 `VideoRefChainHostMixin` 提供，三者都在 `meta_models/minimax_h3_video_ref_streaming_rf.py`。下文的 DMD 复用同一个 pool，不带 resample 前向。

```yaml
meta_model:
  module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_rf
  class_name: MiniMaxH3VideoRefStreamingRF
  resample_timesteps:
    module: dev.yanzuolu.projects.minimax_h3.modeling.training_timesteps
    class_name: PairedLogitNormalTrainingTimesteps
    args:
      T: 1.0
      loc: 0.0
      scale: 1.0
      shift: 0.6
      audio_shift: 0.6
```

`resample_timesteps` 是必填节点，必须提供 `sample_pair`，不支持 `shift_schedule`，因为 `t_s` 按 clip 抽一对并在整条 chain 复用，没有可依附的窗口位置。训练 timestep 仍来自 `diffusion.training_timesteps`。

新增指标 `train/resample_video_timestep` 和 `train/resample_audio_timestep` 记录本 batch `t_s` 的均值。`train/resample_video_error` 和 `train/resample_audio_error` 记录当前块重建与 GT 的 MSE，先在 sample 内取均值再对 sample 平均，没有新增音频的 sample 计 0。`train/window_loss/<ww>` 按两位窗口序号记录本次迭代的总 loss。

## RF chain 上的 TSCD

`MiniMaxH3VideoRefStreamingTSCD` 同时继承 `MiniMaxH3VideoRefStreamingRF` 和 `MiniMaxH3StreamingTSCD`，在上述 chain 上训练 streaming TSCD，entry 使用 `TrajectorySegmentedConsistencyDistillation`。pool、head 规划、`t_s`、窗口条件、SP 广播和写回都与 RF 相同，窗口历史行是 student 对此前窗口的一步 x0 重建。

每次迭代先执行 RF 的 resample 前向，在 `eval()` 和 `no_grad` 下按 clip 固定的 `t_s` 对当前块加噪并预测 x0，噪声来自独立派生的 RNG，不改变 TSCD 的任何抽样。随后按父类依次执行 student 在网格时间 t 的前向、teacher 前向及到 t′ 的一步 DDIM、EMA 在 t′ 的前向和 loss。写回发生在 loss 之后，同一迭代内三个网络读取相同的历史，下一窗口再读取新值。写回的是 resample 重建而不是 student 在 t 的预测，历史分布因此不随所蒸馏的轨迹段变化。网格索引仍按窗口抽取。每次迭代的 collective 与 chain 相位无关，即一次 text encoder 调用、一次 resample 前向、student、teacher 与 EMA 按配置的正负分支前向，以及一次 backward。

student 的窗口 policy 仍由 `data.args` 配置。`meta_model.teacher_streaming` 可以为 teacher 覆盖 `sink_size`、`window_size`、`chunk_size`、`sink_switch_at`、`sink_size_after_switch`、`bootstrap_size`、`audio_lookahead_latents` 和 `audio_right_lookahead_latents`，其它键直接报错。未设置时 teacher 读取 student 的窗口，行为与 `MiniMaxH3StreamingTSCD` 一致。设置后 teacher 在每个 student 起点读取自己的窗口，规划和构造由 `meta_models/streaming_teacher_window.py` 的 `TeacherWindowChainMixin` 实现。历史从同一份工作副本按 teacher 的索引读取，再用派生 RNG 做 0.001 条件加噪，保留音频为原值，noisy 行直接使用 student 的 x_t。reference 行、sink 策略和固定窗口 RoPE 模板都来自 teacher 自己的 plan，picture 与 Qwen 条件和 student 共用。`meta_model.teacher_picture_mode` 决定 picture 在 teacher 窗口中的位置，默认等于 student 的 `picture_mode`，与之不同时必须设置 `teacher_streaming`。宿主可以覆盖 `_teacher_chain_geometry` 和 `_teacher_references`，让 teacher 带 reference 行规划和读取窗口，而 student 窗口没有 reference 行。teacher 的 DDIM 结果按 noisy 行写回 student 窗口，EMA 与 loss 仍使用 student 窗口。

两种 policy 必须共用 C、首块边界和有效音频右 lookahead R，每个窗口的 noisy 视频与音频行因此相同，head 规划时逐窗口核对。student 的 W 与 teacher 不同时，把 `data.args.bootstrap_size` 设为 teacher 的 W+C，首块即与 teacher 完全一致。student 的 S 可以大于首块长度，planner 只读取已经生成的历史。`merge_single_frame_units` 以及 `fixed_window_rope`、`keep_sink_reference`、`separate_reference_rope`、`keep_negative_reference` 等布局开关只在 meta 层设置，两者共用。teacher policy 与 student 不同时要求 `qwen_reference_video: false`，这样 Qwen 条件不依赖窗口，两者可以共用。meta 把 `teacher_streaming` 同步到 `data.args.teacher_streaming_config`，数据集按每个起点上 student 与 teacher 窗口中较大者预留 packing 预算。

训练 UP 大于 1 时沿用 RF 的广播方式。teacher 窗口只由同步后的 payload 和工作副本构造，不读取 pool 条目，组内各 rank 得到相同结果，只有 source rank 写回。validation 按 student policy 流式生成，不使用 teacher，UP 单独配置。

```yaml
entry:
  module: dev.yanzuolu.engines.tscd
  class_name: TrajectorySegmentedConsistencyDistillation
data:
  args:
    sink_size: 1
    window_size: 3
    chunk_size: 2
    bootstrap_size: 7
meta_model:
  module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_tscd
  class_name: MiniMaxH3VideoRefStreamingTSCD
  num_segments: 4
  loss_type: x_t
  teacher_streaming:
    sink_size: 4
    window_size: 5
  resample_timesteps: ...  # 与 RF 相同
```

diffusion 节点和模型配置与 `MiniMaxH3StreamingTSCD` 相同，需要静态网格、`tea_schedule`、`tea_sampler`、冻结的 `models.tea_model` 以及 `backbone` 的 EMA，不需要 `training_timesteps`。CFG fitting 与 teacher 外部 CFG 的开关沿用父类。指标包括 TSCD 的窗口 loss 与上述 RF chain 指标。

## RF chain 上的 DMD

`MiniMaxH3VideoRefStreamingDMD`（`meta_models/minimax_h3_video_ref_streaming_dmd.py`）在上述 chain pool 上训练 `MiniMaxH3StreamingDMD`，entry 使用 `DistributionMatchingDistillation`。窗口带 reference 行、按 `picture_mode` 放置的 picture 和逐窗口 Qwen 条件，与 RF 相同。三个网络都返回 x0：LoRA generator `backbone`、可训练 LoRA fake score `fake_model` 和冻结的 real score `tea_model`。

每次迭代先由 generator 在 `diffusion.sampling_timesteps` 的静态少步网格上 rollout 当前窗口的 noisy 行，sampler 为 `diffusion.sampler`，rollout 结束时把最终 x0 写回工作副本，下一窗口把它当作历史读取。FAKE 与 GEN 的点沿用父类的 `score_path`。默认的 `renoise` 中，`fake_use_trajectory` 为 true 时 FAKE 均匀抽取一个 rollout 阶段的 x0，为 false 时使用最终 x0，两者都按 fake timestep 加新噪声。GEN 在均匀抽取的阶段带梯度重跑 generator，对其 x0 在 score timestep 加噪，再由 fake 与 real score 打分。`ode` 中，FAKE 在该开关为 true 时按视频时刻所在区间选择阶段，读取 generator 的 DDIM rollout 轨迹上的点，为 false 时用最后一步缓存的 x_t 和最终 x0 定义 DDIM 直线，在原始成对时刻取点，允许超出最后一步区间外推。GEN 始终重跑视频时刻所在区间的阶段，并沿其 DDIM 直线构造打分点，不受该开关影响。ODE 的两种 FAKE 设置都不新采高斯噪声，具体的成对时间与区间规则同父类。pool 保存 chain 并支持从断点窗口续跑，meta 把 `data.args.resume_mid_chain` 设为 true，没有 resample 前向，也不需要 `resample_timesteps`。每个窗口的 rollout 都写回，chain 从第二个窗口起读取 generator 自己的输出。父类的语料前缀不适用，`gen_gt_ratio` 非零或设置了 `gt_prefix_window_range` 时抛出 `ValueError`。

generator 的 rollout、写回和 GEN 前向都使用 student 窗口，即 `data.args` 的 policy。设置 `meta_model.teacher_streaming` 后，fake 与 real score 在每个 student 起点读取 teacher 自己的窗口，由 `TeacherWindowChainMixin` 构造：历史从工作副本按 teacher 的索引读取，reference 行、sink 与固定窗口 RoPE 模板来自 teacher 的 plan，picture 与 Qwen 条件与 student 共用，noisy 行就是 generator 的 noisy 行，顺序、取值和时间都相同，两种 `score_path` 的点都放在这些行上。两种 policy 的约束和数据集预算与上节 TSCD 相同。score 需要的负分支都是 teacher 窗口的 captionless 布局，保留 reference 行和 picture。student 的负分支只在 `student_cfg_fitting` 生效时构造。未设置 `teacher_streaming` 时两个 score 读取 student 窗口，负分支规则与父类相同。

每个窗口由 payload 构造一次，rollout、FAKE 和 GEN 读取同一份窗口。payload 的 `window_seed` 由 head 的 prepare seed、`chain_id` 和窗口的数据集序号派生，决定两个窗口中 picture、reference 与历史行的条件加噪，所以训练 UP 大于 1 时组内各 rank 由广播的 payload 得到相同的窗口，续跑也得到相同的窗口。每个 source rank 的窗口在组内依次训练，只有 source rank 持有 pool 条目并写回。每次迭代的 collective 与 chain 相位无关，即一次 text encoder 调用、固定步数的 rollout，以及 FAKE 与 GEN 的固定前向。validation 按 student policy 流式生成，UP 单独配置。

```yaml
entry:
  module: dev.yanzuolu.engines.dmd
  class_name: DistributionMatchingDistillation
data:
  args:
    sink_size: 2
    window_size: 2
    chunk_size: 2
    bootstrap_size: 7
meta_model:
  module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_dmd
  class_name: MiniMaxH3VideoRefStreamingDMD
  teacher_streaming:
    sink_size: 4
    window_size: 5
  student_cfg_fitting: false
  fake_cfg_fitting: true
  teacher_cfg_fitting: false
  cfg_fitting_scale: 3.0
  teacher_guidance_scale: [1.0, 1.0]
  qwen_reference_video: false
engine:
  fake_step_models: [fake_model]
  gen_step_models: [backbone]
```

diffusion 节点、DMD loss、`score_path` 与 CFG fitting 开关沿用 `MiniMaxH3StreamingDMD`，需要静态少步网格、`sampler`、`fake_schedule`、`fake_training_timesteps`、`score_timesteps` 及其音频节点，validation 网格使用与 rollout 相同的步数。指标包括父类的 FAKE 与 DMD loss 以及 chain 的 `self_history` 与窗口序号。

### DMD2

`MiniMaxH3VideoRefStreamingDMD2`（`meta_models/minimax_h3_video_ref_streaming_dmd2.py`）在上述 DMD 上加入 GAN loss，像素条件 generator 对应 `MiniMaxH3VideoRefStreamingPixelDMD2`。fake model 使用 `modeling/build.py` 的 `MiniMaxH3StreamingGANX0DiTSPV2`，其 V2 判别器 `MiniMaxH3SampleChunkDiscriminatorV2` 把每个样本的全部显式行区间合成一个 chunk，每个窗口样本只输出一个 logit，`discriminator_video_chunk_size` 不影响分块。判别器在 fake score 读取的窗口上打分，设置 `teacher_streaming` 时即 teacher 窗口。每次调用都按 fake score 完整条件前向的方式打包窗口，只把当前单元的 noisy 视频行及其 noisy 音频行作为目标行，sink、历史、reference、picture 与文本行只作为上下文。taps 总是来自完整条件前向，与 `cfg_fitting_guidance` 或 `fake_fitting_state` 让 fake score 读取的状态无关。每次判别都是单独的前向，在最后一个 tap 后停止。

fake 样本在 FAKE 中是 rollout 的最终 x0，放在 fake score 窗口的 noisy 行上。GEN 使用 generator 带图的 x0，即 DMD loss 打分的同一张量。real 样本是该窗口目标的语料 latent，放在语料窗口中。语料窗口与 fake score 窗口共用 plan、reference 行和 picture，历史从 chain head 的语料 latent 读取，条件加噪使用相同的噪声抽样。head 的语料 latent 不受写回影响，随每个窗口的 payload 广播，续跑时由恢复的 head 重新带入。chain 的第一个窗口仍读取语料历史，此时两个窗口逐位一致。之后 real 目标延续自己的语料历史，fake 目标延续 generator 写回的历史。判别器输入按 `diffusion.gan_training_timesteps` 抽取成对时间步并经 `fake_schedule` 加噪，`gan_disc_clean_input` 保持干净输入，`gan_r1_lambda` 与 `gan_r1_sigma` 控制近似 R1。`score_path: ode` 下这些新噪声照常抽取，是 rollout 之后仅有的噪声，因为语料目标不在任何 rollout 轨迹上，fake 目标也要与 real 目标以相同方式加噪。

FAKE 返回 score loss、`gan_lambda_disc` 乘以 fake 项、real 项与可选 aR1 三个独立 loss，GEN 在 DMD loss 上加 `gan_lambda_gen` 乘以非饱和 generator loss。GEN 期间 fake model 与判别器冻结，输入梯度传给 generator。每个窗口 FAKE 增加两次判别前向（开启 aR1 时三次），GEN 增加一次。判别器属于 fake model，随其 checkpoint 保存、续跑并由其优化器训练。从不含判别器的 checkpoint 加载 fake 时，在 `allow_missing` 中列出判别器，由 `InitMiniMaxH3Discriminator` 重新初始化。指标为 `fake_losses/gan_loss`、`fake_losses/gan_logits_fake_mean/tap0`、`fake_losses/gan_logits_real_mean/tap0`、`fake_losses/gan_ar1`、`dmd_losses/gan_loss` 与 `dmd_losses/gan_logits_mean/tap0`。

```yaml
diffusion:
  gan_training_timesteps:
    module: dev.yanzuolu.projects.minimax_h3.modeling.training_timesteps
    class_name: PairedUniformTrainingTimesteps
    args: {T: 1.0, shift: 6.0, audio_shift: 1.5}
meta_model:
  module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_dmd2
  class_name: MiniMaxH3VideoRefStreamingDMD2
  gan_lambda_disc: 1.0e-2
  gan_lambda_gen: 1.0
models:
  fake_model:
    module: dev.yanzuolu.projects.minimax_h3_videoref.modeling.build
    class_name: MiniMaxH3StreamingGANX0DiTSPV2
    args:
      discriminator_tap_blocks: [1, 9, 17, 25]
      discriminator_num_queries: 2
    placement:
      fsdp:
        wrap_modules: [MiniMaxH3DiTBlock, MiniMaxH3DiscriminatorBlock]
    adapter:
      target_modules: '^dit\.(?:.*\.)?(qkv_proj|out_proj|fc1|fc2|linear|condition_proj|video_patch_proj|audio_patch_proj|video_out|audio_out|proj_in|proj_out)$'
      modules_to_save: [discriminator]
```

## RF chain 上的蒸馏

`MiniMaxH3VideoRefStreamingDistill`（`meta_models/minimax_h3_video_ref_streaming_distill.py`）同时继承 `MiniMaxH3VideoRefStreamingRF` 和 `MiniMaxH3StreamingDistill`，在上述 chain 上把 student 的单次前向回归到冻结 teacher 的单次前向，entry 使用 `DiffusionDistillation`。pool、head 规划、`t_s`、窗口条件、SP 广播、写回和续跑都与 RF 相同，窗口历史行是 student 对此前窗口的一步 x0 重建，训练 timestep 与 RF 一样取自 `diffusion.training_timesteps` 的全范围。

每次迭代先执行 RF 的 resample 前向，再执行 student 的训练前向和 teacher 在 `no_grad` 下的前向或拟合所需的各分支前向。两者读取相同的 noisy 行、timestep 和历史工作副本，写回发生在 loss 之后。`meta_model.teacher_streaming` 与上文 TSCD 相同，由 `TeacherWindowChainMixin` 在每个 student 起点构造 teacher 自己的窗口，两种 policy 的约束和数据集预算也相同。未设置时 teacher 读取 student 的窗口。loss 是两者预测在 noisy 行上的窗口 MSE，权重与 SFT 相同。每次迭代的 collective 与 chain 相位无关，即一次 text encoder 调用、一次 resample 前向、一次 student 前向、按配置的 teacher 前向和一次 backward。

student 始终读取 raw 前向。teacher 目标默认也是 raw 前向。`meta_model.teacher_cfg_fitting` 开启时，teacher 目标改为 `cfg_fitting_guidance` 写出的训练组合的完整条件拟合，由下文的 `GuidedDistillationFittingMixin` 在 teacher 自己的窗口上计算，每个样本按自身可用条件编译拟合。该开关要求设置 `cfg_fitting_guidance`。沿 `0 --1.5--> T --3.0--> TS` 且 `anchors: guided` 时目标为 `(f(TS) + f(T) + f(0)) / 3`，每个窗口三次 teacher 前向，T 去掉 reference 行，0 再去掉 caption，picture 及其 Qwen 上下文保留在每个分支中。student 因此学到不需要 guidance 的预测，validation 用 guidance 1、每步一次前向采样。validation 与 RF 相同，由 `validate_ema` 与 `validate_backbone` 选择模型。

```yaml
entry:
  module: dev.yanzuolu.engines.diffusion_distillation
  class_name: DiffusionDistillation
data:
  args:
    sink_size: 2
    window_size: 2
    chunk_size: 2
    bootstrap_size: 7
meta_model:
  module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_distill
  class_name: MiniMaxH3VideoRefStreamingDistill
  teacher_streaming:
    sink_size: 4
    window_size: 5
  teacher_cfg_fitting: true
  cfg_fitting_guidance:
    anchors: guided
    branches:
      - {name: "null", drop: [text, reference]}
      - {name: text_only, drop: [reference]}
    stages:
      - {branch: "null", scale: 1.5}
      - {branch: text_only, scale: 3.0}
  resample_timesteps: ...  # 与 RF 相同
```

diffusion 节点与 RF 相同，需要 `training_timesteps`、采样网格、`schedule` 和 `sampler`。模型配置与 TSCD 相同，`backbone` 带 optimizer 和可选 EMA，`tea_model` 冻结且不配置 optimizer。指标包括 SFT 的窗口 loss 与上述 RF chain 指标。

## 语义视频作为当前块起点的 RF

`MiniMaxH3VideoRefStreamingSourceRF` 继承 `MiniMaxH3VideoRefStreamingRF`。数据、picture、Qwen 条件、chain、resample 和 validation request 都与 RF 相同，但窗口内不再有 reference 行。语义视频改为从当前块的 noisy 输入进入，相当于训练过的 SDEdit bridge。target 与语义 latent 共用同一 VAE 和时间网格，当前块的源 `src` 就是语义 latent 在该块 video 索引上的行。

schedule 为 `x_t = (1 - t) x0 + t eps`。`meta_model.source_sigma` 记为 σ，取值 (0, 1]。当前块的起点为 `z = (1 - σ) src + σ eps`，训练时 video 时间取 `t = σ u`，u 是父类成对抽取的 video 时间，noisy video 行为 `x_t = (1 - t/σ) x0 + (t/σ) z`。噪声系数恰好是 t，DiT 仍按预训练 schedule 读取 t，只有信号部分是 target 与源的混合。target 仍是 x0。`target_noises` 保存等效噪声 `x0 + (z - x0)/σ`，`flow_v` 转换因此与 bridge 一致。音频没有源，沿用同一次成对抽取的未缩放时间和原有加噪。历史行、picture、负分支和 fitting 与父类相同，eps 都来自父类的抽取，RNG 流不变。resample 前向以 `σ u_s` 做同样的 bridge 加噪，写回的历史与训练分布一致。

chain head 把每个 clip 的完整语义 latent 保存在条件状态中，每个窗口的 payload 以 `source_latents` 携带。训练 UP 大于 1 时，SP 输入广播把源 rank 的语义 latent 与工作副本一起发给组内各 rank。

validation 中每个窗口的当前块从 z 出发，z 由 request 语义 latent 的对应行和该 request 自己的噪声构成，video 时间为 σ，video 网格为配置网格乘以 σ。音频网格不变，仍从 1 开始。x0 预测上的 DDIM 一步把 `x_t` 移到 `x0 + (t'/t)(x_t - x0)`，正是同一 bridge，因此 sampler 不变。validation 视频仍把语义视频放在左侧、样本放在右侧。

meta 把 `data.args.reference_video_rows` 设为 false，数据集按无 reference 行的窗口预留 packing 预算，并照常读取语义 latent。`qwen_reference_video` 必须为 false。`keep_negative_reference` 必须为 true，因为采样的 captionless 分支保留 picture 行或 keyframe 块。`selective_video_encoding` 与 RF 一样不支持。

`meta_model.picture_mode` 决定 DiT 如何读取 picture，默认 `rows`，即上文在 reference 组前加一帧近 clean 的图像行，它占用文本之后的独立时间槽，后续起点顺延一格。`keyframe` 对应 FL2VA 原生的首帧 keyframe。原生 FL2VA 把 keyframe 作为独立的 clean 条件块放在文本之后，时间坐标等于 target 第 0 帧，空间网格与 target 相同，token tag 为视频，clean 时间为 0.999，所有 target 帧照常生成。keyframe 模式在每个窗口的文本之后放入同样的条件块，其时间坐标为 target 起点加 latent 0 的位置，fixed-window RoPE 下 sink 域不平移，因此每个窗口都钉在第 0 帧的时间上，其余行的坐标与不带 picture 时完全相同。picture 须与 target 共用 latent 网格，它按原生方式拉伸到画布后编码。target latent 0 仍是普通的块行，照常做 bridge 加噪、计入 loss，并由 resample 写回。Qwen 仍按 Picture 1 加 caption 的顺序读取 picture。数据集的 picture 预算与 `rows` 相同，计入条件块行数和 Qwen presentation。validation 中每个窗口都带 keyframe 块，第 0 帧与其余行一样从 z 采样。

```yaml
meta_model:
  module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_source_rf
  class_name: MiniMaxH3VideoRefStreamingSourceRF
  source_sigma: 0.95
  qwen_reference_video: false
  picture_mode: keyframe
  resample_timesteps: ...  # 与 RF 相同
```

## 语义视频按通道拼接的 RF

`MiniMaxH3VideoRefStreamingConcatRF` 继承 `MiniMaxH3VideoRefStreamingSourceRF`，沿用其数据、picture、Qwen 条件、chain、`source_latents` payload、resample 和 validation request，σ 固定为 1。当前块做普通的 t2va 去噪，从 t = 1 的纯噪声出发，使用未缩放的成对时间，不做 bridge。干净的语义 latent 在 video patch embedding 处按通道拼到 target 视频行上，模型输入为 24 + 24 通道。patchify 按通道优先，所以拼接等价于 patchify 后在特征轴上拼成 `[noisy 96 | 语义 96]`。

每个 plan 以 `video_condition_latents` 携带按 `video_indices` 排列的语义 latent。历史行和当前块都读取各自帧的语义，keyframe 或 picture 块、文本、音频和 SP padding 行读取零，拼接列的 eps 为零。训练在加噪前从窗口的 `source_latents` 挂上条件，resample 和 captionless 分支复用这些 plan，因此都带同样的拼接。UP 大于 1 时 SP 输入广播已携带 sources。采样在每个 CFG 分支的 `_state_plan` 中从 request 的语义 latent 挂上条件。行拼装在可复用的 `StreamingVideoConditionMixin`（`modeling/streaming_concat.py`）中完成。

`models.backbone.args.video_condition_channels` 必须等于 `data.args.video_latent_channels`。`MiniMaxH3WeightLoader` 读到较窄的 `video_patch_proj.weight` 时补零新增的输入列，初始模型因此与基座完全相同。patch embedding 通过 adapter 的 `modules_to_save` 整体训练，并从 `target_modules` 中移除，其优化器分组匹配 `*video_patch_proj.modules_to_save.*`。`video_condition_channels` 为 0 时模型与上述各 meta 的行为不变。

```yaml
meta_model:
  module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_concat_rf
  class_name: MiniMaxH3VideoRefStreamingConcatRF
  qwen_reference_video: false
  picture_mode: keyframe
  resample_timesteps: ...  # 与 RF 相同
models:
  backbone:
    args:
      video_condition_channels: 24
    adapter:
      modules_to_save: [video_patch_proj]
    optimizer:
      - class_name: AdamW
        groups:
          - match: ["*video_patch_proj.modules_to_save.*"]
            args: {lr: 1.0e-4, weight_decay: 0.0}
```

## 语义注入的 teacher forcing

两种语义注入各是一个 mixin，定义在 `meta_models/streaming_semantic.py`，同时用于 teacher forcing 和 RF。`SemanticSourceMixin` 是上文的源点 bridge，`SemanticConcatMixin` 是通道拼接。两者都继承 `SemanticLatentMixin`，负责去掉 reference 行、把语义 latent 放进 payload 的 `source_latents`，以及 validation 中的语义来源。RF 类为 mixin 加 `SemanticChainMixin` 加 `MiniMaxH3VideoRefStreamingRF`，teacher forcing 类为 mixin 加 `MiniMaxH3VideoRefStreamingSFT`。

`MiniMaxH3VideoRefStreamingSourceSFT` 和 `MiniMaxH3VideoRefStreamingConcatSFT` 每个样本随机训练一个窗口，bootstrap 与续写按 `bootstrap_probability` 选择。历史行是该 clip 自身的 corpus latent，按 clean 时间锚定。没有 RF chain、resample 或写回。当前块分别做 bridge 加噪或普通 t2va 加噪加通道拼接。`MiniMaxH3VideoRefStreamingSFT` 在 `reference_video_rows` 为 false 时不规划 reference 行，并把 paired latent 放进 payload 的 `source_latents`，训练 UP 大于 1 时随其余输入一起广播。训练样本带 `picture` 时，meta 编码 picture latent，Qwen 按 Picture 1 读取，每个窗口按 `picture_mode` 放置 picture 或 keyframe 块。validation 与对应 RF 相同。拼接模型从只含 LoRA 的导出初始化时，adapter 使用 `modeling/adapters.py` 的 `PartialWeightPeftLoraAdapter`，文件中没有的 adapter 张量（包括 `modules_to_save` 的 patch embedding 副本）保留当前值，日志报告 unexpected 数和保留数。

```yaml
data:
  module: dev.yanzuolu.projects.minimax_h3_videoref.data.video_ref_streaming_long_latent
  class_name: VideoRefStreamingLongLatentT2AVDataset
meta_model:
  module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_source_sft
  class_name: MiniMaxH3VideoRefStreamingSourceSFT  # 或 ..._concat_sft 中的 MiniMaxH3VideoRefStreamingConcatSFT
  source_sigma: 0.95  # 仅 source
  bootstrap_probability: 0.5
  qwen_reference_video: false
  picture_mode: keyframe
```

## 反向光流作为逐 token AdaLN 条件的 RF

`MiniMaxH3VideoRefStreamingGeometryRF`（`meta_models/minimax_h3_video_ref_streaming_geometry_rf.py`）沿用 `MiniMaxH3VideoRefStreamingRF` 的 reference 行、picture、Qwen 条件、chain、resample、captionless 分支和 validation request，另外让每个 target 视频行读取自身 latent 的反向光流。条件只用与游戏无关的信号，即 `flow_bwd`、`occlusion_bwd` 和 `flow_bwd_valid`，不读 car_id、深度、相机或位姿。

DiT 侧的机制是通用的逐 token AdaLN 条件（`minimax_h3/modeling/transformer/adaln_condition.py`）。`adaln_condition_dim` 为 d 时，模型持有一个编码器 `adaln_condition_encoder`，把每个 latent 一张、分辨率为 token 网格 4 倍的 `adaln_condition_channels` 通道图编码成每个视频 token 一个 d 维特征，结构为 conv3x3 C→64、SiLU、stride 2 的 conv3x3 64→128、SiLU、stride 2 的 conv3x3 128→d、SiLU、conv3x3 d→d，最后对通道做 RMSNorm。50 个 block 各有一个 `adaln_condition_head`，即 d→4H 的线性层，weight 和 bias 都初始化为零，输出注意力和 MLP 的 shift、scale 增量。被列出的行按 `norm(x) * (1 + scale[idx] + Δscale) + shift[idx] + Δshift` 调制，其余行保持原 AdaLN，gate 和 final layer 不变。forward 的输入为 `adaln_condition_maps`（每个样本一个 `[n, C, 4h, 4w]`）和 `adaln_condition_rows`（这些 token 在 packed 序列中的行号，按 latent、h、w 顺序，与视频 patchify 的行顺序一致）。编码器每次 forward 只运行一次，每个 SP rank 只保留自身行区间内的特征并换成局部行号，与 decoder 行按同一边界分片。head 位于 block 内，随 block 一起 FSDP 分片，并在 block 的 activation checkpoint 中重算，所以同一时刻只存在一个 block 的增量，且只对被列出的行计算。head 为零时模型逐位等于未加条件的模型。两个字段默认为 0，此时 state dict 与行为都不变。causal DiT 不支持该条件，配置后直接报错。

`StreamingAdalnConditionMixin`（`modeling/streaming_adaln_condition.py`）负责行路由。plan 的 `adaln_condition_maps` 是该样本完整 target 视频的全部 latent 图，构造 kwargs 时才按 plan 当前的 `video_indices` 选取，因此 sink、recent 历史行和当前块都读自身 latent 的图，去掉 reference 或 target 历史的分支也保持对齐。列出的行只有每个样本尾部的 target 视频行，reference、picture 或 keyframe、文本、音频和 SP padding 行都不接收增量。训练在加噪前把 payload 中的图挂到每个窗口 plan 上，resample、条件分支、captionless 分支和每个 guidance 分支都由这些 plan 构造，所以全部带光流，不做随机丢弃或噪声增强。每条 chain 在设备上保存各 clip 的完整图，并放进每个窗口的 payload，训练 UP 大于 1 时 SP 输入广播随之携带。采样在每个 CFG 分支的 `_state_plan` 中挂上 request 的图。

图的打包在 `data/flow_geometry.py`。clip 旁的 `geometry.safetensors` 在比 RGB 粗 8 倍的网格上逐帧保存 `flow_bwd` float16 `[T, 2, H, W]`（帧 t 到 t−1，全分辨率像素，x 向右、y 向下）、`flow_bwd_valid` uint8 `[T]` 和 `occlusion_bwd` uint8 `[T, H, W]`（每格 64 像素中被遮挡的个数），帧序与 reference 视频一致。每帧转成 4 个通道，光流先除以 32（每个 token 的像素数）再取 `sign(v) log1p(|v|)`，无效处置零，随后是遮挡比例和有效位。帧按 codec 的 decode 时间线放进 latent，H3 每 17 帧对应 1、4、4、4、4 帧的 latent。每个 latent 有 4 个帧槽，每槽 4 通道，未用的槽为零，最后 4 个通道标记已用槽，共 20 通道，362 帧得到 107 个 latent。1344×768 时图为 168×96，是 42×24 token 网格的 4 倍。

数据集设置 `data.args.geometry_root` 后，每个样本多出 `adaln_condition_maps` `[T, 20, H/8, W/8]` float16，读自 `<geometry_root>/<artifact_id>/geometry.safetensors`，从条目的 `reference_start_frame` 起只读取本次 crop 需要的帧。未设置时不读取。validation request 用 `geometry_path` 指定 clip 的几何文件，可选的 `geometry_start_frame` 为 request 第一帧对应的几何帧，默认 0。

从没有这些参数的 checkpoint（例如 0924d）继续时，placement 插件 `InitMiniMaxH3AdalnCondition`（`minimax_h3/modeling/adaln_condition_init.py`）必须排在 DCP 加载之前。它在分片前把仍为 meta 的编码器和 head 实体化到 CPU，head 置零，编码器按 `seed` 使用 PyTorch 默认初始化，EMA 和 PEFT 的 `original_module` 副本因此完全相同。DCP 加载用 `allow_missing` 只放行这些参数。LoRA 之外的这两个模块通过 adapter 的 `modules_to_save` 训练并保存。`engine.resume_dir` 恢复旧 checkpoint 时依赖 `persistence.checkpoint.allow_partial_load`，新参数及其优化器状态保持初始化值。d = 256、H = 5376 时 50 个 head 共 276,326,400 个参数，编码器 970,944 个。

```yaml
meta_model:
  module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_geometry_rf
  class_name: MiniMaxH3VideoRefStreamingGeometryRF
data:
  args:
    geometry_root: /home/jinchengy/jinchengy/yanzuolu/data/torcs_clips
validation:
  - requests:
      - geometry_path: /home/jinchengy/jinchengy/yanzuolu/data/torcs_clips/<window>/geometry.safetensors
models:
  backbone:
    args:
      adaln_condition_dim: 256
      adaln_condition_channels: 20
    placement:
      plugins:
        - module: dev.yanzuolu.projects.minimax_h3.modeling.adaln_condition_init
          class_name: InitMiniMaxH3AdalnCondition
          seed: 1019
        - module: dev.yanzuolu.projects.minimax_h3.modeling.fp32_placement
          class_name: MiniMaxH3FP32ModulePlacement
        - module: dev.yanzuolu.common.plugin.dcp_weights
          class_name: ShardedDCPWeights
          path: /path/to/checkpoint
          key: models.backbone
          allow_missing: ['\.adaln_condition_(encoder|head)\.']
    adapter:
      modules_to_save: [adaln_condition_encoder, adaln_condition_head]
```

## 像素条件视频经逐 token 输入条件注入

`MiniMaxH3VideoRefStreamingPixelSFT`（teacher forcing）和 `MiniMaxH3VideoRefStreamingPixelRF`（RF）沿用语义通道拼接的 t2va 布局，窗口没有 reference 行，picture 作为 FL2VA 原生 keyframe，当前块从纯噪声做普通 t2va 去噪，Qwen 不读条件视频，captionless 分支保留条件，validation 视频左侧显示 request 的 `reference_video_path`。区别在于条件不是 VAE latent，而是与 target 逐帧对齐的任意 RGB 视频，由 backbone 的输入条件编码器从像素读取。两者由 `meta_models/streaming_pixel.py` 的 `PixelConditionMixin` 实现，RF 另加 `PixelChainMixin`。

DiT 侧是通用的逐 token 输入条件（`minimax_h3/modeling/transformer/input_condition.py`）。`input_condition_channels` 为 C 时，模型持有 `input_condition_encoder`，读取每个 latent 一张、分辨率为 token 网格 s 倍的 C 通道图，s = 2^(len(`input_condition_widths`) − 1)。编码器是残差 CNN，1x1 stem 到第一级宽度，每级 `input_condition_blocks` 个残差块（GroupNorm(32)、SiLU、3x3 卷积，重复两次，再加恒等），第一级之后每级开头有一个 stride 2 的 3x3 卷积，最后一级落在 token 网格上。之后对通道做 RMSNorm，再用一个 weight 和 bias 都初始化为零的线性层映射到 hidden size，结果加到被列出的行在 patch embedding 之后的输入 embedding 上。forward 的输入为 `input_condition_maps`（每个样本一个 `[n, C, s·h, s·w]`）和 `input_condition_rows`（这些 token 在 packed 序列中的行号，按 latent、h、w 顺序）。未列出的行保持原 embedding，输出层为零时模型逐位等于未加条件的模型。每个 SP rank 只编码与自身行区间有交集的 latent，只把增量加到自己的行上。开启 gradient checkpointing 时编码器逐层重算，只保留层边界。字段默认为 0，此时 state dict 与行为都不变。默认宽度 [256, 512, 768]、每级 2 块、C = 772、hidden 5376 时编码器共 42,101,248 个参数，其中 stem 197,888，三级分别为 2,362,368、9,443,328、21,242,880，两个下采样共 4,719,872，RMSNorm 768，输出层 4,134,144。

像素到条件图的转换在 `data/pixel_condition.py`。帧先缩放到 [-1, 1]，与 VAE 输入一致，再按 u 做无损的 pixel unshuffle 得到 3u² 个通道，通道 c·u² + i·u + j 存放像素 (u·y + i, u·x + j) 的颜色 c。帧按 codec 的 decode 时间线放进 latent，与光流图的打包相同，每个 latent 4 个帧槽，未用的槽为零，最后 4 个通道标记已用槽。s 由宽度级数决定，u = spatial_vae_stride × 2 / s，默认三级时 u = 8，C = 4 × (3 × 64 + 1) = 772，1344×768 的帧得到 96×168 的图。meta 由 backbone 配置推出 u 并检查 C，`video_condition_channels` 必须为 0。

数据集设置 `data.args.pixel_condition: {root, filename}` 后，每个样本多出 `condition_frames`，即 `<root>/<artifact_id>/<filename>` 从条目的 `reference_start_frame` 起、与 target 同分辨率的 uint8 `[F, H, W, 3]` 帧，按解码顺序读取。只有 pack 最终保留的样本才解码，每个 pack 解码一次，RF chain 只在 head 解码。一个 362 帧 1344×768 的 clip 占 1.04 GiB。meta 把整段帧放进 payload 和设备，SP 输入广播不接受 uint8 张量，payload 因此携带同一存储的 int8 视图，读取时再视为 uint8，每个 plan 携带整段帧，构造 kwargs 时才按 `video_indices` 选出窗口 latent 的帧并在设备上转换，所以去掉 target 历史的分支仍然对齐，也只转换窗口所需的帧。reference 被 drop 的 guidance 分支不列出任何条件行，这些行逐位保持原 embedding，其余与不 drop reference 的同一分支相同。validation request 用 `condition_video_path` 指定条件视频，可选 `condition_start_frame`，默认 0，左侧显示的仍是 `reference_video_path`。

从没有编码器的权重初始化时，placement 插件 `InitMiniMaxH3InputCondition`（`minimax_h3/modeling/input_condition_init.py`）排在其他 placement 插件和 DCP 加载之前，在分片前把仍为 meta 的编码器按 `seed` 用 PyTorch 默认初始化实体化并把输出层置零，EMA 和 PEFT 的 `original_module` 副本因此相同。released 权重的分片加载会让缺失的键保持 meta，DCP 加载则用 `allow_missing: ['\.input_condition_encoder\.']` 放行。编码器通过 adapter 的 `modules_to_save` 整体训练，只含 LoRA 的导出由 `PartialWeightPeftLoraAdapter` 加载。

```yaml
meta_model:
  module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_pixel_sft
  class_name: MiniMaxH3VideoRefStreamingPixelSFT  # 或 ..._pixel_rf 中的 MiniMaxH3VideoRefStreamingPixelRF
  qwen_reference_video: false
  picture_mode: keyframe
data:
  args:
    pixel_condition: {root: /home/jinchengy/jinchengy/yanzuolu/data/torcs_clips, filename: semantics.mp4}
validation:
  - requests:
      - reference_video_path: /home/jinchengy/jinchengy/yanzuolu/data/torcs_clips/<window>/semantics.mp4
        condition_video_path: /home/jinchengy/jinchengy/yanzuolu/data/torcs_clips/<window>/semantics.mp4
models:
  backbone:
    args:
      input_condition_channels: 772
      input_condition_widths: [256, 512, 768]
      input_condition_blocks: 2
    placement:
      plugins:
        - module: dev.yanzuolu.projects.minimax_h3.modeling.input_condition_init
          class_name: InitMiniMaxH3InputCondition
          seed: 1019
        - module: dev.yanzuolu.projects.minimax_h3.modeling.fp32_placement
          class_name: MiniMaxH3FP32ModulePlacement
    adapter:
      module: dev.yanzuolu.projects.minimax_h3_videoref.modeling.adapters
      class_name: PartialWeightPeftLoraAdapter
      modules_to_save: [input_condition_encoder]
    optimizer:
      - class_name: AdamW
        groups:
          - match: ["*input_condition_encoder.modules_to_save.*"]
            args: {lr: 1.0e-4, weight_decay: 0.0}
```

`MiniMaxH3VideoRefStreamingPixelTSCD`（`meta_models/minimax_h3_video_ref_streaming_pixel_tscd.py`）在 RF chain 上把 Ref2VA teacher 蒸馏到像素条件 t2va student。student 与 EMA 是上面的 RF 布局，teacher 在每个 student 起点读取自己的窗口：按 `teacher_streaming` 规划，clip 的语义 latent（payload 的 `source_latents`）作为 reference 行，picture 按 `teacher_picture_mode: rows` 作为 Picture 1 放在自己的时间槽。teacher 的 plan 不带 `condition_frames`，其 forward 不接收输入条件。两者的 noisy 行、时间和历史工作副本与上节相同。`qwen_reference_video: false` 时两种 picture 模式的 Qwen 输入都是 Picture 1 加 caption，两者 text 行逐位相同，captionless 分支各自只去掉 caption，teacher 保留 reference 行和 picture，student 保留 keyframe 和条件视频。meta 设置 `data.args.teacher_reference_video_rows: true`，数据集按带 reference 行的 teacher 窗口预留预算，student 窗口仍不带 reference 行。数据集同时提供语义 latent 与条件帧。

```yaml
meta_model:
  module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_pixel_tscd
  class_name: MiniMaxH3VideoRefStreamingPixelTSCD
  picture_mode: keyframe
  teacher_picture_mode: rows
  teacher_streaming: {sink_size: 4, window_size: 5}
models:
  backbone:
    placement:
      plugins:
        - module: dev.yanzuolu.projects.minimax_h3.modeling.input_condition_init
          class_name: InitMiniMaxH3InputCondition
          seed: 1019
        - module: dev.yanzuolu.projects.minimax_h3.modeling.fp32_placement
          class_name: MiniMaxH3FP32ModulePlacement
        - module: dev.yanzuolu.common.plugin.dcp_weights
          class_name: ShardedDCPWeights
          path: /path/to/checkpoint
          key: models.backbone_ema
          allow_missing: ['\.input_condition_encoder\.']
```

`MiniMaxH3VideoRefStreamingPixelDMD`（`meta_models/minimax_h3_video_ref_streaming_pixel_dmd.py`）是同一异构组合的 DMD。generator 是上面的像素条件 t2va student，在自己的窗口上 rollout、写回和训练 GEN，窗口 plan 带 payload 中的条件帧。fake 与 real score 是 Ref2VA 网络，按上节 teacher 的方式读取自己的窗口，语义 latent 作为 reference 行，generator 的 anchored picture 作为 Picture 1，noisy 行、噪声、时间和历史与 generator 相同。其余行为与上文 RF chain 上的 DMD 相同。

蒸馏时每个开启 fitting 的网络都应按其训练时的组合读取。`meta_model.cfg_fitting_guidance` 用 SFT 的键写出该组合，包括 `branches`、`stages` 和 `anchors`，由 `meta_models/streaming_guided_fitting.py` 的 `GuidedDistillationFittingMixin` 实现，videoref 的 TSCD、DMD 与蒸馏都支持。设置后，开启 `*_cfg_fitting` 的网络不再使用单边 `(P + (s - 1) U) / s`，而是按每个样本的可用条件读取 `compile_guidance` 的完整条件拟合 `(P - sum(c_i sg(N_i))) / s`。沿 `0 --1.5--> T --3.0--> TS` 且 `anchors: guided` 时即 `(f(TS) + f(T) + f(0)) / 3`。梯度只经过条件前向，分支前向不带梯度。除下述 fake score 的分支监督外，蒸馏不监督分支，`loss_weight` 必须为 0，也不接受丢弃 `history`。每个网络在自己的布局上丢弃条件。Ref2VA 窗口丢弃 `reference` 时去掉 reference 行，丢弃 `text` 时去掉 caption，picture 和它的 Qwen 上下文除非分支丢弃 `picture` 否则保留，与 `reference_drop_keeps_picture` 的训练一致。像素条件 student 丢弃 `reference` 时去掉条件视频，保留 keyframe。TSCD 中 student、EMA 与 teacher 各自按开关使用该组合，teacher 此时不再叠加外部 CFG，`teacher_guidance_scale` 必须为 1。DMD 中 generator 与 fake score 各自按开关使用该组合，窗口在准备时带上分支窗口而不是 captionless 负分支，real score 必须不做 fitting 且 `teacher_guidance_scale` 为 1。蒸馏中只有 teacher 按 `teacher_cfg_fitting` 使用该组合，student 始终读取 raw 前向。

```yaml
meta_model:
  cfg_fitting_guidance:
    anchors: guided
    branches:
      - {name: "null", drop: [text, reference]}
      - {name: text_only, drop: [reference]}
    stages:
      - {branch: "null", scale: 1.5}
      - {branch: text_only, scale: 3.0}
```

DMD 的 `meta_model.fake_branch_loss` 默认关闭。开启后 fake score 按 SFT 使用 `guidance_branches` 与 `guidance_fitting` 的方式训练，要求 `fake_cfg_fitting` 开启且 `cfg_fitting_guidance` 中至少一个分支的 `loss_weight` 为正。fake loss 在完整条件拟合的 loss 之外，对每个带权分支加上 `loss_weight` 乘以该分支自身拟合的 loss。分支拟合取自 `compile_guidance` 的 `branch_scales` 与 `branch_coefficients`，目标与主项相同，即 generator 样本在 fake 时间步上的目标。每个拟合只对自身状态的预测传梯度，读取的更低状态全部停止梯度，带权分支因此带梯度前向。上例的分支加上 `loss_weight: 0.1` 与 `0.5` 后，fake loss 为 `L((f(TS) + f(T) + f(0)) / 3) + 0.5 L((2 f(T) + f(0)) / 3) + 0.1 L(f(0))`，与该组合的 SFT 相同。各分支 loss 记录为 `running/fake_losses/branches/<name>/video` 与 `audio`。generator、real score 以及 fake score 在 GEN 阶段的读取都不变。

DMD 的 `meta_model.fake_fitting_state` 指定 fake score 读取 `cfg_fitting_guidance` 中哪个分支的状态，FAKE 训练和 GEN 打分读取同一状态。不设置时读取完整条件拟合。设置后读取 SFT 监督该分支时 `compile_guidance` 给出的拟合，只对该分支的预测传梯度，完整条件前向不再运行。上例取 `text_only` 时 fake score 读取 `(2 f(T) + f(0)) / 3`，与 SFT 训练 text_only 分支的拟合相同。T 去掉 score 窗口的 reference 行，0 再去掉 caption，picture 及其 Qwen 上下文都保留。real score 仍读取自身的完整条件预测，DMD 方向因此比较两个状态。要求 `fake_cfg_fitting` 开启、该分支位于 `stages` 上且 `fake_branch_loss` 关闭。

像素条件 DMD 的 generator 开启 `student_cfg_fitting` 并设置 `cfg_fitting_guidance` 后，每次预测都读取完整条件拟合，包括每个 rollout 步（最后一步的 x0 即写回的历史）、GEN 以及 validation。上例即 `(f(TS) + f(T) + f(0)) / 3`，每个去噪步三次前向，只有 GEN 的 `f(TS)` 带梯度。T 去掉条件视频，保留 keyframe 与 caption，0 再去掉 caption，读取只含 picture 的 Qwen 上下文。validation 把三个状态作为权重各 1/3 的采样分支，在同一组 noisy 行和历史上去噪，每个分支使用自己的 Qwen 上下文和条件视频，此时 validation 的 `guidance_scale` 必须为 1。其他 videoref DMD 不支持引导 student 的采样，设置后直接报错。

## 长 GT 母片与逐窗口 caption

`VideoRefStreamingLongLatentChainDataset` 从离线编码的完整连续母片在线裁片，复用上述 RF chain 的打包、相位错开和 worker 恢复协议。先均匀选择能够容纳配置长度的母片，再均匀选择其合法起点。不会预先生成固定训练切片，也不按母片长度增加其抽样概率。母片索引、裁片和 picture 由 `LongLatentCatalogMixin` 提供。teacher forcing 使用同一 mixin 上的 `VideoRefStreamingLongLatentT2AVDataset`（`data/video_ref_streaming_long_latent.py`），每次迭代发出一包完整裁片，窗口由 meta 选择。两者共用 `CausalLatentT2AVDataset._build_pack` 的抽样和打包。

```yaml
data:
  module: dev.yanzuolu.projects.minimax_h3_videoref.data.video_ref_streaming_long_latent_chain
  class_name: VideoRefStreamingLongLatentChainDataset
  args:
    latent_index_path: /path/to/14_long_gt_latents/index.jsonl
    num_frames: 243  # 示例长度，可配置为任意 17n+5。
    height: 704
    width: 1280
    # tokenizer、S/W/C、packing 等参数沿用 RF trial。
```

索引记录 `latent_path`、`num_frames`、几何、音频真实有效长度和格式标识。`latent_path` 可相对索引目录。初始化只读索引并确认候选 `.pt` 已存在，读到样本时使用 mmap，只复制本次 crop 的 normalized target video/audio 和 reference latent。原生 video 起点为 `5k`，对应 RGB 起点 `17k`，样本长度为 `17n+5` RGB、`5n+2` native latent。母片尾部直接截到合法编码长度。音频依据视频起点选择离线保存的三种 PCM 相位，再按整数 audio latent 边界切片，完整监督区间必须落在真实音频范围内。

每个母片的 `caption_segments` 保存局部 24 FPS 半开 RGB 区间及原文 `prompt`。dataset 截取并重基准该时间表，整条 crop 只抽一次 text dropout，并按所有可能 caption 的最大 text/Qwen 长度预留 packing 预算。每个 RF 窗口根据实际 noisy video 区间选择重叠最多的一条 caption，平局取较后一条。S/W 历史和固定窗口 RoPE 不影响选择。重采样与训练共用该条件，caption 在窗口内保持不变。原来的单 prompt 数据路径保持不变。

Validation request 可选配置同样的 `caption_segments`，省略全局 `prompt` 时使用首个非空 caption 作为样本展示文字，全空时间表仍显示空文字。Qwen 每窗口只编码一次，CFG 负分支将整个 caption 时间表置空并保留 reference 条件。关闭 Qwen visual context 时，validation 一次合批编码各条 caption，再按窗口切换 embedding。不同长度 case 的占位轮使用相同选择入口，不增加或减少每轮 collective 调用次数。RF 训练和 validation 都支持 SP。
