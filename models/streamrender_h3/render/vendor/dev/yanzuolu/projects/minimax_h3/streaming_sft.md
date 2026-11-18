# MiniMax H3 streaming SFT

`MiniMaxH3StreamingSFT` 使用纯文本条件训练和生成双向音视频窗口。每个窗口包含文本、保留的音视频历史和当前 noisy target，不需要 reference 视频。共享的窗口、采样与输出组件也供 VideoRef 使用。

## 配置入口

以下字段接入现有 `DiffusionFinetuning` 配置。语料目录、tokenizer、分辨率、帧数、模型权重、优化器和 diffusion 节点仍按已有接口配置。

```yaml
entry:
  module: dev.yanzuolu.engines.diffusion_finetuning
  class_name: DiffusionFinetuning

data:
  module: dev.yanzuolu.projects.minimax_h3.data.streaming
  class_name: StreamingLatentT2AVDataset
  args:
    sink_size: 2
    sink_switch_at: 1
    sink_size_after_switch: null
    window_size: 2
    chunk_size: 1
    bootstrap_size: null
    merge_single_frame_units: false
    audio_lookahead_latents: 17

meta_model:
  module: dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_sft
  class_name: MiniMaxH3StreamingSFT
  bootstrap_probability: 0.5
  guidance_scale: 1.0
  negative_loss_weight: 0.0
  fixed_window_rope: false
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

数据集读取现有 `prompt*.pt` 语料中的 `prompt`、`video` 和 `audio`。音视频 latent 必须已经归一化，形状和时间映射必须与配置的 codec 一致。打包预算使用所有合法窗口中最大的实际 token 数，包含右侧 R 个音频 lookahead，不包含 decoder 左缓存 L 或 reference 行。窗口在 SP 输入同步后抽取，数据集保留现有语料读取和 worker 断点恢复行为。

## 窗口和训练

S、W、C 分别为 `sink_size`、`window_size`、`chunk_size`，单位为窗口 unit。默认一个 unit 对应一个原生视频 latent，保持原行为。S、W 可以为零，C 必须为正。首块默认生成最多 W+C 个 unit，后续每次生成最多 C 个。续块从已经生成的历史中保留前 S 个和最近 W 个 unit，并按全局原生 latent 索引合并去重，再接当前新帧。历史不足 S 时只保留已有帧，不读取未来 GT。短首块、短尾块按真实长度处理，不复制帧补齐。

`data.args.merge_single_frame_units` 默认 `false`。设为 `true` 时，原生 H3 每周期的 1 RGB latent 与其后 4 RGB latent 组成一个 unit，unit 的原生 latent 跨度依次为 `(2,1,1,1)`，沿完整时间轴重复。unit 边界为 `0,2,3,4,5,7,8,9,10…`，不会在每轮续写时重新开始周期。S、W、C 和 `sink_size_after_switch` 都按 unit 计，音频 L、R 仍按 40 Hz audio latent 计。仅支持原生 H3 时序描述符，包括原生 VAE 和普通 TAE，连续 `MiniMaxH3StreamingTAE` 开启时会明确报错。

例如 W6/C1 的首块为 7 个 unit，即 9 个原生 latent、30 RGB。后续 C1 的原生结束边界依次为 `10,12,13,14,15…`，本轮新增 RGB 分别为 `4,5,4,4,4…`。unit 只改变选窗分组，原生 latent 数值、通道、codec 编解码和音视频时间映射保持原样。

`data.args.sink_switch_at` 从第 1 个续写块开始计数，不含首块，默认为 1。`sink_size_after_switch` 默认 `null`，表示继续沿用原 `sink_size`。设置为整数时要求 `0 <= sink_size_after_switch <= sink_size`，从指定续写块开始使用较小的 sink，可以减到 0。例如 `sink_switch_at: 8`、`sink_size_after_switch: 1` 表示前 7 个续块使用原 S，第 8 个起使用 S=1。首块长度与续块起点不随 S 改变。当前块去噪完成后，状态按下一块所需的 S 裁剪视频及对应音频历史。

`data.args.bootstrap_size` 默认 `null`，首块为 W+C 个 unit。设为正整数时首块改为该数量的 unit，续块起点随之平移，之后仍每次推进 C 个 unit。续块读取的 S、W 历史和固定窗口 RoPE 使用的 S+W+C 模板不变，首块短于 W 时第一个续块只读取已有历史。例如 W3/C2 配 `bootstrap_size: 7` 与默认 W5/C2 的首块和全部续块起点一致，使窗口几何不同的模型共用同一个窗口序列。未设置时 plan、打包预算和流式生成与原行为一致，设置后 validation 和流式生成按同一首块推进。

每个窗口是一个完整双向 attention document，历史也能读取当前 noisy target。不同 sample 和 SP padding 彼此隔离。每个窗口重新计算 DiT，不保留跨窗口 transformer KV。

训练从完整 GT latent 序列中抽取首块或续写窗口。首块没有历史，续写窗口用 GT 模拟已经生成的历史。每个 sample 抽取一对配对的视频和音频 timestep。视频历史使用 `0.999 * x0 + 0.001 * noise`，保留音频原值，两者在模型中使用 clean timestep。监督只覆盖当前 noisy 视频和新增 noisy audio。

`loss_prediction_type: x0` 直接拟合 clean latent。`flow_v` 在正噪声时间步下转换为原生 data-ward velocity 后拟合，要求线性插值 schedule。每个 sample 在各模态内先取均值，再对 sample 平均。没有新增 noisy audio 的窗口贡献零音频 loss，仍计入 sample 分母。

`meta_model.bootstrap_loss_weight` 和 `continuation_loss_weight` 默认 1.0，接受有限非负值。`first_frame_loss_weight` 省略或设为 `null` 时继承 bootstrap 权重，不单独改变首个 latent 的权重。显式数值同样必须有限非负，包括 0 和 1，分别覆盖首块视频和音频各自全局索引 0 的权重，与 bootstrap 不相乘。续写块实际监督的 noisy 音视频 latent 使用 continuation 权重，保留的第 0 个 sink 不触发 first 加权。历史、reference 和保留音频不参与 loss。音频在上述加权后统一乘 `audio_loss_weight`。

例如 bootstrap=4、audio=0.1 且未配置 first 或设为 `null` 时，首块视频权重为 `[4, 4, 4, ...]`，音频有效权重为 `[0.4, 0.4, 0.4, ...]`。显式设置 first=10 后，分别为 `[10, 4, 4, ...]` 和 `[1, 0.4, 0.4, ...]`。显式设置 first=1 后，分别为 `[1, 4, 4, ...]` 和 `[0.1, 0.4, 0.4, ...]`。continuation 默认 1 时，续写视频和音频有效权重分别为 1 和 0.1。设 continuation=2.5 后，分别为 2.5 和 0.25。

加权在 x0 或 flow_v 转换后进行，仍以各 sample 原始监督元素数取均值，不用权重和重新归一化。权重 0 会屏蔽对应误差，continuation=0 时续块仍计入原样本分母。视频合并 unit 不改变全局 latent 索引，显式 first 仅覆盖首块的原生视频 latent 0，partner 仍用 bootstrap。音频按自身的全局 latent 索引 0 应用同样规则。

训练 CFG fitting 与采样 CFG 分开配置。训练 `meta_model.guidance_scale = w` 使用

```text
prediction = (conditional + (w - 1) * stopgrad(captionless)) / w
```

w 为 1 时不执行负分支。采样使用通常的 `unconditional + scale * (conditional - unconditional)`，由调用参数或 `validation.guidance_scale` 控制。

`meta_model.negative_loss_weight` 默认 `0.0`，负分支只作为停止梯度的锚点，w 为 1 时不运行。设为正数 λ 后，负分支带梯度前向，并按与主项相同的 x0 或 flow_v 转换和窗口加权计算自身预测的 MSE，以 λ 倍加入总 loss，记录为 `train/negative_video_loss` 与 `train/negative_audio_loss`，音频同样乘 `audio_loss_weight`。fitting 项仍使用停止梯度的负分支，λ 只负责把负分支校准到其残缺条件下的后验均值。λ 为正时即使 w 为 1 也运行负分支，此时等价于带条件 dropout 的普通训练。

`meta_model.keep_negative_reference` 默认 `true`，负分支只移除语言。设为 `false` 时，负分支同时移除 reference 行和为每个窗口编码的 Qwen 媒体前缀，只保留 target 音视频窗口，packing rows 相应减少。该开关同时作用于训练 fitting、streaming TSCD 复用的负分支和采样 CFG。采样时以空语言识别负分支，同一分支内所有 sample 必须一致。纯 T2AV 没有 reference，该开关不改变行为。

纯 T2AV 的视频位置为 `text_len + p(global_video_index)`，音频位置为 `text_len + global_audio_index`。sink 与最近历史之间的时间间隔保留，不加入 reference 的额外位置偏移。

`meta_model.fixed_window_rope` 默认 `false`。设为 `true` 时，首块保持上述布局，续写按当前有效 sink 大小 S_eff 的窗口模板编号。令 B(q) 为 q 个 unit 对应的原生 latent 边界，未合并时 B(q)=q。令 u 为当前窗口起点的 unit 索引，s=B(S_eff)，b=B(max(0,u-W))，r=max(s,b)，T 为文本长度，p(i) 为 codec 声明的原生视频时间位置。

```text
sink video       T + p(i)
WC video         T + p(s) + p(i) - p(r)
sink audio       T + j
WC + R audio     T + p(s) + j - p(r)
```

i、j 仍是原始全局视频、音频索引。sink 行仅包括 `i < min(s,start)` 的已有视频历史，以及 `j < ceil(p(min(s,start)))` 的对应音频。sink 与 recent 重叠时只出现一次，此时平移量为 0，联合窗口保留原生位置和内部时间间距。recent 与 sink 分离后，平移量为 `p(s)-p(b)`，压缩两者之间的时间间隔。本轮新帧即使全局索引小于 s，仍按 WC 坐标处理。音频保留 40 Hz 格点相对视频边界的分数偏移，不重新从零编号。不同 codec 的内部 cadence 仍由 p 决定。sink 切换时模板随 S_eff 更新，短尾不因本次 C 变短而重新缩放。

开关统一作用于训练、validation、实时接口和 CFG 分支，不改变音频生成、提交或 decoder 缓存。纯 T2AV 没有 reference 域偏移，`separate_reference_rope` 和 `keep_sink_reference` 对其不起作用，前者设为 `false` 也不妨碍启用固定窗口。VideoRef 的 reference 选择及域偏移规则见其使用说明。

## 多分支监督与顺序 fitting

SFT 可用 `meta_model.guidance_branches` 声明多种残缺条件，并用独立的 `guidance_fitting.stages` 选择 fitting 路径。每个分支的 `drop` 是 `text`、`reference`、`history` 的非空组合，`loss_weight` 默认 0。`text` 只移除语言，`reference` 移除 VAE reference 和 Qwen 视觉条件，`history` 移除所有已生成的 target 音视频行，包括上轮已预测而尚未提交的音频尾部。当前 noisy target、噪声、时间步和监督范围始终与完整条件分支相同。

例如纯 T2AV 可按「无文字无历史 → 有文字无历史 → 完整条件」拟合，同时单独训练保留历史的 captionless 分支。

```yaml
meta_model:
  guidance_branches:
    - name: unconditional
      drop: [text, history]
      loss_weight: 0.1
    - name: no_history
      drop: [history]
      loss_weight: 0.1
    - name: no_text
      drop: [text]
      loss_weight: 0.1
  guidance_fitting:
    stages:
      - branch: unconditional
        scale: 1.5  # Adds text.
      - branch: no_history
        scale: 2.0  # Adds history and reaches the implicit full-condition state.
```

`stages` 按条件逐渐增加的顺序排列，其 `drop` 集合必须逐级严格缩小，完整条件分支隐式放在最后。每个 `scale` 属于「这一行的分支 → 下一分支」的增量，接受有限且不小于 1 的数。路径可以只包含一个分支，也可以从保留部分条件的分支开始。不在路径上的分支仍可通过 `loss_weight` 独立训练。省略 `guidance_fitting` 时，只计算完整条件的普通监督和各分支的独立监督。

令路径为 N0、N1、…、F，F 表示普通完整条件预测，边强度为 s0、s1、…，目标引导预测为

```text
G = N0 + s0 * (N1 - N0) + ... + s_last * (F - N_last)
  = c_full * F + sum(c_i * N_i)
```

完整条件网络输出 P 通过下面的反解拟合 x0，所有锚点均停止梯度。

```text
Q = (P - sum(c_i * stopgrad(N_i))) / c_full
loss = window_loss(Q) + sum(loss_weight_i * window_loss(N_i))
```

所有边强度相同时，中间项抵消为普通 CFG 的 `N0 + s * (F - N0)`。所有强度为 1 时，主项使用原始 P。只计算一个组合后的主 loss，各负分支监督使用各自原始预测，不对分支数再次平均。x0/flow_v 转换以及音频、首帧、首块、续写权重与已有监督共用。

每个 sample 根据实际存在的条件合并等价状态。首块没有历史时，history 边不产生引导，末条有效边决定 `c_full`。等价于完整条件的负分支不重复监督 P。其它等价分支合并 fitting 系数和监督权重，原 sample 数仍作为 batch 分母。移除历史只裁分支输入，所有保留行的 RoPE 坐标保持不变。

`guidance_fitting.anchors` 默认为 `posterior`，即上面的读法。每个锚点视为其残缺条件的后验均值，分支监督使用原始预测。设为 `guided` 时，任一状态的单次前向 f 视为沿路径截断到该状态的引导输出。每个 sample 合并等价状态并去掉空边后，按下式逐级还原后验均值，主 loss 拟合 u_F，所有低位 f 停止梯度。

```text
u_N0 = f(N0)
u_Ni = u_N(i-1) + (f(Ni) - f(N(i-1))) / s(i-1)
```

带正 `loss_weight` 的分支必须位于 `stages` 上。它的监督拟合 u_Ni，由自身带梯度的 f(Ni) 与停止梯度的更低状态算出，路径起点就是原始预测。两段路径 `N0 → S → F` 的主项为 `f(F)/s1 + (1/s0 - 1/s1) * f(S) + (1 - 1/s0) * f(N0)`，s0 为 1 时退化为单段 fitting。只有一条有效边时两种读法的主项一致。带监督的分支仍只前向一次，其 detach 结果兼作锚点。

只有实际用于 fitting 或具有正 `loss_weight` 的分支需要额外 DiT 前向。纯 fitting 的分支在 no_grad 下运行，带独立监督的分支保留梯度，fitting 中仍使用它的 detach 结果。SP/FSDP 各 rank 统一决定是否运行某个分支，本地无有效样本的 rank 仍参加必要前向及零值反传。分支监督指标写入 `train/branches/<name>/video_loss` 和 `audio_loss`。等价状态使用首个配置分支的名称，指标已经包含合并后的 `loss_weight`，音频在组成总 loss 时再乘 `audio_loss_weight`。

省略 `guidance_branches` 或设为 `null` 时保留原 `guidance_scale`、`negative_loss_weight`、`keep_negative_reference` 路径。空列表 `[]` 是显式的新配置，表示没有负分支。启用新列表时，旧字段须为默认值，避免同时定义两套训练目标。新分支配置仅用于 SFT 训练，streaming TSCD 不接受这些配置。validation 和实时采样仍由原采样参数控制，`guidance_scale: 1.0` 每个去噪步只运行完整条件分支，不执行训练分支。

## 音频与 codec

`data.args.audio_lookahead_latents = L` 控制音频 decoder 保留的连续已提交左缓存，默认 17。可选的 `data.args.audio_right_lookahead_latents = R` 控制 DiT 右侧预测范围、保留的未来音频、noisy mask、窗口 token 预算和 decoder 右上下文。两者以 40 Hz 音频 latent 为单位，接受非负整数。R 省略或为 `null` 时使用同一有效 policy 的 L，旧配置仍得到 L=R 的行为。L 不改变 DiT 的 S/W 历史选择。

例如，以下配置保留 17 个左历史 latent，同时不预测或使用右侧音频。

```yaml
data:
  args:
    audio_lookahead_latents: 17
    audio_right_lookahead_latents: 0
```

生成边界领先当前视频发布边界最多 R 个 audio latent。已经生成的音频保留原值，只给尚未生成的右端新增部分加噪并计算 loss。当前提交音频和新增 noisy audio 使用不同的 mask，因此一个窗口可以发布音频而不生成新音频。

`models.video_vae` 声明的 `VideoTemporalMapping` 决定视频 latent 和音频边界。可选的 `models.video_decoder` 是独立的媒体解码模型节点，未配置时使用 `video_vae`。它按标准模型节点配置类、权重和运行精度，不参与训练目标或 reference 的编码。两者必须使用相同的 normalized latent 空间、时间映射、latent 通道数和空间缩放比例。配置阶段检查时间映射，运行时还检查通道和空间比例。

例如，原生 `MiniMaxH3VideoVAE` 编码后，可以由普通 `MiniMaxH3TAE` 解码。模型输出保持 normalized H3 latent，输出适配器仅使用所选 decoder 的 `latents_mean` 与 `latents_std` 转换一次。普通 TAE 的统计量是零均值和单位标准差，不先叠加原生 encoder 的反归一化。

视频输出能力由选中的 decoder 决定。连续 `MiniMaxH3StreamingTAE` 支持逐步视频解码，原生 H3 VAE 和普通 TAE 在流结束时解码完整视频。音频仍在每次 push 时分块解码，使用最多 L 个连续已提交左历史 latent、当前音频和最多 R 个保留的右侧 latent。首块没有左历史，后续使用 `min(实际已有左历史数, L)`，不为凑足 L 重复 latent 或等待。视频 decoder 只支持最终输出时，PCM 暂存到最终 flush 与视频一起发布。不能把 DiT 的带间隔 S/W 历史当作连续音频。

实时输出和 validation 使用同一个 `StreamingAVDecoder`，按每个 sample 的 policy 传入 L、R。手动创建 decoder 时也需传入这两个字段，未指定的 R 可以保留为 `None`。

```python
from dev.yanzuolu.projects.minimax_h3.modeling.streaming_output import StreamingAVDecoder

decoder = StreamingAVDecoder(
    models.get("video_decoder", models["video_vae"]), models["audio_vae"],
    audio_lookahead_latents=window_config.get("audio_lookahead_latents", 17),
    audio_right_lookahead_latents=window_config.get("audio_right_lookahead_latents"),
)
```

`decoder.audio_lookahead_seconds` 返回 R 对应的秒数，即 R/40。已知总长度时提前截断右侧音频，未知总长度时在最终块裁掉终点外尚未提交的音频。输出不会重新生成已经保留的音频，也不补静音。默认 L=R=17 是经验值，当前 32 kHz BigVGAN 按完整计算图向外取整的上下文覆盖界为左右各 29 个 audio latent。L、R 控制实际上下文长度，不改变感受野，也不保证分块音频与整段解码位级一致。

## 流式接口

`start_stream` 为各 sample 创建独立状态。视频几何是 `(channels, latent_height, latent_width)`，`video_lengths` 和 `audio_lengths` 分别使用各模态的 latent 单位。

只给 `video_lengths` 时，按 codec 对应的视频终点推导缺省音频长度，并提前截断 lookahead。显式 `audio_lengths` 优先。

```python
states = meta.start_stream(
    prompt_embeds=prompt_embeds,
    video_geometries=video_geometries,
    audio_channels=[32] * len(prompt_embeds),
    rngs=sample_rngs,
    streaming_configs=window_configs,
    video_lengths=video_lengths,
    audio_lengths=audio_lengths,
    guidance_scale=3.0,
)

chunk = meta.step(models["backbone"], active_states)
```

不传 `video_counts` 时，首块按 bootstrap units、后续按 C units 的实际原生边界推进。`video_counts`、`video_lengths`、shape、事件 indices 和 start/stop 始终以原生 latent 计，不改成 unit 数。已知 `video_lengths` 时自动截短最终块并标记结束。未知长度时由调用者传入 `video_counts=[...]` 和 `is_last=[...]`，仅最终块允许短于正常窗口。

validation 配置、单个 validation request 和 `start_stream` 的 `streaming_configs` 可以只覆盖部分窗口参数，未指定项继承训练 policy。`sink_size_after_switch: null` 按合并后的 `sink_size` 解释。`audio_right_lookahead_latents` 在这些覆盖合并后解析，值为 `null` 时跟随有效 L。例如训练未显式设 R，validation 只把 L 改为 5 时，验证使用 L=R=5。训练显式设 R=0 时，仅修改 L 不会开启右侧预测。每个 sample 按自己的全局视频起点计算续块编号，长短不同的流互不影响切换时机。

准备完整生成长度时，可以构造 `StreamingBatch` 并使用 `stream_latents` 或 `stream`。

```python
from dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_sft import StreamingBatch

inputs = StreamingBatch(
    prompt_embeds=prompt_embeds,
    reference_latents=None,
    video_shapes=video_shapes,
    audio_shapes=audio_shapes,
    streaming_configs=window_configs,
)
for chunk in meta.stream(models, inputs, sample_rngs, guidance_scale=3.0):
    consume(chunk)
```

`video_shapes` 为 `(C,T,H,W)`，`audio_shapes` 为 `(2,C,T)`。`streaming_configs` 可为各 sample 指定不同 L、R，`stream` 和 validation 回放使用各自的值创建 decoder。事件中的 `sample_indices` 指回最初的 sample 顺序，`video` 和 `audio` 是本次提交的 latents，`audio_lookahead` 是最多 R 个保留的已生成右侧音频，`decoded` 包含本次 codec push 和最终 flush 的输出。

完整 `stream` 路径为提前结束的分布式组执行独立 dummy forward，保持各组 FSDP collective 次序一致，不推进真实 sample 的 RNG。直接调用 `step` 时，由调用者保证参与进程使用一致的模型调用顺序。

VideoRef 的可选 `meta_model.qwen_visual_context` 在选好每个窗口后编码其实际参考 RGB，默认关闭。训练时选窗及 Qwen 编码都在 prepare 内完成，SP 只同步准备好的条件和计划。推理在每个窗口进入 solver 前准备一次条件。该模式需要显式的 `qwen_processor_path`，为布局提供完整 prefix embeddings 和 tags。纯 T2AV 没有 reference，不能启用此开关。详细配置和新增原始 RGB 实时参数见 [VideoRef 使用说明](../minimax_h3_videoref/streaming_sft.md)。

raw VideoRef SFT 还支持默认关闭的 `meta_model.selective_video_encoding`。开启后，prepare 为各 source rank 和 sample 使用独立的窗口 RNG 先选窗口，再并行编码所需视频 latent。SP 同步后复用计划，正常采样 timestep 和噪声。原生 VAE 可以跳过未使用的 temporal blocks，TAE 使用完整编码后选择的兼容路径，音频始终整段编码。已有 latent 的 T2AV/TSCD 输入不适用此选项。

## Validation latent export

每个 validation entry 可设置 `save_latent: true`，沿用原生双向 validation 的导出接口。

```yaml
validation:
  - name: cfg1.0
    guidance_scale: 1.0
    save_latent: true
    latent_dir: runs/streaming_latent_corpus
```

每个样本导出为 `<latent_dir>/validation/backbone_cfg1.0/prompt0000.pt`，EMA 对应 `backbone_ema_cfg1.0`。未配置 entry 的 `name` 时不加名称后缀。`latent_dir` 可省略，默认使用该次 validation 的 `media/<step>` 目录。独立 corpus 目录会跨 step 复用已有结果，不同 checkpoint 的导出应使用不同目录。

文件包含 `video`、`audio`、`prompt` 和 `prompt_idx`。音视频为采样完成、尚未反归一化或解码的 CPU BF16 latent，形状分别为 `[C,T,H,W]` 和 `[2,C,T]`。streaming 各块按原始索引拼成完整时间轴，只保存正式提交的音频，右侧 lookahead 不重复追加。文件采用临时写入后重命名，续跑只根据所有所需 variant 已存在的 `.pt` 跳过样本，`.mp4` 和未完成的临时文件不算完成。

latent 模式跳过音视频解码、MP4、grid 和视频日志。采样、CFG、batch/SP 分组及补齐 forward 沿用原路径。VideoRef 仍会编码生成所需的 reference 和 Qwen 条件，并为同步的单视频 reference 附加可供训练读取的配对信息，见 [VideoRef latent 数据入口](../minimax_h3_videoref/streaming_sft.md#latent-corpus)。纯 T2AV 保持上述四字段格式。默认 `save_latent: false` 保持媒体输出。

## Streaming TSCD

`MiniMaxH3StreamingTSCD` 对 streaming SFT 模型做轨迹分段一致性蒸馏。它沿用上述数据集、codec、窗口、历史加噪、RoPE 和生成接口。student、固定 teacher、EMA target 都使用兼容的 streaming x0 模型，例如 `MiniMaxH3X0DiTSP`，三者读取同一窗口布局和历史条件。

接入现有配置时替换 engine 和 meta 入口，保留所需的窗口与损失权重参数。

```yaml
entry:
  module: dev.yanzuolu.engines.tscd
  class_name: TrajectorySegmentedConsistencyDistillation

meta_model:
  module: dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_tscd
  class_name: MiniMaxH3StreamingTSCD
  num_segments: 4
  loss_type: x_t
  teacher_guidance_scale: [1.0, 1.0]
  student_cfg_fitting: false
  ema_cfg_fitting: false
  teacher_cfg_fitting: false
  cfg_fitting_scale: 3.0
  bootstrap_probability: 0.5
  audio_loss_weight: 1.0
  bootstrap_loss_weight: 1.0
  first_frame_loss_weight: null
  continuation_loss_weight: 1.0
```

TSCD 不需要 SFT 的 `diffusion.training_timesteps`。训练从静态采样网格抽取一对相邻时间步。视频和音频共用网格索引及比较位置的插值比例，各自保留原来的 timestep shift。`num_segments` 必须整除共同的网格步数 N，每段包含 N / num_segments 个网格索引。分段数不会随训练自动变化。

需要以下六个 diffusion 节点，均沿用 `{module, class_name, args}` 配置形式。

| 节点 | 要求 |
| --- | --- |
| `sampling_timesteps` | 静态网格，例如 `TrailingSamplingTimesteps`，`num_sampling_steps: N`，`sampling_skip_max: 1`，保留适合源模型的 T 与 shift |
| `audio_sampling_timesteps` | 静态网格且步数同为 N，使用对应音频的 T 与 shift |
| `schedule` | 与 streaming x0 模型一致，例如 `LinearInterpolationSchedule`，`pred_type: x_0` |
| `sampler` | 供 streaming 生成使用，可以是 DDIM 或 ConsistencySampler，不参与训练比较点投影 |
| `tea_schedule` | 与 teacher 一致的 schedule，`pred_type: x_0`，通常与 student 相同 |
| `tea_sampler` | `dev.yanzuolu.common.diffusion.sampler.ddim.DDIMSampler`，仅执行确定性的相邻一步 |

`sampling_skip_max >= 1` 排除会令相邻 EMA 输入时间步落到 0 的起点。每个 sample 先在 t 上计算 student 预测，再由 teacher 将当前 noisy 行推进到相邻 t′，然后由 EMA 在 t′ 上预测。teacher 不直接推进到分段边界。`loss_type: x_0` 比较两边的 x0，`x_t` 则用独立的确定性 DDIM 将两边投影到共同的 s，再比较结果，其中 s 在该段边界与 t′ 之间。只有 student 保留梯度。

三个模型复用一次准备的历史，不重新抽历史噪声。宿主可以覆盖 `_solver_window`，让 teacher 读取自己布局的窗口，其 noisy 行须与 student 逐行相同，solver 结果仍写回 student 窗口。solver 只更新 `video_noisy_mask` 和 `audio_noisy_mask` 选中的行，再写回完整窗口。已有视频历史、保留音频和 reference 保持原值，新增 noisy audio lookahead 则正常参与 solver 与 loss。`x_0` 和 `x_t` 共用上述 loss 归约，首块音视频使用 `bootstrap_loss_weight`。`first_frame_loss_weight` 缺省或为 `null` 时继承该权重，仅显式数值覆盖各模态全局 latent 0。续块实际监督的音视频使用 `continuation_loss_weight`，音频再统一乘 `audio_loss_weight`。分母仍为各 sample 原始监督元素数及原样本数。

模型配置使用 `models.backbone`、`models.tea_model` 和由 `models.backbone.ema` 构造的 `backbone_ema`。student 和固定 teacher 可以从同一个 streaming SFT checkpoint 初始化，并保留相同的模型 wrapper、adapter、codec 和窗口条件。只为 `backbone` 配置 optimizer，teacher 设置为冻结且不配置 optimizer。启用 `models.backbone.ema: {}` 或相应的 EMA placement 配置，同时设置 `engine.ema_decay.backbone`。新 run 在权重恢复后由 engine 将 student 参数复制到 EMA，resume 时恢复 checkpoint 中的 EMA。checkpoint 路径属于模型加载配置。

`student_cfg_fitting`、`ema_cfg_fitting`、`teacher_cfg_fitting` 分别控制三个训练角色，默认全部关闭。它们共用有限正数 `cfg_fitting_scale = g`，默认为 3.0。令 P、U 为同一模型的正、负条件 x0 预测，开启 fitting 后使用 `Q = (P + (g - 1) * U) / g`。student 的 U 在 `no_grad` 下计算，仅 P 反传。teacher 和 EMA 的两条分支都不反传。g 为 1 时等价于 P，不增加负分支。训练目标仍由 `loss_type` 指定，不使用 SFT 的 `meta_model.guidance_scale` 或 `loss_prediction_type: flow_v`。

`teacher_guidance_scale` 继续表示 teacher 的外部 CFG，默认 `[1.0, 1.0]`，也接受固定 scalar 或采样区间，两模态共用每个 sample 的 w。若同时开启 teacher fitting，先形成 Q，再使用 `U + w * (Q - U)`，等价于 `U + (w/g) * (P - U)`。实现复用 P、U 各一次 forward。固定 w=g 时直接使用 P，范围采样则始终执行两条分支，即使某次抽样恰好使 w=g，也不改变各进程的 forward 次数。全部开关关闭时保留原 TSCD 行为。

正、负条件的 encoder 输出、prefix tags、选定窗口和 `StreamingInputs` 布局在一个训练 ctx 内各准备一次，三个模型复用同一个负条件对象。已经提供负 prefix 时直接使用其 embeddings 和 tags，未提供负 prefix 且未开启视觉上下文时回退为空文本。两条条件分支共享同一份已加噪 reference 和 AV 历史。已有 preselected plan 会直接复用，不在编码 prefix 后重新选窗。每个模型仍通过自己的 condition projection、refiner 和 DiT 计算 hidden，权重不同的模型不共享 hidden 或 KV。每次 forward 的行打包只组合这些已准备好的输入，不重复调用 encoder。

上述开关仅作用于训练。validation 继续支持独立的音视频采样网格、生成 sampler、guidance 和窗口 overrides，结束后恢复训练节点。`validation.guidance_scale: 1` 使用 raw P，需要采样同一 Q 时可显式设为 1/g。默认验证 `backbone`，`validation.validate_ema: true` 会额外验证 `backbone_ema`。实时 `start_stream`、`step`、`stream` 与 SFT 接口相同，也只使用调用者显式传入的 guidance，不自动读取训练 fitting 开关。

## Streaming DMD

`MiniMaxH3StreamingDMD` 用 `engines.dmd` 对少步 streaming student 做纯 DMD，不含 GAN。`backbone` 是 LoRA student，`fake_model` 是可训练的 LoRA fake score，`tea_model` 是冻结的 real score，三者都是 streaming x0 模型。窗口计划、布局、历史加噪、负条件和 validation 沿用 SFT。

训练数据使用 `projects.minimax_h3.data.streaming_chain.StreamingLatentChainDataset`。它把每个 pack 展开为按窗口顺序的一条 chain，head batch 携带完整 pack，之后的 light batch 只携带 `chain_id`、`chain_index`、`chain_length` 和 `chain_windows`。`chain_phase_stagger` 默认开启，让各 rank 与 worker 的第一条 chain 截短到不同长度，从而处在不同窗口相位。续跑时从 head 重新发出当前 chain 的完整 pack。续跑方式由 meta 在代码中通过 `data.args.resume_mid_chain` 决定，trial 不设置该项。为 false 时这条 chain 从窗口 0 重新开始，并截短到剩余窗口数。为 true 时 head 保留中断时的 `chain_index` 和 `chain_length`，供能从 checkpoint 恢复 chain 状态的 meta 从断点窗口接着训练。

meta 在每个 rank 的进程内存中按 `chain_id` 保存 float32 工作副本，不写入 checkpoint，因此把 `data.args.resume_mid_chain` 设为 false，续跑后的 chain 截短重启。light batch 找不到条目时抛出 `RuntimeError`。每次迭代训练每个 clip 的一个窗口，每个窗口在一条 chain 中恰好训练一次，因此不接受 `bootstrap_probability`。`prepare_inputs` 从工作副本读取 sink 与 recent 历史，`rollout` 在返回前把 student 生成的 noisy 行 x0 写回，下一窗口即读取 student 自己的输出。FAKE 与 GEN 只读取 payload，因此 `ga_steps` 与 `offline` 可以大于 1。工作副本不经 SP 同步，`engine.up_size.train` 必须为 1。自带可 checkpoint 的 pool 与输入广播的宿主覆盖 `_configure_chains`，替换上述 pool 约束。text encoder 每次迭代调用一次，rollout 与 FAKE、GEN 的 forward 数与窗口序号无关。

`gen_gt_ratio` 按 clip 在 head 决定是否保留语料前缀，保留时从闭区间 `gt_prefix_window_range` 均匀抽取 k。前 k 个窗口照常生成和训练，但不写回，因此窗口 w 在 w ≤ k 时读取语料历史，之后读取自己生成的 recent 历史，位于前 k 个窗口内的 sink 保持语料值。

rollout 在 student 的静态少步网格上生成当前窗口的 noisy 行，保留每一步的输入 x_t 与 x0。`score_path` 决定 FAKE 在哪些点上训练 fake score，以及两个 score 在哪些点上读取 generator。默认的 `renoise` 中，GEN 在均匀抽取的一步上带梯度重跑 student，对其 x0 按成对的 score 时间步加新噪声，再在同一窗口上计算 fake 与 real 预测。`fake_use_trajectory` 为 true 时，FAKE 均匀抽取一步的 x0，为 false 时使用最终 x0，两者都按成对的 fake 时间步加新噪声。该开关只影响 FAKE，不改变 GEN。

`score_path: ode` 中 FAKE 与 GEN 不再抽取高斯噪声，仍按上述分布抽取成对时间步。除窗口历史的条件加噪外，rollout 的初始噪声是唯一的噪声，之后的状态都由 DDIM 得到。GEN 对每个 sample 取视频时间 s 所在区间 `t_{k+1} <= s < t_k` 对应的一步 k，最后一个区间的下端为 0。音视频共用这一步 k，各自的时间被限制在自己网格上第 k 步的区间内。成对分布与网格使用相同 shift 时成对抽样保持不变，抽到 `t_0` 的时间则移到略低于 `t_0` 的位置。GEN 在第 k 步缓存的 x_t 上带梯度重跑 student，DMD loss 对其 x0 打分，两个 score 读取从该 x_t 沿 detach 后的预测推进到 s 的 DDIM 结果。

ODE 的 FAKE 由 `fake_use_trajectory` 选择 x0。为 true 时，与 GEN 相同，按 s 所在区间选择第 k 步，并将成对时间限制在各自的第 k 步区间内，用 DDIM 把缓存的 x_t 沿该步缓存的预测推进到 s，回归目标是该步的 x0 及其隐含的 x_T。为 false 时，固定使用最后一步缓存的 x_t 和 rollout 最终 x0 定义 DDIM 直线，在原始成对时间 s 处取点，回归目标是最终 x0 及该直线隐含的 x_T。时间不限制到最后一步的区间内，因此允许沿该直线外推。两种设置都不新采高斯噪声，也不改变 GEN 的阶段选择或打分点。

student 开启 fitting 时，缓存与重跑的都是 fitted 预测，fake score 在该点读取的组合与 `renoise` 相同。ODE 要求 DDIM sampler、与 `schedule` 的 T、A、B 相同的 `fake_schedule`、严格递减到 0 的 student 网格，并要求 `fake_grad_enabled` 与 `real_grad_enabled` 都为 false，否则抛出 `ValueError`。

宿主可以用 `StreamingDMDInputs.score_window` 给两个 score 另设窗口，其 noisy 行与 generator 的顺序相同，FAKE 的训练和两个 score 的打分都在该窗口上进行，rollout 与 generator 前向仍使用 student 窗口。两种 `score_path` 的点都放在该窗口的 noisy 行上。loss 只覆盖 noisy 行，每个 sample 的各模态先对元素取均值，再以 `audio_loss_weight` 合并。`dmd_loss` 与 causal DMD 相同，但不支持 `norm_per_chunk` 与 `chunk_wise_weighting`。

`student_cfg_fitting`、`fake_cfg_fitting`、`teacher_cfg_fitting` 与 TSCD 的 fitting 语义相同，共用 `cfg_fitting_scale`，负分支不反传。student fitting 开启时，rollout、写回和 GEN 都使用 fitted 预测。`teacher_guidance_scale` 是 real score 的外部 CFG，在 teacher fitting 之后以 `U + w * (Q - U)` 组合。

```yaml
meta_model:
  module: dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_dmd
  class_name: MiniMaxH3StreamingDMD
  fixed_window_rope: true
  fake_loss_type: x_0
  score_path: renoise
  fake_use_trajectory: true
  gen_gt_ratio: 0.5
  gt_prefix_window_range: [1, 10]
  student_cfg_fitting: true
  fake_cfg_fitting: true
  teacher_cfg_fitting: false
  cfg_fitting_scale: 3.0
  teacher_guidance_scale: [1.0, 1.0]
  dmd_loss: {type: dmd, norm_clip_min: 1.0e-5, phuber_c: 0.001, alpha: 1.0}
```

diffusion 节点与 causal DMD 相同，包括 `sampling_timesteps`、`audio_sampling_timesteps`、`schedule`、`sampler`、`fake_training_timesteps`、`audio_fake_training_timesteps`、`score_timesteps`、`audio_score_timesteps` 和 `fake_schedule`。音视频采样网格必须是静态的且步数相同。

## Streaming 蒸馏

`MiniMaxH3StreamingDistill` 用 `engines.diffusion_distillation` 把 streaming student 的预测回归到冻结 teacher 的预测上。数据集、窗口、按 `diffusion.training_timesteps` 全范围抽取的 timestep、噪声、历史加噪和 validation 都沿用 SFT。student `backbone` 与 teacher `tea_model` 在同一组 noisy 行、同一 timestep 和同一历史上各做一次 raw 前向，teacher 前向在 `no_grad` 下运行。

```yaml
entry:
  module: dev.yanzuolu.engines.diffusion_distillation
  class_name: DiffusionDistillation

meta_model:
  module: dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_distill
  class_name: MiniMaxH3StreamingDistill
  loss_prediction_type: x0
  audio_loss_weight: 1.0
  bootstrap_loss_weight: 1.0
  first_frame_loss_weight: null
  continuation_loss_weight: 1.0
```

engine 每步依次调用 `sample_timesteps`、`add_noise`、`student_forward`、`teacher_forward` 和 `compute_loss`，并在 `no_grad` 下调用 `teacher_forward`。构造、续跑、UP 与 GA、EMA、validation 和 early stop 都沿用 `DiffusionFinetuning`，engine 配置键相同。模型配置与 TSCD 相同，只为 `backbone` 配置 optimizer，`tea_model` 冻结且不配置 optimizer，`backbone` 的 EMA 可选。

loss 沿用 SFT 的窗口 loss，目标换成 teacher 的预测。两边都按 `loss_prediction_type` 读取，只覆盖 noisy 行，首块、首帧、续块与音频权重及分母都与 SFT 相同。SFT 自身的 guidance 目标不适用，设置 `guidance_branches`、`guidance_fitting`、不为 1 的 `guidance_scale` 或正的 `negative_loss_weight` 时抛出 `ValueError`。

宿主可以覆盖 `_teacher_inputs`，给 teacher 另设 noisy 行与 student 逐行相同的窗口。覆盖 `_teacher_target` 或 `_student_prediction` 可以让任一侧读取多次前向的线性组合，例如 teacher 读取外部 CFG `u_c + w (u_c - u_u)` 或训练时的 fitting 组合，engine 不需要改动。
