# Legacy 显式控制对齐草案

## 目标

这份草案不是给 `Qwen3` 路线用的，而是专门对应：

- `legacy/streaming_codec_baseline`

当前这条线的目标不是改成 `decoder-only + interleaved tokens`，而是：

- 保留现有 chunk-based codec baseline
- 保留 chunk 序列上的局部因果注意力
- 把旧的粗条件输入替换成更明确的显式控制

换句话说，这条线要解决的是：

- 当前 chunk 应该看什么控制信息
- 这些控制信息如何从标注里 rule-based 展开
- 模型只学 `control -> codec token` 的映射

而不是：

- 让模型自己学习 duration
- 让模型自己学习 shift
- 让模型自己学习 alignment state transition

## 先讲结论

对 legacy 来说，最合理的改造方向是：

- **显式控制**
- **确定性对齐展开**
- **chunk 级条件生成**

而不是：

- `Voxtream` 那种预测 `pred_shift`
- `CosyVoice 3` 那种固定比例交错

原因很简单：

- 你的 singing 任务已经有显式 `note / phoneme / duration / slur`
- 这意味着对齐状态本来就是已知的
- 没必要再让模型回头学一次

## 为什么不继续用旧的 `cond_num[7]`

旧版 legacy 用的是：

- `phoneme_id`
- `pitch`
- `note_on`
- `note_off`
- `t_since_on`
- `t_to_end`
- `note_prog`
- `ph_prog`

这套设计的问题是：

1. 条件语义不够干净
   - 既有离散符号，又有绝对时间量
2. 不够接近后续 token-based 控制设计
3. 把很多“应该由 scheduler 决定”的东西混进了模型输入

因此 v1 要改成更明确的离散控制集合。

## 参考意义怎么取舍

### `CosyVoice 3`

对 legacy 路线的意义很有限。

它主要解决：

- 流式双流同步协议

但不解决：

- singing 场景里的强结构化 note/phoneme 对齐

### `Voxtream`

对 legacy 的参考点只有一个：

- **显式 alignment state 比固定比例同步更可靠**

但 legacy 这里不需要照搬它的 `pred_shift` 训练目标，因为：

- 你的对齐状态是已知的
- 没必要额外建一个“模型预测状态机”

所以真正该借的是：

- 把 alignment state 显式化

不该借的是：

- 让模型去预测 alignment transition

## Step 单位怎么定义

legacy v1 不把“绝对 step 时长”编码进模型语义。

这里的策略是：

- chunk 本身仍然是训练/推理步长
- chunk 大小由 `audio.chunk_ms` 控制
- 控制状态在每个 chunk 的中心时刻采样

也就是说：

- 一个 model step = 一个 chunk
- 控制由外部 scheduler / 预处理展开
- 模型内部不需要知道“这个 step 在物理时间上是多少毫秒”

这件事很关键，因为以后如果你改 chunk 大小：

- 可以重做 cache
- 但不需要把模型设计成“固定绝对时长语义”

## v1 控制集合

当前 v1 只保留 4 个离散控制：

- `NOTE`
- `PHONEME`
- `SLUR`
- `PHONE_PROGRESS`

对应到 cache 字段就是：

- `note_id`
- `phoneme_id`
- `slur`
- `phone_progress`

## 为什么 v1 不用 `NOTE_PROGRESS`

这个结论来自 `OpenCpop segments` 的展开方式。

在当前标注里：

- 每个展开位置本来就是 `1 phoneme <-> 1 note`
- 如果一个 phoneme 跨多个 note
- 数据不是保留一个“长 phoneme span”
- 而是把 phoneme 重复展开到多个位置
- 并用 `slur=1` 表示这是连音延续

所以对于 legacy v1：

- `NOTE` 已经告诉模型当前音高是谁
- `SLUR` 已经告诉模型是不是连音延续
- `PHONE_PROGRESS` 已经告诉模型当前 phoneme 内部走到哪里

在这个前提下，`NOTE_PROGRESS` 的边际收益不高。

因此 v1 先不上。

## 每个控制字段的含义

### `note_id`

- 当前 chunk 中心时刻对应的活动 note
- 由 `note_pitch_midi` 取整得到
- `0` 保留给 rest / inactive note

### `phoneme_id`

- 当前 chunk 中心时刻对应的活动 phoneme
- 由 phoneme vocabulary adapter 产生

### `slur`

- 当前 note 的连音标志
- `OpenCpop` 可以直接从原始标注带出
- 其它没有 slur 的数据集目前回退为全 `0`

### `phone_progress`

- 当前 phoneme 内的进度离散桶
- 从显式 phoneme interval 规则计算
- 桶数由 `model.phone_progress_bins` 控制

## Preprocess 责任

legacy v1 的预处理要负责这几件事：

1. 用冻结 EnCodec 把 wav 编成离散 `codes`
2. 按 chunk 划分 `codes`
3. 在每个 chunk 中心时刻计算：
   - `note_id`
   - `phoneme_id`
   - `slur`
   - `phone_progress`
4. 保留边界诊断字段：
   - `note_on_boundary`
   - `note_off_boundary`

也就是说：

- 预处理负责“显式控制展开”
- 模型不负责“学习如何展开”

## Cache Schema

v1 cache item 结构为：

- `utt_id`
- `codes`: `[T, F, K]`
- `note_id`: `[T]`
- `phoneme_id`: `[T]`
- `slur`: `[T]`
- `phone_progress`: `[T]`
- `note_on_boundary`: `[T]`
- `note_off_boundary`: `[T]`

这里：

- `T` = chunk 数
- `F` = chunk 内 codec frame 数
- `K` = codebook 数

## 模型结构怎么理解

legacy 现在仍然是：

- chunk-level condition encoder
- chunk 序列上的 local causal attention
- chunk 内 `(F, K)` 槽位并行分类

所以这条线的建模形式依然是：

```text
explicit chunk control + local chunk history -> current chunk codec logits
```

不是：

```text
control token / audio token interleaved decoder-only LM
```

## 当前历史建模是什么

当前历史建模已经不是最早的单 `prev_repr` 递推了，而是：

- 先对上一 chunk code 做摘要
- 再在 chunk 序列上做 `history_window` 范围内的局部因果注意力

这条线的意义是：

- 控制仍然是显式给定
- 局部历史只负责跨 chunk 连续性

## 多码本预测方式

当前 legacy **没有做多码本自回归**。

它现在是：

- 先得到当前 chunk 的 hidden
- 再加 slot embedding
- 对所有 `(frame, codebook)` 槽位并行分类

也就是：

- 不是 `codebook 0 -> codebook 1 -> ...`
- 不是 coarse-to-fine
- 不是 delayed pattern

而是：

- **parallel classification over all codebooks**

v1 先保持这个不变。

## 为什么 v1 先不动多码本策略

因为当前最需要验证的是：

- 显式控制是不是比旧 `cond_num[7]` 更合理

如果这一步都还没跑通，就没必要同时引入：

- 更细的多码本自回归
- 更复杂的 coarse-to-fine 结构

所以当前优先级是：

1. 新控制 schema 跑通
2. 观察 token CE / token acc / boundary acc
3. 再决定是否改多码本预测顺序

## 关键超参

当前 v1 主要相关的超参是：

- `audio.chunk_ms`
- `model.note_vocab_size`
- `model.phone_progress_bins`
- `model.history_window`

其中最重要的是：

- `phone_progress_bins`

因为它直接决定了 `PHONE_PROGRESS` 的离散粒度。

## 当前的实现边界

v1 当前已经做到了：

- cache schema 改成离散控制
- adapter 可以带 `OpenCpop` 的 slur
- model condition encoder 改成 discrete embeddings
- train / infer 接口改为读取 `note_id / phoneme_id / slur / phone_progress`

但还没做的是：

- 端到端重跑 preprocess + train smoke test
- 对新旧条件输入做对比实验

## 下一步建议

最务实的顺序是：

1. 重跑新版 preprocess，生成 v2 cache
2. 跑最小训练 smoke test
3. 确认 loss / acc 正常
4. 再决定是否：
   - 增加 `NOTE_PROGRESS`
   - 改 chunk 大小
   - 改多码本预测方式

## 一句话总结

legacy v1 的正确方向不是“让模型学对齐”，而是：

- **外部显式展开控制**
- **模型只学 control-to-codec**
- **局部注意力只负责跨 chunk 连续性**

当前最小有效控制集合就是：

- `NOTE`
- `PHONEME`
- `SLUR`
- `PHONE_PROGRESS`
