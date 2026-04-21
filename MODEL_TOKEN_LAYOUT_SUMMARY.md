# 主流语音模型 Token 排列方式总结（面向当前项目）

本文档整理了当前仓库里已拉取并调研过的主流路线，重点关注：
- 输入协议是偏 `instruction` 还是结构化控制
- 音频 token 是如何排列与预测的
- 是否使用“交错”策略

## 1. 总览表

| 模型/路线 | 主要输入范式 | 音频输出 token 排列 | 交错类型 | instruction 式 |
|---|---|---|---|---|
| CosyVoice 2/3 | 文本（可含控制 tag）+ 参考音频/说话人条件 | 先生成单路 speech token 序列，再经 flow/vocoder 还原波形 | 不做多码本显式交错（多在后端解码） | 是 |
| MOSS-TTS Delay | 文本/指令前缀 + 语音条件 | RVQ 多码本“延迟对角”并行头预测（delay pattern） | 是（码本间错位交错） | 是 |
| MOSS-TTS Local | 文本/指令前缀 + 语音条件 | 同一时间步内按码本层级逐层预测（time-synchronous） | 部分（步内层级 AR，不是全局对角） | 是 |
| MOSS-TTS Realtime | 分层文本 + 历史语音上下文 | AR 生成 RVQ token（实时增量） | 主要是分层条件，不是显式 control/audio 交错 | 是 |
| VALL-E / VALL-E X | 文本 token + 声学 prompt token | codec token AR（常见按帧展开） | 通常不做控制 token 交错 | 弱 instruction（更像条件生成） |
| VITA-QinYu（本地版本） | SenseVoice（理解输入）+ XY tokenizer（目标语音 token） | LLM 侧生成语音相关 token 序列 | 非典型多码本交错 AR | 是（对话式） |
| Qwen3-Omni / AUT 路线（当前调研结论） | 多模态对话 instruction | 倾向统一音频表示接口（非传统 RVQ 多头 AR） | 非典型 RVQ 交错 | 是 |

## 2. 结论（对你当前项目最相关）

- 主流方案几乎都支持 `instruction` 式输入。
- 真正“多码本显式交错”最典型的是 `MOSS-TTS Delay`。
- 你当前在做的“控制 token + 音频 token 交错 AR”属于更强可控的工程化方案，不是现成默认模板。

## 3. 流式相关判断

- `instruction` 本身不阻碍流式；瓶颈在于首包延迟和控制确定性。
- 纯 instruction 更容易出现在线更新不稳定。
- 工程上更稳的组合是：`instruction（高层语义） + 结构化 control token（低层强约束） + KV cache 增量解码`。

## 4. 对当前 legacy 路线的建议

- 继续保持：`decoder-only + 滑动窗口局部注意力 + 每 step 结构化控制`。
- 可把 `MOSS Delay` 作为多码本预测参考，把 `CosyVoice` 作为流式工程（token2wav、hop/chunk）参考。

## 5. 交错序列里的「首组/BOS/EOS」处理（代码核对版）

这一节专门回答：交错 token 路线里，第一组 token 怎么起步？有没有 BOS？结束怎么做？

### 5.1 MOSS-TTS Delay（典型多码本交错）

- 不是“每个 step 一个 BOS”设计。  
- 用的是显式边界 token：
  - `audio_start_token`
  - `audio_assistant_gen_slot_token`
  - `audio_assistant_delay_slot_token`
  - `audio_end_token`
- 音频段是由 `audio_start -> (gen_slot * length + delay_slot * (n_vq-1)) -> audio_end` 构造出来的。
- 真正的 delay 交错是把 `[T, K]` 码本矩阵变为 `[T+K-1, K]`：
  - 第 0 行只有 codebook0 有效，其他列是 `pad_code`
  - 后续行按对角线逐步填满
  - 也就是“首组天然是不完整组”，通过 `pad_code` 显式表示。
- 推理时没有单独 BOS；靠 `audio_start_token` 进入音频态，并在状态机里逐步输出 `gen_slot/delay_slot`，到满足条件后发 `audio_end_token`。
- 另外有一个首步保护：`time_step == 0` 时会禁止输出 delay slot，避免一上来先发错位占位。

关键代码：
- `moss_tts_delay/processing_moss_tts.py` 的 `_replace_audio_placeholders`、`apply_delay_pattern`
- `moss_tts_delay/modeling_moss_tts.py` 的 `generate`（`time_step==0` 屏蔽 delay slot、`audio_start/audio_end` 状态迁移）

### 5.2 CosyVoice（bistream 交错，不是 RVQ delay 交错）

- 这条不是多码本 delay 矩阵，而是“文本块 + 语音 token 块”的双流交错（`mix_ratio`）。
- 有显式 `sos` 和 `eos_token`，并引入 `fill_token` 作为块间调度符号。
- 起始是 `sos`，然后按比例插入 text/speech；当到达设定位置会强制产出 `fill_token`，驱动下一轮块拼接。
- 结束由 `eos_token` 控制。

关键代码：
- `cosyvoice/llm/llm.py` 中 `prepare_lm_input_target`、`inference_bistream`

### 5.3 和我们之前讨论点的对应关系

- “首组没有完整上下文怎么办？”  
  - MOSS Delay 用 `pad_code` + delay slot 机制处理首组不完整，不需要每 step BOS。
- “一定要 BOS 吗？”  
  - 主流实现里，常见是“序列级 SOS/BOS 或显式 start token”，而不是“每个 step 都加 BOS”。
- “结束怎么稳定？”  
  - 主流是显式 `audio_end/eos`，并配合状态机或规则（而不是隐式长度猜测）。

## 6. 对你当前路线的直接启发（简版）

- 如果坚持 `control + audio` 交错，建议沿用：
  - 全局一次性起始符（或虚拟前导 step），不要每 step BOS；
  - 首组无上下文位置用占位（`pad`/`null`/`audio_start`）；
  - 明确的结束机制（step 计数上限 + 显式结束 token 至少二选一，最好两者都有）。
