# Codec Route Comparison

这个表只看对你当前路线最重要的三件事：

1. 是否支持流式解码  
2. 第一层 / 第 0 层码本有没有显式语义约束  
3. 参数量级大概在什么档位

说明：

- 我只把“真正像 codec / tokenizer / 语音输出链”的路线放进主表。
- `AudioLM`、`Bark` 这类更像**生成框架**，不是单独 codec，本表放到最后备注。
- 参数量级只写当前能从本地代码或公开材料**较稳地确认**的范围；拿不到的明确写 `未公开/当前材料不足`。

## 主表

| 路线 | 类型 | 离散结构 | 是否支持流式解码 | 第 0 层/第一层是否有显式语义约束 | 参数量级 | 备注 |
|---|---|---|---|---|---|---|
| `EnCodec` | neural audio codec | 多码本 RVQ | `是`。原论文明确是 streamable，24kHz 最小步长约 `13.3ms` | `否`。标准 RVQ 粗到细，但没有单独把第 0 层蒸馏成语义层 | `小到中等`，大致是 codec-only 的 `tens of M` 档；当前材料里没有稳定单一官方数字 | 优点是成熟、标准、工程接口相对清晰；缺点是第一层不是专门 semantic codebook |
| `Mimi` | streaming speech codec | `1` 个 semantic VQ + `7` 个 acoustic RVQ（常见 `Q=8`） | `是`。Moshi/Mimi 路线就是为实时对话设计 | `是`。公开材料明确写第一层专门做 semantic projection，并从 `WavLM` 蒸馏 | `未在当前材料里看到稳定公开总参数`；从系统定位看是 `codec-only` 中小型模块，不是大模型级 | 这是目前最典型的“显式 semantic first codebook + streaming”路线之一 |
| `LMCodec` | codec + causal token predictor | 基于 `SoundStream` RVQ，显式 coarse-to-fine | `偏是`。论文是 causal low-bitrate speech codec，面向实时/低时延方向 | `弱显式`。不是蒸馏 semantic 第 0 层，而是明确用 coarse tokens 预测 fine tokens | `未查到稳定公开总参数` | 更像“在已有 RVQ codec 上做显式 coarse-to-fine 预测”的代表，不是单独 semantic first codebook |
| `X-Codec` | semantic-aware speech/audio codec | 多码本离散 token | `当前不把它当流式主打能力`；公开材料更强调 semantic shortcoming 修复 | `是，但不是简单第 0 层蒸馏`。核心是增强 codec 的 semantic ability | `未在当前材料里拿到稳定总参数` | 适合你如果后面发现 `EnCodec` token 语义性不够强 |
| `XY-Tokenizer` | semantic-acoustic speech tokenizer | 多码本离散 token | `当前公开材料没有把严格 stateful streaming 当核心卖点`；更偏高质量 tokenizer | `是`。论文摘要明确说它在 text alignment 上强于 distillation-based Mimi / SpeechTokenizer | `中大型，保守估计数亿参数`。官方未直接给总参数；结合公开结构和本地 `xy_tokenizer.ckpt≈2.14GB`，更像 `300M-600M` 这一档，而不是几十 M 小 codec | 官方 `XY-Tokenizer` 训练数据是 `Emilia`； `MOSS-TTSD/VITA` 使用的 `TTSD` 版本 README 写的是 `Emilia + internal general audio` |
| `Qwen3-Omni` 语音链路 | 系统级：Talker + residual predictor + codec decoder | 主 token + residual codebooks | `是`。论文明确写 `streaming multi-codebook codec decoder` | `不靠第 0 层 semantic 蒸馏`，而是主 `Talker` 出第一层、`MTP` 补 residual | `AuT encoder` 已公开两档：`180M` / `300M`；整套系统是多模块、多 B 级 | 更像工程化系统参考，不是单独 codec 论文 |
| `VITA-QinYu` | 系统级：Qwen3 + SenseVoice + XY + speaker stack | 主音频 token + `7` 个 residual audio heads | `是，但更像 chunked streaming-style`；repo 有 streaming full-duplex demo | `不是第 0 层语义蒸馏`；它更像第一层主 token + 其余层补头 | 主 LLM 明确有 `4B / 8B` 两档；本地 `qwen3_sensevoice_xy` 配置接近 `8B` 档；`XY tokenizer` 精确参数未公开 | 适合借它的“主 token + residual heads”输出协议，不建议直接无脑整套照搬 |

## 我对每条路线的判断

### 1. `EnCodec`

- 最稳
- 现成工程接口成熟
- 适合你现在的 `legacy`
- 缺点是：
  - 第 0 层没有专门 semantic 蒸馏
  - 多码本直接做 AR 时，语义和细节混在一起

适合：

- 先把 `legacy` 训通
- 先测 `flatten AR`
- 再测 `codebook0 -> residual codebooks`

### 2. `Mimi`

- 这是最值得看的“下一代 speech codec”替代
- 关键优势不是重建质量本身，而是：
  - 第一层就被设计成 semantic layer
  - 整套系统本来就是为 streaming 设计

适合：

- 你后面如果决定不再坚持 `EnCodec`
- 又明确需要流式 + semantic first codebook

### 3. `LMCodec`

- 不像 `Mimi` 那样在 codec 结构里直接做 semantic first
- 但非常适合参考：
  - **在 RVQ codec 上显式做 coarse-to-fine 预测**

适合：

- 你现在不想换 codec
- 只想测试：
  - `coarse tokens -> fine tokens`
  这条建模路线是否有效

### 4. `X-Codec`

- 更像“如果你觉得普通 codec 语义性不够，该怎么办”的路线
- 它对你最有价值的不是流式，而是：
  - **语义增强**

适合：

- 你后面发现 `EnCodec` 很难把音高/音素/歌词信息学稳

### 5. `XY-Tokenizer`

- 更强调 semantic-acoustic conflict 的解决
- 在 text alignment 上比 Mimi 这类纯 distillation 方案更强
- 但当前我不建议你马上把它作为第一选择，因为：
  - 你已经有 `EnCodec` 路线
  - 直接切它会让变量增多

### 6. `Qwen3-Omni` / `VITA-QinYu`

这两个更像**系统工程参考**，不是单独 codec 选型。

它们真正值得借的是：

- 第一层主音频 token 由主 LLM 负责
- 其余 residual codebooks 用小头补
- 后端 codec decoder 独立、可流式

也就是说，对你最有价值的是：

- **输出协议**
- **系统分工**

不是直接把它们当成一个“codec 模型”。

## 如果按你当前路线给建议

### 现在最稳的实验顺序

1. 继续用 `EnCodec`
2. 先做当前 `legacy` 的纯 decoder-only AR baseline
3. 再测一个最小 coarse-to-fine：
   - `codebook 0`
   - `codebook 1..K-1`
4. 如果 `EnCodec` 语义确实不够，再看 `Mimi` 或 `X-Codec`

### 不建议现在立刻做的

- 直接切 `XY-Tokenizer`
- 直接切 `Qwen3-Omni` 整套链路
- 一边换 codec、一边换大模型、一边改流式协议

## 为什么没把 `AudioLM` / `Bark` 放进主表

因为它们更像：

- hierarchical token generation framework
- coarse/fine 生成协议参考

而不是你现在要选的“底层 codec/tokenizer 本体”。

它们对你当然有参考价值，但主要参考的是：

- 是否值得 coarse-to-fine 生成
- semantic / coarse / fine 三段式是否有效

不是：

- 底层 codec 到底该选哪个

## 结论

如果只给一句话建议：

- **短期**：继续用 `EnCodec`，先测 `codebook0 -> residual codebooks`
- **中期替代**：优先看 `Mimi`
- **如果你判断语义性是主要瓶颈**：再认真看 `X-Codec`
- **系统结构参考**：借 `Qwen3-Omni` / `VITA-QinYu` 的“主 token + residual heads + streaming decoder”
