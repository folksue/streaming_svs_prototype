## 71. OpenCpop 标注格式：segments vs raw TextGrid

### Question
OpenCpop 的官方 `segments` 和 `raw TextGrid` 各是什么格式？`segments` 里是不是显式 phoneme 时间戳？一个 phoneme 会不会跨多个 note？

### Code Evidence
- 官方 README 明确给出 `segments` 的字段：
  - [datasets/opencpop/readme.md:12](datasets/opencpop/readme.md#L12)
  - [datasets/opencpop/readme.md:18](datasets/opencpop/readme.md#L18)
- `segments/train.txt` 的实际样例：
  - [datasets/opencpop/segments/train.txt:1](datasets/opencpop/segments/train.txt#L1)
- `raw TextGrid` 的 tier：
  - [datasets/opencpop/textgrids/2001.TextGrid:11](datasets/opencpop/textgrids/2001.TextGrid#L11)
  - [datasets/opencpop/textgrids/2001.TextGrid:181](datasets/opencpop/textgrids/2001.TextGrid#L181)
  - [datasets/opencpop/textgrids/2001.TextGrid:2331](datasets/opencpop/textgrids/2001.TextGrid#L2331)
  - [datasets/opencpop/textgrids/2001.TextGrid:4481](datasets/opencpop/textgrids/2001.TextGrid#L4481)
  - [datasets/opencpop/textgrids/2001.TextGrid:6631](datasets/opencpop/textgrids/2001.TextGrid#L6631)
  - [datasets/opencpop/textgrids/2001.TextGrid:8781](datasets/opencpop/textgrids/2001.TextGrid#L8781)
  - [datasets/opencpop/textgrids/2001.TextGrid:12487](datasets/opencpop/textgrids/2001.TextGrid#L12487)

### Conclusion
- Confirmed:
  - `segments` 是 utterance-level 的扁平训练标注，一行字段为：
    - `wav_name | text | phoneme | note | note duration | phoneme duration | slur`
  - `segments` 不是显式 `(start, end)` 时间戳，而是把对齐压成 `duration` 列表。
  - `raw TextGrid` 才保留了显式时间边界，tier 包括：
    - `句子`
    - `汉字`
    - `音节`
    - `音高`
    - `音长`
    - `音素`
    - `连音`
- Confirmed:
  - `segments` 的**单个位置**总是 `1 phoneme <-> 1 note`。
  - 但从真实 singing 事件看，一个 phoneme，尤其是元音，在 `slur=1` 时可以跨多个 note；数据通过“重复 phoneme + 不同 note + slur=1”表示。
  - 例如 [datasets/opencpop/segments/train.txt:5](datasets/opencpop/segments/train.txt#L5) 里：
    - `ai -> A4`, `slur=1`
    - `ai -> G#4/Ab4`, `slur=1`
    表示同一发音在连音情况下跨多个 note 延续。
- Inference:
  - `segments` 更适合直接训练。
  - `raw TextGrid` 更适合重建更强的对齐协议。

## 72. CosyVoice：是否是 text-speech 交错自回归，是否有强语义对齐

### Question
CosyVoice，尤其是 CosyVoice 3，是否采用了 text-speech 交错自回归？它有没有显式强语义/时间对齐？

### Paper Evidence
Source: `papers/cosyvoice3.pdf`
- Link index: [papers/cosyvoice3.pdf#page=4](papers/cosyvoice3.pdf#page=4)
- Page 4: `we insert a Finite Scalar Quantization (FSQ) module ... enabling the extraction of low-bitrate speech tokens carrying rich semantics and paralinguistic information`
- Link index: [papers/cosyvoice3.pdf#page=5](papers/cosyvoice3.pdf#page=5)
- Page 5: `directly optimize the LLM to align the output tokens with ASR preference`

### Code Evidence
- `CosyVoice 1` 风格主链更像 prefix-conditioned speech-token AR：
  - [third_party/CosyVoice/cosyvoice/llm/llm.py:91](third_party/CosyVoice/cosyvoice/llm/llm.py#L91)
  - [third_party/CosyVoice/cosyvoice/llm/llm.py:118](third_party/CosyVoice/cosyvoice/llm/llm.py#L118)
  - [third_party/CosyVoice/cosyvoice/llm/llm.py:200](third_party/CosyVoice/cosyvoice/llm/llm.py#L200)
- `CosyVoice 3` 最新代码明确使用固定比例双流混合：
  - [third_party/CosyVoice/cosyvoice/llm/llm.py:317](third_party/CosyVoice/cosyvoice/llm/llm.py#L317)
  - [third_party/CosyVoice/examples/libritts/cosyvoice3/conf/cosyvoice3.yaml:29](third_party/CosyVoice/examples/libritts/cosyvoice3/conf/cosyvoice3.yaml#L29)
- streaming 推理里也按 `mix_ratio` 和 `fill_token` 工作：
  - [third_party/CosyVoice/cosyvoice/llm/llm.py:592](third_party/CosyVoice/cosyvoice/llm/llm.py#L592)

### Conclusion
- Confirmed:
  - `CosyVoice 3` 最新官方代码确实使用固定比例双流混合，默认 `mix_ratio=[5,15]`。
  - 它的对齐增强主要来自：
    - 更语义化的 speech tokenizer
    - `mix_ratio + fill_token` 的局部同步协议
    - DiffRO 的序列级 token-text reward
- Confirmed:
  - 它不是你想要的那种显式强 step-level 对齐协议。
  - 我没有看到：
    - `note-step`
    - `phoneme-span`
    - 显式 shared alignment object
    - 每个 speech token 对应哪个 phoneme/note 的硬对齐表
- Inference:
  - `CosyVoice 3` 更像是“流式同步协议 + 语义增强 token”，不是结构化强对齐系统。

## 73. Decoder-only 交错 Token：为什么 Voxtream 的对齐比 CosyVoice 3 更值得给 singing 参考

### Question
如果坚持 `decoder-only + 交错 token` 范式，不回到传统 duration/length-regulator，那么 `Voxtream` 的对齐机制是否比 `CosyVoice 3` 更值得参考？它算不算“强语义对齐”？

### Paper Evidence
Source: `papers/cosyvoice3.pdf`
- Link index: [papers/cosyvoice3.pdf#page=4](papers/cosyvoice3.pdf#page=4)
- Page 4: `we insert a Finite Scalar Quantization (FSQ) module ... enabling the extraction of low-bitrate speech tokens carrying rich semantics and paralinguistic information`
- Link index: [papers/cosyvoice3.pdf#page=5](papers/cosyvoice3.pdf#page=5)
- Page 5: `directly optimize the LLM to align the output tokens with ASR preference`

### Code Evidence
- `CosyVoice 3` 最新代码仍然使用固定比例双流混合：
  - [third_party/CosyVoice/cosyvoice/llm/llm.py:317](third_party/CosyVoice/cosyvoice/llm/llm.py#L317)
  - [third_party/CosyVoice/examples/libritts/cosyvoice3/conf/cosyvoice3.yaml:29](third_party/CosyVoice/examples/libritts/cosyvoice3/conf/cosyvoice3.yaml#L29)
- `Voxtream` 训练时直接提供 phoneme 到音频时间步的显式索引：
  - [third_party/voxtream/voxtream/dataset.py:54](third_party/voxtream/voxtream/dataset.py#L54)
  - [third_party/voxtream/voxtream/dataset.py:183](third_party/voxtream/voxtream/dataset.py#L183)
- `Voxtream` 模型用 `reorder_phone_emb(...)` 按这些索引重排 phoneme embedding：
  - [third_party/voxtream/voxtream/model.py:102](third_party/voxtream/voxtream/model.py#L102)
- `Voxtream` 推理时额外预测 `pred_shift`，runtime 用它更新下一帧的 phone pointer / indices：
  - [third_party/voxtream/voxtream/model.py:214](third_party/voxtream/voxtream/model.py#L214)
  - [third_party/voxtream/voxtream/utils/generator/runtime.py:25](third_party/voxtream/voxtream/utils/generator/runtime.py#L25)
  - [third_party/voxtream/voxtream/utils/generator/runtime.py:49](third_party/voxtream/voxtream/utils/generator/runtime.py#L49)

### Conclusion
- Confirmed:
  - `CosyVoice 3` 的强项是：更语义化的 speech tokenizer、固定比例 `mix_ratio` 的双流混合序列、以及序列级 `Token2Text` reward alignment。
  - `Voxtream` 的强项不是泛泛的“强语义”，而是更明确的 **时间/状态对齐**：训练时有 `phoneme_embedding_indices`，推理时有 `pred_shift` 驱动的在线状态机。
  - 如果目标是 singing，并且还想坚持 `decoder-only + 交错 token`，那么更值得借鉴的是 `Voxtream` 的“显式 alignment state”思想，而不是 `CosyVoice 3` 的固定比例协议。
- Inference:
  - 对 singing 来说，最自然的折中方案是：保留 `decoder-only + interleaved tokens`，但把 `NOTE_PTR / PHONE_PTR / PROGRESS / SLUR` 这类对齐状态显式 token 化。
- Open uncertainty:
  - `Voxtream` 这套状态机更接近 phoneme-frame TTS；迁移到 singing 时，`note-step` 应该成为主状态单位还是 `phoneme-step` 应该成为主状态单位，还需要实验决定。

## 74. Voxtream 的对齐链：训练时怎么构造，推理时怎么推进

### Question
Voxtream 的对齐到底是怎么做的？训练时有哪些显式监督，推理时 runtime 又是怎么在线维护对齐状态的？

### Code Evidence
- 训练配置直接要求数据提供：
  - `phoneme_embedding_indices`
  - `semantic_label_shifts`
  - `punctuation`
  见 [third_party/voxtream/configs/train.yaml:37](third_party/voxtream/configs/train.yaml#L37), [third_party/voxtream/configs/train.yaml:38](third_party/voxtream/configs/train.yaml#L38), [third_party/voxtream/configs/train.yaml:40](third_party/voxtream/configs/train.yaml#L40)
- `TrainDataset.__getitem__` 会同步裁切：
  - `audio_code`
  - `phoneme_embedding_index`
  - `semantic_label_shift`
  见 [third_party/voxtream/voxtream/dataset.py:170](third_party/voxtream/voxtream/dataset.py#L170), [third_party/voxtream/voxtream/dataset.py:183](third_party/voxtream/voxtream/dataset.py#L183), [third_party/voxtream/voxtream/dataset.py:204](third_party/voxtream/voxtream/dataset.py#L204)
- 训练标签不是只有 semantic token，还把 `semantic_label_shift` 合成到联合标签里：
  - [third_party/voxtream/voxtream/dataset.py:279](third_party/voxtream/voxtream/dataset.py#L279)
  - [third_party/voxtream/voxtream/dataset.py:304](third_party/voxtream/voxtream/dataset.py#L304)
- 模型里 `reorder_phone_emb(...)` 直接按 `phoneme_embedding_indices` 将 phone embedding 重排到音频时间轴：
  - [third_party/voxtream/voxtream/model.py:118](third_party/voxtream/voxtream/model.py#L118)
- 推理时 `generate_frame(...)` 会同时输出：
  - `frame`
  - `pred_shift`
  见 [third_party/voxtream/voxtream/model.py:193](third_party/voxtream/voxtream/model.py#L193), [third_party/voxtream/voxtream/model.py:301](third_party/voxtream/voxtream/model.py#L301)
- runtime 用 `pred_shift` 查 `phoneme_index_map`，更新下一帧的 `phone_emb_indices`：
  - [third_party/voxtream/voxtream/utils/generator/runtime.py:37](third_party/voxtream/voxtream/utils/generator/runtime.py#L37)
  - [third_party/voxtream/voxtream/utils/generator/runtime.py:49](third_party/voxtream/voxtream/utils/generator/runtime.py#L49)
- generator 主循环每一帧都执行：
  - `reorder_phone_emb`
  - `generate_frame`
  - `update_indices_and_tokens`
  见 [third_party/voxtream/voxtream/generator.py:242](third_party/voxtream/voxtream/generator.py#L242), [third_party/voxtream/voxtream/generator.py:245](third_party/voxtream/voxtream/generator.py#L245), [third_party/voxtream/voxtream/generator.py:279](third_party/voxtream/voxtream/generator.py#L279)

### Conclusion
- Confirmed:
  - `Voxtream` 的对齐不是“让 LM 自己猜 text/speech 比例”，而是训练数据里就显式给出每个音频时间步该去取哪些 phoneme embedding。
  - 它的对齐监督至少包含两部分：
    - `phoneme_embedding_indices`：当前音频步绑定哪几个 phoneme
    - `semantic_label_shift`：当前步的离散状态标签
  - 模型训练时学的是：
    - 当前帧看哪几个 phoneme
    - 当前帧对应哪个 semantic token
    - 当前帧的状态/shift 是什么
- Confirmed:
  - 推理时它不是依赖固定比例，也不是一次性知道整条对齐表。
  - 它通过 `pred_shift -> phoneme_index_map -> phone_emb_indices` 形成一个在线状态机，逐帧推进 phoneme pointer。
- Inference:
  - 如果迁移到 singing，最值得借的是这套“显式对齐索引 + 推理时状态推进”的模式，而不是它具体使用 phoneme-frame 作为主状态单位。

## 75. Voxtream 的“状态机”具体是什么

### Question
Voxtream 里说的在线状态机具体怎么实现？它到底维护了哪些状态、每一步如何更新？

### Code Evidence
- 生成循环初始化的 `FrameState`：
  - [third_party/voxtream/voxtream/generator.py:185](third_party/voxtream/voxtream/generator.py#L185)
- 每帧生成后调用 `update_indices_and_tokens(...)`：
  - [third_party/voxtream/voxtream/generator.py:279](third_party/voxtream/voxtream/generator.py#L279)
- `update_indices_and_tokens(...)` 的状态转移逻辑：
  - [third_party/voxtream/voxtream/utils/generator/runtime.py:29](third_party/voxtream/voxtream/utils/generator/runtime.py#L29)
  - [third_party/voxtream/voxtream/utils/generator/runtime.py:37](third_party/voxtream/voxtream/utils/generator/runtime.py#L37)
  - [third_party/voxtream/voxtream/utils/generator/runtime.py:49](third_party/voxtream/voxtream/utils/generator/runtime.py#L49)
  - [third_party/voxtream/voxtream/utils/generator/runtime.py:66](third_party/voxtream/voxtream/utils/generator/runtime.py#L66)
- `pred_shift` 是由 `generate_frame(...)` 返回的：
  - [third_party/voxtream/voxtream/model.py:244](third_party/voxtream/voxtream/model.py#L244)
  - [third_party/voxtream/voxtream/model.py:301](third_party/voxtream/voxtream/model.py#L301)

### Conclusion
- Confirmed:
  - 这个“状态机”不是一个抽象概念，而是 runtime 里一个明确的数据结构 `FrameState` 加上 `phoneme_index_map` 驱动的转移规则。
  - 它维护的核心状态有四个：
    - `phone_emb_indices`：下一帧该取哪些 phoneme embedding
    - `phone_emb_max_idx`：当前已经推进到的最大 phoneme 索引
    - `eos_idx`：何时应该结束
    - `state_counter`：某个 `(start, end)` phone 对齐状态已经重复了几次
- Confirmed:
  - 每一帧模型会输出 `pred_shift`，runtime 通过 `ctx.phoneme_index_map[str(pred_shift)]` 查到：
    - `shift`
    - `num_tokens`
  - 然后用：
    - `start = frame_state.phone_emb_max_idx + shift`
    - `val = range(start, start + num_tokens)`
    构造下一帧的 `phone_emb_indices`
- Confirmed:
  - 如果某个 `(start, end)` 状态重复太多次，并且已经接近结尾，runtime 会强制把 `start` 再往前推一步，避免卡死在同一个状态。
- Inference:
  - 这本质上就是一个“单调前进的 phoneme pointer 状态机”：
    - 模型不直接预测完整对齐表
    - 只预测当前这一步该不该前进、前进多少、下一帧看几个 phoneme

## 76. Anticipatory Music Transformer 里的 control 是什么

### Question
`Anticipatory Music Transformer` 所谓的 `control` 具体控制什么？是不是像 singing 里的 note/phoneme control？

### Paper Evidence
Source: `papers/anticipatory_music_transformer.pdf`
- Link index: [papers/anticipatory_music_transformer.pdf#page=1](papers/anticipatory_music_transformer.pdf#page=1)
- Page 1: `Generating an accompaniment to a given melody is an example of a control task: we seek the ability to generate an accompaniment (the events) conditioned on a given melody (the controls).`
- Link index: [papers/anticipatory_music_transformer.pdf#page=3](papers/anticipatory_music_transformer.pdf#page=3)
- Page 3: `We focus on infilling control, whereby the controls u1:K share a vocabulary with the events e1:N. Given a user-specified set of K events u1:K, we would like to generate a complete realization of the process e1:N such that u1:K ⊆ e1:N .`
- Link index: [papers/anticipatory_music_transformer.pdf#page=1](papers/anticipatory_music_transformer.pdf#page=1)
- Page 1: `The controls are a subset of the events themselves`

### Conclusion
- Confirmed:
  - 这篇里的 `control` 不是像 TTS/SVS 那样的 `phoneme / note duration / slur` 结构化控制字段。
  - 它的 `control` 更具体地说是：**已经给定的一部分音乐事件本身**，典型例子就是“给定 melody，生成 accompaniment”。
  - 论文关注的是 `infilling control`：给定部分事件 `u1:K`，要求生成完整事件序列 `e1:N`，并保证这些给定控制事件包含在结果里。
- Inference:
  - 所以这篇更适合给你参考的是：
    - control/event 怎么交错序列化
    - 异步 control 怎么放进 decoder-only 自回归建模
  - 不太适合直接照搬它的 control 定义，因为你的 singing control 是显式结构化字段，不是“部分已知音乐事件子集”。

## 77. Qwen3 路线该用什么音频编码器，Qwen3-Omni 实际用了什么

### Question
如果后续做 `Qwen3` singing 路线，是否需要类似 `EnCodec` 之外的音频编码器？`Qwen3-Omni` 在音频输入和语音输出这两侧实际用了什么结构？

### Paper Evidence
Source: `papers/qwen3omni.pdf`
- Link index: [papers/qwen3omni.pdf#page=1](papers/qwen3omni.pdf#page=1)
- Page 1: `the Talker autoregressively predicts discrete speech codecs using a multi-codebook scheme`
- Link index: [papers/qwen3omni.pdf#page=4](papers/qwen3omni.pdf#page=4)
- Page 4: `Qwen3-Omni employs the AuT encoder as the audio encoder to obtain general purpose audio representations at a token rate of 12.5Hz.`
- Page 4: `Audio Transformer (AuT) is an attention-encoder-decoder model ... trained from scratch on 20 million hours of supervised audio data.`
- Page 4: `filter bank features of the audio are downsampled 8 times using Conv2D blocks ... reducing the token rate to 12.5 Hz`
- Link index: [papers/qwen3omni.pdf#page=5](papers/qwen3omni.pdf#page=5)
- Page 5: `each frame of the audio representation corresponds to approximately an 80 ms segment`
- Link index: [papers/qwen3omni.pdf#page=6](papers/qwen3omni.pdf#page=6)
- Page 6: `once Talker generates the first token, the MTP module predicts the remaining tokens for the current frame`
- Page 6: `The MTP Module is an ultra-lightweight fixed-step autoregressive dense transformer`
- Page 6: `streaming multi-codebook codec decoder that only attends to the left context`

### Code Evidence
- Talker 主头先预测主 codec token，再挂一个 residual code predictor：
  - [`.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py:2968`](.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L2968)
- `code_predictor` 为剩余 code groups 维护独立 embedding / lm_head：
  - [`.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py:2456`](.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L2456)
- `Code2Wav` 用多码本 embedding 均值后接 Transformer + causal Conv/TransposedConv 解 wav：
  - [`.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py:3704`](.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L3704)
- 推理时 `talker.generate(...) -> talker_codes -> code2wav.chunked_decode(...)`：
  - [`.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py:4035`](.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L4035)
  - [`.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py:4047`](.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L4047)

### Conclusion
- Confirmed:
  - `Qwen3-Omni` 的**输入音频编码器**不是 Whisper，而是它自己训练的 `AuT encoder`。它把 filter-bank 音频特征经 `Conv2D + attention encoder-decoder` 压到 `12.5Hz`，也就是大约 `80ms` 一帧的通用音频表征。
  - `Qwen3-Omni` 的**输出语音侧**不是直接预测 waveform，而是：
    - `Talker` 先 AR 预测当前帧的主 codec token
    - `MTP/code_predictor` 再补剩余 residual codebooks
    - `Code2Wav` 用 left-context-only 的流式多码本 codec decoder 解成波形
- Confirmed:
  - 它的生成侧是典型的“高层 LLM + 轻量 residual code predictor + 流式 codec decoder”三段式，不是让一个大模型直接 AR 全部底层 wav/token。
- Inference:
  - 对后续 `Qwen3` singing 路线，确实需要一个**音频编码器**，但用途应主要是：
    - 编码参考音频 / prompt audio / 风格音频
    - 或做通用音频理解输入
  - 如果输入本身已经是显式 `NOTE / PHONEME / SLUR / PROGRESS` 控制，则不需要再用一个大 ASR encoder 去重新“理解”这些结构化条件。
  - 更合理的设计是：
    - 显式控制直接 token/embedding 化
    - 可选再加一个小型参考音频 encoder
    - `Qwen3` 只负责高层 semantic/singing token
    - 后端再接轻量 streaming acoustic / codec decoder

## 78. VITA-QinYu 用的什么音频编码方式，AuT 有没有独立论文

### Question
`third_party/VITA-QinYu` 用的是什么音频编码方式？`AuT` 有没有独立原始论文，它大概是什么结构和参数量级？

### Paper Evidence
Source: `papers/qwen3omni.pdf`
- Link index: [papers/qwen3omni.pdf#page=4](papers/qwen3omni.pdf#page=4)
- Page 4: `Audio Transformer (AuT) is an attention-encoder-decoder model ...`
- Page 4: `downsampled 8 times using Conv2D blocks ... reducing the token rate to 12.5 Hz`
- Page 4: `Qwen3-Omni employs the AuT encoder as the audio encoder`

Source: `papers/qwen3_asr.pdf`
- Link index: [papers/qwen3_asr.pdf#page=2](papers/qwen3_asr.pdf#page=2)
- Page 2: `Speech to recognize is first fed to AuT encoder`
- Page 2: `AuT is an attention-encoder-decoder (AED) based ASR model which conducts 8 times downsampling to Fbank feature with 128 dimensions, yielding a 12.5Hz token rate audio encoder.`
- Page 2: `Qwen3-ASR-1.7B ... an AuT encoder with 300M parameters and 1024 hidden size.`
- Page 2: `Qwen3-ASR-0.6B ... an AuT encoder with 180M parameters and a hidden size of 896.`
- Page 2: `AuT pretraining ... approximately 40 million hours of pseudo-labeled ASR data`

### Code Evidence
- `VITA-QinYu` README 直接写的是 `parallel multi-codebook audio tokens`：
  - [third_party/VITA-QinYu/README.md:18](third_party/VITA-QinYu/README.md#L18)
- tokenizer 类型是 `sensevoice_xytokenizer_speaker`：
  - [third_party/VITA-QinYu/configs/config.yaml:26](third_party/VITA-QinYu/configs/config.yaml#L26)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer.py:44](third_party/VITA-QinYu/vita_qinyu/tokenizer.py#L44)
- 该 tokenizer 增加了 `8` 个 codebook、每层 `1024` + `pad` 的音频 token：
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:58](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L58)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:82](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L82)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:171](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L171)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:323](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L323)
- `SenseVoiceXYSpkTokenizer` 连续输入侧返回的是 `SenseVoice` 风格 fbank/speech features，而离散输出侧才是 XY tokenizer 的 8 层 audio codes：
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:174](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L174)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:208](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L208)
- 生成时主文本头先给第一层音频 token，再补其余 `7` 层：
  - [third_party/VITA-QinYu/utils/xy_inference_utils.py:96](third_party/VITA-QinYu/utils/xy_inference_utils.py#L96)
  - [third_party/VITA-QinYu/utils/xy_inference_utils.py:139](third_party/VITA-QinYu/utils/xy_inference_utils.py#L139)
  - [third_party/VITA-QinYu/vita_qinyu/models/qwen3_sensevoice_xy_v4_51_3/modeling_qwen3.py:951](third_party/VITA-QinYu/vita_qinyu/models/qwen3_sensevoice_xy_v4_51_3/modeling_qwen3.py#L951)
  - [third_party/VITA-QinYu/vita_qinyu/models/qwen3_sensevoice_xy_v4_51_3/modeling_qwen3.py:1100](third_party/VITA-QinYu/vita_qinyu/models/qwen3_sensevoice_xy_v4_51_3/modeling_qwen3.py#L1100)
  - [third_party/VITA-QinYu/vita_qinyu/models/qwen3_sensevoice_xy_v4_51_3/modeling_qwen3.py:1131](third_party/VITA-QinYu/vita_qinyu/models/qwen3_sensevoice_xy_v4_51_3/modeling_qwen3.py#L1131)

### Conclusion
- Confirmed:
  - `VITA-QinYu` 不是 `EnCodec`，也不是 `Mimi`。它用的是自家的 `sensevoice_xytokenizer_speaker`，核心是 `XY_Tokenizer`，离散音频 token 共有 `8` 个 codebook，每层 `1024` 个 token，加少量 pad token。
  - `VITA-QinYu` 的输入音频和输出音频不是完全同一种表示：
    - 用户侧连续输入走 `SenseVoice` / fbank / speech feature 路线
    - 生成侧输出的是 XY tokenizer 的多码本离散 audio tokens
  - 它的生成也不是单纯 flatten 全码本 AR，而是：
    - 主 LM 先出第一层音频 token
    - 剩余 `7` 层通过 `audio_heads` / MTP 分支补出来
- Confirmed:
  - 目前公开可查到的 `AuT` 最完整来源不是独立“原始论文”，而是 `Qwen3-Omni Technical Report` 与 `Qwen3-ASR Technical Report`。
  - 从 `Qwen3-ASR` 报告看，AuT 是：
    - `attention-encoder-decoder (AED)` ASR 模型
    - 输入 `128-dim Fbank`
    - `Conv2D` 前端做 `8x` 下采样
    - 输出 `12.5Hz` 音频表征
    - 动态 attention window `1s - 8s`
  - 已公开的量级有两档：
    - `AuT-300M`, hidden size `1024`（配 Qwen3-ASR-1.7B）
    - `AuT-180M`, hidden size `896`（配 Qwen3-ASR-0.6B）
- Open uncertainty:
  - 我没有找到一篇独立署名的 “AuT original paper”。当前公开材料里，AuT 更像 Qwen 音频路线内部命名的 encoder family，而不是一篇单独发表的经典架构论文。

## 79. VITA-QinYu 的 SenseVoice + XY tokenizer 具体是什么结构，参数量级如何

### Question
`VITA-QinYu` 里说的 `SenseVoice + XY tokenizer` 到底是不是一个单一音频编码器？它内部由哪些部分组成，结构和参数量级大概是什么水平？

### Code Evidence
- `VITA-QinYu` 配置把音频 tokenizer 路径写成三段：`xy_tokenizer.ckpt`、`campplus_cn_common.bin`、`SenseVoiceSmall`，类型是 `sensevoice_xytokenizer_speaker`：
  - [third_party/VITA-QinYu/configs/config.yaml:10](third_party/VITA-QinYu/configs/config.yaml#L10)
  - [third_party/VITA-QinYu/configs/config.yaml:14](third_party/VITA-QinYu/configs/config.yaml#L14)
  - [third_party/VITA-QinYu/configs/config.yaml:26](third_party/VITA-QinYu/configs/config.yaml#L26)
- `get_audio_tokenizer(...)` 直接返回 `SenseVoiceXYSpkTokenizer(...)`，说明这是一层组合封装，不是单个模型：
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer.py:44](third_party/VITA-QinYu/vita_qinyu/tokenizer.py#L44)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer.py:62](third_party/VITA-QinYu/vita_qinyu/tokenizer.py#L62)
- `SenseVoiceXYSpkTokenizer.load_model()` 内部实际加载了三件东西：
  - `SenseVoiceSmall.from_pretrained(...)`
  - `XY_Tokenizer.load_from_checkpoint(...)`
  - `CAMPPlus(feat_dim=80, embedding_size=192)`
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:135](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L135)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:143](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L143)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:149](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L149)
- 离散音频 token 空间是 `8` 个 codebook，每层 `1024`，再加 `8` 个 pad token：
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:58](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L58)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:67](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L67)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:80](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L80)
- 连续输入侧并不是直接离散 token 化，而是把音频重采样到 `16kHz`，再用 `extract_fbank(...)` 得到 `SenseVoice` 风格的连续 speech features：
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:95](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L95)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:174](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L174)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:193](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L193)
- 离散输出侧通过 `self.xy_tokenizer.encode(...)` 得到 `codes_list`，解码时再把 `8 x T` 码本 token 送回 `xy_tokenizer.decode(...)`：
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:167](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L167)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:208](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L208)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:249](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L249)
- speaker 侧另外走 `CAMPPlus`，输出 `192` 维 speaker embedding：
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:149](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L149)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:255](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L255)
- `SenseVoiceSmall` 本地可见结构不是超大模型，encoder 默认是：
  - `input_size=80`
  - `output_size=256`
  - `attention_heads=4`
  - `linear_units=2048`
  - `num_blocks=6`
  - [third_party/VITA-QinYu/vita_qinyu/models/vita_utu_mtp_sensevoice_xy_v4_51_3/modeling_sensevoice.py:523](third_party/VITA-QinYu/vita_qinyu/models/vita_utu_mtp_sensevoice_xy_v4_51_3/modeling_sensevoice.py#L523)
  - [third_party/VITA-QinYu/vita_qinyu/models/vita_utu_mtp_sensevoice_xy_v4_51_3/modeling_sensevoice.py:660](third_party/VITA-QinYu/vita_qinyu/models/vita_utu_mtp_sensevoice_xy_v4_51_3/modeling_sensevoice.py#L660)
- 主 `VITA-QinYu` LLM 这份本地配置对应的量级是：
  - `hidden_size=2560`
  - `intermediate_size=9728`
  - `num_hidden_layers=40`
  - `num_attention_heads=32`
  - `num_key_value_heads=8`
  - `num_codebook=8`
  - `codebook_size=1032`
  - `speaker_embedding_size=192`
  - [third_party/VITA-QinYu/vita_qinyu/models/vita_utu_mtp_sensevoice_xy_v4_51_3/config.json](third_party/VITA-QinYu/vita_qinyu/models/vita_utu_mtp_sensevoice_xy_v4_51_3/config.json)
  - [third_party/VITA-QinYu/vita_qinyu/models/vita_utu_mtp_sensevoice_xy_v4_51_3/configuration_utu_v1.py:169](third_party/VITA-QinYu/vita_qinyu/models/vita_utu_mtp_sensevoice_xy_v4_51_3/configuration_utu_v1.py#L169)
- 生成时第一层音频 token 走主 `lm_head`，其余 `7` 层走单独的 `audio_heads`：
  - [third_party/VITA-QinYu/utils/xy_inference_utils.py:96](third_party/VITA-QinYu/utils/xy_inference_utils.py#L96)
  - [third_party/VITA-QinYu/utils/xy_inference_utils.py:118](third_party/VITA-QinYu/utils/xy_inference_utils.py#L118)
  - [third_party/VITA-QinYu/utils/xy_inference_utils.py:139](third_party/VITA-QinYu/utils/xy_inference_utils.py#L139)
  - [third_party/VITA-QinYu/vita_qinyu/models/qwen3_sensevoice_xy_v4_51_3/modeling_qwen3.py:930](third_party/VITA-QinYu/vita_qinyu/models/qwen3_sensevoice_xy_v4_51_3/modeling_qwen3.py#L930)
  - [third_party/VITA-QinYu/vita_qinyu/models/qwen3_sensevoice_xy_v4_51_3/modeling_qwen3.py:943](third_party/VITA-QinYu/vita_qinyu/models/qwen3_sensevoice_xy_v4_51_3/modeling_qwen3.py#L943)
  - [third_party/VITA-QinYu/vita_qinyu/models/qwen3_sensevoice_xy_v4_51_3/modeling_qwen3.py:1131](third_party/VITA-QinYu/vita_qinyu/models/qwen3_sensevoice_xy_v4_51_3/modeling_qwen3.py#L1131)

### Conclusion
- Confirmed:
  - `SenseVoice + XY tokenizer` 在 `VITA-QinYu` 里不是一个单一编码器，而是一个三件套封装：
    1. `SenseVoiceSmall`：处理连续用户音频，输出连续 speech features
    2. `XY_Tokenizer`：把生成音频表示成 `8` 层多码本离散 token，并负责解码回 waveform
    3. `CAMPPlus`：提取 `192` 维 speaker embedding
  - 所以它的“音频编码方式”是非对称的：
    - 输入理解侧偏 `SenseVoice`
    - 输出生成侧偏 `XY` 多码本 codec/tokenizer
- Confirmed:
  - 本地能直接确认的 `SenseVoiceSmall` 结构是一个比较小的前端 encoder，不是大模型量级；可见默认结构大约是 `80-dim 输入 + 6 个 block + 256 hidden + 4 heads + 2048 FFN`。
  - 主 `VITA-QinYu` LLM 则明显大得多；本地这份配置是 `40` 层、`2560` hidden、`32` heads，属于多十亿参数级大模型配套结构。
- Inference:
  - `XY_Tokenizer` 自身应该是一个中小型专用 codec/tokenizer 模块，但当前仓库没有 `xy_tokenizer.model` 的源码细节和 checkpoint 元数据，所以我没法从本地证据给出它的精确总参数量。
  - 从系统设计上看，`VITA-QinYu` 不是“一个音频 encoder 统一搞定所有事”，而是把：
    - 连续理解
    - 离散声码器
    - speaker 表征
    拆成了三个专用模块。
- Open uncertainty:
  - `XY_Tokenizer` 的精确层数、hidden size、总参数量级，当前本地代码里拿不到，只能确认它通过 checkpoint 加载、负责 `8 x 1024(+pad)` 多码本离散音频 token 的编解码。

## 80. AuT 输出的是连续编码还是离散 token，这些编码怎么送进 Qwen3-Omni

### Question
`Qwen` 的 `AuT` 输出到底是连续编码还是离散 token？如果是连续编码，它是怎么送给大模型的？

### Paper Evidence
Source: `papers/qwen3_asr.pdf`
- Link index: [papers/qwen3_asr.pdf#page=2](papers/qwen3_asr.pdf#page=2)
- Page 2: `Speech to recognize is first fed to AuT encoder`
- Page 2: `yielding a 12.5Hz token rate audio encoder`

Source: `papers/qwen3omni.pdf`
- Link index: [papers/qwen3omni.pdf#page=4](papers/qwen3omni.pdf#page=4)
- Page 4: `Qwen3-Omni employs the AuT encoder as the audio encoder`

### Code Evidence
- `get_audio_features(...)` 的 docstring 直接写的是：`Encodes audios into continuous embeddings that can be forwarded to the language model.`
  - [/.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py:1905](.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L1905)
- 具体流程是：
  - 先 `self.audio_tower(...)`
  - 再取 `audio_outputs.last_hidden_state`
  - [/.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py:1922](.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L1922)
  - [/.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py:1927](.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L1927)
- 在 thinker `forward(...)` 里，先得到文本 `inputs_embeds`，再把 `audio_features` 直接 `masked_scatter` 到音频占位符位置：
  - [/.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py:2062](.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L2062)
  - [/.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py:2070](.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L2070)
  - [/.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py:2077](.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L2077)
- placeholder mask 是按 `audio_token_id` 在输入序列里找出来的：
  - [/.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py:1951](.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L1951)
  - [/.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py:1959](.venv_pdf/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L1959)

### Conclusion
- Confirmed:
  - `AuT` 输出的是**连续 audio embeddings**，不是离散 codec token。
  - 这些连续编码不会先量化成离散 id 再送入大模型，而是直接作为 `inputs_embeds` 的一部分塞进 `Thinker`。
  - 具体做法是：文本序列里先放音频占位符 `audio_token_id`，然后用 `audio_tower` 产生的 `audio_features` 直接替换这些占位符位置上的 embedding。
- Inference:
  - `Qwen3-Omni` 对输入音频采用的是“连续 memory 注入”路线，而不是像 `VITA-QinYu` 输出侧那样的多码本离散 token 路线。
  - 所以 `AuT` 更像多模态理解前端，而不是可生成的音频 codec/tokenizer。

## 81. CosyVoice3 有没有多层码本，用的是什么编码器

### Question
`CosyVoice3` 有用多层码本吗？它的音频编码器 / speech tokenizer 到底是什么？

### Paper Evidence
Source: `papers/cosyvoice3.pdf`
- Link index: [papers/cosyvoice3.pdf#page=4](papers/cosyvoice3.pdf#page=4)
- Page 4: `we insert a Finite Scalar Quantization (FSQ) module into the voice encoder of MinMo`
- Page 4: `enabling the extraction of low-bitrate speech tokens carrying rich semantics and paralinguistic information`
- Page 4: `Speech tokenizer training consists of two stages`

### Code Evidence
- `CosyVoice3` 初始化时加载的是 `speech_tokenizer_v3.onnx`，不是多码本 codec checkpoint：
  - [third_party/CosyVoice/cosyvoice/cli/cosyvoice.py:202](third_party/CosyVoice/cosyvoice/cli/cosyvoice.py#L202)
  - [third_party/CosyVoice/cosyvoice/cli/cosyvoice.py:205](third_party/CosyVoice/cosyvoice/cli/cosyvoice.py#L205)
- 运行时 `token2wav_cosyvoice3.py` 里，audio tokenizer 是 `s3tokenizer.load_model(...)`，然后直接 `quantize(...)` 返回 `prompt_speech_tokens` 和长度，后面按一条 token 序列取出：
  - [third_party/CosyVoice/runtime/triton_trtllm/token2wav_cosyvoice3.py:129](third_party/CosyVoice/runtime/triton_trtllm/token2wav_cosyvoice3.py#L129)
  - [third_party/CosyVoice/runtime/triton_trtllm/token2wav_cosyvoice3.py:225](third_party/CosyVoice/runtime/triton_trtllm/token2wav_cosyvoice3.py#L225)
  - [third_party/CosyVoice/runtime/triton_trtllm/token2wav_cosyvoice3.py:228](third_party/CosyVoice/runtime/triton_trtllm/token2wav_cosyvoice3.py#L228)
- `CosyVoice3LM` 的 LLM 输出头就是单个 `speech_token_size + 200` 词表，不是多码本并行头：
  - [third_party/CosyVoice/cosyvoice/llm/llm.py:687](third_party/CosyVoice/cosyvoice/llm/llm.py#L687)
  - [third_party/CosyVoice/cosyvoice/llm/llm.py:696](third_party/CosyVoice/cosyvoice/llm/llm.py#L696)
- 训练和流式推理里，`speech_token` 都按一维序列处理：
  - 训练：`speech_token_emb = self.speech_embedding(speech_token)`  
    [third_party/CosyVoice/cosyvoice/llm/llm.py:374](third_party/CosyVoice/cosyvoice/llm/llm.py#L374)
  - 流式：逐个 `top_ids` 产出并 `yield`  
    [third_party/CosyVoice/cosyvoice/llm/llm.py:635](third_party/CosyVoice/cosyvoice/llm/llm.py#L635)
  - 后续一步一步把 `top_ids` 查 embedding 回灌  
    [third_party/CosyVoice/cosyvoice/llm/llm.py:641](third_party/CosyVoice/cosyvoice/llm/llm.py#L641)

### Conclusion
- Confirmed:
  - `CosyVoice3` 在 LLM 侧**不是多层码本生成**。它的 `speech_token` 在代码里表现为一条单序列 token stream，不是 `K` 个 codebook 并行或分层输出。
  - `CosyVoice3` 的 speech tokenizer v3 论文里明确是：
    - 以 `MinMo` 的 voice encoder 为基础
    - 在 voice encoder 里插入 `FSQ`
    - 训练成低码率、带语义和副语言信息的 speech token 提取器
- Inference:
  - `CosyVoice3` 更像“单序列语义 speech token + flow/vocoder 渲染”路线，而不是 `EnCodec/Mimi/VITA-QinYu` 这种多码本音频 token 直接生成路线。
  - 所以如果你关心的是多码本自回归协议，`CosyVoice3` 对这件事的参考价值不大；它更适合参考的是“语义化 speech tokenizer + streaming text/speech LM”。

## 82. VITA-QinYu 里的 SenseVoice 和 XY tokenizer 到底是什么，怎么训练，原模型有没有音频输入能力

### Question
`VITA-QinYu` 里的 `SenseVoice` 和 `XY tokenizer` 各自负责什么？它们公开能确认的训练信息是什么？原模型本身有没有原生音频输入能力？

### Paper Evidence
Source: `https://huggingface.co/FunAudioLLM/SenseVoiceSmall/raw/main/README.md`
- Link index: [https://huggingface.co/FunAudioLLM/SenseVoiceSmall/raw/main/README.md](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/raw/main/README.md)
- Line 0: `SenseVoice is a speech foundation model with multiple speech understanding capabilities, including automatic speech recognition (ASR), spoken language identification (LID), speech emotion recognition (SER), and audio event detection (AED).`
- Line 8: `Trained with over 400,000 hours of data, supporting more than 50 languages`
- Line 9: `The SenseVoice-Small model utilizes a non-autoregressive end-to-end framework`
- Line 23: `Although trained exclusively on speech data, SenseVoice can still function as a standalone event detection model.`

### Code Evidence
- `SenseVoiceXYSpkTokenizer` 同时加载三块模型：
  - `SenseVoiceSmall`
  - `XY_Tokenizer`
  - `CAMPPlus`
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:135](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L135)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:138](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L138)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:141](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L141)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:149](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L149)
- `XY` 音频 token 空间是 `8 x (1024 + 8)`，也就是 8 个 codebook、每层 1024 个有效 token 外加 8 个 pad：
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:82](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L82)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:85](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L85)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:86](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L86)
- `is_discrete=True` 时，音频被 `xy_tokenizer.encode(...)` 编成 `8 x T` 的离散码本 token：
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:165](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L165)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:168](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L168)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:249](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L249)
- `is_contiguous=True` 时，音频经过重采样和 `extract_fbank(...)`，返回连续 speech features，而不是离散 token：
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:175](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L175)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:198](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L198)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:204](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L204)
- 角色分配上，离散音频默认用于 `assistant/gpt`，连续音频默认用于 `user/human`：
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:217](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L217)
  - [third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py:222](third_party/VITA-QinYu/vita_qinyu/tokenizer_sensevoice_xyspk.py#L222)
- `SenseVoiceSmall` 本地配置能确认它是一个非小型的连续音频理解前端：
  - `output_size: 512`
  - `attention_heads: 4`
  - `num_blocks: 50`
  - [third_party/VITA-QinYu/pretrained_meta/SenseVoiceSmall/config.yaml:1](third_party/VITA-QinYu/pretrained_meta/SenseVoiceSmall/config.yaml#L1)
  - [third_party/VITA-QinYu/pretrained_meta/SenseVoiceSmall/config.yaml:6](third_party/VITA-QinYu/pretrained_meta/SenseVoiceSmall/config.yaml#L6)
- `XY tokenizer` 的配置能确认它本身是一整套 codec，而不是一个轻量 residual predictor：
  - semantic encoder: `d_model=768`, `12` 层
  - acoustic encoder: `d_model=768`, `12` 层
  - `num_quantizers: 8`, `codebook_size: 1024`
  - acoustic decoder: `12` 层
  - vocos: `dim=512`, `num_layers=30`
  - [third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml:25](third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml#L25)
  - [third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml:50](third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml#L50)
  - [third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml:79](third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml#L79)
  - [third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml:103](third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml#L103)
  - [third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml:117](third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml#L117)

### Conclusion
- Confirmed:
  - `SenseVoice` 在 `VITA-QinYu` 里负责**连续音频理解前端**，不是生成 codec。公开模型卡说明它是一个多任务 speech foundation model，训练于 `40万小时+` 语音数据，重点是 ASR/LID/SER/AED，而且公开材料明确写了它是**speech-only training**。
  - `XY tokenizer` 在 `VITA-QinYu` 里负责**离散语音生成 token**。它输出 `8 x T` 的 RVQ 风格多码本 token，并能把这些 token decode 回音频。
  - 原模型**有原生音频输入能力**，而且是两条输入路径：
    - 用户侧音频：连续 `fbank/speech features`（`is_contiguous=True`）
    - 助手侧音频：离散 `XY` 音频 token（`is_discrete=True`）
- Inference:
  - `VITA-QinYu` 不是单一“codec+LLM”结构，而是一个 hybrid speech-text stack：`SenseVoice` 负责听、`XY tokenizer` 负责说、`Qwen3/UTU` 主干负责统一语言建模。
  - 这也是为什么它比单纯的 `EnCodec + LLM` 更像完整 spoken language model。
- Open uncertainty:
  - 公开代码和配置能看清 `XY tokenizer` 的结构，但我目前没有看到它的独立论文或官方训练数据说明，所以**不能严谨断言它具体用了哪些语料、是否包含歌声数据**。目前能确认的只是它的架构和 token 协议，而不是完整训练 recipe。

## 83. XY tokenizer 的结构到底是什么

### Question
`XY tokenizer` 内部到底是什么结构？它是不是单纯的 RVQ codec，还是做了语义/声学双通道设计？

### Paper Evidence
Source: `https://huggingface.co/gaoyang07/XY_Tokenizer`
- Link index: [https://huggingface.co/gaoyang07/XY_Tokenizer](https://huggingface.co/gaoyang07/XY_Tokenizer)
- Model card: `Dual-channel modeling: Simultaneously captures semantic meaning and acoustic details`
- Model card: `1kbps bitrate with RVQ8 quantization at 12.5Hz frame rate`
- Model card: `24kHz output`

Source: `https://arxiv.org/abs/2506.23325`
- Link index: [https://arxiv.org/abs/2506.23325](https://arxiv.org/abs/2506.23325)
- Abstract: `mitigates the conflict between semantic and acoustic capabilities through multi-stage, multi-task learning`

### Code Evidence
- `VITA-QinYu` 本地配置已经把它拆得很清楚：
  - semantic encoder 一路
  - acoustic encoder 一路
  - 之后融合，再进 RVQ
  - 再上采样、acoustic decoder、vocos
  - [third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml:25](third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml#L25)
  - [third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml:50](third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml#L50)
  - [third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml:66](third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml#L66)
  - [third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml:79](third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml#L79)
  - [third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml:98](third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml#L98)
  - [third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml:117](third_party/VITA-QinYu/pretrained/xy_tokenizer_config.yaml#L117)
- 关联的公开 `MOSS-TTSD` 源码页面里，`XY_Tokenizer` 的主结构是：
  - `semantic_encoder = OmniAudioEncoder(...)`
  - `semantic_encoder_adapter = Transformer(...)`
  - `acoustic_encoder = OmniAudioEncoder(...)`
  - `pre_rvq_adapter = Transformer(...)`
  - `downsample = ResidualDownConv(...)`
  - `quantizer = ResidualVQ(...)`
  - `post_rvq_adapter = Transformer(...)`
  - `upsample = UpConv(...)`
  - `acoustic_decoder = OmniAudioDecoder(...)`
  - `enhanced_vocos = Vocos(...)`
  - Source: [turn1view0 lines 93-117](https://huggingface.co/spaces/fnlp/MOSS-TTSD/blob/main/XY_Tokenizer/xy_tokenizer/model.py)
- tokenize 路径是：
  - `feature_extractor`
  - semantic encoder
  - acoustic encoder
  - concat 两路
  - pre-RVQ adapter
  - downsample 到 `12.5Hz`
  - `ResidualVQ`
  - Source: [turn1view0 lines 147-183](https://huggingface.co/spaces/fnlp/MOSS-TTSD/blob/main/XY_Tokenizer/xy_tokenizer/model.py)
- detokenize 路径是：
  - `decode_codes`
  - post-RVQ adapter
  - upsample
  - acoustic decoder
  - vocos
  - Source: [turn1view0 lines 216-224](https://huggingface.co/spaces/fnlp/MOSS-TTSD/blob/main/XY_Tokenizer/xy_tokenizer/model.py)

### Conclusion
- Confirmed:
  - `XY tokenizer` 不是“普通单路 RVQ codec”，而是**语义通道 + 声学通道双路编码**，在量化前做融合。
  - 它的主干结构是：`semantic encoder + acoustic encoder -> fusion -> RVQ8@12.5Hz -> acoustic decoder + vocos`。
  - 所以它确实是在 codec 里显式解决“语义 vs 声学”冲突，而不是把所有信息都压在一个 encoder 上。
- Inference:
  - 这解释了为什么它比 `EnCodec` 更适合你想做的“主模型只预测 coarse token、剩余 token 另一个模块补”的路线：它本来就把语义和声学做了更强的结构分离。
  - 但它的复杂度也明显更高，因为它不是轻量 tokenizer，而是一整套大型 codec。

## 84. MOSS-TTSD 公开出来的训练链路到底是什么，XY tokenizer 有没有公开训练

### Question
`MOSS-TTSD` 论文和官方代码里，哪些训练是公开的？`v0.7` 的模型微调怎么做？`XY tokenizer` 自己的预训练有没有公开？

### Paper / README Evidence
Source: [third_party/MOSS-TTSD/README.md:55](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTSD/README.md#L55)
- README 明确写了：`We open-source the fine-tuning code for MOSS-TTSD v0.5, supporting full-parameter fine-tuning, LoRA fine-tuning, and multi-node training.`

Source: [third_party/MOSS-TTSD/legacy/v0.7/README.md:300](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTSD/legacy/v0.7/README.md#L300)
- `v0.7` README 公开的是 `finetune/` 目录下的**模型微调脚本和数据预处理工具**。

Source: [third_party/MOSS-TTSD/legacy/v0.7/README.md:32](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTSD/legacy/v0.7/README.md#L32)
- 模型总览写的是：`Built on unified semantic-acoustic neural audio codec, a pre-trained large language model, millions of hours of TTS data and conversational speech`
- 这说明主系统训练依赖大规模 TTS / conversational speech，但 README 没把完整 pretraining recipe 展开。

### Code Evidence
- `v0.7` 公开微调流程是两步：
  1. `finetune/data_preprocess.py`
  2. `finetune/finetune.py`
  见 [third_party/MOSS-TTSD/legacy/v0.7/README.md:343](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTSD/legacy/v0.7/README.md#L343) 和 [third_party/MOSS-TTSD/legacy/v0.7/README.md:391](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTSD/legacy/v0.7/README.md#L391)
- `data_preprocess.py` 做的事情是：
  - 用 `XY_Tokenizer` 把目标音频编码成 `8` 通道 speech token
  - 把 prompt/style、文本、`<|begin_of_speech|>` 这些部分拼成多通道 `input_ids`
  - 只对音频 token 和结束标记计算 loss
  - 见 [third_party/MOSS-TTSD/legacy/v0.7/finetune/data_preprocess.py:33](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTSD/legacy/v0.7/finetune/data_preprocess.py#L33)
  - [third_party/MOSS-TTSD/legacy/v0.7/finetune/data_preprocess.py:116](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTSD/legacy/v0.7/finetune/data_preprocess.py#L116)
  - [third_party/MOSS-TTSD/legacy/v0.7/finetune/data_preprocess.py:146](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTSD/legacy/v0.7/finetune/data_preprocess.py#L146)
- `finetune.py` 做的事情是：
  - 加载 `AsteroidTTSInstruct`
  - 可选 LoRA
  - 用 HuggingFace `Trainer` 训练
  - 数据是 `.pkl + _metas.npy`
  - 见 [third_party/MOSS-TTSD/legacy/v0.7/finetune/finetune.py:134](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTSD/legacy/v0.7/finetune/finetune.py#L134)
  - [third_party/MOSS-TTSD/legacy/v0.7/finetune/finetune.py:232](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTSD/legacy/v0.7/finetune/finetune.py#L232)
  - [third_party/MOSS-TTSD/legacy/v0.7/finetune/finetune.py:262](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTSD/legacy/v0.7/finetune/finetune.py#L262)
- `AsteroidTTSInstruct` 本身是**多通道因果 LM**：
  - 每个通道一个 `lm_head`
  - 总 loss 是按通道加权求和
  - 默认权重是前一层更重
  - 见 [third_party/MOSS-TTSD/legacy/v0.7/modeling_asteroid.py:485](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTSD/legacy/v0.7/modeling_asteroid.py#L485)
  - [third_party/MOSS-TTSD/legacy/v0.7/modeling_asteroid.py:514](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTSD/legacy/v0.7/modeling_asteroid.py#L514)
- 训练配置默认非常直接：
  - `per_device_train_batch_size: 1`
  - `num_train_epochs: 50`
  - `learning_rate: 1e-4`
  - `bf16: true`
  - 见 [third_party/MOSS-TTSD/legacy/v0.7/finetune/training_config.yaml:1](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTSD/legacy/v0.7/finetune/training_config.yaml#L1)
- `XY_Tokenizer` 本地仓库只公开了：
  - inference README
  - 模型结构
  - 配置
  - quantizer 实现
  - 但没有看到独立的 tokenizer 训练脚本
  - 见 [third_party/MOSS-TTSD/legacy/v0.7/XY_Tokenizer/README.md:1](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTSD/legacy/v0.7/XY_Tokenizer/README.md#L1)
  - [third_party/MOSS-TTSD/legacy/v0.7/XY_Tokenizer/xy_tokenizer/model.py:21](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTSD/legacy/v0.7/XY_Tokenizer/xy_tokenizer/model.py#L21)
  - [third_party/MOSS-TTSD/legacy/v0.7/XY_Tokenizer/config/MOSS_TTSD_tokenizer.yaml:1](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTSD/legacy/v0.7/XY_Tokenizer/config/MOSS_TTSD_tokenizer.yaml#L1)

### Conclusion
- Confirmed:
  - `MOSS-TTSD v0.7` 公开的是**模型微调链路**，不是完整从零预训练链路。
  - 公开微调方式是：先用 `XY_Tokenizer` 把目标音频转成 8 通道离散 speech token，再把文本前缀和音频 token 拼成多通道训练样本，最后用多通道因果 LM 做 next-token prediction。
  - 这个 LM 不是“主 token + residual heads”分工，而是**8 通道一起建模、分头输出、加权求 loss**。
- Inference:
  - 如果你要借 `MOSS-TTSD`，当前最有价值的是它的**多通道 speech token 训练格式**和 `XY tokenizer` 结构，不是现成的 tokenizer 预训练代码。
  - 公开仓库里我没有看到 `XY tokenizer` 的独立训练脚本，所以**XY tokenizer 预训练 recipe 目前仍然是不透明的**。

## 85. MOSS-Audio-Tokenizer 是不是 XY tokenizer，它怎么训练

### Question
`MOSS-Audio-Tokenizer: Scaling Audio Tokenizers for Future Audio Foundation Models` 这篇论文里的 tokenizer，是不是 `XY tokenizer`？它的训练方式和结构是什么？

### Paper Evidence
Source: [papers/moss_audio_tokenizer.pdf](/mnt/c/streaming_svs_prototype/papers/moss_audio_tokenizer.pdf)
- 摘要明确写：
  - `CAT (Causal Audio Tokenizer with Transformer), a purely Transformer-based architecture`
  - `jointly optimizes the encoder, quantizer, and decoder from scratch`
  - `1.6 billion parameters`
  - `pre-trained on 3 million hours of diverse, general audio data`
  - `speech, sound, and music`
- 第 3 页摘要/贡献段继续写：
  - `12.5 Hz`
  - `0.125 kbps to 4 kbps`
  - `low-latency, frame-level streaming encoding and decoding`

### Code / README Evidence
- 主仓库 README 明确写：
  - `MOSS-Audio-Tokenizer` 是基于 `CAT` 的统一离散音频 tokenizer
  - `1.6B combined parameters (Encoder + Decoder)`
  - `32-layer Residual Vector Quantizer (RVQ)`
  - `Trained on 3 million hours of diverse audio data`
  - `Fully Trained From Scratch`
  - `All components—including the encoder, quantizer, decoder, discriminator, and a decoder-only LLM for semantic alignment—are optimized jointly in a single unified training pipeline.`
  - 见 [third_party/MOSS-Audio-Tokenizer/README.md:17](/mnt/c/streaming_svs_prototype/third_party/MOSS-Audio-Tokenizer/README.md#L17)
- 本地配置能确认它和 `XY tokenizer` 完全不是一套结构：
  - encoder 是多级 `PatchedPretransform + Transformer`
  - decoder 也是对称的纯 Transformer 结构
  - `quantizer_type` 默认是 `rlfq`
  - 下采样率 `1920`
  - 见 [third_party/MOSS-Audio-Tokenizer/configuration_moss_audio_tokenizer.py:72](/mnt/c/streaming_svs_prototype/third_party/MOSS-Audio-Tokenizer/configuration_moss_audio_tokenizer.py#L72)
  - [third_party/MOSS-Audio-Tokenizer/configuration_moss_audio_tokenizer.py:96](/mnt/c/streaming_svs_prototype/third_party/MOSS-Audio-Tokenizer/configuration_moss_audio_tokenizer.py#L96)
- README 还明确写：
  - `MOSS-Audio-Tokenizer` 支持简单流式 encode/decode
  - streaming 通过 `chunk_duration`
  - `chunk_duration <= causal_transformer_context_duration`
  - 见 [third_party/MOSS-Audio-Tokenizer/README.md:144](/mnt/c/streaming_svs_prototype/third_party/MOSS-Audio-Tokenizer/README.md#L144)

### Conclusion
- Confirmed:
  - `MOSS-Audio-Tokenizer` **不是** `XY tokenizer`。
  - `XY tokenizer` 是旧的 `legacy/v0.7` 双通道 semantic/acoustic codec；`MOSS-Audio-Tokenizer` 是新的、**纯 Transformer、端到端从零联合训练**的 `CAT` 路线。
  - 它的训练方式是：
    - 纯 Transformer encoder + quantizer + decoder
    - 再加 discriminator 和一个 decoder-only LLM 做 semantic alignment
    - 所有模块联合优化
    - 在 `3M` 小时通用音频上从零预训练
- Inference:
  - 如果你后面想跟 `Qwen3` 主线更对齐，`MOSS-Audio-Tokenizer` 比 `XY tokenizer` 更像 OpenMOSS 当前真正押注的 tokenizer 方向。
  - 但它也明显更重：`1.6B` 参数，不是轻量 codec。

## 86. XY-Tokenizer 有没有官方论文

### Question
`XY tokenizer` 到底有没有官方论文或报告？

### Evidence
Source: [papers/xy_tokenizer.pdf](/mnt/c/streaming_svs_prototype/papers/xy_tokenizer.pdf)
- 论文标题就是：
  - `XY-Tokenizer: Mitigating the Semantic-Acoustic Conflict in Low-Bitrate Speech Codecs`
- arXiv ID:
  - `2506.23325`

Source: `arXiv abstract / model card`
- 官方摘要明确写：
  - `we propose XY-Tokenizer, a novel codec that mitigates the conflict between semantic and acoustic capabilities through multi-stage, multi-task learning`
- 公开代码链接也在摘要里给出：
  - `https://github.com/gyt1145028706/XY-Tokenizer`

Source: [third_party/XY-Tokenizer/readme.md](/mnt/c/streaming_svs_prototype/third_party/XY-Tokenizer/readme.md#L1)
- 本地已拉下官方仓库，能和论文题目、模型卡对应上。

### Conclusion
- Confirmed:
  - `XY tokenizer` **有官方论文**，不是只有模型卡。
  - 这篇论文是 arXiv 论文，不是我前面说的“没有独立论文”。那句需要修正。
  - 公开仓库也单独存在，不只是被 `MOSS-TTSD` 或 `VITA-QinYu` 间接引用。

## 87. MusicGen 对多层码本预测方式的实验结论是什么

### Question
`MusicGen` 对多层 RVQ/codebook 的预测方式做了哪些探索？哪种方法最好，哪种更适合当前想做的语音/歌声 codec 自回归？

### Paper Evidence
Source: [papers/musicgen.pdf](/mnt/c/streaming_svs_prototype/papers/musicgen.pdf)
- 第 3 页明确说明 RVQ 的内部结构：
  - `the first codebook is the most important one`
  - `quantized values for different codebooks are in general not independent`
- 第 4 页定义了多种 interleaving pattern：
  - `parallel`
  - `delay`
  - `partial delay`
  - `coarse first`
  - `flattening`
- 第 5 页给出作者最终训练选择：
  - `We use the "delay" interleaving pattern`
  - `This translates 30 seconds of audio into 1500 autoregressive steps`
- 第 9 页表 4 给出核心消融结论：
  - `Flattening` 得分最好
  - `Delay` 和 `Partial flattening` 接近
  - `Parallel` 明显更差
  - 文字总结是：
    - `While flattening improves generation, it comes at a high computational cost`
    - `similar performance can be reached for a fraction of this cost using a simple delay approach`
- 第 14 页附录解释 `partial delay` / `partial flattening` 的直觉：
  - `the first codebook from RVQ is the most important one`
  - 希望单独处理第一层，同时把其余层并行预测以减少开销

### Conclusion
- Confirmed:
  - `MusicGen` 系统性比较了 `flattening / delay / partial delay / parallel / coarse first / partial flattening`。
  - 如果只看质量，`flattening` 最好。
  - 如果看质量与算力/时延折中，论文自己的结论是 `delay` 最划算，也是他们最终实际采用的方案。
  - `parallel` 最差，说明“同一时刻所有码本完全并行出”会明显损伤建模质量。
- Inference:
  - 对你当前想做的 `Qwen3 + 主模型预测 coarse token` 路线，`MusicGen` 的证据更支持两件事：
    1. 不要做完全 parallel 的多码本预测。
    2. 如果不想付出 full flatten 的长序列代价，`delay pattern` 是最稳的主流折中。
  - 它没有直接证明“coarse token + 单独 residual predictor”一定优于 delay；它证明的是：在单模型框架内，`delay` 是比 `parallel` 更靠谱、比 `flattening` 更便宜的方案。
- Open uncertainty:
  - `MusicGen` 的结论来自 music generation，不是 speech/singing；是否原样迁移到语音/歌声，还要结合 codec 本身的语义层设计一起判断。

## 88. MusicGen 的错位模式和 MOSS-TTS Delay 的多头机制具体怎么做

### Question
`MusicGen` 里的错位 / delay pattern 具体是什么自回归顺序？`MOSS-TTS Delay` 里是否也用了多头，具体怎么做？

### Paper Evidence
Source: [papers/musicgen.pdf](/mnt/c/streaming_svs_prototype/papers/musicgen.pdf)
- 第 4 页定义了 delay interleaving：
  - `introduce a "delay" between the codebooks`
  - 目标是在单模型里做一种 inexact autoregressive decomposition
- 第 9 页表 4 和正文说明：
  - `delay` 是作者最终采用的 pattern
  - 比 `parallel` 明显更好，接近 `flattening`

### Code / README Evidence
- `MossTTSDelay` README 明确写：
  - `Single Transformer backbone with Multi-Head Parallel Prediction and a Delay-Pattern scheduling mechanism`
  - `1 + N_q` 个 heads，`N_q = 32`
  - `Head 1 predicts Layer 1 of Frame t`
  - `Head 2 predicts Layer 2 of Frame t-1`
  - `Head 3 predicts Layer 3 of Frame t-2`
  - 见 [third_party/MOSS-TTS/moss_tts_delay/README.md:13](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTS/moss_tts_delay/README.md#L13)
  - [third_party/MOSS-TTS/moss_tts_delay/README.md:18](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTS/moss_tts_delay/README.md#L18)
  - [third_party/MOSS-TTS/moss_tts_delay/README.md:52](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTS/moss_tts_delay/README.md#L52)
- `delay_state.py` 注释更直接：
  - `Head 0 predicts text at t, Head k predicts codebook k-1 at t-(k-1).`
  - 见 [third_party/MOSS-TTS/moss_tts_delay/llama_cpp/delay_state.py:8](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTS/moss_tts_delay/llama_cpp/delay_state.py#L8)
- 实现层面：
  - 输入 shape 是 `(batch, seq_len, 1 + n_vq)`
  - 主模型确实维护 `1 + n_vq` 个 `lm_heads`
  - 见 [third_party/MOSS-TTS/moss_tts_delay/modeling_moss_tts.py:185](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTS/moss_tts_delay/modeling_moss_tts.py#L185)
  - [third_party/MOSS-TTS/moss_tts_delay/modeling_moss_tts.py:243](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTS/moss_tts_delay/modeling_moss_tts.py#L243)
- 训练/推理时的 delay 格式就是把每个 codebook 列沿时间轴平移：
  - `delayed[i : i + T, i] = codes[:, i]`
  - 见 [third_party/MOSS-TTS/moss_tts_delay/llama_cpp/delay_state.py:267](/mnt/c/streaming_svs_prototype/third_party/MOSS-TTS/moss_tts_delay/llama_cpp/delay_state.py#L267)

### Conclusion
- Confirmed:
  - `MusicGen` 的 `delay` 不是“另一个模型补 fine”，而是把不同 codebook 按时间错位后交给同一个自回归模型建模。
  - `MOSS-TTS Delay` 进一步把这件事做成了**单 backbone + 多头并行输出**：
    - 一个 text/main head
    - `N_q` 个 audio heads
    - 每个 head 负责不同 codebook 的不同时间偏移位置
  - 所以它每一步不是只出一个 token，而是同时出一组 token；其中组内不同位置对应不同延迟后的 codebook 层。
- Inference:
  - `MusicGen delay` 更像“单头单序列 pattern 设计”。
  - `MOSS-TTS Delay` 更像“多头版 delay”，计算上更激进，直接一拍并行出整组音频 head。
  - 这两者都不是你之前说的“coarse 模型 + fine 模型”两阶段方案。

## 89. SoulX 的数据到底有没有公开“爬取与清洗”代码

### Question
`SoulX-Singer` 仓库是否公开了数据爬取/清洗代码？如果有，边界在哪里？

### Code / README Evidence
- 仓库首页明确定位是 **inference code**：
  - 标题写的是 `Official inference code for SoulX-Singer`，见 [third_party/SoulX-Singer/README.md:4](/mnt/c/streaming_svs_prototype/third_party/SoulX-Singer/README.md#L4)
- 仓库公开了一个 `preprocess` 子模块，且 README 明确是“转写与编辑工具链”：
  - `singing transcription and editing toolkit`，见 [third_party/SoulX-Singer/preprocess/README.md:3](/mnt/c/streaming_svs_prototype/third_party/SoulX-Singer/preprocess/README.md#L3)
  - 自动流程包含：分离/去混响、F0+VAD、歌词转写、音符转写，见 [third_party/SoulX-Singer/preprocess/README.md:79](/mnt/c/streaming_svs_prototype/third_party/SoulX-Singer/preprocess/README.md#L79)
- `pipeline.py` 的具体实现也对应这个边界：
  - 调用 `VocalSeparator / F0Extractor / VocalDetector / LyricTranscriber / NoteTranscriber`，见 [third_party/SoulX-Singer/preprocess/pipeline.py:10](/mnt/c/streaming_svs_prototype/third_party/SoulX-Singer/preprocess/pipeline.py#L10)
  - 输出 `metadata.json` 并可转 MIDI 做人工修订，见 [third_party/SoulX-Singer/preprocess/pipeline.py:117](/mnt/c/streaming_svs_prototype/third_party/SoulX-Singer/preprocess/pipeline.py#L117)
- 仓库中未看到“网页抓取/版权过滤/大规模去重”的采集流水线脚本；更多是给定音频后的标注生成与编辑。

### Conclusion
- Confirmed:
  - SoulX 开源了“预处理与标注工具链”代码（可把输入音频转成可用于 SVS 的 metadata）。
  - 这套代码主要覆盖：音频预处理、歌词/音符转写、MIDI 编辑回写。
  - 没有在仓库中公开 42k 小时训练集对应的完整“数据爬取+全量清洗”生产流水线。
- Inference:
  - 你可以直接复用它的 `preprocess` 作为“给定音频后的自动标注器”，但“数据源采集与合规清洗”需要你自己补。
- Open uncertainty:
  - 官方是否在内网有完整数据工程系统不可见；公开仓库层面无法验证。

## 90. SoulX 原始论文对“数据构建”是怎么写的

### Question
SoulX 原始论文是否说明了数据怎么爬/怎么构建？具体写到了哪一步？

### Paper Evidence
Source: `third_party/SoulX-Singer/assets/technical-report.pdf`
- [third_party/SoulX-Singer/assets/technical-report.pdf#page=3](third_party/SoulX-Singer/assets/technical-report.pdf#page=3)
  - Section 2.1 写的是数据处理工作流：`vocal separation -> lyric transcription -> note transcription`，并提到抽取 F0。
  - Figure 2 标题明确是 `Pipeline for large-scale singing data curation: from raw audio extraction to time-aligned MIDI and text formulation.`
- [third_party/SoulX-Singer/assets/technical-report.pdf#page=4](third_party/SoulX-Singer/assets/technical-report.pdf#page=4)
  - `Using the aforementioned processing methods, we obtained approximately 42,000 hours ...`
  - 给了语种规模分布（中/英各约 20k 小时，粤语约 2k 小时）。
- [third_party/SoulX-Singer/assets/technical-report.pdf#page=6](third_party/SoulX-Singer/assets/technical-report.pdf#page=6)
  - 对评测集来源写得更具体：Mandarin 来自征集并同意开源的 singers；English 从 Mixing Secrets 切分过滤。

### Conclusion
- Confirmed:
  - 论文公开说明了“处理与标注流水线”细节（分离、转写、对齐、F0）。
  - 论文给了训练数据总量与语种分布（42k+ 小时）。
  - 论文对评测集数据来源有较明确描述。
- Inference:
  - 论文没有展开公开“全量训练数据来源采集/网页爬取/版权清洗”的工程细节与脚本；公开信息更偏向数据后处理与标注构建方法。
- Open uncertainty:
  - 42k 训练语料的具体来源分布、授权策略、去重/过滤规则在论文中未细化到可复现实操级别。
