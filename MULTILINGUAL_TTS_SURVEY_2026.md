# 多语言 TTS 路线速查（含本地下载/代码状态）

更新时间：2026-04-19  
目标：给当前项目做“多语言处理方案”对比，重点看 `语言建模方式 / tokenization / 是否流式 / 是否可复用到你当前路线`。

## 已下载论文

目录：`papers/multilingual_tts/`

- `yourtts_icml2022.pdf`
- `mms_jmlr2024.pdf`
- `xtts_interspeech2024.pdf`
- `vallex_arxiv2023.pdf`
- `voicecraftx_emnlp2025.pdf`
- `omnivoice_arxiv2026.pdf`

## 已就绪代码仓库（本地）

- `third_party/yourtts`（已 clone）
- `third_party/VALL-E-X`（已 clone）
- `third_party/Qwen3-Omni`（此前已存在）
- `third_party/MOSS-TTS`（此前已存在）
- `third_party/MOSS-TTSD`（此前已存在）
- `third_party/MOSS-Audio-Tokenizer`（此前已存在）

备注：`coqui-ai/TTS` 仓库本次网络环境下 checkout 不稳定，未成功保留本地副本；XTTS 结论主要来自论文与公开文档。

## 对比表（多语言处理重点）

| 模型 | 时间 | 多语言输入处理 | 语言控制方式 | codec/tokenizer 路线 | 流式/实时 | 本地证据 |
|---|---|---|---|---|---|---|
| YourTTS | 2022 | 直接 raw text（不强依赖 phoneme） | 显式 language embedding；推理常用 `language_idx` | VITS 系路线（非 codec-LM） | 非主打流式 | 论文：`papers/multilingual_tts/yourtts_icml2022.pdf`；代码：`third_party/yourtts/README.md` |
| VALL-E X | 2023 | 多语言 G2P，文本可用 `[EN]/[ZH]` 等段落标记 | 显式 language ID + 代码切换标注 | Neural codec LM（离散声学 token） | 非实时优先 | 论文：`papers/multilingual_tts/vallex_arxiv2023.pdf`；代码：`third_party/VALL-E-X/README.md` |
| XTTS | 2024 | 文本 token（BPE），强调大规模多语 zero-shot | 显式语言条件（推理接口常见 language 参数） | GPT 风格 text->audio code 预测 | 非实时优先 | 论文：`papers/multilingual_tts/xtts_interspeech2024.pdf` |
| MMS-TTS | 2024 | 超大语言覆盖，前端偏字符/转写（含 uroman） | 按语言配置/模型路由（工程覆盖优先） | VITS 家族（更偏每语种扩展） | 非实时优先 | 论文：`papers/multilingual_tts/mms_jmlr2024.pdf` |
| VoiceCraft-X | 2025 | 明确“phoneme-free cross-lingual text processing” | 统一 AR codec LM + 跨语音色克隆 | codec token + token 重排 | 非实时优先 | 论文：`papers/multilingual_tts/voicecraftx_emnlp2025.pdf` |
| OmniVoice | 2026 | 直接用 LLM 子词 tokenizer，尽量去语言专用 G2P | 以统一模型覆盖 600+ 语言 | diffusion language model + codec token 路线 | 论文强调规模与零样本，非工程实时优先 | 论文：`papers/multilingual_tts/omnivoice_arxiv2026.pdf` |
| Qwen3-Omni | 2025 | 统一多模态输入；文本 119 语言，语音输入/输出多语 | 指令+系统提示控制，语种能力内建 | Thinker-Talker + AuT，多码本低时延语音输出 | 主打实时交互 | 代码：`third_party/Qwen3-Omni/README.md`；论文：`papers/qwen3omni.pdf` |
| MOSS-TTS Family | 2026 | 支持多语/码切换；显式提到 phoneme/pinyin 细粒度控制 | 语言代码/提示控制 + 声学 prompt | Delay/Local/Reatime 架构 + MOSS Audio Tokenizer | `Realtime` 分支主打流式 | 代码：`third_party/MOSS-TTS/README.md`；论文：`papers/moss_tts.pdf` |
| MOSS-TTSD | 2026 | 面向多说话人长对话，多语言 continuation | speaker+语言脚本控制 | 依赖 MOSS Audio Tokenizer + TTSD 生成器 | 支持流式推理版本 | 代码：`third_party/MOSS-TTSD/README.md`；论文：`papers/moss_ttsd.pdf` |

## 你当前项目最相关的结论（可直接落地）

1. 主流方案几乎都用了“显式语种条件”  
- language ID / language embedding / language code prompt 至少一种。  
- 你的“联合词表 + language tag”是主流兼容方案。

2. 是否强依赖 phoneme 正在分化  
- 传统 TTS（YourTTS、部分 VALL-E 系）仍常用 G2P/phoneme。  
- 新模型（VoiceCraft-X、OmniVoice）更偏“phoneme-free + 大模型 tokenizer”。

3. 想保留你现在的 codec + AR 路线，最稳是：
- 继续离散 codec token
- 增加 `lang_id` 控制
- 文本前端先做“联合词表（带语言前缀）”
- 不急于完全切到纯子词文本路线

## 建议补充（下一步）

- 增加一个小实验配置：`lang_id + 前缀音素词表` vs `无lang_id`，对比跨语种稳定性。
- 对 `OpenCpop + M4Singer` 先做“单语中文稳定基线”，再加入英文数据扩展。
- 若后续切 Qwen3 路线，再考虑把前端从 phoneme 逐步迁移到子词/字节方案。
