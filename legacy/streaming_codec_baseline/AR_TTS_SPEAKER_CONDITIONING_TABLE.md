# AR-TTS / SVS 说话人（音色）条件方式对比

> 目标：回答“自回归范式里 speaker 怎么处理”，并给你当前路线一个对齐基线。

## 1) 仓库内可核对的实现

| 系统 | 是否 AR 离散 token | speaker 条件注入方式 | 推理是否需要参考音频 | 是否显式 speaker ID embedding | 证据 |
|---|---|---|---|---|---|
| 你的 `legacy/streaming_codec_baseline` | 是（decoder-only） | `singer_id -> Embedding`，与 `note/phoneme/slur/progress` 拼接后过 MLP | 不需要（可选固定 `singer_id`） | 是 | `model.py`, `train.py`, `infer.py --singer-id` |
| MOSS-TTS | 是（LLM + codec token） | 主要是 `reference audio` prompt（voice clone）/ 也支持无参考 direct TTS | voice clone 场景需要 | 通常不是固定 speaker_id lookup | `third_party/MOSS-TTS/README.md` |
| MOSS-TTS-Nano | 是（~0.1B AR） | `prompt_audio_path` 或内置 voice preset，走 voice cloning conditioning | voice clone 场景需要 | 不强调固定 speaker_id 表 | `third_party/MOSS-TTS-Nano/infer.py`, `README.md` |
| MOSS-TTSD | 是（延迟式 AR） | 多说话人 `[S1..S5]` + 每个 speaker 的 `prompt_audio_speakerN/prompt_text_speakerN` | 需要（多说话人克隆） | 以 prompt 条件为主 | `third_party/MOSS-TTSD/README.md` |
| VALL-E-X（开源实现） | 是（NCLM 系） | prompt 驱动为主，代码里也定义了 speaker class 配置（`4096 x 64`） | 零样本场景需要 | 代码配置中有 | `third_party/VALL-E-X/models/macros.py` |
| VISinger2（SVS） | 否（非 codec AR 主干） | `emb_spk = nn.Embedding(n_speakers, spk_channels)`，注入 duration/f0/decoder/flow 等模块 | 不需要参考音频（用 spk_id） | 是 | `third_party/VISinger2/egs/visinger2_flow/models.py` |
| SoulX-Singer（SVS） | 否（flow matching） | 以 prompt 音频音色条件为主（zero-shot timbre cloning） | 需要（zero-shot） | 非固定 ID 为主 | `third_party/SoulX-Singer/README.md` |

## 2) 外部代表性 AR 论文（补充）

| 工作 | speaker 条件核心思路 | 备注 |
|---|---|---|
| VALL-E (2023) | 文本 + `acoustic prompt`（3s 参考音频对应 codec token）联合条件，实现 zero-shot speaker cloning | 经典“prompt-based speaker conditioning”，不是单纯 speaker_id 表 |
| VALL-T (2024) | 保留 prompt-based zero-shot speaker adaptation，同时优化对齐/稳健性 | 仍然是 prompt 条件主导 |
| VALL-E R (2024) | 沿 VALL-E 路线，强调鲁棒性与效率，speaker similarity 用 speaker vector 评测 | 推理 speaker 条件仍来自 acoustic prompt |
| VALL-E 2 (2024) | 继续强化 zero-shot speaker similarity 与自然度 | 仍依赖 prompt 条件 |

参考链接：
- VALL-E: https://arxiv.org/abs/2301.02111
- VALL-T: https://arxiv.org/abs/2401.14321
- VALL-E R: https://arxiv.org/abs/2406.07855
- VALL-E 2: https://arxiv.org/abs/2406.05370

## 3) 对你当前路线的直接结论

- 多数据集混训时，**显式 singer_id embedding 是建议保留的**（你现在已接入）。  
- 如果以后要 zero-shot 克隆，再叠加 `reference-audio prompt` 分支，而不是替换掉 singer_id。  
- 最稳的工程路径通常是两路并存：  
  1) `singer_id`（稳定身份锚点）  
  2) `prompt audio`（细节音色/韵律补偿）  

