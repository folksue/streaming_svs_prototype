# AR Discrete TTS Scaling 调研表（2026-04）

本文聚焦：`TTS / 语音生成` 中采用 `离散 token + 自回归（AR）` 的代表工作，整理其参数量、训练数据规模，以及是否依赖纯文字 token 预训练。

## 口径说明

- “离散+AR”指主生成链路里，模型对离散语音/音频 token 做自回归建模。
- “数据规模”优先使用论文或官方页面明确给出的小时数。
- “是否纯文字 token 预训练”仅统计：主 backbone 是否先在纯文本 LM 任务上预训练。
- `token/param` 的讨论这里仅做趋势判断，不把 Chinchilla 文本口径生硬套用到所有语音系统。

## 对比表

| 模型 | 是否离散+AR | codec/tokenizer 参数 | LLM主干参数 | 训练数据规模（公开） | 等效 coarse / 语义 token（估算） | coarse token / param（估算） | 主干是否纯文字 token 预训练 | 备注 |
|---|---|---:|---:|---:|---:|---:|---|---|
| VALL-E (2023) | 是（Neural codec LM） | EnCodec 24k：约 23M（按 HF `model.safetensors` 93.1MB 折算 FP32） | 未公开 | 60K 小时（LibriLight） | 16.2B（按 75Hz） | N/A | 否（未报告 text-only LM 预训练） | 典型“codec token + AR”早期大规模路线 |
| BASE-TTS (2024) | 是（AR 预测离散 speechcodes） | 未公开（自研 speech tokenizer，WavLM 特征 + VQ） | 1B | 100K 小时 | N/A（speechcodes 含 BPE 压缩，公开信息不足以统一换算） | N/A | 否（论文明确 from scratch，不做 text-pretrain） | 大模型配大数据，强调 emergent 能力 |
| MOSS-TTS (2026) | 是（离散 token + AR） | 1.6B（MOSS-Audio-Tokenizer；enc/dec 各约 0.8B） | 1.6B | “millions of hours” 级（报告描述） | >=45.0B（按 1M 小时下界、12.5Hz） | >=28.1 | 未明确纯文本预训练 | 语音侧规模化预训练范式 |
| MOSS-TTSD (2026) | 是（离散 token + AR + delay） | 1.6B（沿用 MOSS-Audio-Tokenizer 体系） | 8B（论文 Qwen3-8B-base）或 1.7B（官网发布线） | 官网给出约 1M 小时单人 + 400K 小时对话（版本相关） | 63.0B（按 1.4M 小时、12.5Hz） | 7.9（按 8B）/ 37.1（按 1.7B） | 是（基于 Qwen3-base） | 典型“LLM backbone + 语音 continued pretrain” |
| VoXtream2 (2026) | 是（全流式 AR，离散 token） | Mimi：约 96M（按 HF `model.safetensors` 384.6MB 折算 FP32） | 462M | 40K 小时 | 1.8B（按 12.5Hz） | 3.9 | 未报告 text-only 预训练 | 流式低延迟场景，数据规模仍很大 |
| CosyVoice3 (2025，混合） | 部分（AR + CFM 两阶段） | MinMo voice encoder + FSQ（公开材料未给 tokenizer 总参数） | 0.5B → 1.5B | 10K → 1M 小时 | N/A（论文未给统一可对齐的语义 token 速率） | N/A | 非纯文本（MinMo/语音多模态预训练） | 不是纯离散 AR 单模型，但规模趋势具有参考价值 |

估算口径：
- `coarse_tokens ≈ hours × 3600 × token_rate_hz`
- 这里统一用文中常见的 `75Hz`（EnCodec coarse）或 `12.5Hz`（MOSS/XY/Mimi 常见语义层级）做近似。
- 若论文未公开可对齐的有效 token 速率（或存在额外 BPE 压缩），记为 `N/A`。
- 对 codec 参数的“文件大小折算”采用近似：`params ≈ bytes / 4`（按 FP32 等效）。若实际权重不是 FP32，参数估算会偏差一个倍率。

## 对你当前实验的对照结论

- 当前配置（OpenCpop + M4Singer）约 `35.02 小时`，按 `75 Hz` 粗算是 `9.46M coarse tokens`。
- 当前主线模型约 `100M` 参数，`coarse token / param ≈ 0.095`。
- 对照上表主流 AR 离散 TTS 的训练规模，你当前组合属于明显“小数据 + 相对大模型 from scratch”，过拟合风险高（与你当前实验现象一致）。

## 实操建议（针对当前仓库路线）

1. 保持 100M 参数时，优先加数据规模（至少一个数量级）。
2. 若短期无法扩数据，先把模型降到更小区间再观察泛化曲线。
3. 采用两阶段训练（先 coarse，再扩 fine）通常比直接全码本联合训练更稳。
4. 若切换到 LLM 路线，优先考虑“已有文本/语音预训练 backbone + continued pretraining”，而不是完全 from scratch。

## 参考链接

- VALL-E: https://arxiv.org/abs/2301.02111
- BASE-TTS: https://arxiv.org/abs/2402.08093
- MOSS-TTS: https://arxiv.org/abs/2603.18090
- MOSS-TTSD: https://arxiv.org/abs/2603.19739
- MOSS-TTSD 官方页: https://openmoss.ai/en/moss-ttsd/
- MOSS-Audio-Tokenizer: https://arxiv.org/abs/2602.10934
- MOSS-Audio-Tokenizer 代码: https://github.com/OpenMOSS/MOSS-Audio-Tokenizer
- VoXtream2: https://arxiv.org/abs/2603.13518
- CosyVoice3: https://arxiv.org/abs/2505.17589
- EnCodec 24k HF 模型页: https://huggingface.co/facebook/encodec_24khz
- Mimi HF 模型页: https://huggingface.co/kyutai/mimi
