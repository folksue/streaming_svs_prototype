# 数据集语言清单（SVS / TTS 相关）

更新时间：2026-04-22  
说明：本表覆盖我们前面讨论过的那批数据集，重点列出“语言”字段，便于你快速做数据配比和多语规划。

## A. 唱歌 / SVS 直接相关

| 数据集 | 总时长（小时） | 语言 |
|---|---:|---|
| OpenCpop | 5.25 | 中文（普通话 / Mandarin） |
| M4Singer | 29.77 | 中文（普通话 / Mandarin） |
| GTSinger | 80.59 | 中文、英文、日文、韩文、俄文、西班牙文、法文、德文、意大利文 |
| ACE-Opencpop | 128.9 | 中文（普通话 / Mandarin） |
| ACE-KiSing | 32.5 | 中文（普通话）+ 英文 |
| OpenSinger | 50.0（常见论文口径） | 中文（普通话） |
| NUS-48E | 1.91 | 英文 |
| NHSS | 7.0 | 英文 |
| JVS-MuSiC | 2.28 | 日文 |

## B. 语音预训练常用（可做扩展预训练）

| 数据集 | 总时长（小时） | 语言 |
|---|---:|---|
| Emilia | 101k+（Emilia-Large 216k+） | 英文、中文、德文、法文、日文、韩文 |
| WenetSpeech | 10k+ 强标注（总计约 22.4k） | 中文（普通话 / Mandarin） |
| MLS (Multilingual LibriSpeech) | 约 50k（英文约 44.5k + 其他语种约 6k） | 英文、德文、法文、西班牙文、意大利文、葡萄牙文、波兰文、荷兰文 |
| VoxPopuli | 400k（无标注语音） | 23 种欧洲语言（含英文、德文、法文、西班牙文、意大利文、波兰文、葡萄牙文等） |
| GigaSpeech | 10k（标注）+ 33k（半/无监督） | 英文 |
| LibriTTS-R | 585 | 英文 |
| HiFiTTS-2 | 36.7k（22.05k 训练版）/ 31.7k（44.1k 训练版） | 英文 |

## C. 仓库内相关汇总文档索引（已整理）

| 文件 | 主要内容 | 适用场景 |
|---|---|---|
| `AR_DISCRETE_TTS_SCALING_TABLE.md` | AR 离散语音/TTS 代表工作对比（模型规模、数据规模、coarse token 估算、是否 text 预训练） | 估算你当前数据量下的合理模型参数档位 |
| `CODEC_ROUTE_COMPARISON.md` | 多种 codec/tokenizer 路线对比（EnCodec、XY/MOSS、Qwen3-Omni/VITA 方案） | 选 tokenizer/codec 技术路线 |
| `MODEL_TOKEN_LAYOUT_SUMMARY.md` | 主流语音模型 token 排列方式（交错/instruction/delay）和起止策略总结 | 设计交错 AR 序列协议、BOS/EOS/状态机 |
| `MULTILINGUAL_TTS_SURVEY_2026.md` | 多语言 TTS 近期模型调研（含本地论文/代码落地状态） | 多语言扩展和前端 tokenization 方案对比 |
| `legacy/streaming_codec_baseline/M4SINGER_INTERFACE_NOTE.md` | M4Singer 原始标注与可落地转换接口说明 | 写/改 M4Singer adapter 与数据转换 |

备注：如果你希望，我可以把 A/B 两张表再导出一份“仅中文优先训练集”子表（含建议采样权重）。

## 主要来源

- OpenCpop: https://github.com/wenet-e2e/opencpop
- M4Singer（含 OpenSinger / OpenCpop 语言对照表）: https://proceedings.neurips.cc/paper_files/paper/2022/file/2de60892dd329683ec21877a4e7c3091-Paper-Datasets_and_Benchmarks.pdf
- GTSinger: https://proceedings.neurips.cc/paper_files/paper/2024/file/023d2c1a17cf35b11a0cbb43a0677c91-Paper-Datasets_and_Benchmarks_Track.pdf
- ACE-Opencpop / ACE-KiSing: https://arxiv.org/abs/2401.17619
- NHSS: https://arxiv.org/abs/2012.00337
- JVS-MuSiC: https://arxiv.org/abs/2001.07044
- Emilia: https://arxiv.org/abs/2501.15907
- WenetSpeech: https://arxiv.org/abs/2110.03370
- MLS: https://arxiv.org/abs/2012.03411
- VoxPopuli: https://github.com/facebookresearch/voxpopuli
- GigaSpeech: https://arxiv.org/abs/2106.06909
- LibriTTS-R: https://arxiv.org/abs/2305.18802
- HiFiTTS-2: https://arxiv.org/abs/2506.04152
