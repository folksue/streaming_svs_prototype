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
