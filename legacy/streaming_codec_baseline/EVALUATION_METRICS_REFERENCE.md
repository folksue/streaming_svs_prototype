# 评测指标参考（SoulX / VISinger2）与当前落地方案

本文档总结了我们参考的两条线：
- `SoulX-Singer` 技术报告（`third_party/SoulX-Singer/assets/technical-report.pdf`）
- `VISinger2` 论文（`papers/visinger2.pdf`）

并给出当前 legacy 路线可直接运行的一套客观指标实现。

## 1) 参考工作里用到的指标

### SoulX-Singer（报告中明确提到）
- `FFE`（F0 Frame Error，旋律准确性）
- `SIM`（说话人/歌手相似度，通常基于 speaker embedding）
- `WER/CER`（可懂度）
- `SingMOS`（歌声主观质量预测）
- `Sheet-SSQA`（零样本感知质量）

### VISinger2（论文中明确提到）
- `MOS`（主观自然度）
- `F0-RMSE`（音高误差）
- `dur-RMSE`（时值误差）

## 2) 当前仓库可直接落地的客观指标（已实现）

实现文件：
- `legacy/streaming_codec_baseline/eval_audio_metrics.py`

当前实现的指标：
- `f0_rmse_hz`：有声帧上的 F0 RMSE（Hz）
- `f0_cents_rmse`：有声帧上的 F0 RMSE（cents）
- `f0_corr`：log-F0 相关系数（有声帧）
- `vuv_acc`：有声/无声判别一致率（V/UV）
- `ffe_proxy`：FFE 近似代理指标（V/UV 错配 + gross pitch error，阈值 50 cents）
- `logmel_l1` / `logmel_l2`：log-mel 频谱距离
- `mcd_like`：MFCC L2 的 MCD-like 代理（未做 DTW）

## 3) 为什么先用这组指标

- 你当前路线是离散 codec + AR，首要风险是“可听性差、音高崩、全噪声”。
- 这组指标可以直接反映：
  - 音高是否跟住（F0 相关指标）
  - 频谱是否接近（log-mel / mcd-like）
  - 发声结构是否正常（vuv / ffe_proxy）
- 不依赖外部大模型服务，能先跑稳定批量评测闭环。

## 4) 暂未在本仓实现的指标（后续可加）

- `WER/CER`：需要接 ASR（如 Paraformer/Whisper）
- `SIM`：需要 speaker encoder（如 WeSpeaker/ECAPA）
- `SingMOS/Sheet`：需要对应官方模型或评测工具
- 严格 `MCD`：建议加 DTW 对齐（当前是 mcd-like proxy）
- `dur-RMSE`：需要稳定的音素/音符对齐路径输出

## 5) 使用方法

### A. 目录匹配模式（按同名 wav 匹配）
```bash
python3 legacy/streaming_codec_baseline/eval_audio_metrics.py \
  --ref_dir /path/to/ref_wavs \
  --pred_dir /path/to/pred_wavs \
  --out_json artifacts/eval/audio_metrics.json
```

### B. 显式 pairs json 模式
`pairs.json` 结构示例：
```json
[
  {
    "utt_id": "utt_0001",
    "ref_wav": "/abs/path/ref/utt_0001.wav",
    "pred_wav": "/abs/path/pred/utt_0001.wav"
  }
]
```

运行：
```bash
python3 legacy/streaming_codec_baseline/eval_audio_metrics.py \
  --pairs_json /abs/path/pairs.json \
  --out_json artifacts/eval/audio_metrics.json
```

## 6) 解释口径（论文里建议怎么写）

建议在实验部分把指标分成两层：
- 主指标（本仓稳定可复现）：`f0_cents_rmse`, `f0_corr`, `vuv_acc`, `logmel_l1`
- 扩展指标（后续对齐 SOTA）：`WER/CER`, `SIM`, `SingMOS`, `Sheet`

这样可以先保证你的 ablation 和训练迭代有稳定、低依赖的定量依据。

