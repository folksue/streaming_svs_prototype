# Streaming SVS Prototype
# 流式歌声合成原型

This repository contains two related research routes:
1. The **root route** is a chunk-based streaming SVS baseline built on frozen discrete codec targets.
2. The **legacy route** in `legacy/streaming_codec_baseline` is the newer step-wise decoder-only line with local attention and RoPE.

本仓库包含两条相关但不同的研究路线：
1. **根目录路线** 是基于冻结离散 codec 目标的 chunk 级流式 SVS 基线。
2. `legacy/streaming_codec_baseline` 下的 **legacy 路线** 是 step 级 decoder-only 路线，使用局部注意力和 RoPE。

## What This Root Route Is / 根目录路线是什么

The root codebase is a chunk-wise discrete-codec SVS prototype.
It preprocesses audio into EnCodec code chunks, conditions on phoneme and note features, and predicts the next chunk with a chunk-local classifier.

根目录代码是一个 chunk 级离散 codec SVS 原型。
它会把音频预处理成 EnCodec code chunk，在音素和音符特征条件下预测下一个 chunk，核心是 chunk-local classifier。

### Core properties / 核心特性
- Frozen EnCodec targets, precomputed offline
- 冻结的 EnCodec 目标，离线预处理
- Chunk-level symbolic conditioning
- chunk 级符号条件
- Previous-chunk context as streaming history
- 以前一个 chunk 的表示作为流式历史
- Masked token cross-entropy training
- mask token 交叉熵训练
- Single-GPU friendly research baseline
- 适合单卡复现实验的研究基线

## When to Use Which Route / 什么时候用哪条路线

- Use the **root route** if you want the original chunk-based baseline.
- 如果你要的是最初那条 chunk-based 基线，就用根目录路线。
- Use the **legacy route** if you want step-wise autoregressive decoding, local attention, and the current decoder-only design.
- 如果你要 step-wise 自回归、局部注意力和当前 decoder-only 设计，就用 `legacy` 路线。

## Repository Layout / 仓库结构

- `preprocess_encodec.py`: waveform + alignment manifest -> chunk cache
- `preprocess_encodec.py`：音频 + 对齐标注 -> chunk 缓存
- `dataset.py`: variable-length cache dataset + padding/mask collation
- `dataset.py`：变长缓存数据集 + padding/mask 拼批
- `metadata_adapters.py`: adapters for real dataset metadata
- `metadata_adapters.py`：真实数据集元信息适配器
- `model.py`: condition encoder + previous-chunk encoder + codec classifier
- `model.py`：条件编码器 + 上一 chunk 编码器 + codec 分类器
- `train.py`: teacher-forcing training, validation, checkpoints
- `train.py`：Teacher Forcing 训练、验证、断点保存
- `infer.py`: autoregressive inference and optional wav decode
- `infer.py`：自回归推理与可选 wav 解码
- `logging_utils.py`: console / TensorBoard / W&B logging
- `logging_utils.py`：控制台 / TensorBoard / W&B 日志
- `examples/manifest_example.json`: manifest schema example
- `examples/manifest_example.json`：manifest 格式样例
- `legacy/streaming_codec_baseline/`: step-wise decoder-only route
- `legacy/streaming_codec_baseline/`：step 级 decoder-only 路线

## Root Model Summary / 根目录模型概览

### Inputs / 输入
- `phoneme_id`: `[B, T]`
- `note_id` / pitch-related condition: `[B, T]`
- timing / progress features: `[B, T, ...]`
- `phoneme_id`: `[B, T]`
- `note_id` / 音高相关条件: `[B, T]`
- 时序 / 进度特征: `[B, T, ...]`

### Targets / 目标
- `codes`: `[B, T, F, K]`
- `T`: chunks
- `F`: EnCodec frames per chunk
- `K`: EnCodec codebooks
- `codes`: `[B, T, F, K]`
- `T`: chunk 数
- `F`: 每个 chunk 的 EnCodec frame 数
- `K`: EnCodec codebook 数

### Computation / 计算方式
- Condition encoder maps symbolic inputs to `cond`
- 条件编码器把符号输入映射为 `cond`
- Previous generated chunk is embedded and pooled into streaming context
- 上一个生成 chunk 会先 embedding，再池化成流式上下文
- Chunk predictor expands the fused state to frame/codebook slots
- chunk predictor 会把融合后的状态扩展到 frame/codebook 槽位
- Final head classifies each slot over the codebook vocabulary
- 最后分类头对每个槽位做 codebook 词表分类

## Data Format / 数据格式

Each manifest item should contain at least:
每条 manifest 至少应包含：
- `utt_id`
- `audio_path`
- `phoneme_ids` or `phoneme_symbols`
- `phoneme_intervals`
- `note_pitch_midi`
- `note_intervals`
- optional `duration`
- 可选 `duration`

Optional speaker fields are supported and normalized into `singer_id` when present.
如果包含可选说话人字段，预处理会统一规范化为 `singer_id`。

Supported adapters include:
支持的适配器包括：
- `manifest`
- `m4singer`
- `opencpop`
- `kising`
- `soulx-singer-eval`
- `ace-opencpop-segments`
- `ace-kising-segments`

## Quick Start / 快速开始

### 1) Build cache / 构建缓存
```bash
python preprocess_encodec.py --config config.yaml --split both
```

### 2) Train / 训练
```bash
python train.py --config config.yaml
```

### 3) Inference / 推理
```bash
python infer.py --checkpoint artifacts/checkpoints/best.pt --cache artifacts/data/valid_chunks.pt --index 0 --decode_wav
```

If you want the mixed-data setup, use one of the mixed configs such as `mixed_2ds_official_config.yaml` or `mixed_tiny_4m.yaml`.
如果你要混合数据配置，请使用 `mixed_2ds_official_config.yaml` 或 `mixed_tiny_4m.yaml` 这类配置。

## Notes / 说明

- EnCodec is frozen by design.
- EnCodec 设计上保持冻结。
- This root route does not use raw text directly; it uses phoneme IDs.
- 根目录路线不直接使用原始文本，而是使用音素 ID。
- The step-wise decoder-only route is documented separately in `legacy/streaming_codec_baseline/README.md`.
- step 级 decoder-only 路线单独记录在 `legacy/streaming_codec_baseline/README.md`。
- This repository is a research baseline, not a production SVS system.
- 这个仓库是研究基线，不是生产级 SVS 系统。
