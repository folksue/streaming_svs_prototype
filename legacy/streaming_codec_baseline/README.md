# Streaming SVS Prototype (EnCodec Discrete Codec Streaming)
# 流式歌声合成原型（EnCodec 离散码流式建模）

Research prototype for chunk-wise streaming singing synthesis.
分块流式歌声合成研究原型。

## Design Choices / 设计要点
- **Acoustic space**: frozen pretrained EnCodec discrete codec ids (preprocessed offline)
- **声学空间**：使用冻结的预训练 EnCodec 离散码（离线预处理）
- **Chunking**: default 100 ms
- **分块方式**：默认 100 ms
- **Inputs**: phoneme IDs + note/pitch/slur + integer phone-progress bucket features
- **输入条件**：音素 ID + 音符/音高/slur + 整数化的音素进度桶特征
- **Streaming context**: previous-chunk token embedding summary
- **流式上下文**：上一分块 token 嵌入摘要
- **Generator**: chunk-local residual classifier over frame/codebook slots
- **生成器**：面向分块内 frame/codebook 槽位的残差分类器
- **Training loss**: masked token-level cross-entropy (with token accuracy metrics)
- **训练目标**：带 mask 的 token 级交叉熵（并记录 token 准确率）

## Files / 文件说明
- `preprocess_encodec.py`: waveform + alignment manifest -> chunk sequence cache (`.pt`)
- `preprocess_encodec.py`：音频与对齐标注 -> 分块序列缓存（`.pt`）
- `metadata_adapters.py`: real dataset metadata adapters for M4Singer / OpenCpop
- `metadata_adapters.py`：M4Singer / OpenCpop 真实数据元信息适配器
- `dataset.py`: variable-length sequence dataset + padding/mask collate
- `dataset.py`：变长序列数据集与 padding/mask 拼批
- `model.py`: condition encoder + previous-chunk context encoder + discrete codec classifier
- `model.py`：条件编码器 + 上一分块上下文编码器 + 离散码分类器
- `train.py`: teacher-forcing training loop + validation + checkpoint
- `train.py`：Teacher Forcing 训练、验证与断点保存
- `logging_utils.py`: console / TensorBoard / W&B metric logging
- `logging_utils.py`：控制台 / TensorBoard / W&B 指标日志
- `infer.py`: autoregressive generation + diagnostics + optional wav decode
- `infer.py`：自回归生成 + 诊断指标 + 可选 wav 解码
  - current inference path now supports chunk-wise streaming-style generation and chunk-wise decode
  - 当前推理路径已支持逐 chunk 生成与逐 chunk 解码
- `artifacts/`: generated caches, checkpoints, TensorBoard runs, inference outputs
- `artifacts/`：缓存、检查点、训练日志与推理输出
- `config.yaml`: experiment config
- `config.yaml`：实验配置
- `Dockerfile.gpu`: GPU training/inference image
- `Dockerfile.gpu`：GPU 训练/推理镜像
- `compose.gpu.yaml`: GPU compose launcher
- `compose.gpu.yaml`：GPU compose 启动配置
- `DOCKER.md`: container build/run notes
- `DOCKER.md`：容器构建与运行说明
- `examples/manifest_example.json`: sample manifest schema
- `examples/manifest_example.json`：样例 manifest 格式

## Model Architecture (Detailed) / 模型结构（详细）
This section describes the exact model currently implemented in `model.py`.
本节对应 `model.py` 当前实际实现。

### 1) Input and target definitions / 输入与目标定义
- Chunk-level symbolic conditions:
- 分块级符号条件：
  - `note_id`: `[B, T]`
  - `phoneme_id`: `[B, T]`
  - `slur`: `[B, T]`
  - `phone_progress`: `[B, T]`
- Target acoustic tokens:
- 目标声学 token：
  - `codes`: `[B, T, F, K]`
  - `T`: number of chunks / 分块数量
  - `F`: EnCodec frames per chunk / 每分块的 EnCodec 帧数
  - `K`: number of EnCodec codebooks / EnCodec codebook 数量

Training objective is autoregressive next-token prediction over the flattened chunk audio sequence.
训练目标是在展平后的 chunk 音频序列上做自回归下一 token 预测。

### 2) Condition encoder / 条件编码器
- `note_id` -> embedding via `nn.Embedding(note_vocab_size, note_emb_dim)`
- `note_id` 通过 `nn.Embedding(note_vocab_size, note_emb_dim)` 编码
- `phoneme_id` -> embedding via `nn.Embedding(phoneme_vocab_size, phoneme_emb_dim)`
- `phoneme_id` 通过 `nn.Embedding(phoneme_vocab_size, phoneme_emb_dim)` 编码
- `slur` -> embedding
- `slur` -> embedding
- `phone_progress` bucket -> embedding
- `phone_progress` bucket -> embedding
- Concatenate and project to condition hidden state: `cond` `[B, T, cond_dim]`
- 拼接后投影为条件隐状态：`cond` `[B, T, cond_dim]`

### 3) Previous-chunk context encoder / 上一分块上下文编码器
- Discrete code ids are embedded by shared token embedding:
- 离散码 ID 先经过共享 token embedding：
  - `nn.Embedding(codebook_size, token_emb_dim)`
- Mean-pool embeddings over all frames/codebooks per chunk:
- 在每个分块内对 frame/codebook 维度做均值池化：
  - `[B, T, F, K, E] -> [B, T, E]`
- Project pooled representation with `prev_proj` MLP:
- 经 `prev_proj` MLP 投影后得到：
  - `chunk_repr`: `[B, T, prev_chunk_dim]`
- Teacher forcing (train) uses shift-right with learnable `bos_chunk`:
- 训练时 Teacher Forcing 采用带可学习 `bos_chunk` 的右移：
  - `prev_repr[:, 0] = bos_chunk`, `prev_repr[:, t] = chunk_repr[:, t-1]`

### 4) Chunk audio decoder / 分块音频解码器
- Fuse condition and previous-chunk context:
- 融合条件与上一分块上下文：
  - `chunk_in([cond, prev_repr]) -> h`, shape `[B, T, model_dim]`
- Flatten one chunk from `[F, K]` to `[F*K]`
- 将一个 chunk 从 `[F, K]` 展平到 `[F*K]`
- Use shifted-right audio token embeddings with learned `audio_bos`
- 使用带可学习 `audio_bos` 的右移音频 token embedding
- Add slot position embeddings for the flattened chunk token order
- 为展平后的 chunk token 顺序加入位置 embedding
- Run local-causal decoder-only self-attention over chunk audio tokens
- 在 chunk 内音频 token 序列上运行局部因果 decoder-only 自注意力
- Final linear classifier over codebook vocabulary:
- 最后线性分类到 codebook 词表：
  - logits: `[B, T, F, K, codebook_size]`

### 5) Loss and metrics / 损失与指标
- Loss: masked token-level cross-entropy over valid chunk positions
- 损失：仅对有效分块位置计算 mask token 级交叉熵
- Main metric: masked token accuracy
- 主指标：mask token 准确率
- Boundary diagnostics: on-boundary / off-boundary token accuracy
- 边界诊断：音符起止边界 token 准确率

### 6) Autoregressive inference path / 自回归推理路径
- Initialize previous context with `bos_chunk`
- 用 `bos_chunk` 初始化上一分块上下文
- For each chunk `t`:
- 对每个分块 `t`：
  - compute current chunk context from explicit control + previous chunk history
  - 先由显式控制和上一分块历史得到当前 chunk context
  - decode the flattened chunk audio sequence autoregressively
  - 再在展平后的 chunk 音频序列上做 decoder-only 自回归解码
  - slot order is frame-major, then codebook-major inside each frame
  - 槽位顺序为 frame 优先，同一 frame 内按 codebook 索引递增
  - re-encode predicted chunk to update previous context for next chunk
  - 将预测分块重新编码，更新下一分块所需上下文
- Output predicted codes: `[B, T, F, K]`
- 输出预测码：`[B, T, F, K]`
- In the current inference script, generation is performed chunk-by-chunk and optional decoding can also be invoked per chunk, then concatenated.
- 当前推理脚本会逐 chunk 生成；若启用 `decode_wav`，也会逐 chunk 解码后再拼接。

### 7) Why this structure / 结构设计动机
- Streaming-friendly: depends on current condition + previous generated chunk representation
- 适合流式：仅依赖当前条件与上一分块生成表示
- Stable boundaries: explicit previous-chunk token context improves transitions
- 边界更稳：显式上一分块 token 上下文有助于分块衔接
- Efficient baseline: no long-history recurrent stack, single-GPU friendly
- 计算高效：无长历史循环堆叠，单卡易复现

## Manifest Schema / Manifest 格式
Each utterance record should include:
每条样本应包含：
- `utt_id`
- `audio_path`
- `phoneme_ids`
- `phoneme_intervals`: `[[start, end], ...]`
- `note_pitch_midi`
- `note_intervals`: `[[start, end], ...]`
- optional `duration`
- 可选 `duration`

Adapters can also build this schema from real dataset metadata:
也可通过适配器从真实数据集元信息转换到上述格式：
- `manifest`: existing JSON list in the schema above
- `m4singer`: real `meta.json` with `item_name / phs / ph_dur / notes / notes_dur`
- `opencpop`: OpenCpop `transcriptions.txt`-style metadata
- `soulx-singer-eval`: JSON or JSONL with GTSinger-style `phs / ph_dur / notes / notes_dur`

For real metadata adapters, preprocessing builds/reuses phoneme vocab at `data.phoneme_vocab_path`.
对于真实数据适配器，预处理会在 `data.phoneme_vocab_path` 生成/复用音素词表。

If `data.auto_split_valid: true` and valid manifest resolves to zero entries, preprocessing will deterministically carve out validation set from train entries using `utt_id`.
若配置 `data.auto_split_valid: true` 且 valid 清单为空，预处理会基于 `utt_id` 从训练集确定性切分验证集。

## Logging / 日志
Training supports metric backends via `logging.backends`:
训练通过 `logging.backends` 支持多日志后端：
- `console`
- `tensorboard`
- `wandb`

Example / 示例：
```yaml
logging:
  backends: ["console", "tensorboard", "wandb"]
  tensorboard_dir: artifacts/runs/exp_m4
  wandb_project: streaming-svs
  run_name: m4-baseline
  wandb_mode: online
```

TensorBoard tags include step loss, epoch averages, validation metrics, and LR.
TensorBoard 包含 step 损失、epoch 均值、验证指标与学习率。

## Real Dataset Examples / 真实数据集示例
M4Singer:
```yaml
data:
  train_manifest: artifacts/data/m4/train.json
  valid_manifest: artifacts/data/m4/valid.json
  train_manifest_adapter: m4singer
  valid_manifest_adapter: m4singer
  train_audio_root: /path/to/M4Singer
  valid_audio_root: /path/to/M4Singer
  phoneme_vocab_path: artifacts/data/m4_phoneme_vocab.json
```

OpenCpop:
```yaml
data:
  train_manifest: artifacts/data/opencpop/train.txt
  valid_manifest: artifacts/data/opencpop/test.txt
  train_manifest_adapter: opencpop
  valid_manifest_adapter: opencpop
  train_audio_root: /path/to/opencpop/wavs
  valid_audio_root: /path/to/opencpop/wavs
  phoneme_vocab_path: artifacts/data/opencpop_phoneme_vocab.json
```

SoulX-Singer-Eval mixed into training with automatic validation split:
SoulX-Singer-Eval 混入训练并自动切分验证集：
```yaml
data:
  train_manifest: /path/to/soulx_singer_eval/train.jsonl
  valid_manifest: /path/to/soulx_singer_eval/empty.jsonl
  train_manifest_adapter: soulx-singer-eval
  valid_manifest_adapter: soulx-singer-eval
  train_audio_root: /path/to/soulx_singer_eval
  valid_audio_root: /path/to/soulx_singer_eval
  auto_split_valid: true
  valid_ratio: 0.05
  min_valid_items: 32
  phoneme_vocab_path: artifacts/data/soulx_phoneme_vocab.json
```

For auto-split, `valid_manifest` can be an empty JSON list file:
自动切分时，`valid_manifest` 可以是空 JSON 列表文件：
```json
[]
```

## Pipeline / 流程
1. Build train/valid manifests / 构建训练与验证 manifest
2. Preprocess to EnCodec discrete-code chunk cache / 预处理为 EnCodec 离散码分块缓存
3. Train / 训练
4. Inference + diagnostics / 推理与诊断

## Cache Compatibility Checks / 缓存兼容性校验
Cache files include schema metadata (`cache_format`, `cache_format_version`, `cond_dim`, shape meta).
缓存文件包含 schema 元信息（`cache_format`、`cache_format_version`、`cond_dim`、shape 元数据）。

During training startup, the loader validates:
训练启动时会校验：
- payload schema and required keys
- payload 结构与必要字段
- item tensor ranks/shapes and time-dimension consistency
- item 张量维度/形状与时间维一致性
- train/valid cache meta compatibility (`num_codebooks`, `frames_per_chunk`, `codebook_size`, etc.)
- train/valid 缓存元信息一致性（如 `num_codebooks`、`frames_per_chunk`、`codebook_size`）

If checks fail, training exits early with clear error messages.
若校验失败，训练会在早期直接退出并给出明确报错。

## Commands / 常用命令
```bash
python preprocess_encodec.py --config config.yaml --split both
python train.py --config config.yaml
python infer.py --checkpoint artifacts/checkpoints/best.pt --cache artifacts/data/valid_chunks.pt --index 0 --decode_wav
```

## Notes / 说明
- This is a research baseline, not production SVS.
- 本项目是研究基线，不是生产级 SVS 系统。
- EnCodec is frozen by design.
- EnCodec 在设计上保持冻结。
- No raw text is used; only phoneme IDs.
- 不使用原始文本，仅使用音素 ID。
