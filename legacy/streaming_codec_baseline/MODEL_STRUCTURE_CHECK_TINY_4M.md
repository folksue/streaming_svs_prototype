# Legacy 单流模型结构核对（Tiny 4M 配置）

对应实现文件：
- `legacy/streaming_codec_baseline/model.py`

对应配置文件：
- `legacy/streaming_codec_baseline/config_tiny_4m.yaml`

本文结构与 `MODEL_STRUCTURE_CHECK.md` 保持同一口径，用于小模型配置核对。

## 1. 总体结论

当前 Tiny 配置仍是同一条主线：
- 单流 decoder-only 自回归
- 每 step：`[CONTROL][AUDIO...]` 交错
- control/audio 在同一注意力流内建模
- 局部因果滑窗注意力 + RoPE

与 100M 配置的区别仅在容量缩放，不改协议和数据形状。

## 2. 记号与默认配置（Tiny）

记号：
- `B`: batch size
- `T`: step 数
- `S`: `tokens_per_step`
- `K`: 码本数（来自 cache）
- `V`: 码本词表大小（`codebook_size`，来自 cache）
- `H`: `model_dim`
- `L`: 展平后 token 总长度

Tiny 默认（`config_tiny_4m.yaml`）：
- `note_vocab_size=256`
- `phoneme_vocab_size=1024`
- `note_emb_dim=64`
- `phoneme_emb_dim=64`
- `slur_emb_dim=8`
- `phone_progress_bins=8`
- `phone_progress_emb_dim=16`
- `cond_dim=128`
- `token_emb_dim=96`
- `model_dim(H)=192`
- `attn_heads=6`
- `ffn_mult=2.6666666667`
- `ffn_hidden=int(192*2.666...) = 512`
- `attn_layers=2`
- `num_blocks=2`
- `total_layers=4`
- `S=1`（默认）
- `K` 由 cache 决定（EnCodec 常见 8）

派生长度：
- `seq_len_per_step = 1 + S`
- `L = (T + P) * (1 + S)`，`P=ablation_prefix_pad_steps`
- 在默认 `S=1, P=0` 下，`L=2T`

## 3. 按层输入输出维度

### 3.1 ConditionEncoder

输入：
- `note_id [B,T]`
- `phoneme_id [B,T]`
- `slur [B,T]`
- `phone_progress [B,T]`

子层：
- `Embedding(256,64)` -> `[B,T,64]`
- `Embedding(1024,64)` -> `[B,T,64]`
- `Embedding(2,8)` -> `[B,T,8]`
- `Embedding(8,16)` -> `[B,T,16]`

拼接：
- `[B,T,152]`

MLP：
- `Linear(152,128)` + SiLU
- `Linear(128,128)`
- 输出 `cond [B,T,128]`

### 3.2 control token 投影

`control_in`：
- `Linear(128,192)` + SiLU
- `Linear(192,192)`
- 输出 `control_tok [B,T,192]`

### 3.3 audio frame 输入投影

目标：
- `codes [B,T,S,K]`

处理：
- frame 轴 shift-right（首位 BOS）得到 `frame_inputs [B,T,S,K]`
- 每个 codebook embedding 后求和 -> `[B,T,S,96]`
- `audio_in: Linear(96,192)` -> `frame_tok [B,T,S,192]`

### 3.4 交错成单序列

每 step：
- 位置 0: control
- 位置 1..S: audio frame slot

展平：
- `seq_in [B,L,192]`

### 3.5 位置/类型编码

叠加：
- `token_type_emb: Embedding(2,192)`
- `within_step_pos_emb [1+S,192]`
- 注意力内部 Q/K 用 RoPE（`rope_base=10000`）

### 3.6 Transformer blocks

每层输入输出：
- 输入 `x [B,L,192]`
- 局部因果自注意力（`heads=6`, `head_dim=32`）
- SwiGLU FFN（`192 -> 512 -> 192`）
- 残差后仍 `[B,L,192]`

总层数：
- `total_layers=4`

### 3.7 输出层

- `out_norm: RMSNorm(192)`
- 取 frame 位置 hidden -> `[B,T,S,192]`
- `audio_heads[k]: Linear(192,V), k=0..K-1`
- 输出 `logits [B,T,S,K,V]`

## 4. 注意力窗口规则

token 轴上，位置 `i` 可见 `j` 满足：
- 因果：`j <= i`
- 局部：`i-j < token_history_window`

其中：
- `token_history_window = history_window * (1 + S)`
- 当前默认 `history_window=16, S=1`，故 `token_history_window=32`

## 5. 自回归顺序

step 内：
- frame 维串行（`f0 -> f1 -> ...`）
- 每个 frame 内 `K` 个码本由并行头一次性预测

## 6. 参数量拆分（Tiny 默认估算）

估算条件：
- `K=8`
- `V=1024`
- `S=1`

总参数约：
- **4.34M**

主要分项（约值）：

| 模块 | 参数量 |
|---|---:|
| 4 个 Transformer blocks | 1.77M |
| audio_heads | 1.58M |
| codebook token embeddings | 0.79M |
| control_in | 0.06M |
| ConditionEncoder MLP | 0.04M |
| 其余（emb / audio_in / type-pos / norm） | 0.10M |

说明：
- 在小模型下，`audio_heads + codebook embeddings` 仍占比较高。
- 若 `K` 或 `V` 变化，总参数会明显变化（与 cache 强相关）。

## 7. 适用场景建议

在当前 `~9.46M coarse tokens` 数据规模下，这个 4M 级配置比 100M 更符合“先训稳、先看泛化”的目的。

