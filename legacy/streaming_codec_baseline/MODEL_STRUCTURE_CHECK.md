# Legacy 单流模型结构核对（中文详细版）

对应实现文件：
- `legacy/streaming_codec_baseline/model.py`

本文用于做结构审查：逐层输入/输出维度 + 参数量拆分。

## 1. 总体结论

当前实现已经是你要求的路线：
- **单流 decoder-only 自回归**
- 每个 step 序列按 `[CONTROL][AUDIO...]` 交错
- control 与 audio 在**同一注意力流**里参与计算
- 使用统一的**局部因果滑动窗口注意力**

不再是之前“外层 step transformer + 内层 audio transformer”的两级结构。

## 2. 记号与默认配置

记号：
- `B`：batch size
- `T`：step 数
- `S`：`tokens_per_step`
- `K`：码本数（来自 cache meta）
- `V`：码本词表大小（`codebook_size`）
- `H`：`model_dim`
- `L`：展平后的总 token 长度

当前默认配置（`config.yaml`）：
- `note_vocab_size=256`
- `phoneme_vocab_size=256`
- `note_emb_dim=192`
- `phoneme_emb_dim=192`
- `slur_emb_dim=32`
- `phone_progress_bins=8`
- `phone_progress_emb_dim=64`
- `cond_dim=384`
- `token_emb_dim=192`
- `model_dim(H)=768`
- `attn_heads=12`
- `ffn_mult=2.6666666667`
- `ffn_hidden=int(768*2.666...) = 2048`
- `attn_layers=6`
- `num_blocks=8`
- `total_layers=14`
- `S=1`（默认）
- `K` 由 cache 决定（EnCodec 常见是 8）

派生长度：
- 每 step token 数：`seq_len_per_step = 1 + S`（1 个 control + S 个 frame slot）
- 总长度：`L = (T + P) * (1 + S)`，其中 `P=ablation_prefix_pad_steps`
- 若 `S=1`：`seq_len_per_step=2`，`L=2T`
  - 若 `P>0`，会在真实 step 前插入 `P` 个“虚拟 step”（learned padding token），用于消融“前置历史”效果

## 3. 按层输入输出维度

## 3.1 ConditionEncoder（控制编码）

输入：
- `note_id [B,T]`
- `phoneme_id [B,T]`
- `slur [B,T]`
- `phone_progress [B,T]`

子层输出：
- `Embedding(256,192)` -> `[B,T,192]`（note）
- `Embedding(256,192)` -> `[B,T,192]`（phoneme）
- `Embedding(2,32)` -> `[B,T,32]`（slur）
- `Embedding(8,64)` -> `[B,T,64]`（progress）

拼接：
- `[B,T,192+192+32+64] = [B,T,480]`

MLP：
- `Linear(480,384)` + SiLU -> `[B,T,384]`
- `Linear(384,384)` -> `cond [B,T,384]`

## 3.2 control token 投影

`control_in`：
- `Linear(384,768)` + SiLU -> `[B,T,768]`
- `Linear(768,768)` -> `control_tok [B,T,768]`

可选方案（新加开关）：
- `control_encoder_type=convnext` 时，改为：
  - `Linear(cond_dim, H)` 投影
  - 再过 `control_conv_layers` 层 1D ConvNeXtV2 风格 block（沿 step 轴）
  - 输出仍是 `[B,T,H]`
- 相关配置：
  - `control_conv_layers`
  - `control_conv_hidden_mult`
  - `control_conv_kernel_size`
  - `control_conv_dilation`

## 3.3 audio frame 输入投影（step 内 frame 串行）

原始目标：
- `codes [B,T,S,K]`

重排：
- `frame_targets_flat [B,T*S,K]`
- 左移一位并补 BOS（按 frame 轴）-> `frame_inputs_flat [B,T*S,K]`
- reshape -> `frame_inputs [B,T,S,K]`

embedding/proj：
- `codebook_token_emb[k]: Embedding(V,192), k=0..K-1`
- 每个 frame 把 K 个码本 embedding 求和 -> `[B,T,S,192]`
- `audio_in: Linear(192,768)` -> `frame_tok [B,T,S,768]`

## 3.4 交错成单序列

每个 step：
- 第 0 位：control token
- 第 1..S 位：frame token

展平后：
- `seq_in [B,L,768]`

训练监督：
- 直接对 `codes [B,T,S,K]` 做 K 头分类监督（不是单个展平 token target）

## 3.5 类型/位置编码

叠加项：
- `token_type_emb: Embedding(2,768)`（control=0, audio=1）
- `within_step_pos_emb [1+S,768]`（每 step 复用）
- 注意力内部使用 `RoPE`（基于全局 token 位置 index 旋转 Q/K，`rope_base` 可配）

叠加后维度不变：
- `seq_in [B,L,768]`

## 3.6 Transformer Block（每层）

输入：
- `x [B,L,768]`
- `attn_mask [L,L]`（局部 + 因果）
- `key_padding_mask [B,L]`

子结构：
- RMSNorm -> `[B,L,768]`
- 自定义 QKV 注意力 + RoPE + SDPA（`heads=12`, `head_dim=64`）-> `[B,L,768]`
- 残差加和 -> `[B,L,768]`
- RMSNorm -> `[B,L,768]`
- SwiGLU FFN：
  - gate/up: `Linear(768,2048)` 两路
  - 逐元素门控后 `[B,L,2048]`
  - down: `Linear(2048,768)`
- 残差加和 -> `[B,L,768]`

总层数：
- `total_layers = attn_layers + num_blocks = 14`

## 3.7 输出层

- `out_norm: RMSNorm(768)` -> `[B,L,768]`
- 提取 frame 位置 hidden -> `[B,T,S,768]`
- `audio_heads[k]: Linear(768,V), k=0..K-1`
- 堆叠得到 `logits [B,T,S,K,V]`

`forward_train` 返回就是：
- `logits [B,T,S,K,V]`

## 4. 注意力窗口规则

flatten token 轴上，位置 `i` 只能看 `j`，满足：
- 因果：`j <= i`
- 局部：`i-j < token_history_window`

其中：
- `token_history_window = history_window * (1 + S)`

即 `history_window` 先按 step 定义，再换算成 token 窗口。

## 5. 自回归顺序（step 内）

step 内顺序是：
- 先按 frame 串行（`f0 -> f1 -> ... -> f(S-1)`）
- 同一 frame 内 `K` 个 codebook 由并行头一次性预测（不是 `k0..kK-1` 串行）

若 `S=2, K=3`，step 内 audio 顺序为：
1. 预测 `f0` 的 `(k0,k1,k2)`（并行）
2. 预测 `f1` 的 `(k0,k1,k2)`（并行）

完整 step 序列为：
- `[CONTROL], f0, f1, ...`

## 6. 推理路径说明

`generate_stream` 逐 step 输出 `[B,S,K]`。

当前实现为了复用训练同一路径，在每次采样时会构建“部分已生成 + 未来填 BOS”的临时流并前向，功能正确；还未做 KV-cache 优化。

## 7. 按层参数量拆分（公式）

下表用 `V` 表示 codebook size，`K` 表示码本数。

| 模块 | 参数公式 |
|---|---|
| note embedding | `note_vocab_size * note_emb_dim` |
| phoneme embedding | `phoneme_vocab_size * phoneme_emb_dim` |
| slur embedding | `2 * slur_emb_dim` |
| progress embedding | `phone_progress_bins * phone_progress_emb_dim` |
| ConditionEncoder MLP | `(480*384+384) + (384*384+384)` |
| codebook token embeddings | `K * (V * 192)` |
| audio_in | `192*768 + 768` |
| control_in | `(384*768+768) + (768*768+768)` |
| token_type_emb | `2*768` |
| within_step_pos_emb | `(1+S)*768` |
| 单个 Transformer block | `2*768 + 4*768*768 + 3*768*2048` |
| 全部 blocks | `total_layers * 单层参数` |
| out_norm | `768` |
| audio_heads | `K * (768*V + V)` |

备注：
- 单层 block 里 `4*768*768` 来自 MHA 的 `qkv + out` 权重（无 bias）。
- FFN 用 SwiGLU，两条上投影 + 一条下投影，共 `3` 个线性层（无 bias）。

## 8. 按层参数量拆分（默认配置示例）

示例取常见 EnCodec 设置：
- `V=1024`
- `K=8`
- `S=1`

得到约：
- **总参数约 109.40M**

分项如下（约值）：

| 模块 | 参数量 |
|---|---:|
| 14 个 Transformer blocks | 98.13M |
| audio_heads | 6.30M |
| codebook token embeddings | 1.57M |
| control_in | 0.89M |
| ConditionEncoder MLP | 0.33M |
| audio_in | 0.15M |
| note embedding | 0.05M |
| phoneme embedding | 0.05M |
| within_step_pos_emb (`1+1=2`) | <0.01M |
| 其余（type emb / norm / slur/progress emb） | <0.01M |

说明：
- 你之前约 109M 的版本是“两级结构”的统计；当前改成单流后，参数重心更集中在统一 blocks。
- 实际总参数会随 `V/K/S` 和配置变动。

## 9. 核对结论

和你本轮要求一致：
- control 与 audio 直接交错并参与同一注意力
- 不再是外层/内层分离
- 局部滑窗注意力是统一作用在单一 token 流上
