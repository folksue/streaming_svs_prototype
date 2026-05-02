# 当前模型结构与参数说明（mixed_4ds_ace_100m_8cq_singerfix_rerun）

对应文件：
- 结构实现：[model.py](/mnt/streaming_svs_prototype/legacy/streaming_codec_baseline/model.py)
- 训练入口：[train.py](/mnt/streaming_svs_prototype/legacy/streaming_codec_baseline/train.py)
- 实验配置：[mixed_4ds_ace_100m_8cq_singerfix_rerun.yaml](/mnt/streaming_svs_prototype/legacy/streaming_codec_baseline/configs/experiments/mixed_4ds_ace_100m_8cq_singerfix_rerun.yaml)
- 模型配置：[mixed_100m_8cq.yaml](/mnt/streaming_svs_prototype/legacy/streaming_codec_baseline/configs/model/mixed_100m_8cq.yaml)
- 基础模型配置：[default_100m.yaml](/mnt/streaming_svs_prototype/legacy/streaming_codec_baseline/configs/model/default_100m.yaml)
- runtime 配置：[runtime_mixed_4ds_100m_8cq.yaml](/mnt/streaming_svs_prototype/legacy/streaming_codec_baseline/configs/base/runtime_mixed_4ds_100m_8cq.yaml)

本文是对“当前正在跑的这版模型”做结构和参数核对。它不是泛化描述，而是按当前实际配置、当前数据缓存 meta、当前 singer 设置写的。

## 1. 当前结论

当前这版模型仍然是：
- 单流 decoder-only 自回归
- 每个 step 按 `[CONTROL][AUDIO]` 交错成一条序列
- control 和 audio 共享同一套局部因果注意力块
- 每个 audio frame 内，`K=8` 个 codebook 由 8 个 head 并行预测

当前这版和你之前那轮最大的差别：
- singer conditioning 已开启
- `singer_vocab_size=64`
- `singer_emb_dim=32`
- 当前数据里实际 `max_singer_id=52`，即真实 singer 类别数是 53（含 `0`），配置中的 64 是带冗余的上限

## 2. 当前实际配置

来自当前配置链和 cache meta 的实际值：

### 2.1 训练配置

- `run_name = mixed_4ds_ace_100m_8cq_singerfix_rerun`
- `batch_size = 24`
- `max_seq_len = 256`
- `epochs = 80`
- `save_every_epoch_until = 20`
- `ckpt_interval_epochs = 10`
- `qc.enabled = true`
- `qc.interval_epochs = 10`

### 2.2 音频/cache meta

- `sample_rate = 24000`
- `encodec_model = facebook/encodec_24khz`
- `encodec_bandwidth = 6.0`
- `num_codebooks (K) = 8`
- `tokens_per_step (S) = 1`
- `codebook_size (V) = 1024`
- `max_note_id = 81`
- `max_singer_id = 52`

### 2.3 模型配置

- `note_vocab_size = 256`
- `phoneme_vocab_size = 2048`
- `note_emb_dim = 192`
- `phoneme_emb_dim = 192`
- `slur_emb_dim = 32`
- `phone_progress_bins = 8`
- `phone_progress_emb_dim = 64`
- `singer_vocab_size = 64`
- `singer_emb_dim = 32`
- `cond_dim = 384`
- `token_emb_dim = 192`
- `prev_step_dim = 384`
  说明：当前单流实现里该字段仅保留配置兼容性，实际不参与计算图
- `model_dim = 768`
- `attn_heads = 12`
- `attn_layers = 6`
- `num_blocks = 8`
- `total_layers = 14`
- `history_window = 16`
- `token_history_window = 16 * (1 + S) = 32`
- `audio_history_window = 64`
  说明：当前单流实现里该字段也只是兼容旧配置，不参与核心结构
- `control_encoder_type = mlp`
- `control_conv_layers = 0`
- `ablation_prefix_pad_steps = 0`
- `rope_base = 10000.0`
- `ffn_mult = 2.6666666667`
- `ffn_hidden = int(768 * 2.6666666667) = 2048`

## 3. 当前序列组织方式

## 3.1 每个 step 的 token 组成

当前：
- `tokens_per_step = 1`
- 所以每个 step 只有 1 个 audio slot

因此每个 step 的展平序列长度是：
- `seq_len_per_step = 1 + audio_slots_per_step = 2`

也就是每个 step 实际是：
- 位置 0：`CONTROL`
- 位置 1：`AUDIO`

所以若一条样本有 `T` 个 step：
- 展平总长度 `L = T * 2`

若以后改成 `tokens_per_step = 2`，则会变成：
- `[CONTROL][AUDIO_0][AUDIO_1]`

但当前这版不是这样，当前就是最简单的 `1 control + 1 audio frame`。

## 3.2 codebook 的预测方式

虽然每个 step 只有 1 个 audio frame slot，但这个 frame 内有：
- `K = 8` 个 EnCodec codebook

当前实现不是串行预测 `k0 -> k1 -> ... -> k7`，而是：
- 用同一个 frame hidden
- 经过 8 个 `audio_head`
- 一次并行输出 8 个 codebook 的 logits

所以一步的 audio 预测形状是：
- `logits_step: [B, S=1, K=8, V=1024]`

## 4. 当前按层结构

## 4.1 ConditionEncoder

输入：
- `note_id [B,T]`
- `phoneme_id [B,T]`
- `slur [B,T]`
- `phone_progress [B,T]`
- `singer_id [B,T]`

各 embedding 输出：
- note: `Embedding(256, 192)` -> `[B,T,192]`
- phoneme: `Embedding(2048, 192)` -> `[B,T,192]`
- slur: `Embedding(2, 32)` -> `[B,T,32]`
- progress: `Embedding(8, 64)` -> `[B,T,64]`
- singer: `Embedding(64, 32)` -> `[B,T,32]`

拼接后维度：
- `192 + 192 + 32 + 64 + 32 = 512`

然后过两层 MLP：
- `Linear(512, 384)`
- `SiLU`
- `Linear(384, 384)`

输出：
- `cond [B,T,384]`

说明：
- 这和没开 singer 时不同。没开 singer 时这里的输入总维度只有 `480`。
- 当前因为打开了 `singer_emb_dim=32`，所以 ConditionEncoder 的输入拼接维度变成了 `512`。

## 4.2 control token 投影

当前 `control_encoder_type = mlp`，所以 control token 的构造是：
- `Linear(384, 768)`
- `SiLU`
- `Linear(768, 768)`

输出：
- `control_tok [B,T,768]`

说明：
- 当前没有走 `convnext` 分支
- `control_conv_layers = 0`

## 4.3 audio token 输入投影

GT/历史 codes 的 shape：
- `codes [B,T,S=1,K=8]`

训练时：
- 会先做 frame 级 shift-right
- 用上一个 frame 的 codes 当当前 frame 的输入
- 起始用 `audio_bos_id = 0`

每个 codebook 的 token embedding：
- 8 个 `Embedding(1024, 192)`

8 个 embedding 求和后：
- `frame_emb [B,T,S=1,192]`

再过输入投影：
- `audio_in: Linear(192, 768)`

得到：
- `frame_tok [B,T,S=1,768]`

## 4.4 单流展平

每个 step：
- `control_tok [B,1,768]`
- `frame_tok [B,1,768]`

拼成：
- `[CONTROL][AUDIO]`

整段展平后：
- `seq_in [B,L,768]`

其中：
- `L = 2T`

## 4.5 位置与类型编码

当前叠加的内容：
- `token_type_emb: Embedding(2, 768)`
  - `0 = control`
  - `1 = audio`
- `within_step_pos_emb: Parameter[2, 768]`
  - 对应 step 内两个位置
- `RoPE`
  - 在注意力里作用于 Q/K

另外还有：
- `prefix_control_pad [768]`
- `prefix_frame_pad [768]`

但当前：
- `prefix_pad_steps = 0`

所以这两个 learned virtual pad 参数当前存在，但不会实际参与前向。

## 4.6 Transformer 主干

当前总层数：
- `attn_layers + num_blocks = 6 + 8 = 14`

14 层都放在：
- `self.blocks: ModuleList[LocalCausalSelfAttentionBlock]`

每层结构：
- `RMSNorm(768)`
- 局部因果自注意力
- 残差
- `RMSNorm(768)`
- `SwiGLU FFN(768 -> 2048 -> 768)`
- 残差

注意力细节：
- `attn_heads = 12`
- 每头维度 `head_dim = 768 / 12 = 64`
- 局部窗口大小是 `token_history_window = 32`

当前窗口含义：
- 不是 32 个 step
- 是 32 个 token
- 因为每个 step 有 2 个 token，所以等价于最多看回 `16` 个 step

## 4.7 输出层

主干输出：
- `hidden [B,L,768]`

提取 audio 位置后：
- `frame_hidden [B,T,S=1,768]`

然后走 8 个 head：
- `audio_heads[k]: Linear(768,1024), k=0..7`

最终输出：
- `logits [B,T,S=1,K=8,V=1024]`

训练时：
- 直接对这个 logits 做 cross-entropy

推理时：
- 对每个 `k` 取 `argmax` 或采样
- 得到 `codes_pred [B,T,1,8]`
- 再解码成音频

## 5. 当前参数量（实际值）

下面是按当前实际配置构建模型后统计出的参数量。

总参数量：
- `108,814,144`
- 约 `108.81M`

### 5.1 主要模块参数量

| 模块 | 参数量 |
|---|---:|
| `blocks`（14 层主干） | 99,111,936 |
| `audio_heads` | 6,299,648 |
| `codebook_token_emb`（8 个码本 embedding，总和） | 1,572,864 |
| `control_in` | 886,272 |
| `ConditionEncoder` 两层 MLP（`cond_enc.out`） | 344,832 |
| `audio_in` | 148,224 |
| `cond_enc.phoneme_emb` | 393,216 |
| `cond_enc.note_emb` | 49,152 |
| `cond_enc.singer_emb` | 2,048 |
| `token_type_emb` | 1,536 |
| `within_step_pos_emb` | 1,536 |
| `prefix_control_pad + prefix_frame_pad` | 1,536 |
| `out_norm` | 768 |
| `cond_enc.phone_progress_emb` | 512 |
| `cond_enc.slur_emb` | 64 |

### 5.2 参数量结论

参数几乎都集中在三块：
- 主干 14 层 attention+FFN
- 8 个输出 head
- 8 个 codebook token embeddings

singer 分支本身很小：
- `singer_emb = 64 * 32 = 2048`

所以：
- 把 `singer_vocab_size` 从 53 改到 64，影响极小
- 把 `singer_emb_dim` 从 16 改到 32，也不是显存主因
- 这两个改动主要影响 conditioning 能力，不会显著改变总参数量

## 6. 当前模型为什么还是 108.81M

虽然你把 singer conditioning 打开了，但总参数量仍然大约是 `108.81M`，原因是：

1. 主干参数占了绝大头  
14 层 blocks 就有 `99.11M`

2. singer 分支很小  
就算从：
- `singer_emb_dim = 0` 变成 `32`

增加的也主要只是：
- singer embedding 本身
- ConditionEncoder 第一层输入维度从 `480` 到 `512`

这部分只增加了很少的参数，不会把 108M 量级改成别的数量级。

## 7. 当前训练相关的关键结构点

和你最近排查问题最相关的是这几个点。

### 7.1 singer conditioning 现在是启用的

当前 `ConditionEncoder` 里：
- `use_singer = singer_vocab_size > 0 and singer_emb_dim > 0`

而当前配置是：
- `singer_vocab_size = 64`
- `singer_emb_dim = 32`

所以当前训练这轮一定会把 `singer_id` 拼进 condition。

### 7.2 history window 仍然偏小

当前：
- `history_window = 16`
- `seq_len_per_step = 2`
- 所以 `token_history_window = 32`

也就是模型在注意力里最多回看大约 `16` 个 step 的历史。

这对你的问题意味着：
- control 条件虽然是全序列给进去的
- 但生成时 audio token 的局部历史仍然是受限的

### 7.3 `prev_step_dim` 和 `audio_history_window` 当前不参与核心结构

在 [model.py](/mnt/streaming_svs_prototype/legacy/streaming_codec_baseline/model.py:332) 的构造函数里：
- `del prev_step_dim, audio_history_window`

这说明当前实现已经不是旧的“双层结构”，而是单流结构。

所以如果你还在按旧理解看这两个参数，就会误判当前模型实际结构。

## 8. 当前这版相对旧 run 的差异

旧 run 的问题：
- checkpoint 里没开 singer embedding
- 实际训练时没把 singer 条件真正接到模型里

当前这版的变化：
- `singer_vocab_size: 64`
- `singer_emb_dim: 32`
- batch 改成 `24`
- 训练另开了 rerun 目录，避免覆盖旧 run

所以当前这版才是“speaker/singer conditioning 修正后”的正式版本。

## 9. 总结

当前模型的核心画像可以概括为：

- 单流 decoder-only
- `[CONTROL][AUDIO]` 两 token/step
- `K=8` 个 codebook 并行输出
- `V=1024`
- `14` 层主干
- `H=768`
- `ffn_hidden=2048`
- `token_history_window=32`
- `singer_vocab_size=64`
- `singer_emb_dim=32`
- 总参数 `108.81M`

如果你后面继续做结构调优，最值得优先关注的不是 singer embedding 本身大小，而是：
- AR 训练/推理 gap
- history window 是否过小
- fine codebook 学习是否掉队
- `max_seq_len=256` 与实际推理长度的偏差
