# Legacy Streaming SVS Handoff

This handoff tracks the current production branch for:

- [legacy/streaming_codec_baseline](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline)

Scope is still legacy controllable SVS baseline (not `qwen3_8b_singing` route).

## Current Mainline (Important)

Current model is now **single-stream decoder-only interleaving**:

- step sequence layout: `[CONTROL], [FRAME_0], [FRAME_1], ... [FRAME_(S-1)]`
- each frame slot predicts **all codebooks in parallel** (`K` linear heads)
- no more “codebook-0 AR + residual fine heads” path
- local/sliding causal attention on token stream
- RoPE enabled
- inference supports online incremental generation with KV cache

So this line is:

- explicit symbolic control per step
- fixed step grouping via `tokens_per_step = S`
- parallel multi-codebook prediction per frame slot
- streaming-friendly incremental decode

## Control Protocol (v1)

Per-step control fields:

- `note_id`
- `phoneme_id`
- `slur`
- `phone_progress`
- `note_on_boundary`
- `note_off_boundary`

Still using `PHONE_PROGRESS`, and still **not** introducing `NOTE_PROGRESS` in current legacy line.

## Cache Format

Step cache format currently required by dataset/model:

- `CACHE_FORMAT = "svs_step_codes"`
- `CACHE_FORMAT_VERSION = 3`

If old chunk cache is used, regenerate with:

- [preprocess_encodec.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/preprocess_encodec.py)

## Key Hyperparameters

From current default config:

- `audio.tokens_per_step: 1` (current default mainline)
- `model.model_dim: 768`
- `model.attn_heads: 12`
- `model.attn_layers: 6`
- `model.num_blocks: 8`
- `model.history_window: 16`
- `model.rope_base: 10000.0`
- `model.ablation_prefix_pad_steps: 0` (ablation knob)

Note:

- `tokens_per_step` defines per-step audio slot count (`S`)
- `K` comes from cached codec metadata (`num_codebooks`)

## Streaming Behavior

Implemented in:

- [model.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/model.py)

Entry points:

- `forward_train(...)` for offline training
- `generate_stream(...)` for online step-by-step generation
- `generate(...)` wrapper over `generate_stream`

Inference script:

- [infer.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/infer.py)

`infer.py` can decode per-step outputs to waveform with EnCodec decoder calls. This is streaming-style script orchestration (step loop), not persistent internal EnCodec decode state.

## Prefix-Pad Ablation

Ablation support exists for synthetic history at sequence start:

- `model.ablation_prefix_pad_steps`

Behavior:

- prepends learned virtual steps before first real step
- applies in train and stream generation paths

## Mixed-Dataset Path

Training can use mixed manifests via config-level manifests/adapters in:

- [configs/train_100m_mixed_official.yaml](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/configs/train_100m_mixed_official.yaml)

Primary train launcher:

- [train.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/train.py)

## Architecture / Docs

Model structure note (updated):

- [ARCHITECTURE_NOTE.md](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/ARCHITECTURE_NOTE.md)

Main related files:

- [preprocess_encodec.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/preprocess_encodec.py)
- [dataset.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/dataset.py)
- [metadata_adapters.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/metadata_adapters.py)
- [model.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/model.py)
- [train.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/train.py)
- [infer.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/infer.py)
- [logging_utils.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/logging_utils.py)

## Logging / Repro

Logger backends:

- `console`
- `tensorboard`
- `wandb`

For current stage, default recommendation remains `console + tensorboard`; `wandb` optional.

## Docker

GPU Docker assets remain in:

- [Dockerfile.gpu](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/Dockerfile.gpu)
- [compose.gpu.yaml](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/compose.gpu.yaml)
- [DOCKER.md](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/DOCKER.md)

They package source/runtime deps; datasets/caches/checkpoints are expected as mounted host volumes.

## Recent Commits On Main (Context)

- `fce18a0` `legacy: switch to frame-AR + parallel codebook heads, add RoPE and online streaming decode`
- `308c3c2` `chore: ignore local research assets for cleaner pull/rebase workflow`
- `c69d70b` `docs: add project note handoff summary`
- `4c212d9` `legacy: add gpu docker packaging and handoff docs`
- `8525c3b` `legacy: switch codec baseline to step-level decoding`
- `088b10e` `legacy: add step smoke config and notes`

## What Is Still Outdated

Some docs still describe the old chunk-based behavior and need refresh if they are going to be used as source-of-truth:

- [README.md](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/README.md)
- [CONTROL_PROTOCOL_V1.md](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/CONTROL_PROTOCOL_V1.md)
- [LEGACY_ALIGNMENT_CONDITIONING_DRAFT.md](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/LEGACY_ALIGNMENT_CONDITIONING_DRAFT.md)

The code is current; those docs are not fully current.

## Recommended Next Steps

If continuing the legacy line:

1. decide whether to keep the current:
   - `codebook0 AR + residual parallel heads`
   or move to:
   - stronger RVQ-style residual decoding
2. add a cleaner experiment matrix for:
   - current residual-head baseline
   - delay pattern baseline if still desired
   - future XY-tokenizer / Qwen route comparisons
3. refresh stale docs so they match the step-level model

If pivoting toward the future Qwen route:

- keep this legacy path as a validated EnCodec baseline
- do not mix the legacy step-baseline refactor with the future XY/Qwen architecture in one branch of reasoning
