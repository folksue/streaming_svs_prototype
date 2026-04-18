# Legacy Streaming SVS Handoff

This handoff is for continuing work on:

- [legacy/streaming_codec_baseline](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline)

It reflects the current state after the step-level refactor and smoke-test validation.

## Scope

This is **not** the `qwen3_8b_singing` route.

The active target is still the legacy EnCodec baseline, but it is no longer chunk-semantic. The current line is:

- EnCodec discrete codec tokens
- explicit symbolic control
- step-level local/sliding causal attention
- main-codebook autoregression with residual parallel heads

## Current Design

The legacy path now treats each training unit as a **step**, not a chunk.

Per step:

- one control state
- one grouped audio target of `tokens_per_step` consecutive EnCodec frames

The grouping hyperparameter is:

- `audio.tokens_per_step`

This is only an engineering grouping factor. It replaces the old chunk packaging role, but it is **not** a semantic chunk boundary in the model.

## Current Control Schema

The active v1 control set is:

- `NOTE`
- `PHONEME`
- `SLUR`
- `PHONE_PROGRESS`

Concrete cache fields:

- `note_id`
- `phoneme_id`
- `slur`
- `phone_progress`
- `note_on_boundary`
- `note_off_boundary`

We still do **not** use `NOTE_PROGRESS`.

Reason:

- for the current OpenCpop-style flattened supervision, `SLUR + NOTE + PHONEME + PHONE_PROGRESS` is the smallest useful set

## Current Model Behavior

The old outer chunk-history mechanism has been removed.

The current model is:

- step-level local causal attention over explicit control + previous-step summary
- within each step:
  - `codebook 0` is decoded sequentially
  - `codebook 1..K-1` are predicted by parallel residual heads

So the current decoding protocol is:

- **not** full flatten-all-codebooks autoregression
- **not** delay pattern
- **not** fully parallel all-codebook prediction

It is:

- **main codebook AR + same-step residual parallel heads**

This choice is intentional because the model has strong per-step control inputs, and delay scheduling would misalign control with finer codebooks from shifted time positions.

## Cache Format

Cache format has changed again and old caches are invalid.

Current values:

- `CACHE_FORMAT = "svs_step_codes"`
- `CACHE_FORMAT_VERSION = 3`

This means:

- any earlier chunk-based cache must be regenerated

## What “Regenerate Cache” Means

It means rerunning:

- [preprocess_encodec.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/preprocess_encodec.py)

to rebuild:

- train cache
- valid cache

with the new step-based schema.

Old chunk-based `.pt` cache files are not compatible with the new code.

## Files Most Relevant Now

Core implementation:

- [preprocess_encodec.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/preprocess_encodec.py)
- [metadata_adapters.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/metadata_adapters.py)
- [dataset.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/dataset.py)
- [model.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/model.py)
- [train.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/train.py)
- [infer.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/infer.py)
- [config.yaml](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/config.yaml)
- [resume_soulx_train.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/resume_soulx_train.py)
- [soulx_train_config.yaml](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/soulx_train_config.yaml)

Current architecture note:

- [ARCHITECTURE_NOTE.md](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/ARCHITECTURE_NOTE.md)

Smoke config:

- [examples/config_smoke.yaml](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/examples/config_smoke.yaml)
- [examples/opencpop_smoke_train.txt](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/examples/opencpop_smoke_train.txt)
- [examples/opencpop_smoke_valid.txt](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/examples/opencpop_smoke_valid.txt)

## Important Bug Fix Included

OpenCpop note parsing needed a compatibility fix for note spellings like:

- `G#4/Ab4`

This was fixed in:

- [metadata_adapters.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/metadata_adapters.py)

The parser now splits on `/` and uses the first spelling before converting to MIDI.

## Current Training / Inference Semantics

### Training

- data is stored as step sequences
- local causal attention is applied directly during training
- there is no special extra “simulate truncation” branch
- the training-time attention constraint is already the same local-window assumption used by inference

### Inference

- continuous step-by-step generation
- one control state per step
- same sliding/local context rule as training
- EnCodec wav decoding can still be invoked step-by-step at script level

## Smoke Test Status

This has already been validated locally on CPU.

Environment used:

- temporary venv at `/tmp/legacy_svs_smoke`

Verified dependencies:

- `torch 2.11.0+cpu`
- `torchaudio 2.11.0+cpu`
- `transformers 5.5.4`
- `soundfile`
- `numpy`

### Smoke preprocess

Command:

```bash
cd /mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline
/tmp/legacy_svs_smoke/bin/python preprocess_encodec.py --config examples/config_smoke.yaml --split both
```

Result:

- succeeded
- wrote:
  - [train_steps.pt](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/artifacts_smoke/data/train_steps.pt)
  - [valid_steps.pt](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/artifacts_smoke/data/valid_steps.pt)

### Smoke train

Command:

```bash
cd /mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline
/tmp/legacy_svs_smoke/bin/python train.py --config examples/config_smoke.yaml
```

Result:

- succeeded
- tiny smoke model trained for one epoch
- reported:
  - `Model params: 1.32M`
  - final validation metrics were emitted without shape/runtime failure

This is enough to say:

- preprocess works
- new cache schema works
- training forward/backward works
- the step-level refactor is not broken at basic runtime level

## Logging

There is already a metric logger abstraction in:

- [logging_utils.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/logging_utils.py)

Supported backends:

- `console`
- `tensorboard`
- `wandb`

Practical recommendation for this stage:

- use `console + tensorboard`
- `wandb` is optional and not required yet

## Docker

GPU Docker packaging already exists for the legacy baseline:

- [Dockerfile.gpu](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/Dockerfile.gpu)
- [compose.gpu.yaml](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/compose.gpu.yaml)
- [DOCKER.md](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/DOCKER.md)

It packages:

- the legacy baseline source

It does **not** package by default:

- datasets
- generated caches
- checkpoints

Those are still expected to be mounted from host.

## Git Commits Relevant to This Work

Already on remote `main`:

- `0e07119` `legacy: switch chunk conditioning to explicit note/phoneme controls`
- `6e1f02f` `legacy: document control protocol v1`
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
