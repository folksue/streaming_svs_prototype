# Legacy Streaming SVS Handoff

This file is a handoff summary for continuing work on:

- `legacy/streaming_codec_baseline`

It is intended for a fresh Codex / server-side coding session.

## Scope

The current work is **not** about the `qwen3_8b_singing` route.

The active target is:

- the legacy EnCodec chunk-based baseline

The immediate goal was:

- keep the existing chunk-based local-attention baseline
- replace the old mixed numeric control input with a simpler explicit control schema
- add GPU Docker packaging for this legacy baseline

## Current Design Decision

Do **not** make the legacy path learn duration or shift.

Instead:

- explicit control is generated rule-based from aligned labels
- the model only learns `control -> codec token prediction`
- chunk history only handles short-range continuity

## New Legacy V1 Control Schema

The old legacy condition was:

- `phoneme_id + cond_num[7]`

This has been replaced by four discrete controls:

- `NOTE`
- `PHONEME`
- `SLUR`
- `PHONE_PROGRESS`

Concrete cache fields are now:

- `note_id`
- `phoneme_id`
- `slur`
- `phone_progress`

We intentionally do **not** use `NOTE_PROGRESS` in v1.

Reason:

- OpenCpop `segments` is already flattened as `1 phoneme <-> 1 note`
- if one phoneme spans multiple notes, the dataset duplicates the phoneme and uses `slur=1`
- therefore `NOTE + PHONEME + SLUR + PHONE_PROGRESS` is the smallest useful control set for legacy v1

## Important Behavioral Choice

The legacy model remains:

- chunk-based
- local-causal over chunk history
- parallel classifier over all `(frame, codebook)` slots in the chunk

It is **not**:

- decoder-only interleaved-token training
- codebook autoregressive
- coarse-to-fine multi-codebook decoding

Current multi-codebook behavior is still:

- predict all codebooks in parallel inside each chunk

## Files Changed

### Core implementation

- [legacy/streaming_codec_baseline/preprocess_encodec.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/preprocess_encodec.py)
- [legacy/streaming_codec_baseline/metadata_adapters.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/metadata_adapters.py)
- [legacy/streaming_codec_baseline/model.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/model.py)
- [legacy/streaming_codec_baseline/dataset.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/dataset.py)
- [legacy/streaming_codec_baseline/train.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/train.py)
- [legacy/streaming_codec_baseline/infer.py](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/infer.py)
- [legacy/streaming_codec_baseline/config.yaml](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/config.yaml)

### Documentation

- [legacy/streaming_codec_baseline/CONTROL_PROTOCOL_V1.md](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/CONTROL_PROTOCOL_V1.md)
- [legacy/streaming_codec_baseline/LEGACY_ALIGNMENT_CONDITIONING_DRAFT.md](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/LEGACY_ALIGNMENT_CONDITIONING_DRAFT.md)
- [legacy/streaming_codec_baseline/README.md](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/README.md)

### Docker

- [legacy/streaming_codec_baseline/Dockerfile.gpu](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/Dockerfile.gpu)
- [legacy/streaming_codec_baseline/compose.gpu.yaml](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/compose.gpu.yaml)
- [legacy/streaming_codec_baseline/.dockerignore](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/.dockerignore)
- [legacy/streaming_codec_baseline/DOCKER.md](/mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline/DOCKER.md)

## What Preprocess Now Does

For each chunk center time, preprocess extracts:

1. `note_id`
   - integerized MIDI pitch
   - `0` reserved for rest / inactive note

2. `phoneme_id`
   - from phoneme vocabulary

3. `slur`
   - for OpenCpop, directly from official metadata
   - for datasets without slur, current fallback is all zeros

4. `phone_progress`
   - bucketized progress inside current phoneme
   - controlled by `model.phone_progress_bins`

Boundary diagnostics are still kept:

- `note_on_boundary`
- `note_off_boundary`

## Cache Format Change

Cache format version was bumped.

Current values:

- `CACHE_FORMAT = "svs_chunk_codes"`
- `CACHE_FORMAT_VERSION = 2`

This means:

- old caches built with the previous `cond_num[7]` format are no longer compatible
- preprocess must be rerun

## Dataset / Model Expectations

Current training/inference expects cache items to contain:

- `codes`
- `note_id`
- `phoneme_id`
- `slur`
- `phone_progress`
- `note_on_boundary`
- `note_off_boundary`

The condition encoder is now fully discrete:

- note embedding
- phoneme embedding
- slur embedding
- phone-progress embedding

then projected to `cond_dim`

## Current Model Structure

The current model is still a relatively small chunk model:

- `model_dim=512`
- `attn_layers=4`
- `attn_heads=8`
- `num_blocks=6`
- `history_window=8`

This is not a several-hundred-million-parameter model.

It is still a lightweight legacy baseline.

## OpenCpop Conclusions Used Here

Important dataset interpretation that drove the control design:

- `segments/train.txt` is flattened
- each flattened position corresponds to one `phoneme` and one `note`
- if a phoneme continues across notes, OpenCpop duplicates the phoneme and uses `slur=1`

That is the reason `NOTE_PROGRESS` was dropped from legacy v1.

## Docker Packaging Decision

The Docker setup is for **GPU** training/inference.

It packages:

- the legacy baseline source code

It does **not** package by default:

- the entire repository
- datasets
- generated caches
- checkpoints

Those are expected to be mounted from host.

Build:

```bash
docker build -f legacy/streaming_codec_baseline/Dockerfile.gpu -t legacy-svs:gpu .
```

Run:

```bash
docker compose -f legacy/streaming_codec_baseline/compose.gpu.yaml run --rm legacy-gpu
```

## Git State

Two commits were created and pushed:

- `0e07119` `legacy: switch chunk conditioning to explicit note/phoneme controls`
- `6e1f02f` `legacy: document control protocol v1`

At the time of writing, Docker files and the new legacy draft were added **after** those pushed commits and may still require an additional commit/push depending on repository state.

Check with:

```bash
git status
git log --oneline -5
```

## Important Repo Hygiene Notes

The repository root is dirty with many unrelated changes.

Do not blindly commit everything.

Only touch / stage files that are actually part of the current task.

Known local-only ignore additions:

- `.paper_note_venv/`
- `legacy/streaming_codec_baseline/.venv_smoke/`

`note.md` and `questions.md` were intentionally left trackable.

## Smoke Test Status

Python syntax check already passed for the modified legacy files.

What has **not** been completed yet:

- full preprocess rerun with the new cache schema
- training smoke test with the new cache
- end-to-end inference verification on the updated cache

There was an attempt to build a small local venv for smoke testing:

- `legacy/streaming_codec_baseline/.venv_smoke/`

but this was only for local CPU testing and is not part of the intended production path.

## Recommended Next Steps

1. Rebuild caches with the new preprocess schema:

```bash
python preprocess_encodec.py --config config.yaml --split both
```

2. Run a minimal train smoke test:

```bash
python train.py --config config.yaml
```

3. Verify inference reads the new cache format:

```bash
python infer.py \
  --checkpoint artifacts/checkpoints/best.pt \
  --cache artifacts/data/valid_chunks.pt \
  --index 0 \
  --temperature 0.0
```

4. Only after that, decide whether to change:

- chunk size
- `phone_progress_bins`
- multi-codebook prediction order

## What Not To Do Next

Do not immediately:

- turn legacy into decoder-only interleaved-token training
- add `NOTE_PROGRESS` before validating v1
- redesign multi-codebook decoding before validating the new control schema

The highest-value next validation is simply:

- does the new explicit control schema outperform or stabilize the old `cond_num[7]` baseline?
