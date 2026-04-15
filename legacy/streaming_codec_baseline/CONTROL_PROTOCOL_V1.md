# Legacy Control Protocol V1

This document records the control-conditioning scheme currently used by the `legacy/streaming_codec_baseline` path.

## Goal

Replace the old mixed numeric condition vector

- `phoneme_id + cond_num[7]`

with a simpler explicit control schema that is closer to the intended singing-conditioning design:

- `NOTE`
- `PHONEME`
- `SLUR`
- `PHONE_PROGRESS`

The legacy model remains chunk-based. This protocol does **not** convert the model into a decoder-only interleaved-token system. It only changes how each chunk is conditioned.

## Step Unit

The current legacy path still uses the existing chunk lattice:

- one model step = one EnCodec chunk
- chunk size is controlled by `audio.chunk_ms`

The longer-term design idea of `tokens_per_step` is intentionally kept out of this legacy implementation. For this path, control is sampled at each chunk center time.

## Control Fields

Each chunk now carries these discrete controls:

1. `note_id`
   - Current active note at chunk center.
   - Derived from `note_pitch_midi`.
   - `0` is reserved for rest / inactive note.

2. `phoneme_id`
   - Current active phoneme at chunk center.
   - Comes from the phoneme vocabulary adapter.

3. `slur`
   - Current note slur flag.
   - For OpenCpop this comes directly from the official flattened metadata.
   - For datasets without slur annotation, the current fallback is all zeros.

4. `phone_progress`
   - Bucketized progress inside the active phoneme.
   - Computed rule-based from explicit phoneme intervals.
   - Number of buckets is controlled by `model.phone_progress_bins`.

## Why `NOTE_PROGRESS` Is Not Used In V1

For the OpenCpop-style flattened supervision used here:

- each row position is already `1 phoneme <-> 1 note`
- if one phoneme continues across multiple notes, the dataset duplicates that phoneme and marks continuation with `slur=1`

Because of that, `NOTE_PROGRESS` was intentionally dropped from the legacy V1 condition set. The combination

- `NOTE`
- `PHONEME`
- `SLUR`
- `PHONE_PROGRESS`

is the smallest control set that still preserves the intended singing structure.

## Preprocess Output Schema

Each cached sequence item now stores:

- `codes`: `[T, F, K]`
- `note_id`: `[T]`
- `phoneme_id`: `[T]`
- `slur`: `[T]`
- `phone_progress`: `[T]`
- `note_on_boundary`: `[T]`
- `note_off_boundary`: `[T]`

The boundary tensors are kept for evaluation only.

## Model Input

The condition encoder now embeds four discrete controls:

- note embedding
- phoneme embedding
- slur embedding
- phone-progress embedding

and projects the concatenated embedding to the model condition space.

## Training Target

The legacy model still predicts EnCodec code indices for the current chunk.

Nothing about the prediction target changed:

- controls are inputs only
- the model predicts chunk audio codes

## Key Hyperparameters

The relevant V1 control hyperparameters are:

- `audio.chunk_ms`
- `model.note_vocab_size`
- `model.phone_progress_bins`

`phone_progress_bins` is the main granularity knob for this protocol.
