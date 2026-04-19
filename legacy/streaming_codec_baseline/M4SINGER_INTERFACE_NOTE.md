# M4Singer Interface Note

This note records the official `M4Singer` raw-data shape that the legacy baseline
should expect.

## Official layout

The public docs and downstream dataset guides agree on the following structure:

```text
M4Singer/
  SingerA#SongX/
    0000.wav
    0000.mid
    0000.TextGrid
    ...
  SingerB#SongY/
    0000.wav
    ...
  meta.json
```

The important point for this baseline is that `meta.json` is a list of utterance
records, while audio is stored under `Singer#Song/<sent_id>.wav`.

## Official metadata fields

The official preprocessing code reads these keys from `meta.json`:

- `item_name`
- `wav_fn`
- `txt`
- `phs`
- `ph_dur`
- `notes`
- `notes_dur`
- `is_slur`

Observed from:

- `third_party/M4Singer_official/code/data_gen/singing/binarize.py`

## What the legacy adapter consumes

The legacy baseline converts each item into:

- `utt_id`
- `audio_path`
- `phoneme_symbols`
- `phoneme_intervals`
- `note_pitch_midi`
- `note_intervals`
- `note_slur`

`note_slur` now reads the official `is_slur` field directly instead of forcing
all zeros.

## Path resolution rule

If `wav_fn` exists in `meta.json`, it is used first.

If not, the adapter falls back to:

```text
{singer}#{song}/{sent_id}.wav
```

derived from `item_name = "{singer}#{song}#{sent_id}"`.

## Example command

```bash
cd legacy/streaming_codec_baseline
python preprocess_encodec.py --config your_m4singer_config.yaml --split train
```

Recommended config fields:

```yaml
data:
  train_manifest_adapter: m4singer
  valid_manifest_adapter: m4singer
  train_manifest: /abs/path/to/M4Singer/meta.json
  valid_manifest: /abs/path/to/M4Singer/meta.json
  train_audio_root: /abs/path/to/M4Singer
  valid_audio_root: /abs/path/to/M4Singer
```

## Local reference assets

Public demo audio examples downloaded for quick inspection:

- `/mnt/c/streaming_svs_prototype/datasets_meta/m4singer_samples/1.wav`
- `/mnt/c/streaming_svs_prototype/datasets_meta/m4singer_samples/2.wav`
- `/mnt/c/streaming_svs_prototype/datasets_meta/m4singer_samples/3.wav`
