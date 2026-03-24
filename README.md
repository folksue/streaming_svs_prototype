# Streaming SVS Prototype (EnCodec Latent + Flow-Style Generator)

Research prototype for chunk-wise streaming singing synthesis.

## Design choices
- **Acoustic space**: frozen pretrained EnCodec latent (preprocessed offline)
- **Chunking**: default 100 ms
- **Inputs**: phoneme IDs + note/pitch/timing/progress features
- **Streaming context**: 2-layer GRU over previous latent chunks
- **Generator**: lightweight flow-matching style residual MLP backbone
- **Training loss**: latent L1

## Files
- `preprocess_encodec.py`: waveform + alignment manifest -> chunk sequence cache (`.pt`)
- `metadata_adapters.py`: real dataset metadata adapters for M4Singer / OpenCpop
- `dataset.py`: variable-length sequence dataset + padding/mask collate
- `model.py`: condition encoder + GRU context + flow-style generator
- `train.py`: teacher-forcing training loop + validation + checkpoint
- `logging_utils.py`: console / TensorBoard / W&B metric logging
- `infer.py`: autoregressive generation + diagnostics + optional wav decode
- `artifacts/`: generated caches, checkpoints, TensorBoard runs, inference outputs
- `config.yaml`: experiment config
- `examples/manifest_example.json`: sample manifest schema

## Manifest schema
Each utterance record should include:
- `utt_id`
- `audio_path`
- `phoneme_ids`
- `phoneme_intervals`: `[[start, end], ...]`
- `note_pitch_midi`
- `note_intervals`: `[[start, end], ...]`
- optional `duration`

Adapters can also build this schema from real dataset metadata:
- `manifest`: existing JSON list in the schema above
- `m4singer`: real `meta.json` with `item_name / phs / ph_dur / notes / notes_dur`
- `opencpop`: OpenCpop `transcriptions.txt`-style metadata
- `soulx-singer-eval`: JSON or JSONL with GTSinger-style `phs / ph_dur / notes / notes_dur` fields

For real metadata adapters, preprocessing builds and reuses a shared phoneme vocab JSON at `data.phoneme_vocab_path`.
If `data.auto_split_valid: true` and the valid manifest resolves to zero entries, preprocessing will deterministically carve out a validation split from the training entries using `utt_id`.

Recommended layout:
```text
streaming_svs_prototype/
  artifacts/
  datasets/
    data/
    checkpoints/
    runs/
    outputs/
  examples/
  *.py
```

## Logging
Training supports multiple metric backends via `logging.backends`:
- `console`
- `tensorboard`
- `wandb`

Example:
```yaml
logging:
  backends: ["console", "tensorboard", "wandb"]
  tensorboard_dir: artifacts/runs/exp_m4
  wandb_project: streaming-svs
  run_name: m4-baseline
  wandb_mode: online
```

TensorBoard scalar tags include train step loss, epoch averages, validation metrics, and learning rate.

## Real Dataset Examples
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

SoulX-Singer-Eval mixed into training with an automatic validation split:
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

For the auto-split case, `valid_manifest` can be an empty JSON list file:
```json
[]
```

## Pipeline
1. Build train/valid manifest JSON
2. Preprocess to EnCodec latent chunk cache
3. Train
4. Inference + diagnostics

## Commands
```bash
python preprocess_encodec.py --config config.yaml --split both
python train.py --config config.yaml
python infer.py --checkpoint artifacts/checkpoints/best.pt --cache artifacts/data/valid_chunks.pt --index 0 --decode_wav
```

## Notes
- This is a research baseline, not production SVS.
- EnCodec is frozen by design.
- No raw text is used; only phoneme IDs.
