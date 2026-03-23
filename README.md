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
- `dataset.py`: variable-length sequence dataset + padding/mask collate
- `model.py`: condition encoder + GRU context + flow-style generator
- `train.py`: teacher-forcing training loop + validation + checkpoint
- `infer.py`: autoregressive generation + diagnostics + optional wav decode
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

## Pipeline
1. Build train/valid manifest JSON
2. Preprocess to EnCodec latent chunk cache
3. Train
4. Inference + diagnostics

## Commands
```bash
python preprocess_encodec.py --config config.yaml --split both
python train.py --config config.yaml
python infer.py --checkpoint checkpoints/best.pt --cache data/valid_chunks.pt --index 0 --decode_wav
```

## Notes
- This is a research baseline, not production SVS.
- EnCodec is frozen by design.
- No raw text is used; only phoneme IDs.
