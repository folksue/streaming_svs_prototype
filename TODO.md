# Streaming SVS Prototype TODO

Last updated: 2026-03-23

## A. Original Requirements Breakdown

- [x] Use **chunk-wise generation** (streaming)
- [x] Use **symbolic singing conditions** (phoneme/pitch/timing)
- [x] Use **pretrained frozen EnCodec** latents (not mel)
- [x] Use **flow-style latent generator**
- [x] Use **streaming context from previous chunk**
- [x] Use **phoneme IDs** as input (no raw text)
- [x] Keep system **modular, debuggable, single-GPU runnable**

## B. Implemented Modules

### 1) Data & Preprocess
- [x] `preprocess_encodec.py`
  - [x] Load JSON manifest
  - [x] Load waveform
  - [x] Encode with frozen EnCodec
  - [x] Group latent frames into ~100ms chunks
  - [x] Build aligned chunk conditions from note/phoneme intervals
  - [x] Save sequence cache to `.pt`

- [x] `dataset.py`
  - [x] Load sequence cache
  - [x] Variable-length sequence support
  - [x] Padding + mask collate

### 2) Model
- [x] `model.py`
  - [x] Condition encoder
    - [x] phoneme embedding
    - [x] pitch projection
    - [x] timing/progress projection
  - [x] 2-layer GRU streaming context
  - [x] lightweight flow-matching style generator (residual MLP stack)
  - [x] output latent chunk with exact chunk shape

### 3) Training
- [x] `train.py`
  - [x] Teacher forcing (ground-truth previous chunk)
  - [x] Latent-space L1 loss
  - [x] Variable-length masking
  - [x] AdamW + warmup cosine LR
  - [x] Gradient clipping
  - [x] AMP mixed precision option
  - [x] Validation logging
  - [x] Checkpoint save (`last.pt` / `best.pt`)

### 4) Inference & Eval
- [x] `infer.py`
  - [x] Autoregressive chunk generation (use generated prev chunk)
  - [x] Latent L1 metric
  - [x] Chunk continuity metric
  - [x] Note-boundary diagnostics (on/off)
  - [x] Optional EnCodec decode to wav

### 5) Documentation / Config
- [x] `README.md`
- [x] `config.yaml`
- [x] `examples/manifest_example.json`
- [x] `requirements.txt`

## C. Paper Sync

- [x] Update Method section to EnCodec latent + flow-style generation
- [x] Update Experiments section to match implemented setup
- [x] Recompile paper PDF

## D. Pending / Optional Next Tasks

- [ ] Add dedicated adapters for real M4Singer metadata
- [ ] Add dedicated adapters for real Opencpop metadata
- [ ] Add richer logging backend (TensorBoard/W&B)
- [ ] Add ablation runner (conditioning variants)
- [ ] Add pitch-aligned visualization script

## E. Quick Run Order

1. `python preprocess_encodec.py --config config.yaml --split both`
2. `python train.py --config config.yaml`
3. `python infer.py --checkpoint checkpoints/best.pt --cache data/valid_chunks.pt --index 0 --decode_wav`
