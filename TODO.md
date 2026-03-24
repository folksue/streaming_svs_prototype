# Streaming SVS Prototype TODO

Last updated: 2026-03-24

## A. Original Requirements Breakdown

- [x] Use **chunk-wise generation** (streaming)
- [x] Use **symbolic singing conditions** (phoneme/pitch/timing)
- [x] Use **pretrained frozen EnCodec** discrete codes (not mel)
- [x] Use **next-chunk discrete codec prediction**
- [x] Use **streaming context from previous chunk**
- [x] Use **phoneme IDs** as input (no raw text)
- [x] Keep system **modular, debuggable, single-GPU runnable**

## B. Implemented Modules

### 1) Data & Preprocess
- [x] `preprocess_encodec.py`
  - [x] Load JSON manifest
  - [x] Load waveform
  - [x] Encode with frozen EnCodec
  - [x] Group codec frames into ~100ms chunks
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
  - [x] previous-chunk token embedding encoder
  - [x] chunk-local residual classifier head
  - [x] output discrete codec logits with exact chunk shape

### 3) Training
- [x] `train.py`
  - [x] Teacher forcing (ground-truth previous chunk)
  - [x] Discrete codec cross-entropy loss
  - [x] Variable-length masking
  - [x] AdamW + warmup cosine LR
  - [x] Gradient clipping
  - [x] AMP mixed precision option
  - [x] Validation logging
  - [x] Checkpoint save (`last.pt` / `best.pt`)

### 4) Inference & Eval
- [x] `infer.py`
  - [x] Autoregressive chunk generation (use generated prev chunk)
  - [x] Token accuracy metric
  - [x] Note-boundary token diagnostics (on/off)
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

- [x] Add dedicated adapters for real M4Singer metadata
- [x] Add dedicated adapters for real Opencpop metadata
- [x] Add richer logging backend (TensorBoard/W&B)
- [x] Replace flow-style latent regression with discrete next-chunk codec prediction using only the previous chunk for boundary stabilization
- [ ] Add ablation runner (conditioning variants)
- [ ] Add pitch-aligned visualization script

## E. Current Training Direction

- Use discrete EnCodec codes as targets instead of regressing normalized code ids
- Predict the next chunk from current phoneme/MIDI/timing conditions plus only the previous chunk
- Use previous-chunk token embeddings as a local continuity signal rather than a GRU over all past chunks
- Train with masked cross-entropy and token accuracy metrics

## F. Quick Run Order

1. `python preprocess_encodec.py --config config.yaml --split both`
2. `python train.py --config config.yaml`
3. `python infer.py --checkpoint artifacts/checkpoints/best.pt --cache artifacts/data/valid_chunks.pt --index 0 --decode_wav`
