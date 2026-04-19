# TODO: Step-Level 8-Codebook Training Alignment

- [ ] Force legacy training path for this line of work:
  use `legacy/streaming_codec_baseline/{preprocess_encodec.py,train.py,infer.py}` instead of top-level chunk pipeline.
- [ ] Add explicit `audio.target_bandwidth` to legacy preprocessing config and pass it into EnCodec encode call.
- [ ] Rebuild legacy train/valid caches and verify `meta.num_codebooks == 8`.
- [ ] Keep step-level data unit with `tokens_per_step` as the only temporal grouping hyperparameter.
- [ ] Ensure model head layout follows handoff design:
  `codebook0` autoregressive + residual parallel heads for `codebook1..7`.
- [ ] Run mixed training with official OpenCPOP split (`train.txt`/`test.txt`) and official M4Singer split manifests.
- [ ] Validate training startup logs print `num_codebooks=8` before long run.
