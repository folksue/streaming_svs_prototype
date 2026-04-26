---
name: legacy-checkpoint-qc
description: Use when working on legacy streaming codec baseline checkpoint listening tests, especially to compare many checkpoints on fixed train/valid samples, extend an existing comparison table without re-running old checkpoints, and regenerate comparison.md/comparison.html with playable audio.
---

# Legacy Checkpoint QC

Use this skill for checkpoint listening tests under `legacy/streaming_codec_baseline`.

## Workflow

1. Prefer the bundled script `scripts/extend_legacy_checkpoint_compare.py` when a comparison page already exists and only missing checkpoints need to be added.
2. Use the repo script `legacy/streaming_codec_baseline/scripts/sample_checkpoint_compare.py` only for brand-new comparison pages.
3. Keep sample indices stable once a page exists. Extend the existing table instead of changing the sampled rows.
4. Generate both `comparison.md` and `comparison.html`. The HTML page is the primary listening surface.

## Existing layout

- Checkpoints usually live under:
  `legacy/streaming_codec_baseline/artifacts/checkpoints/<run_name>`
- Comparison pages usually live under:
  `legacy/streaming_codec_baseline/artifacts/outputs/<run_name>/<comparison_dir>`

## Extend Existing Page

Run the bundled script from repo root:

```bash
python .codex/skills/legacy-checkpoint-qc/scripts/extend_legacy_checkpoint_compare.py \
  --config legacy/streaming_codec_baseline/mixed_mid_27m.yaml \
  --split train \
  --comparison-dir legacy/streaming_codec_baseline/artifacts/outputs/mixed_mid_27m/checkpoint_compare_train5_random
```

Then do the same for `--split valid`.

## Notes

- The script auto-discovers `best.pt`, `last.pt`, and `epoch_*.pt` from the checkpoint directory in the config.
- Existing columns are preserved as-is.
- Missing columns are inferred and appended in checkpoint order.
