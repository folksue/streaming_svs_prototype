# Legacy Model Architecture Note

This note summarizes the current `legacy/streaming_codec_baseline` architecture after removing the outer chunk lattice.

## One-line Summary

The current legacy model is:

- **step-autoregressive with local/sliding causal attention**
- **one explicit control state per step**
- **codebook-0 autoregressive inside each step**
- **codebooks `1..K-1` predicted in parallel as residual heads**

`tokens_per_step` is now only a grouping hyperparameter for how many consecutive EnCodec frame-groups belong to one model step.

## Data Shape

Training and inference now operate on:

- controls:
  - `note_id [B, T]`
  - `phoneme_id [B, T]`
  - `slur [B, T]`
  - `phone_progress [B, T]`
- audio codes:
  - `codes [B, T, S, K]`

Where:

- `T` = number of model steps
- `S` = `tokens_per_step`
- `K` = number of EnCodec codebooks

Each step has exactly one control state and `S` grouped codec frames.

## Outer Autoregression

The outer sequence model is step-based:

1. discrete control is embedded into `cond [B, T, cond_dim]`
2. previous-step main-stream summary is shift-righted
3. local causal self-attention over the step sequence produces `step_h [B, T, H]`

So the outer recurrence is:

```text
previous step summaries + current step control -> current step context
```

The attention mask is directly local-causal. Training and inference use the same visibility rule; there is no separate "simulate truncation" path.

## Within-step Decode

Inside each step:

1. only codebook 0 is teacher-forced / autoregressive
2. the model decodes the `S` frame-groups sequentially for codebook 0
3. once the frame hidden state is available, codebooks `1..K-1` are predicted in parallel

So the inner recurrence is:

```text
previous codebook-0 tokens in current step -> next frame hidden
next frame hidden -> codebook 0
next frame hidden + codebook-0 embedding -> codebooks 1..K-1
```

## Local Attention Meaning

There are two local windows:

- `history_window`
  - how many previous steps the current step may attend to
- `audio_history_window`
  - how many previous frame-groups inside the current step codebook-0 stream may be seen

Both are causal and both are active during training.

## Multi-codebook Strategy

The current model does **not** flatten all codebooks into one long decoder-only stream.

Instead it uses:

- one autoregressive main head for codebook 0
- one residual-conditioning MLP
- one parallel head per remaining codebook

The effective order is:

```text
step t, frame 0: predict k0 -> predict k1..kK-1
step t, frame 1: predict k0 -> predict k1..kK-1
...
```

So this is best described as:

- **step-level local causal AR**
- **main-stream decoder-only on codebook 0**
- **parallel residual decode for finer codebooks**
