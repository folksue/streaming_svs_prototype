# Docker Usage

This baseline now includes a GPU-oriented Docker image definition:

- `Dockerfile.gpu`
- `compose.gpu.yaml`

## What Gets Packaged

This image copies **the legacy baseline source code** into the image:

- `legacy/streaming_codec_baseline/*`

It does **not** bake your whole repository, datasets, caches, or checkpoints into the image unless you explicitly change the Dockerfile to copy them.

In other words:

- source code: packaged
- config files inside this legacy baseline: packaged
- datasets / wavs / manifests / generated caches / checkpoints: expected to be mounted from host

## Build

From the repo root:

```bash
docker build -f legacy/streaming_codec_baseline/Dockerfile.gpu -t legacy-svs:gpu .
```

## Run Interactive Shell

The baseline expects paths like `artifacts/...` relative to its own working directory, so the simplest approach is to mount the legacy folder back onto the same path in the container.

```bash
docker run --rm -it \
  --gpus all \
  -v /mnt/c/streaming_svs_prototype/legacy/streaming_codec_baseline:/workspace/streaming_svs_prototype/legacy/streaming_codec_baseline \
  -w /workspace/streaming_svs_prototype/legacy/streaming_codec_baseline \
  legacy-svs:gpu
```

## Run With Compose

From the repo root:

```bash
docker compose -f legacy/streaming_codec_baseline/compose.gpu.yaml run --rm legacy-gpu
```

If your Docker installation still uses the old standalone command:

```bash
docker-compose -f legacy/streaming_codec_baseline/compose.gpu.yaml run --rm legacy-gpu
```

## Typical Commands

Preprocess:

```bash
python preprocess_encodec.py --config config.yaml --split both
```

Train:

```bash
python train.py --config config.yaml
```

Infer:

```bash
python infer.py \
  --checkpoint artifacts/checkpoints/best.pt \
  --cache artifacts/data/valid_chunks.pt \
  --index 0 \
  --temperature 0.0 \
  --decode_wav \
  --out_dir artifacts/outputs
```

## Why Mount Instead Of Copy For Data

Large wavs, generated caches and checkpoints should stay outside the image:

- faster rebuilds
- smaller image size
- easier iteration on manifests and outputs

If you really want a fully self-contained image, you can change the Dockerfile to `COPY` dataset and artifact directories too, but that is usually the wrong default for research iteration.
