# FaceFormer on MEAD3D

This directory contains the script-based experimental PyTorch code used to train
and evaluate FaceFormer-style speech-driven 3D face animation on MEAD3D. The
code keeps the research workflow direct: preprocess data, launch a training
script, and run demo inference.

## What Is Tracked

Tracked files are source code, small shell entry points, and documentation.
Generated data is intentionally not tracked:

- `audio_cache/`, `emotion_cache/`, `vertices_npy_flat/`
- `logs/`, `result*/`, `save*/`, `demo/`
- `*.pth` checkpoints and local model weights
- local symlinks such as `templates.pkl`, `wav`, and `vertices_npy`

## Expected Data Layout

Set `FACEFORMER_MEAD_DATASET_DIR` to the root of the processed MEAD3D data:

```bash
export FACEFORMER_MEAD_DATASET_DIR=$HOME/datasets/mead_clean
```

The directory is expected to contain:

```text
mead_clean/
  wav/
    M003_021_0_0.wav
    ...
  vertex/
    M003_021_0_0.npy
    ...
  templates.pkl
  flame_sample.ply
```

`templates.pkl` stores personalized subject templates. `flame_sample.ply` is
used by `demo.py` for rendering predicted vertices.

## External Model Weights

Pretrained audio and emotion encoders are not committed to Git. Point the code
to local copies with environment variables:

```bash
export FACEFORMER_WAV2VEC2_PATH=/path/to/wav2vec2-large-xlsr-53
export FACEFORMER_EMOTION2VEC_PATH=/path/to/emotion2vec_plus_large
```

If unset, the scripts fall back to sibling directories at the repository root:
`wav2vec2-large-xlsr-53/` and `emotion2vec_plus_large/`.

## Preprocess

Flatten MEAD vertex sequences from `(T, V, 3)` to `(T, V*3)`:

```bash
cd modelmead
python3 preprocess_mead3d_flatten.py \
  --dataset "$FACEFORMER_MEAD_DATASET_DIR" \
  --vertices_path vertex \
  --output_vertices_path vertices_npy_flat
```

Optional wav2vec2 audio cache:

```bash
python3 precompute_audio_cache.py \
  --dataset "$FACEFORMER_MEAD_DATASET_DIR" \
  --wav_path wav \
  --audio_cache_dir audio_cache \
  --processor_path "$FACEFORMER_WAV2VEC2_PATH"
```

Optional emotion2vec cache:

```bash
python3 precompute_emotion_cache.py \
  --dataset "$FACEFORMER_MEAD_DATASET_DIR" \
  --wav_path wav \
  --emotion_cache_dir emotion_cache \
  --model_dir "$FACEFORMER_EMOTION2VEC_PATH" \
  --jobs 1
```

## Train

Basic FaceFormer training:

```bash
cd modelmead
bash train_full.sh
```

Training with emotion2vec fusion:

```bash
cd modelmead
bash train_emotion2vec.sh
```

Both scripts accept extra `main.py` arguments after the script name. For a quick
smoke test:

```bash
bash train_full.sh \
  --max_epoch 1 \
  --num_workers 0 \
  --pin_memory false \
  --persistent_workers false \
  --debug_max_train_batches 1 \
  --debug_max_valid_batches 1 \
  --debug_max_test_batches 1
```

The scripts use these environment variables when present:

- `PYTHON_BIN`: Python executable, default `python3`
- `FACEFORMER_MEAD_DATASET_DIR`: MEAD data root
- `FACEFORMER_WAV2VEC2_PATH`: wav2vec2 checkpoint directory
- `FACEFORMER_EMOTION2VEC_PATH`: emotion2vec checkpoint directory
- `SKIP_MOTION_PREPROCESS`: set to `1` to skip vertex flattening
- `SKIP_AUDIO_CACHE_PRECOMPUTE`: set to `0` to precompute audio cache
- `SKIP_EMOTION_CACHE_PRECOMPUTE`: set to `0` to precompute emotion cache

## Demo

Run inference and render a video from a trained checkpoint:

```bash
cd modelmead
MODEL_PATH=save_full/30_model.pth \
DEMO_WAV_PATH="$FACEFORMER_MEAD_DATASET_DIR/wav/M003_021_0_0.wav" \
bash demo_mead.sh
```

For direct control, call `demo.py` and pass `--model_path`, `--wav_path`,
`--condition`, and output directories explicitly.
