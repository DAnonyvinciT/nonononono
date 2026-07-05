# emotion2vec Extension

This package adds optional emotion2vec conditioning to the modelmead
FaceFormer pipeline. It is intentionally small and script-friendly: the data
loader returns emotion features, the model fuses them with wav2vec2 audio
memory, and the training/demo scripts enable the branch with CLI flags.

## Files

- `emotion2vec.py`: local emotion2vec feature reader and cache I/O.
- `fusion_module.py`: sequence-length alignment and the cross-attention fusion block.
- `__init__.py`: exports the public extension symbols.

## Runtime Paths

The emotion2vec model directory is resolved in this order:

1. explicit CLI argument such as `--emotion_model_dir` or `--model_dir`
2. `FACEFORMER_EMOTION2VEC_PATH`
3. `<repo_root>/emotion2vec_plus_large`

The model weights are not committed to Git. Keep them outside the repository or
in an ignored local directory.

## Data Flow

`data_loader.py` returns:

```python
audio, vertice, template, one_hot, file_name, emotion_features
```

`emotion_features` is always shaped as `(batch, time, dim)` after collation.
When no emotion branch is active, the loader returns a zero placeholder so the
training and inference loops keep one stable tuple format.

Feature sources:

- `--use_emotion_cache true`: load `.npy` features from `emotion_cache/`.
- `--use_emotion_fusion true`: extract features online if cache is missing.
- neither enabled: use the zero placeholder.

## Training Flags

Common flags:

```bash
--use_emotion_fusion true
--emotion_feature_dim 1024
--emotion_model_dir "$FACEFORMER_EMOTION2VEC_PATH"
--use_emotion_cache true
--emotion_cache_dir emotion_cache
--freeze_epochs 1
```

`freeze_epochs` freezes the wav2vec2 encoder during the early fusion stage.
`emotion_consistency_weight` and `smoothness_weight` are optional losses and
default to `0.0`.

## Cache Precomputation

Use the top-level batch cache script:

```bash
python3 precompute_emotion_cache.py \
  --dataset "$FACEFORMER_MEAD_DATASET_DIR" \
  --wav_path wav \
  --emotion_cache_dir emotion_cache \
  --model_dir "$FACEFORMER_EMOTION2VEC_PATH" \
  --jobs 1
```

The cache is saved as `(time, dim)` `.npy` files. The loader expands it back to
`(1, time, dim)` before PyTorch collation.

## Smoke Test

After preprocessing vertices, a minimal run can be launched from `modelmead/`:

```bash
bash train_emotion2vec.sh \
  --max_epoch 1 \
  --num_workers 0 \
  --pin_memory false \
  --persistent_workers false \
  --debug_max_train_batches 1 \
  --debug_max_valid_batches 1 \
  --debug_max_test_batches 1
```

For full experiments, precompute emotion cache first or expect online
emotion2vec extraction to be slow.
