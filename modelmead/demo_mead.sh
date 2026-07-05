#!/usr/bin/env bash
set -euo pipefail

# modelmead 推理脚本（固定参数）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
WAV2VEC2_MODEL_PATH="${FACEFORMER_WAV2VEC2_PATH:-${WAV2VEC2_MODEL_PATH:-$PROJECT_ROOT/wav2vec2-large-xlsr-53}}"
EMOTION2VEC_MODEL_PATH="${FACEFORMER_EMOTION2VEC_PATH:-${EMOTION2VEC_MODEL_PATH:-$PROJECT_ROOT/emotion2vec_plus_large}}"
DATASET_DIR="${FACEFORMER_MEAD_DATASET_DIR:-$HOME/datasets/mead_clean}"
MODEL_PATH="${MODEL_PATH:-$SCRIPT_DIR/save_full/30_model.pth}"
DEMO_WAV_PATH="${DEMO_WAV_PATH:-$DATASET_DIR/wav/M003_021_0_0.wav}"
RESULT_PATH="${RESULT_PATH:-demo/result}"
OUTPUT_PATH="${OUTPUT_PATH:-demo/output}"

"$PYTHON_BIN" demo.py \
  --dataset modelmead \
  --dataset_dir "$DATASET_DIR" \
  --model_path "$MODEL_PATH" \
  --vertice_dim 15069 \
  --feature_dim 128 \
  --period 25 \
  --wav2vec2_model_path "$WAV2VEC2_MODEL_PATH" \
  --emotion_model_dir "$EMOTION2VEC_MODEL_PATH" \
  --train_subjects "M003 M005 M007 M009 M011 M012 M013 M019 M022 M023 M024 M025 M026 M027 M028 M029 M030 M031 W009 W011 W014 W015 W016 W018 W019 W021 W023 W024 W025 W026 W028 W029" \
  --test_subjects "M037 M039 M040 M041 M042 W037 W038 W040" \
  --wav_path "$DEMO_WAV_PATH" \
  --result_path "$RESULT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --condition M003 \
  --subject M003 \
  "$@"
