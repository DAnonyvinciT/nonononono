#!/usr/bin/env bash
set -euo pipefail

# emotion2vec 版本全量训练脚本（modelmead）
# 用法：
#   bash train_emotion2vec.sh
#   bash train_emotion2vec.sh --max_epoch 50 --save_path save_emotion2vec_v2
# 说明：
#   1) 默认开启 emotion2vec 融合，并优先使用 emotion cache；缓存缺失时会自动回退到在线提特征。
#   2) 数据由 data_loader.py 按说话人列表划分，并保留全部情绪/强度样本。
#   3) 其余参数可通过命令行追加覆盖。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
WAV2VEC2_MODEL_PATH="${FACEFORMER_WAV2VEC2_PATH:-${WAV2VEC2_MODEL_PATH:-$PROJECT_ROOT/wav2vec2-large-xlsr-53}}"
EMOTION2VEC_MODEL_PATH="${FACEFORMER_EMOTION2VEC_PATH:-${EMOTION2VEC_MODEL_PATH:-$PROJECT_ROOT/emotion2vec_plus_large}}"
DATASET_DIR="${FACEFORMER_MEAD_DATASET_DIR:-$HOME/datasets/mead_clean}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d%H%M%S)}"

# 是否跳过离线音频缓存预处理：0 表示执行，1 表示跳过
SKIP_AUDIO_CACHE_PRECOMPUTE="${SKIP_AUDIO_CACHE_PRECOMPUTE:-1}"
if [[ "$SKIP_AUDIO_CACHE_PRECOMPUTE" != "1" ]]; then
    "$PYTHON_BIN" precompute_audio_cache.py \
        --dataset "$DATASET_DIR" \
        --wav_path wav \
        --audio_cache_dir audio_cache \
        --processor_path "$WAV2VEC2_MODEL_PATH" \
        --cache_dtype float32 \
        --overwrite false
fi

# 是否跳过顶点与模板展平预处理：0 表示执行，1 表示跳过
SKIP_MOTION_PREPROCESS="${SKIP_MOTION_PREPROCESS:-0}"
if [[ "$SKIP_MOTION_PREPROCESS" != "1" ]]; then
    "$PYTHON_BIN" preprocess_mead3d_flatten.py \
        --dataset "$DATASET_DIR" \
        --vertices_path vertex \
        --output_vertices_path vertices_npy_flat \
        --overwrite false
fi

# 是否跳过 emotion2vec 缓存预计算：0 表示执行，1 表示跳过
SKIP_EMOTION_CACHE_PRECOMPUTE="${SKIP_EMOTION_CACHE_PRECOMPUTE:-1}"
EMOTION_CACHE_JOBS="${EMOTION_CACHE_JOBS:-1}"
EMOTION_CACHE_CHUNK_SIZE="${EMOTION_CACHE_CHUNK_SIZE:-8}"
if [[ "$SKIP_EMOTION_CACHE_PRECOMPUTE" != "1" ]]; then
    "$PYTHON_BIN" precompute_emotion_cache.py \
        --dataset "$DATASET_DIR" \
        --wav_path wav \
        --emotion_cache_dir emotion_cache \
        --model_dir "$EMOTION2VEC_MODEL_PATH" \
        --jobs "$EMOTION_CACHE_JOBS" \
        --chunk_size "$EMOTION_CACHE_CHUNK_SIZE" \
        --overwrite false
fi

"$PYTHON_BIN" main.py \
    --dataset modelmead \
    --dataset_dir "$DATASET_DIR" \
    --vertice_dim 15069 \
    --feature_dim 128 \
    --period 25 \
    --train_subjects "M003 M005 M007 M009 M011 M012 M013 M019 M022 M023 M024 M025 M026 M027 M028 M029 M030 M031 W009 W011 W014 W015 W016 W018 W019 W023 W024 W025 W026 W028 W029" \
    --val_subjects "M032 M033 M034 M035 W033 W035 W036" \
    --test_subjects "M037 M039 M040 M041 M042 W037 W038 W040" \
    --wav_path wav \
    --vertices_path "$SCRIPT_DIR/vertices_npy_flat" \
    --gradient_accumulation_steps 1 \
    --max_epoch 100 \
    --template_file templates.pkl \
    --teacher_forcing false \
    --num_workers 4 \
    --pin_memory true \
    --persistent_workers true \
    --prefetch_factor 2 \
    --use_audio_cache true \
    --audio_cache_dir audio_cache \
    --wav2vec2_model_path "$WAV2VEC2_MODEL_PATH" \
    --use_emotion_fusion true \
    --emotion_feature_dim 1024 \
    --emotion_model_dir "$EMOTION2VEC_MODEL_PATH" \
    --use_emotion_cache true \
    --emotion_cache_dir emotion_cache \
    --freeze_epochs 1 \
    --save_path "save_emotion2vec/${RUN_TIMESTAMP}" \
    --result_path "result_emotion2vec/${RUN_TIMESTAMP}" \
    "$@"
