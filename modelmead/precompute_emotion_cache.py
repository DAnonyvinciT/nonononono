#!/usr/bin/env python3
"""批量预计算 emotion2vec 帧级特征并保存为 .npy 缓存。"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from tqdm import tqdm
from data_paths import (
    DEFAULT_MEAD_DATASET_DIR,
    resolve_data_path,
    resolve_dataset_dir,
    resolve_work_path,
)

try:
    from emo_extension import DEFAULT_EMOTION2VEC_DIR, Emotion2VecFeatureReader
except Exception:
    DEFAULT_EMOTION2VEC_DIR = None
    Emotion2VecFeatureReader = None


_WORKER_READER = None


def resolve_dataset_base(dataset_arg):
    return resolve_dataset_dir(dataset_arg)


def parse_bool(value):
    v = str(value).strip().lower()
    return v in ("1", "true", "t", "yes", "y", "on")


def collect_wav_files(audio_root):
    wav_files = []
    for root, _, files in os.walk(audio_root):
        for file_name in files:
            if file_name.endswith(".wav"):
                wav_files.append(os.path.join(root, file_name))
    wav_files.sort()
    return wav_files


def build_cache_path(wav_path, audio_root, cache_root):
    rel_path = os.path.relpath(wav_path, audio_root)
    cache_rel = os.path.splitext(rel_path)[0] + ".npy"
    return os.path.join(cache_root, cache_rel)


def get_worker_reader(model_dir):
    global _WORKER_READER
    if _WORKER_READER is None:
        if Emotion2VecFeatureReader is None:
            raise RuntimeError("无法导入 Emotion2VecFeatureReader。")
        _WORKER_READER = Emotion2VecFeatureReader(model_dir=model_dir)
    return _WORKER_READER


def precompute_one(task):
    wav_path, cache_path, model_dir, overwrite = task
    if os.path.exists(cache_path) and (not overwrite):
        return "skipped", wav_path, None

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    reader = get_worker_reader(model_dir)
    reader.save_cache(wav_path, cache_path)
    return "processed", wav_path, None


def chunked(items, chunk_size):
    for index in range(0, len(items), chunk_size):
        yield items[index:index + chunk_size]


def main():
    parser = argparse.ArgumentParser(description="离线预计算 emotion2vec 帧级特征缓存")
    parser.add_argument("--dataset", type=str, default=DEFAULT_MEAD_DATASET_DIR, help="数据集根目录，默认 ~/datasets/mead_clean")
    parser.add_argument("--wav_path", type=str, default="wav", help="音频目录，基于数据集根目录")
    parser.add_argument("--emotion_cache_dir", type=str, default="emotion_cache", help="情感特征缓存输出目录（相对于 modelmead）")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_EMOTION2VEC_DIR, help="emotion2vec 模型目录")
    parser.add_argument("--overwrite", type=parse_bool, default=False, help="是否覆盖已存在的缓存")
    parser.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)), help="并行进程数，建议在 CPU 预计算时提高；GPU 场景建议设为 1")
    parser.add_argument("--chunk_size", type=int, default=8, help="每个并行任务包含的 wav 数量，适当增大可减少进程调度开销")
    args = parser.parse_args()

    dataset_base = resolve_dataset_base(args.dataset)
    audio_root = resolve_data_path(args.wav_path, dataset_base)
    cache_root = resolve_work_path(args.emotion_cache_dir)

    if not os.path.isdir(audio_root):
        raise FileNotFoundError(f"音频目录不存在: {audio_root}")

    wav_files = collect_wav_files(audio_root)
    if not wav_files:
        raise RuntimeError(f"未在目录中找到 wav 文件: {audio_root}")

    os.makedirs(cache_root, exist_ok=True)

    tasks = []
    for wav_path in wav_files:
        cache_path = build_cache_path(wav_path, audio_root, cache_root)
        if os.path.exists(cache_path) and (not args.overwrite):
            continue
        tasks.append((wav_path, cache_path, args.model_dir, args.overwrite))

    processed = 0
    skipped = len(wav_files) - len(tasks)
    errors = 0

    if not tasks:
        print(f"情感缓存已全部存在，无需生成: cache_dir={Path(cache_root).resolve()}")
        return

    if args.jobs <= 1:
        reader = Emotion2VecFeatureReader(model_dir=args.model_dir)
        for wav_path, cache_path, _, overwrite in tqdm(tasks, desc="PrecomputeEmotionCache"):
            if os.path.exists(cache_path) and (not overwrite):
                skipped += 1
                continue
            try:
                # save_cache 保证保存为 (time, dim) 磁盘格式
                reader.save_cache(wav_path, cache_path)
                processed += 1
            except Exception as error:
                errors += 1
                tqdm.write(f"[ERROR] 处理失败: {wav_path} -> {error}")
    else:
        ctx = get_context("spawn")
        chunks = list(chunked(tasks, max(1, args.chunk_size)))
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as executor:
            futures = [executor.submit(_run_chunk, chunk) for chunk in chunks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="PrecomputeEmotionCache"):
                chunk_processed, chunk_skipped, chunk_errors = future.result()
                processed += chunk_processed
                skipped += chunk_skipped
                errors += chunk_errors

    print(
        f"情感缓存完成: processed={processed}, skipped={skipped}, errors={errors}, total={len(wav_files)}, cache_dir={Path(cache_root).resolve()}, jobs={args.jobs}"
    )


def _run_chunk(chunk):
    chunk_processed = 0
    chunk_skipped = 0
    chunk_errors = 0
    for wav_path, cache_path, model_dir, overwrite in chunk:
        if os.path.exists(cache_path) and (not overwrite):
            chunk_skipped += 1
            continue
        try:
            reader = get_worker_reader(model_dir)
            reader.save_cache(wav_path, cache_path)
            chunk_processed += 1
        except Exception:
            chunk_errors += 1
    return chunk_processed, chunk_skipped, chunk_errors


if __name__ == "__main__":
    main()
