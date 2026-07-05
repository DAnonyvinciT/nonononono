#!/usr/bin/env python3
"""
预处理 mead3d 数据，将顶点序列展平为模型可直接读取的格式。

输出默认写入：
- vertices_npy_flat/
"""

import argparse
import os

import numpy as np
from tqdm import tqdm
from data_paths import (
    DEFAULT_MEAD_DATASET_DIR,
    MODELMEAD_ROOT,
    resolve_data_path,
    resolve_dataset_dir,
)


def parse_bool(value):
    """解析命令行布尔参数。"""
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"不支持的布尔值: {value}")


def resolve_dataset_base(dataset_arg):
    """解析数据集根目录。"""
    return resolve_dataset_dir(dataset_arg)


def flatten_motion_array(array, source_name):
    """将 (T, V, 3) 或 (V, 3) 之类的数组展平为最后一维 3*V。"""
    array = np.asarray(array)

    if array.ndim == 1:
        return array

    if array.shape[-1] != 3:
        raise ValueError(f"{source_name} 不是末维为 3 的顶点/模板数组，实际形状: {array.shape}")

    return array.reshape(*array.shape[:-2], -1)


def flatten_vertices_tree(input_root, output_root, overwrite=False):
    """遍历顶点目录并保存展平后的 .npy 文件。"""
    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"输入顶点目录不存在: {input_root}")

    os.makedirs(output_root, exist_ok=True)

    npy_files = []
    for root, _, files in os.walk(input_root):
        for file_name in files:
            if file_name.endswith(".npy"):
                npy_files.append(os.path.join(root, file_name))

    npy_files.sort()
    if not npy_files:
        raise RuntimeError(f"未在目录中找到 .npy 文件: {input_root}")

    processed = 0
    skipped = 0

    for input_path in tqdm(npy_files, desc="FlattenVertices"):
        rel_path = os.path.relpath(input_path, input_root)
        output_path = os.path.join(output_root, rel_path)
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(output_path) and (not overwrite):
            skipped += 1
            continue

        vertices = np.load(input_path, allow_pickle=False)
        flattened_vertices = flatten_motion_array(vertices, input_path)
        np.save(output_path, flattened_vertices)
        processed += 1

    return processed, skipped, len(npy_files)


def main():
    parser = argparse.ArgumentParser(description="预处理 mead3d 顶点，生成扁平化数据文件")
    parser.add_argument("--dataset", type=str, default=DEFAULT_MEAD_DATASET_DIR, help="数据集根目录，默认 ~/datasets/mead_clean")
    parser.add_argument("--vertices_path", type=str, default="vertex", help="原始顶点目录，基于数据集根目录")
    parser.add_argument(
        "--output_vertices_path",
        type=str,
        default=os.path.join(MODELMEAD_ROOT, "vertices_npy_flat"),
        help="输出顶点目录，默认写入 modelmead/vertices_npy_flat",
    )
    parser.add_argument("--overwrite", type=parse_bool, default=False, help="是否覆盖已存在的预处理结果")
    args = parser.parse_args()

    dataset_base = resolve_dataset_base(args.dataset)
    input_vertices_root = resolve_data_path(args.vertices_path, dataset_base)
    output_vertices_root = os.path.abspath(
        os.path.expanduser(args.output_vertices_path)
    )

    processed_vertices, skipped_vertices, total_vertices = flatten_vertices_tree(
        input_vertices_root,
        output_vertices_root,
        overwrite=args.overwrite,
    )

    print(
        f"预处理完成: vertices processed={processed_vertices}, skipped={skipped_vertices}, total={total_vertices}"
    )
    print(f"输出顶点目录: {os.path.abspath(output_vertices_root)}")


if __name__ == "__main__":
    main()
