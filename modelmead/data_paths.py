"""modelmead 数据与实验输出路径的统一解析。"""

import os


MODELMEAD_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASETS_ROOT = os.path.expandvars(
    os.path.expanduser(os.environ.get("FACEFORMER_DATASETS_ROOT", "~/datasets"))
)
DEFAULT_MEAD_DATASET_DIR = os.path.abspath(
    os.path.expanduser(
        os.environ.get(
            "FACEFORMER_MEAD_DATASET_DIR",
            os.path.join(DEFAULT_DATASETS_ROOT, "mead_clean"),
        )
    )
)


def resolve_dataset_dir(dataset_dir=None):
    """返回 MEAD 数据集根目录，支持环境变量、`~` 和绝对路径。"""
    path = dataset_dir or DEFAULT_MEAD_DATASET_DIR
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def resolve_data_path(path, dataset_dir=None):
    """解析数据文件路径；相对路径以数据集根目录为基准。"""
    path = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.join(resolve_dataset_dir(dataset_dir), path)


def resolve_work_path(path):
    """解析缓存、日志、权重和预测结果路径；相对路径以 modelmead 为基准。"""
    path = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.join(MODELMEAD_ROOT, path)
