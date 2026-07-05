import os
from functools import lru_cache
import logging
from contextlib import redirect_stderr, redirect_stdout
import io
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from .fusion_module import align_sequence_length


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_EMOTION2VEC_DIR = os.path.abspath(
    os.path.expandvars(
        os.path.expanduser(
            os.environ.get(
                "FACEFORMER_EMOTION2VEC_PATH",
                os.path.join(PROJECT_ROOT, "emotion2vec_plus_large"),
            )
        )
    )
)


def _silence_emotion2vec_logs():
    for logger_name in ("funasr", "modelscope"):
        logger = logging.getLogger(logger_name)
        logger.disabled = True


def _load_emotion2vec_model(model_dir):
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    previous_warning_filters = warnings.filters[:]
    sink = io.StringIO()
    try:
        root_logger.setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")
        with redirect_stdout(sink), redirect_stderr(sink):
            from funasr import AutoModel

            return AutoModel(model=model_dir, disable_update=True)
    finally:
        root_logger.setLevel(previous_level)
        warnings.filters[:] = previous_warning_filters


def _load_yaml_embed_dim(config_path):
    try:
        import yaml
    except Exception:
        return None

    if not os.path.exists(config_path):
        return None

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    try:
        return int(config["model_conf"]["embed_dim"])
    except Exception:
        return None


class Emotion2VecFeatureReader:
    """本地 emotion2vec 帧级特征读取器。"""

    def __init__(self, model_dir=DEFAULT_EMOTION2VEC_DIR, device=None):
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = _load_yaml_embed_dim(os.path.join(model_dir, "config.yaml")) or 1024
        self._model = None

    @property
    def model(self):
        if self._model is not None:
            return self._model

        try:
            _silence_emotion2vec_logs()
        except Exception as error:
            raise RuntimeError(
                "缺少 funasr，无法加载 emotion2vec。请先安装 funasr 后再运行情绪特征提取。"
            ) from error

        self._model = _load_emotion2vec_model(self.model_dir)
        return self._model

    def prepare_audio(self, wav_path):
        if not os.path.exists(wav_path):
            raise FileNotFoundError(wav_path)

        try:
            import soundfile as sf
        except Exception as error:
            raise RuntimeError("缺少 soundfile，无法读取 wav 音频。") from error

        wav, sample_rate = sf.read(wav_path)
        if sample_rate != 16000:
            try:
                import librosa

                wav = librosa.resample(wav.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
            except Exception as error:
                raise RuntimeError("音频不是 16kHz，且当前环境无法自动重采样。") from error

        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        return torch.from_numpy(np.asarray(wav, dtype=np.float32))

    def extract_frame_features(self, wav_path):
        """从单个 wav 文件提取 50Hz 帧级情绪特征。"""
        result = self.model.generate(wav_path, granularity="frame", extract_embedding=True)
        if not isinstance(result, list) or not result:
            raise RuntimeError("emotion2vec 返回值格式异常。")

        feats = result[0].get("feats")
        if feats is None:
            raise RuntimeError("emotion2vec 未返回 feats 字段。")

        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        elif not torch.is_tensor(feats):
            feats = torch.tensor(feats)

        feats = feats.float()
        # 统一为三维 (batch, time, dim)
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        elif feats.dim() == 1:
            feats = feats.view(1, 1, -1)

        # 强校验最后一维是否匹配预期的 feature_dim
        if feats.shape[-1] != self.feature_dim:
            # 如果 yaml 中读取的 dim 与模型返回不一致，尝试警告但不中断（以防实验环境差异）
            import warnings

            warnings.warn(f"emotion2vec 特征维度 {feats.shape[-1]} 与期望 {self.feature_dim} 不一致。")

        return feats

    def load_cache(self, cache_path):
        if not os.path.exists(cache_path):
            return None
        cache = np.load(cache_path, allow_pickle=False)
        cache = torch.from_numpy(cache).float()
        if cache.dim() == 2:
            cache = cache.unsqueeze(0)
        return cache

    def save_cache(self, wav_path, cache_path, target_length=None):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        feats = self.extract_frame_features(wav_path)
        if target_length is not None:
            feats = align_sequence_length(feats, target_length)
        np.save(cache_path, feats.squeeze(0).detach().cpu().numpy())
        return feats
