import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
from transformers import Wav2Vec2FeatureExtractor
import librosa
from model_paths import resolve_wav2vec2_model_path
from data_paths import resolve_data_path, resolve_work_path

try:
    from emo_extension import Emotion2VecFeatureReader
except Exception:
    Emotion2VecFeatureReader = None

class Dataset(data.Dataset):
    """支持流式加载的数据集。保持简单、可验证的脚本化行为。

    Emotion 特征统一为三维张量 `(batch, time, dim)`。
    """
    def __init__(self, data, subjects_dict, data_type="train", dataset_name="vocaset", emotion_feature_dim=1024, wav2vec2_model_path=None):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.dataset_name = dataset_name
        self.use_audio_cache = False
        self.use_emotion_cache = False
        self.use_emotion_fusion = False
        self.emotion_cache_dir = "audio_cache"
        self.emotion_reader = None
        self.emotion_model_dir = None
        # emotion feature last-dimension (默认与 emotion2vec large 一致)
        self.emotion_feature_dim = emotion_feature_dim
        self.wav2vec2_model_path = resolve_wav2vec2_model_path(wav2vec2_model_path)
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        # 延迟初始化 processor，降低多进程加载时的瞬时内存压力
        self._processor = None
        self._cache_miss_warned = False

    def set_audio_cache(self, use_audio_cache):
        """设置是否优先读取离线音频缓存。"""
        self.use_audio_cache = use_audio_cache

    def set_emotion_cache(self, use_emotion_cache, emotion_cache_dir="emotion_cache"):
        """设置是否优先读取离线 emotion2vec 缓存。"""
        self.use_emotion_cache = use_emotion_cache
        self.emotion_cache_dir = emotion_cache_dir

    def set_emotion_reader(self, use_emotion_fusion, emotion_model_dir=None):
        """设置是否启用在线 emotion2vec 特征提取。"""
        self.use_emotion_fusion = use_emotion_fusion
        self.emotion_model_dir = emotion_model_dir
        if not use_emotion_fusion:
            self.emotion_reader = None
            return

        self.emotion_reader = None

    def _get_emotion_reader(self):
        if self.emotion_reader is not None:
            return self.emotion_reader

        if Emotion2VecFeatureReader is None:
            raise RuntimeError("缺少 emotion2vec 读取器，无法启用情绪融合。")

        self.emotion_reader = Emotion2VecFeatureReader(model_dir=self.emotion_model_dir)
        return self.emotion_reader

    @property
    def processor(self):
        if self._processor is None:
            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(self.wav2vec2_model_path)
        return self._processor

    def __getitem__(self, index):
        """按需读取单条样本（音频 + 顶点 + 模板）。"""
        item = self.data[index]
        file_name = item["name"]

        # 优先读取离线缓存，缺失时回退到在线解码
        audio = None
        audio_cache_path = item.get("audio_cache_path")
        if self.use_audio_cache and audio_cache_path and os.path.exists(audio_cache_path):
            audio = np.load(audio_cache_path, allow_pickle=False)
        else:
            if self.use_audio_cache and (not self._cache_miss_warned):
                print("[WARN] 音频缓存缺失，已回退到在线解码。")
                self._cache_miss_warned = True
            speech_array, _ = librosa.load(item["audio_path"], sr=16000)
            audio = np.squeeze(self.processor(speech_array, sampling_rate=16000).input_values)

        vertice = np.load(item["vertice_path"], allow_pickle=True)

        template = item["template"]

        emotion_features = None
        emotion_cache_path = item.get("emotion_cache_path")
        if self.use_emotion_cache and emotion_cache_path and os.path.exists(emotion_cache_path):
            emotion_features = np.load(emotion_cache_path, allow_pickle=False)
            # 磁盘上通常保存为二维 (time, dim)，在内存里统一扩成 (1, time, dim)
            if getattr(emotion_features, "ndim", None) == 2:
                emotion_features = np.expand_dims(emotion_features, 0)
        elif self.use_emotion_fusion:
            # 在线提取保证返回形状为 (1, time, dim)
            emotion_features = self._get_emotion_reader().extract_frame_features(item["audio_path"]).cpu().numpy()

        if self.data_type == "train":
            # mead3d format: M003_002_0_0 -> M003; vocaset format: FaceTalk_xxx_Sentence01 -> FaceTalk_xxx
            subject = self._extract_subject(file_name)
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels

        output = [torch.FloatTensor(audio),
            torch.FloatTensor(vertice),
            torch.FloatTensor(template),
                torch.FloatTensor(one_hot),
                file_name]

        if emotion_features is None:
            # 保证缺失时也返回三维张量 (1, 1, dim)
            emotion_features = np.zeros((1, 1, self.emotion_feature_dim), dtype=np.float32)
        output.append(torch.FloatTensor(emotion_features))
        return tuple(output)

    def _extract_subject(self, filename):
        """从文件名中提取说话人 ID。"""
        # mead3d: M003_002_0_0.npy -> M003 (first field)
        # vocaset: FaceTalk_xxx_Sentence01.npy -> FaceTalk_xxx (all but last field)
        if self.dataset_name in ["modelmead", "."]:
            return filename.split("_")[0]
        return "_".join(filename.split("_")[:-1])

    def __len__(self):
        return self.len

def read_data(args):
    print("Loading data (streaming mode)...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    # 原始数据统一从 --dataset_dir 读取；绝对路径参数仍可覆盖默认目录。
    dataset_dir = getattr(args, "dataset_dir", None)
    audio_path = resolve_data_path(args.wav_path, dataset_dir)
    vertices_path = resolve_data_path(args.vertices_path, dataset_dir)

    # 先构建说话人划分字典
    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]
    all_subjects = set(subjects_dict["train"] + subjects_dict["val"] + subjects_dict["test"])

    template_file = resolve_data_path(args.template_file, dataset_dir)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    # 先按说话人名单做一级过滤
    splits = {'vocaset': {'train': range(1, 41), 'val': range(21, 41), 'test': range(21, 41)},
              'BIWI': {'train': range(1, 33), 'val': range(33, 37), 'test': range(37, 41)}}
    dataset_key = args.dataset if args.dataset != "." else "."

    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(sorted(fs)):
            if f.endswith("wav"):
                key = f.replace("wav", "npy")
                # mead3d: M003_002_0_0 -> M003；vocaset: FaceTalk_xxx_Sentence01 -> FaceTalk_xxx
                if args.dataset in ["modelmead", "."]:
                    subject_id = key.split("_")[0]
                    parts = os.path.splitext(f)[0].split("_")
                    if len(parts) < 4:
                        continue
                    try:
                        int(parts[1])
                        int(parts[2])
                        int(parts[3])
                    except ValueError:
                        continue
                else:
                    subject_id = "_".join(key.split("_")[:-1])
                    sentence_id = int(key.split(".")[0][-2:])

                # 不在 train/val/test 说话人名单内则跳过
                if subject_id not in all_subjects:
                    continue

                # 根据 sentence_id 判断当前样本归属 split
                if args.dataset in ["modelmead", "."]:
                    # MEAD3D 使用完整情绪数据，按说话人列表划分 train/val/test。
                    in_train = subject_id in subjects_dict["train"]
                    in_val = subject_id in subjects_dict["val"]
                    in_test = subject_id in subjects_dict["test"]
                else:
                    in_train = subject_id in subjects_dict["train"] and sentence_id in splits[dataset_key]['train']
                    in_val = subject_id in subjects_dict["val"] and sentence_id in splits[dataset_key]['val']
                    in_test = subject_id in subjects_dict["test"] and sentence_id in splits[dataset_key]['test']

                if not (in_train or in_val or in_test):
                    continue

                wav_path = os.path.join(r, f)
                vertice_path = os.path.join(vertices_path, f.replace("wav", "npy"))

                # 顶点文件缺失时跳过
                if not os.path.exists(vertice_path):
                    continue

                # 计算离线音频缓存路径
                rel_wav_path = os.path.relpath(wav_path, audio_path)
                cache_rel_path = os.path.splitext(rel_wav_path)[0] + ".npy"
                audio_cache_path = os.path.join(
                    resolve_work_path(args.audio_cache_dir), cache_rel_path
                )
                emotion_cache_path = os.path.join(
                    resolve_work_path(getattr(args, "emotion_cache_dir", "emotion_cache")),
                    cache_rel_path,
                )

                # 仅当模板存在时写入样本
                if subject_id in templates:
                    temp = templates[subject_id]
                    data[key]["name"] = f
                    data[key]["template"] = temp.reshape((-1))
                    data[key]["audio_path"] = wav_path
                    data[key]["vertice_path"] = vertice_path
                    data[key]["audio_cache_path"] = audio_cache_path
                    data[key]["emotion_cache_path"] = emotion_cache_path

                    if in_train:
                        train_data.append(data[key])
                    if in_val:
                        valid_data.append(data[key])
                    if in_test:
                        test_data.append(data[key])

    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    return train_data, valid_data, test_data, subjects_dict

def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)

    dataset_name = args.dataset if args.dataset != "." else "modelmead"

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    wav2vec2_model_path = getattr(args, "wav2vec2_model_path", None)

    train_dataset = Dataset(
        train_data,
        subjects_dict,
        "train",
        dataset_name,
        emotion_feature_dim=getattr(args, "emotion_feature_dim", 1024),
        wav2vec2_model_path=wav2vec2_model_path,
    )
    train_dataset.set_audio_cache(args.use_audio_cache)
    train_dataset.set_emotion_cache(getattr(args, "use_emotion_cache", False), getattr(args, "emotion_cache_dir", "emotion_cache"))
    train_dataset.set_emotion_reader(getattr(args, "use_emotion_fusion", False), getattr(args, "emotion_model_dir", None))
    dataset["train"] = data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, **loader_kwargs)

    valid_dataset = Dataset(
        valid_data,
        subjects_dict,
        "val",
        dataset_name,
        emotion_feature_dim=getattr(args, "emotion_feature_dim", 1024),
        wav2vec2_model_path=wav2vec2_model_path,
    )
    valid_dataset.set_audio_cache(args.use_audio_cache)
    valid_dataset.set_emotion_cache(getattr(args, "use_emotion_cache", False), getattr(args, "emotion_cache_dir", "emotion_cache"))
    valid_dataset.set_emotion_reader(getattr(args, "use_emotion_fusion", False), getattr(args, "emotion_model_dir", None))
    dataset["valid"] = data.DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, **loader_kwargs)

    test_dataset = Dataset(
        test_data,
        subjects_dict,
        "test",
        dataset_name,
        emotion_feature_dim=getattr(args, "emotion_feature_dim", 1024),
        wav2vec2_model_path=wav2vec2_model_path,
    )
    test_dataset.set_audio_cache(args.use_audio_cache)
    test_dataset.set_emotion_cache(getattr(args, "use_emotion_cache", False), getattr(args, "emotion_cache_dir", "emotion_cache"))
    test_dataset.set_emotion_reader(getattr(args, "use_emotion_fusion", False), getattr(args, "emotion_model_dir", None))
    dataset["test"] = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, **loader_kwargs)

    return dataset

if __name__ == "__main__":
    get_dataloaders()
