import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import logging
import sys
import re
from datetime import datetime

import torch
import torch.nn as nn

from data_loader import get_dataloaders
from faceformer import Faceformer
from model_paths import DEFAULT_WAV2VEC2_MODEL_DIR
from data_paths import (
    DEFAULT_MEAD_DATASET_DIR,
    MODELMEAD_ROOT,
    resolve_work_path,
)
from emo_extension import DEFAULT_EMOTION2VEC_DIR


class TqdmLoggingHandler(logging.Handler):
    """使用 tqdm.write 输出日志，避免打断进度条。"""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


class StreamToLogger:
    """将 stdout/stderr 重定向到 logging，过滤空行与进度条控制符。"""

    ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    IGNORE_PATTERNS = (
        "torchaudio/_backend/utils.py",
        "torchaudio/_backend/ffmpeg.py",
        "StreamingMediaDecoder has been deprecated",
        "load_with_torchcodec",
        "warnings.warn(",
        "UserWarning:",
    )

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message):
        if not isinstance(message, str):
            message = str(message)
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._log_line(line)

    def flush(self):
        if self._buffer:
            self._log_line(self._buffer)
            self._buffer = ""

    def _log_line(self, line):
        line = self.ANSI_ESCAPE_PATTERN.sub("", line).rstrip("\r")
        stripped = line.strip()
        if not stripped:
            return
        # tqdm 进度条和光标控制序列不写入日志
        if stripped in {"[A", "]"} or "\r" in line:
            return
        if any(pattern in stripped for pattern in self.IGNORE_PATTERNS):
            return

        level = self.level
        if level >= logging.ERROR and ("Traceback (most recent call last):" in stripped or "Error" in stripped or "Exception" in stripped):
            level = logging.ERROR
        elif level >= logging.ERROR:
            level = logging.WARNING

        self.logger.log(level, stripped)


def get_progress_stream():
    """优先将 tqdm 输出到终端设备，避免被 stderr 重定向后写入日志。"""
    try:
        return open("/dev/tty", "w", encoding="utf-8", buffering=1)
    except OSError:
        return sys.__stderr__


def sanitize_name(value):
    """将名称转换为安全文件名片段。"""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_") or "unknown"


def build_log_file_path(args):
    """按规则生成日志文件名：train_模型名_数据集_年月日时分.log。"""
    log_dir = resolve_work_path(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    model_name = sanitize_name(args.model_name)
    dataset_name = sanitize_name(os.path.basename(os.path.normpath(args.dataset_dir)))
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    base_name = f"train_{model_name}_{dataset_name}_{timestamp}"
    log_file = os.path.join(log_dir, f"{base_name}.log")

    if not os.path.exists(log_file):
        return log_file

    suffix = 1
    while True:
        candidate = os.path.join(log_dir, f"{base_name}_{suffix:02d}.log")
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


def setup_logger(args):
    """配置训练日志：终端输出与文件输出同时开启。"""
    log_file = build_log_file_path(args)

    logger = logging.getLogger("faceformer_train")
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    logger.propagate = True

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(stream=sys.__stdout__)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    if root_logger.handlers:
        root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # 让常见第三方库透传到根日志器，避免绕过本地日志文件
    for lib_logger_name in ["transformers", "torch", "urllib3", "matplotlib"]:
        lib_logger = logging.getLogger(lib_logger_name)
        lib_logger.handlers.clear()
        lib_logger.propagate = True

    return logger, log_file


def redirect_std_streams(logger):
    """将标准输出与错误输出统一收集到日志。"""
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)


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


def trainer(args, train_loader, dev_loader, model, optimizer, criterion, logger, epoch=100):
    save_path = resolve_work_path(args.save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    iteration = 0
    progress_stream = get_progress_stream()
    for e in range(epoch+1):
        stage = "fusion" if e < args.freeze_epochs else "joint"
        model.set_training_stage(stage)
        loss_log = []
        # train
        model.train()
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            dynamic_ncols=True,
            file=progress_stream,
            mininterval=0.5,
        )
        optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name, emotion_features) in pbar:
            iteration += 1
            # to gpu
            audio, vertice, template, one_hot, emotion_features = (
                audio.to(device="cuda"),
                vertice.to(device="cuda"),
                template.to(device="cuda"),
                one_hot.to(device="cuda"),
                emotion_features.to(device="cuda"),
            )
            loss = model(audio, template,  vertice, one_hot, criterion, teacher_forcing=args.teacher_forcing, emotion_features=emotion_features)
            loss.backward()

            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad()

            if args.debug_max_train_batches > 0 and (i + 1) >= args.debug_max_train_batches:
                break

            pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}".format((e+1), iteration ,np.mean(loss_log)))
        pbar.close()
        # validation
        valid_loss_log = []
        model.eval()
        for i, (audio, vertice, template, one_hot_all, file_name, emotion_features) in enumerate(dev_loader):
            # to gpu
            audio, vertice, template, one_hot_all, emotion_features = (
                audio.to(device="cuda"),
                vertice.to(device="cuda"),
                template.to(device="cuda"),
                one_hot_all.to(device="cuda"),
                emotion_features.to(device="cuda"),
            )
            if args.dataset in ["modelmead", "."]:
                train_subject = file_name[0].split("_")[0]
            else:
                train_subject = "_".join(file_name[0].split("_")[:-1])
            if train_subject in train_subjects_list:
                condition_subject = train_subject
                cond_idx = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:, cond_idx, :]
                loss = model(audio, template,  vertice, one_hot, criterion, emotion_features=emotion_features)
                valid_loss_log.append(loss.item())
            else:
                for cond_idx in range(one_hot_all.shape[-1]):
                    condition_subject = train_subjects_list[cond_idx]
                    one_hot = one_hot_all[:, cond_idx, :]
                    loss = model(audio, template,  vertice, one_hot, criterion, emotion_features=emotion_features)
                    valid_loss_log.append(loss.item())

            if args.debug_max_valid_batches > 0 and (i + 1) >= args.debug_max_valid_batches:
                break
                        
        current_loss = np.mean(valid_loss_log)
        train_loss = np.mean(loss_log)
        
        if (e > 0 and e % 25 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path,'{}_model.pth'.format(e)))

        logger.info(
            "Epoch %d, iteration %d TRAIN LOSS:%.7f VALID LOSS:%.7f",
            e + 1,
            iteration,
            train_loss,
            current_loss,
        )
    if progress_stream not in {sys.__stdout__, sys.__stderr__}:
        progress_stream.close()
    return model

@torch.no_grad()
def test(args, model, test_loader,epoch):
    result_path = resolve_work_path(args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = resolve_work_path(args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    model = model.to(torch.device("cuda"))
    model.eval()
   
    for i, (audio, vertice, template, one_hot_all, file_name, emotion_features) in enumerate(test_loader):
        # to gpu
        audio, vertice, template, one_hot_all, emotion_features = (
            audio.to(device="cuda"),
            vertice.to(device="cuda"),
            template.to(device="cuda"),
            one_hot_all.to(device="cuda"),
            emotion_features.to(device="cuda"),
        )
        if args.dataset in ["modelmead", "."]:
            train_subject = file_name[0].split("_")[0]
        else:
            train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            cond_idx = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:, cond_idx, :]
            prediction = model.predict(audio, template, one_hot, emotion_features=emotion_features)
            prediction = prediction.squeeze() # (seq_len, V*3)
            np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
        else:
            for cond_idx in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[cond_idx]
                one_hot = one_hot_all[:, cond_idx, :]
                prediction = model.predict(audio, template, one_hot, emotion_features=emotion_features)
                prediction = prediction.squeeze() # (seq_len, V*3)
                np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())

        if hasattr(args, "debug_max_test_batches") and args.debug_max_test_batches > 0 and (i + 1) >= args.debug_max_test_batches:
            break
         
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="modelmead", help='数据集类型，用于选择时序与文件名处理逻辑')
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=DEFAULT_MEAD_DATASET_DIR,
        help='MEAD 数据集根目录，默认 ~/datasets/mead_clean',
    )
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=128, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default="wav", help='音频目录；相对路径基于 dataset_dir')
    parser.add_argument(
        "--vertices_path",
        type=str,
        default=os.path.join(MODELMEAD_ROOT, "vertices_npy_flat"),
        help='展平顶点目录；默认使用 modelmead 中已有的预处理结果',
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--num_workers", type=int, default=4, help='DataLoader 工作进程数')
    parser.add_argument("--pin_memory", type=parse_bool, default=True, help='是否启用 DataLoader pin_memory')
    parser.add_argument("--persistent_workers", type=parse_bool, default=True, help='是否启用 DataLoader persistent_workers')
    parser.add_argument("--prefetch_factor", type=int, default=2, help='每个 worker 的预取批次数')
    parser.add_argument("--use_audio_cache", type=parse_bool, default=False, help='是否优先读取离线音频缓存')
    parser.add_argument("--audio_cache_dir", type=str, default="audio_cache", help='离线音频缓存目录（相对 modelmead 路径）')
    parser.add_argument("--wav2vec2_model_path", type=str, default=DEFAULT_WAV2VEC2_MODEL_DIR, help='wav2vec2-large-xlsr-53 权重目录')
    parser.add_argument("--use_emotion_fusion", type=parse_bool, default=False, help='是否启用 emotion2vec 融合块')
    parser.add_argument("--emotion_feature_dim", type=int, default=1024, help='emotion2vec 帧级特征维度')
    parser.add_argument("--emotion_fusion_heads", type=int, default=4, help='情绪 cross-attention 头数')
    parser.add_argument("--emotion_fusion_dropout", type=float, default=0.1, help='情绪融合 dropout')
    parser.add_argument("--emotion_fusion_ffn_ratio", type=int, default=2, help='情绪融合 FFN 宽度倍率')
    parser.add_argument("--freeze_epochs", type=int, default=0, help='先冻结 wav2vec/emotion2vec 的训练轮数')
    parser.add_argument("--emotion_consistency_weight", type=float, default=0.0, help='情绪一致性损失权重')
    parser.add_argument("--smoothness_weight", type=float, default=0.0, help='时序平滑损失权重')
    parser.add_argument("--emotion_cache_dir", type=str, default="emotion_cache", help='离线 emotion2vec 缓存目录（相对 modelmead 路径）')
    parser.add_argument("--emotion_model_dir", type=str, default=DEFAULT_EMOTION2VEC_DIR, help='emotion2vec 模型目录')
    parser.add_argument("--use_emotion_cache", type=parse_bool, default=False, help='是否优先读取离线 emotion2vec 缓存')
    parser.add_argument("--debug_max_train_batches", type=int, default=0, help='调试时限制训练 batch 数，0 表示不限制')
    parser.add_argument("--debug_max_valid_batches", type=int, default=0, help='调试时限制验证 batch 数，0 表示不限制')
    parser.add_argument("--debug_max_test_batches", type=int, default=0, help='调试时限制测试 batch 数，0 表示不限制')
    parser.add_argument("--teacher_forcing", type=parse_bool, default=False, help='是否在训练时使用 teacher forcing')
    parser.add_argument("--model_name", type=str, default="faceformer", help='模型名称（用于日志文件命名）')
    parser.add_argument("--log_dir", type=str, default="logs", help='日志目录（相对 modelmead 路径）')
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help='日志等级')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--template_file",
        type=str,
        default="templates.pkl",
        help='个性化模板文件；相对路径基于 dataset_dir，默认读取 MEAD 数据目录下的 templates.pkl',
    )
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="M003 M005 M007 M009 M011 M012 M013 M019 M022 M023 M024 M025 M026 M027 M028 M029 M030 M031 W009 W011 W014 W015 W016 W018 W019 W023 W024 W025 W026 W028 W029")
    parser.add_argument("--val_subjects", type=str, default="M032 M033 M034 M035 W033 W035 W036")
    parser.add_argument("--test_subjects", type=str, default="M037 M039 M040 M041 M042 W037 W038 W040")
    args = parser.parse_args()

    logger, log_file = setup_logger(args)
    redirect_std_streams(logger)
    logger.info("训练启动，日志文件: %s", log_file)

    #build model
    model = Faceformer(args)
    logger.info("model parameters: %d", count_parameters(model))

    model.use_emotion_fusion = args.use_emotion_fusion

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device("cuda"))
    
    #load data
    dataset = get_dataloaders(args)
    # loss
    criterion = nn.MSELoss()

    # Train the model
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)
    model = trainer(args, dataset["train"], dataset["valid"],model, optimizer, criterion, logger, epoch=args.max_epoch)
    
    test(args, model, dataset["test"], epoch=args.max_epoch)
    
if __name__=="__main__":
    main()
