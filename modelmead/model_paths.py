import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WAV2VEC2_MODEL_DIR = os.path.abspath(
    os.path.expandvars(
        os.path.expanduser(
            os.environ.get(
                "FACEFORMER_WAV2VEC2_PATH",
                os.path.join(PROJECT_ROOT, "wav2vec2-large-xlsr-53"),
            )
        )
    )
)


def resolve_wav2vec2_model_path(model_path=None):
    """统一解析 modelmead 使用的 wav2vec2 权重目录。"""
    model_path = model_path or DEFAULT_WAV2VEC2_MODEL_DIR
    return os.path.abspath(os.path.expandvars(os.path.expanduser(model_path)))
