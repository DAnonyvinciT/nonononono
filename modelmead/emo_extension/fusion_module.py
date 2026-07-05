import torch
import torch.nn as nn
import torch.nn.functional as F


def align_sequence_length(features, target_length):
    """将帧级序列线性插值到指定时间长度。"""
    if features is None:
        return None
    if features.dim() != 3:
        raise ValueError(f"期望输入为三维张量 (batch, time, dim)，但得到 {tuple(features.shape)}")
    if features.shape[1] == target_length:
        return features
    if features.shape[1] == 1:
        return features.repeat(1, target_length, 1)

    features = features.transpose(1, 2)
    aligned = F.interpolate(features, size=target_length, mode="linear", align_corners=True)
    return aligned.transpose(1, 2)


class EmotionFusionBlock(nn.Module):
    """单层 cross-attention 情绪融合块。"""

    def __init__(self, audio_dim, emotion_dim, num_heads=4, dropout=0.1, ffn_ratio=2):
        super().__init__()
        if audio_dim % num_heads != 0:
            raise ValueError(f"audio_dim={audio_dim} 必须能被 num_heads={num_heads} 整除")

        self.emotion_proj = nn.Identity() if emotion_dim == audio_dim else nn.Linear(emotion_dim, audio_dim)
        self.cross_attn = nn.MultiheadAttention(audio_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(audio_dim)
        self.dropout1 = nn.Dropout(dropout)

        ffn_hidden_dim = audio_dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(audio_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, audio_dim),
        )
        self.norm2 = nn.LayerNorm(audio_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, audio_features, emotion_features=None):
        """输入 Xw / Xe，输出融合后的 Xf。"""
        if emotion_features is None:
            return audio_features

        emotion_features = align_sequence_length(emotion_features, audio_features.shape[1])
        emotion_features = self.emotion_proj(emotion_features)
        attn_output, _ = self.cross_attn(audio_features, emotion_features, emotion_features, need_weights=False)
        fused = self.norm1(audio_features + self.dropout1(attn_output))

        ffn_output = self.ffn(fused)
        fused = self.norm2(fused + self.dropout2(ffn_output))
        return fused
