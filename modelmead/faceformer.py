import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from wav2vec import Wav2Vec2Model
from emo_extension import EmotionFusionBlock, align_sequence_length
from model_paths import resolve_wav2vec2_model_path

# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    # modelmead 的逻辑与 BIWI 保持一致：每个顶点帧对应两个连续的音频步。
    if dataset in ["modelmead", "."]:
        for i in range(T):
            start = i * 2
            if start >= S:
                break
            end = min(start + 2, S)
            mask[i, start:end] = 0
    return (mask==1).to(device=device)

# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        pe = self.pe
        x = x + pe[:, :x.size(1), :]
        return self.dropout(x)

class Faceformer(nn.Module):
    def __init__(self, args):
        super(Faceformer, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.dataset = args.dataset
        wav2vec2_model_path = resolve_wav2vec2_model_path(getattr(args, "wav2vec2_model_path", None))
        self.audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec2_model_path)
        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        # XLSR-large 输出维度为 1024；从 config 动态读取，避免再次硬编码 768。
        self.audio_feature_map = nn.Linear(self.audio_encoder.config.hidden_size, args.feature_dim)
        self.use_emotion_fusion = getattr(args, "use_emotion_fusion", False)
        self.emotion_feature_dim = getattr(args, "emotion_feature_dim", args.feature_dim)
        self.emotion_fusion = EmotionFusionBlock(
            audio_dim=args.feature_dim,
            emotion_dim=self.emotion_feature_dim,
            num_heads=getattr(args, "emotion_fusion_heads", 4),
            dropout=getattr(args, "emotion_fusion_dropout", 0.1),
            ffn_ratio=getattr(args, "emotion_fusion_ffn_ratio", 2),
        )
        self.emotion_consistency_head = nn.Linear(args.feature_dim, self.emotion_feature_dim)
        self.emotion_consistency_weight = getattr(args, "emotion_consistency_weight", 0.0)
        self.smoothness_weight = getattr(args, "smoothness_weight", 0.0)
        # motion encoder
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        # periodic positional encoding 
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period)
        # temporal bias
        self.biased_mask = init_biased_mask(n_head = 4, max_seq_len = 600, period=args.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=2*args.feature_dim, batch_first=True)        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        # motion decoder
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
        # style embedding
        self.obj_vector = nn.Linear(len(args.train_subjects.split()), args.feature_dim, bias=False)
        self.device = args.device
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def set_training_stage(self, stage):
        """控制预训练编码器和融合模块的训练阶段。"""
        trainable_audio_encoder = stage != "fusion"
        for parameter in self.audio_encoder.parameters():
            parameter.requires_grad = trainable_audio_encoder

    def _normalize_emotion_features(self, emotion_features):
        """将 emotion2vec cache/在线特征统一成 (batch, time, dim)。"""
        if emotion_features is None:
            return None
        if emotion_features.dim() == 4 and emotion_features.shape[1] == 1:
            # Dataset 返回 (1, time, dim)，DataLoader 会再补 batch 维。
            emotion_features = emotion_features.squeeze(1)
        if emotion_features.dim() == 2:
            emotion_features = emotion_features.unsqueeze(0)
        assert emotion_features.dim() == 3, f"emotion_features 期望为三维张量 (batch, time, dim)，但得到 {tuple(emotion_features.shape)}"
        return emotion_features

    def encode_audio_memory(self, audio, frame_num, emotion_features=None):
        hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state
        if self.dataset in ["modelmead", "."] and hidden_states.shape[1] < frame_num * 2:
            frame_num = hidden_states.shape[1] // 2
        hidden_states = self.audio_feature_map(hidden_states)

        if self.use_emotion_fusion and emotion_features is not None:
            emotion_features = self._normalize_emotion_features(emotion_features)
            emotion_features = align_sequence_length(emotion_features, hidden_states.shape[1])
            hidden_states = self.emotion_fusion(hidden_states, emotion_features)

        return hidden_states, frame_num

    def forward(self, audio, template, vertice, one_hot, criterion, teacher_forcing=True, emotion_features=None):
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        template = template.unsqueeze(1) # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)#(1, feature_dim)
        vertice_emb = obj_embedding.unsqueeze(1)
        style_emb = vertice_emb
        vertice_out = None
        frame_num = vertice.shape[1]
        hidden_states, frame_num = self.encode_audio_memory(audio, frame_num, emotion_features=emotion_features)
        if self.dataset in ["modelmead", "."] and hidden_states.shape[1] < vertice.shape[1] * 2:
            vertice = vertice[:, :hidden_states.shape[1]//2]

        if teacher_forcing:
            vertice_input = torch.cat((template,vertice[:,:-1]), 1) # shift one position
            vertice_input = vertice_input - template
            vertice_input = self.vertice_map(vertice_input)
            vertice_input = vertice_input + style_emb
            vertice_input = self.PPE(vertice_input)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
        else:
            for i in range(frame_num):
                if i==0:
                    vertice_input = self.PPE(style_emb)
                else:
                    vertice_input = self.PPE(vertice_emb)
                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
                memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                vertice_out = self.vertice_map_r(vertice_out)
                new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = vertice_out + template
        loss = torch.mean(criterion(vertice_out, vertice)) # (batch, seq_len, V*3)

        if self.smoothness_weight > 0 and vertice_out.shape[1] > 1:
            smoothness_loss = torch.mean((vertice_out[:, 1:] - vertice_out[:, :-1]) ** 2)
            loss = loss + self.smoothness_weight * smoothness_loss

        if self.use_emotion_fusion and self.emotion_consistency_weight > 0 and emotion_features is not None:
            emotion_features = self._normalize_emotion_features(emotion_features)
            aligned_emotion = align_sequence_length(emotion_features, hidden_states.shape[1])
            assert aligned_emotion is not None
            fused_summary = self.emotion_consistency_head(hidden_states).mean(dim=1)
            emotion_summary = aligned_emotion.mean(dim=1)
            emotion_consistency_loss = torch.mean((fused_summary - emotion_summary) ** 2)
            loss = loss + self.emotion_consistency_weight * emotion_consistency_loss

        return loss

    def predict(self, audio, template, one_hot, emotion_features=None):
        template = template.unsqueeze(1) # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        vertice_emb = obj_embedding.unsqueeze(1)
        style_emb = vertice_emb
        vertice_out = None
        if self.dataset in ["modelmead", "."]:
            frame_num = hidden_states.shape[1] // 2
        else:
            frame_num = hidden_states.shape[1]
        hidden_states = self.audio_feature_map(hidden_states)
        if self.use_emotion_fusion and emotion_features is not None:
            emotion_features = self._normalize_emotion_features(emotion_features)
            emotion_features = align_sequence_length(emotion_features, hidden_states.shape[1])
            hidden_states = self.emotion_fusion(hidden_states, emotion_features)

        for i in range(frame_num):
            if i==0:
                vertice_input = self.PPE(style_emb)
            else:
                vertice_input = self.PPE(vertice_emb)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
            new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
            new_output = new_output + style_emb
            vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = vertice_out + template
        return vertice_out
