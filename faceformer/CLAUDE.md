# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FaceFormer is a PyTorch deep learning project (CVPR 2022) that generates 3D facial animations from speech audio. It uses Wav2Vec 2.0 for audio encoding and a Transformer Decoder with periodic positional encoding for sequence-to-sequence prediction.

## Environment Setup

conda activate faceformer

## Common Commands

### Training and Testing
```bash
# Train on VOCASET
python main.py --dataset vocaset --vertice_dim 15069 --feature_dim 64 --period 30 \
  --train_subjects "FaceTalk_170728_03272_TA ..." \
  --val_subjects "FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA" \
  --test_subjects "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA"

# Train on BIWI
python main.py --dataset BIWI --vertice_dim 70110 --feature_dim 128 --period 25 \
  --train_subjects "F2 F3 F4 M3 M4 M5" \
  --val_subjects "F2 F3 F4 M3 M4 M5" \
  --test_subjects "F1 F5 F6 F7 F8 M1 M2 M6"

# Visualize results
python render.py --dataset vocaset --vertice_dim 15069 --fps 30
python render.py --dataset BIWI --vertice_dim 70110 --fps 25
```

### Demo/Inference
```bash
python demo.py --model_name vocaset --wav_path "demo/wav/test.wav" --dataset vocaset \
  --vertice_dim 15069 --feature_dim 64 --period 30 --fps 30 \
  --train_subjects "..." --test_subjects "..." --condition FaceTalk_170913_03279_TA \
  --subject FaceTalk_170809_00138_TA
```

## Architecture

**Core Model** ([faceformer.py](faceformer.py)):
- Wav2Vec2 audio encoder (via [wav2vec.py](wav2vec.py)) → 768-dim audio features
- Periodic Positional Encoding (PPE) for vertex sequences
- Single Transformer Decoder layer with ALiBi temporal bias
- Cross-attention between audio features and vertex embeddings

**Key Files**:
- [faceformer.py](faceformer.py) - Main model architecture
- [wav2vec.py](wav2vec.py) - Wav2Vec2 audio encoder wrapper
- [data_loader.py](data_loader.py) - Dataset and DataLoader implementation
- [main.py](main.py) - Training and testing entry point
- [demo.py](demo.py) - Single-audio inference with rendering
- [render.py](render.py) - Batch rendering of predictions to video

**Data Flow**: Raw audio → Wav2Vec2 → audio features → Transformer Decoder → vertex offsets → add to template → rendered mesh

## Datasets

Three supported datasets with different mesh topologies:
- **VOCASET**: FLAME topology, 15069 vertices, --period 30
- **BIWI**: 70110 vertices, --period 25
- **modelmead**: mead3d dataset (see [modelmead/CLAUDE.md](modelmead/CLAUDE.md))

Each dataset needs: vertices_npy/, wav/, templates.pkl, and pretrained .pth model file. Refer to README for full data preparation instructions.

## Important Notes

- No formal test framework exists; validation is done through rendering and visual inspection
- Model uses ALiBi (Attention with Linear Biases) for handling variable-length sequences
- The `init_biased_mask()` function in faceformer.py computes the ALiBi attention bias matrix
- Download pretrained models from Google Drive links in README
