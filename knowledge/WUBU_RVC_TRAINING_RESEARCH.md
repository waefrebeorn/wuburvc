# WuBuRVC Research Notes — Best Base Models & Training Pipeline (2025-2026)

## Sources

### 1. ArkanDash/Advanced-RVC-Inference (GitHub)
- **URL:** https://github.com/ArkanDash/Advanced-RVC-Inference
- **Key innovations:**
  - Auto pretrained model download (RMVPE/FCPE + HuBERT/ContentVec) before training
  - 4 vocoders: HiFi-GAN NSF (Default), **BigVGAN**, **MRF-HiFi-GAN**, RefineGAN
  - 5 optimizers: AdamW, RAdam, AnyPrecisionAdamW, AdaBelief, AdaBeliefV2
  - `--fast_train` flag: 3x faster (TF32 matmul + cuDNN benchmark + torch.compile)
  - `--bf16_adamw` flag: bf16 autocast on Ampere+ (A100/H100/RTX 30xx+/40xx+)

### 2. RVC-Project/retrieval-based-Voice-Conversion-WebUI
- **RVCv3** pre-trained models are experimental (issue #2013)
- Performance improvements "have not met expectations" yet
- Key bottleneck: model size vs. dataset quality tradeoff

### 3. Politrees/RVC_resources (HuggingFace)
- Host ContentVec base, HuBERT models, pre-trained HiFi-GAN
- R-Kentaren's Ultimate-RVC-Models collection: 3400+ base inference models

### 4. YouTube: "How to Make the PERFECT Dataset for RVC AI Voice Training"
- **URL:** https://www.youtube.com/watch?v=9lsSSPnF67Q
- Key insight: Dataset quality > quantity. 6-10 hours of clean, consistent recordings.
- Analyze TensorBoard to detect overfitting, phoneme coverage gaps.

---

## Pre-Trained Base Models (for fine-tuning)

### Embedder Models (HuBERT/ContentVec/WavLM)
| Model | Size | Layers | Hidden | Best For |
|---|---|---|---|---|
| HuBERT Base (`hubert_base.pt`) | 12-layer, 768-dim | 12 | 768 | General voice conversion |
| ContentVec (`contentvec_base_500.pt`) | 12-layer, 768-dim | 12 | 768 | RVC v2 default |
| **WavLM Base** (`wavlm_base.pt`) | 12-layer, 768-dim | 12 | 768 | Noise-robust VC (22.6% better downstream) |
| WavLM Large | 24-layer, 1024-dim | 24 | 1024 | High-fidelity (too heavy for real-time) |

### Pre-Trained G/D Models
| Model | Epochs | F0 Method | Description |
|---|---|---|---|
| G256k | 256k | RMVPE | High-quality, general purpose (RVC v2) |
| D256k | 256k | RMVPE | Discriminator for G256k |
| G1060k | 1060k | FCPE | Extended training, RVCv3 candidate |
| D1060k | 1060k | FCPE | Matching discriminator |

### Vocoders (HiFi-GAN variants)
| Vocoder | Key Feature | WuBuRVC Plan |
|---|---|---|
| HiFi-GAN NSF | Neural Sine Filter, pitch injection | Default (RVC v2 compat) |
| BigVGAN | SnakeBeta + AMP activations | Monolithic kernel support |
| MRF-HiFi-GAN | Multi-Receptive Field fusion | Best quality target |
| RefineGAN | Post-filter refinement | Optional denoising |

---

## Training Pipeline Improvements

### Vocal Extraction (pre-training)
Our `wubu_wubu.c` supports these separators (enum in `wubu_wubu.h`):

```c
#define WUBU_VOCAL_SEP_NONE   0
#define WUBU_VOCAL_SEP_UVR5   1
#define WUBU_VOCAL_SEP_DEMucs 2
#define WUBU_VOCAL_SEP_MDX    3
#define WUBU_VOCAL_SEP_ROFORMER 4
```

**Best order 2025:** MDX-Net → Demucs → Roformer → BS-Roformer (ArkanDash approach)
- MDX-Net: Best for karaoke/vocal isolation (6+ hours training data)
- Demucs: Real-time capable, good for streaming
- Roformer: Low distortion, preserves speaker characteristics

### Dataset Preparation
- Target: 6-10 hours clean mono audio at 44.1kHz
- Split into 5-10s chunks (RVC v2 expects 22050Hz, 58866 samples per chunk)
- Use `wubu_detect.c` to verify file types before training
- Check TensorBoard for phoneme coverage gaps

### Fine-Tuning Strategy
1. **Base model:** G256k (RVC v2) or G1060k (RVCv3 experimental)
2. **Optimizer:** AnyPrecisionAdamW with bf16 (Ampere+ GPUs)
3. **Vocoder:** BigVGAN or MRF-HiFi-GAN for best quality
4. **Speed:** TF32 + cuDNN benchmark + torch.compile equivalent (monolithic kernels)
5. **Dataset:** 6-10 hours high-quality isolated vocals

---

## Implementation Status

| Component | Status | File |
|---|---|---|
| HuBERT/ContentVec/WavLM loader | ✅ Parity achieved (0.4/0.3/0.3 coefficients) | `wubu_rvc_parity.c` |
| Mind-meld encoder fusion | ✅ `wubu_content_mind_meld()` | `wubu_rvc_parity.c` |
| Monolithic CUDA kernel | ✅ Drafted (mel→flow→HiFi-GAN fusion) | `wubu_rvc_mono.cu` |
| RVCv2 .pth backward compat | ✅ Load + upgrade to .wubu | `wubu_wubu.c` |
| .wubu training format | ✅ Container spec + save/load | `wubu_wubu.c` |
| Vocal separation (Demucs/MDX) | ✅ Enum + training metadata | `wubu_wubu.h` |
| Model dock (10 recent) | ✅ LRU + RCU hot-swap | `wubu_model_dock.c` |
| Drag-drop model import | ✅ File type detection | `wubu_detect.c` + `wubugui.c` |
| Pre-warmed model cache | ✅ `wubu_model_dock_set_prewarm()` | `wubu_model_dock.c` |

## Next Steps

1. **nvcc CUDA kernel:** Compile `wubu_rvc_mono.cu` with nvcc on RTX 2080 (sm_75)
2. **WavLM integration:** Our WavLM shares HuBERT architecture (12-layer/768-dim) — existing .pth weights compatible
3. **Dataset download:** Need to fetch RVCv3 base models from HuggingFace Politrees/RVC_resources
4. **Training speed:** Monolithic kernel + TF32 (RTX 2080 has limited TF32, use mixed precision)
