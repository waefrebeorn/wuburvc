# WUBU RVC Architecture Research & Improvement Plan

## Current State

### What Works (Verified)
- **Cartman v2 model** (40k, hidden=256, content_dim=768, rates=[10,10,2,2])
  - HuBERT content encoder: 12-layer transformer, layer 12 output (768-dim)
  - SineGen ported: `_f02sine` with rad=phase, sin(2*pi*rad)*0.1*uv
  - GeneratorNSF: enc_p + flow + upsamples + MRF + conv_post + PixelShuffle
  - **Parity: PASS** — SNR=29.79 dB, correlation=0.999928
  - **CLI works**: `wubu_rvc_cli_fixed.exe` converts WAV → Cartman voice, 0% clipping

### Hardcoded Values (Fixed)
All previously hardcoded values are now dynamic:
- **Upsample rates** `[10,10,2,2]` → inferred from kernel sizes (k=16→10, k=4→2) or config section
- **Sample rate** `40000` → read from model config
- **Content dim** `768` → dynamic based on RVC version (v1:256, v2:768)
- **Model paths** → dynamic directory scan for `*.pth` + `*.index`
- **WAV reader**: `char`→`unsigned char` (sign extension bug fixed)

## RVC v1 vs v2 Architecture Differences

### Key Differences (from reference code in knowledge/rvc_mangio, knowledge/rvc_original)

| Feature | RVC v1 | RVC v2 |
|---------|--------|--------|
| Content dim | 256 (final_proj 768→256) | 768 (raw layer 12) |
| HuBERT layer | 9 | 12 |
| TextEncoder | TextEncoder (emb_phone: Linear(256,h)) | TextEncoder768 (emb_phone: Linear(768,h)) |
| Speaker embedding | None | spk_embed_dim=109 |
| Generator | conv_post=(1,256,7) | conv_post=(1,32,7) + PixelShuffle |
| Sample rates | 22050, 25600 | 40000, 48000 |

### Config Array Mapping (from .pth)
```
[0]  = spec_channels (1025 mel bins)
[1]  = segment_size (32)
[2]  = inter_channels / hidden_channels (192)
[3]  = hidden_channels (192, duplicate)
[4]  = content_dim (768 for v2, 256 for v1)
[5]  = ? (2)
[6]  = ? (6)
[7]  = ? (3)
[8]  = ? (0)
[9]  = version string ('1' or '2')
[10] = resblock_kernels ([3, 7, 11])
[11] = resblock_dilation_sizes ([[1,3,5], [1,3,5], [1,3,5]])
[12] = upsample_rates ([10, 10, 2, 2] for 40k, [12, 10, 2, 2] for 48k)
[13] = upsample_initial_channel (512)
[14] = upsample_kernel_sizes ([16, 16, 4, 4])
[15] = spk_embed_dim (109)
[16] = hidden_channels (256)
[17] = sample_rate (40000 for Cartman, 48000 for Miku)
```

## WUBU Binary Format (current)
```
[4B]  magic = "WUBU"
[4B]  n_tensors (uint32)
For each tensor:
  [1B]  name_len
  [NB]  name (ASCII)
  [4B]  n_dims
  [4B*n_dims] dims
  [4B]  data_len (bytes)
  [DL]  raw float32 data
[4B]  config_len (uint32) — 0 if no config
[CL]  config data:
  For each field:
    [1B]  field_id
      1 = upsample_rates (list of int32)
      2 = sample_rate (int32)
      3 = hidden_channels (int32)
      4 = mel_channels (int32)
      5 = version (int32)
    [4B]  n_values
    For each value:
      [1B]  type (0=int32, 1=float32)
      [4B]  value
```

## Wubuwizard C11 Primitives Available for Porting

From `wubuwizard/src/lfm2_math.c` + `lfm2_conv.c`:
- `lfm2_matmul_f32(x, W, M, K, N, y)` — self-contained matmul
- `lfm2_rmsnorm(x, gamma, n, eps)` — RMS normalization

These are cleaner than our current implementations and should be port candidates.

## Missing Features / Improvement Targets

### 1. RVC v1 Support (`TextEncoder256`)
- Current C11 `wubu_rvc_real.c` only handles v2's `emb_phone(768→192)`
- v1 uses `emb_phone(256→192)` — need version detection and dynamic content_dim
- **Status**: `content_dim` is already dynamic based on `rvc->rvc_version`, but HuBERT layer selection needs v1 path (layer 9 + final_proj)

### 2. HuBERT v1 Content Path (layer 9 + final_proj 768→256)
- Current HuBERT always uses layer 12 (768-dim)
- v1 RVC models need layer 9 output, then `final_proj(768→256)`
- **Status**: `wubu_hubert_extract_real` exists but may not have final_proj loaded

### 3. Speaker Embedding Support (v2 multi-speaker)
- Miku model: `spk_embed_dim=109, gin_channels=256`
- v2 models with spk_embed need `dec.cond` (Conv1d(256, 512, 1))
- **Status**: C11 `wubu_generator_nsf` has `if (g)` path but CLI doesn't pass speaker

### 4. Noise Injection in NSF
- PyTorch: `noise_amp * torch.randn_like(sine_waves)` — stochastic
- C11: currently 0 (deterministic, for parity testing)
- **Status**: Deterministic is correct for parity; noise toggle needed for final output

### 5. HuBERT Transformer Full 12 Layers
- Current C11 implementation uses only 4 layers (limited for CPU speed)
- **Status**: `HUBERT_N_LAYERS=12` is defined but loop is capped at 4
- wubuwizard has optimized matmul that could help

### 6. PixelShuffle/Unshuffle for Generator
- v2: `dec.conv_post` outputs (1, 32, T) → PixelShuffle(2) → (1, 1, 2T)
- v1: `dec.conv_post` outputs (1, 256, T) → no PixelShuffle
- **Status**: C11 `wubu_flow_reverse` has PixelShuffle logic but needs to detect v1 vs v2

### 7. FAISS Index for Retrieval (v2)
- Current: loads index but doesn't use it in synthesis
- PyTorch: uses index for feature retrieval → blend with content
- **Status**: Index parser exists, integration into synthesis pipeline pending

## RVC v1 vs v2 Tensor Name Differences

### Posterior Encoder (enc_p)
- v1: `enc_p.emb_phone.weight: (256, hidden)`, no transformer layers
- v2: `enc_p.emb_phone.weight: (768, hidden)`, transformer encoder layers

### Generator (dec)
- v1: `dec.conv_post.weight: (1, 256, 7)` — direct output
- v2: `dec.conv_post.weight: (1, 32, 7)` + PixelShuffle(2)

### Flow
- Both: `flow.flows.0.post.weight: (96, 192, 1)`, `flow.flows.2/4/6.post`
- v2: `flow.flows.0/`, `.2/`, `.4/`, `.6/` → 4 flow layers

## Improvement Roadmap

### Phase 1: Full Loader Compatibility
- Parse config from .pth `config` list properly
- Auto-detect v1 vs v2 from config[4] (content_dim) or tensor names
- Store config in WuBu binary format

### Phase 2: HuBERT Full Transformer
- Port wubuwizard's `lfm2_matmul_f32` for faster matmul
- Run all 12 transformer layers (with OpenMP parallelization)
- Implement v1 layer-9 + final_proj path

### Phase 3: Speaker Support
- Load `dec.cond` weights for multi-speaker models
- Integrate FAISS index retrieval into synthesis
- Add speaker ID selection CLI arg

### Phase 4: Noise + Stochastic Generation
- Add `noise_amp * randn()` to NSF sine generation
- Match PyTorch's `noise_std=0.003` default

### Phase 5: Training Pipeline
- Implement gradient-based training in C11 (GradRetentionNet integration)
- Support fine-tuning from existing models
- Export trained models back to .pth + .bin format
