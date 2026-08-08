# WuBuRVC

> **From-scratch C11 RVC — Retrieval-based Voice Conversion, inference + training + mastering, one binary.**
> Architecture-agnostic. Parity-verified against PyTorch to **corr 0.9999**. Runs on Windows today, WuBuOS tomorrow.

WuBuRVC is the voice engine of the WuBu AGI stack — a complete, self-contained
C11 implementation of the RVC v2 pipeline: **HuBERT content → RMVPE pitch →
flow (prior + posterior) → NSF vocoder**, plus a **full mastering suite** and a
**WordVoice TTS bridge** for character voices.

No Python at runtime. No torch. No "third party" — *we properly make it*.

## Why it's amazing

| Property | Value |
|---|---|
| **Engine** | 100% C11, opaque structs, minimal includes, self-contained modules |
| **Arch-agnostic** | Every parameter read from model config/weights — 32 kHz, 40 kHz, 768-dim, any upsample/MRF topology |
| **Pitch** | Full RMVPE port (DeepUnet+BiGRU+cnn), **corr 0.9999** vs PyTorch, 0.19 Hz mean diff |
| **Parity** | Output **corr 0.9999 / SNR 29.7 dB** vs the PyTorch reference with reference inputs |
| **Off-key fix** | f0 median filter (`filter_radius`, default 3) kills octave jumps, keeps vibrato |
| **Glitch fix** | True-peak (4× inter-sample) limiter, −1 dBFS ceiling — no hidden clipping |
| **Sample rates** | Windowed-sinc (Kaiser) resampler — consistent pitch across model kHz |
| **Dynamics** | `rms_mix_rate` (0.25) — converted voice follows the input's volume envelope |
| **TTS** | WordVoice (CosyVoice3 0.5B) bridge: word-level pitch/tone/energy/duration control, zero-shot cloning |
| **Speed** | OpenMP on all convs; RMVPE/GRU parallelized; CLI at -O3 |
| **License** | WaefreBeorn Umbrella License v3.0 (source-available, attribution) |

## Verified by the numbers

```
Pitch conditioning (coarse-bin agreement with training f0):
  YIN 22.8%  →  RMVPE 96.0%  →  RMVPE + median filter 99.2%

Output correlation vs PyTorch reference:
  old pipeline bugs 0.004  →  RMVPE f0 0.70  →  exact f0 0.9999

RMVPE port (C11 vs Python):     corr 0.9999, 168/168 voiced, 0.19 Hz diff
Generator (C11 vs torch):       corr 0.9999, SNR 29.7 dB
Mastering suite:                peak under −1 dBTP, RMS target ±1.5 dB
```

## The pipeline (all C11)

```
input.wav ── sinc resample → 16 kHz
    ├─ HuBERT v2 content (layer 12, 768-dim) ── nearest ×2 ──┐
    └─ RMVPE f0 (DeepUnet+cnn+BiGRU, 100 fps) ── median filter ── coarse bins ──┴─→ flow
                                                                                       │
posterior encoder + flow (prior/posterior, affine coupling) ◄── f0 + content ──────────┤
                                                                                       ▼
NSF vocoder (sine + noise source → BigVGAN-style generator with Snake) ── rms_mix ──► out.wav
```

Then the **C11 mastering suite**: RBJ EQ (double-precision biquads) → RMS
compressor → saturation → stereo width → **true-peak limiter** → loudness.

## Build

### Windows (MSYS2 / MinGW-w64)

```bash
export PATH="/c/msys64/mingw64/bin:$PATH"

# RVC CLI (RMVPE + HuBERT + flow + NSF)
gcc -std=c11 -O3 -I src -fopenmp src/wubu_rvc_cli.c src/wubu_rvc.c \
    src/wubu_rvc_parity.c src/wubu_rvc_weights.c src/wubu_rvc_kernels_exact.c \
    src/wubu_rvc_real.c src/wubu_rvc_hubert.c src/wubu_rvc_f0.c \
    src/wubu_postproc.c src/wubu_rmvpe.c src/wubu_stft.c src/wubu_gru.c \
    src/wubu_audioio.c -lm -o build/wubu_rvc

# RMVPE parity test (needs models + reference)
cc -std=c11 -O2 -I src -fopenmp tests/test_rmvpe.c src/wubu_rmvpe.c \
    src/wubu_stft.c src/wubu_gru.c -lm -o build/test_rmvpe

# Mastering suite
cc -std=c11 -O2 -I src src/test_master.c src/wubu_master.c src/wubu_audioio.c \
    -lm -o build/test_master
cc -std=c11 -O2 -I src tools/wubu_mixmaster.c src/wubu_master.c \
    src/wubu_audioio.c -lm -o build/wubu_mixmaster
```

### Linux
Same commands (gcc, OpenMP). CUDA training kernels (`src/wubu_rvc_kernels.cu`)
build via `nvcc` with `Makefile.win cuda` (sm_75+).

## Models

The repo carries **no weights** — download RVC v2 `.pth` models and extract
them to the self-describing binary format the engine loads:

```bash
python tools/extract_rvc_weights.py model.pth model.pth.weights.bin
python tools/extract_rmvpe_weights.py models/rvc/rmvpe.pt  # pitch extractor
python tools/extract_hubert_weights.py ...                 # content encoder
```

Config fields: `1`=upsample rates, `2`=sample rate, `3`=hidden, `4`=mel,
`5`=version, `6`=MRF resblock kernels, `7`=MRF dilations. The engine reads
EVERYTHING from the config/weights — a 32 kHz Bart model and a 40 kHz Freddie
model both load into the same binary.

## Usage

```bash
# Convert a vocal through any RVC v2 model
build/wubu_rvc in.wav models/rvc/cartman out.wav \
    --model models/rvc/cartman/model.pth --noise 0.33333

# Full control
build/wubu_rvc in.wav models/rvc/bart out.wav \
    --model models/rvc/bart/model.pth \
    --f0filter 3      # median filter on f0 (kills off-key jumps)
    --rmsmix 0.25     # output follows input volume envelope
    --f0 yin          # fallback pitch extractor
```

## Character voices — WordVoice TTS bridge

`tools/wubu_wordvoice_jokes.py` synthesizes emotional speech (word-level
duration/pitch/tone/energy/pause control, zero-shot voice cloning from a
10 s prompt) and converts it through the C11 engine to any character voice:

```bash
python tools/wubu_wordvoice_jokes.py --voice cartman --joke 0 \
    --out out/demo/joke_cartman.wav
```

## Parity — we make our own, and prove it

The `knowledge/` tree documents the full verification chain: `test_rmvpe`
proves the pitch port to 3–4 decimals; `tools/gen_reference_real.py` produces
the PyTorch reference; `compare_outputs.py` scores correlation/SNR. A claim
isn't done until a runtime artifact shows the number.

## Repo layout

```
src/         C11 engine — RVC core, RMVPE, HuBERT, STFT, GRU, flow, NSF,
             audio I/O (robust WAV), mastering chain, training (CPU+CUDA)
tools/       extractors (weights → self-describing bins), parity generators,
             comparison/statistics, A/B video, TTS jokes, charts
tests/       parity + unit tests (RMVPE vs Python, mastering, training)
knowledge/   research + the PyTorch reference implementation (parity source)
examples/    runnable examples
models/      where weights go (README inside)
```

## License

**WaefreBeorn Umbrella License v3.0** (`SPDX-License-Identifier: WaefreBeorn-UMV3`).
Source-available; non-commercial / personal / research / educational use with
attribution. See `LICENSE`.

---

*Part of the WuBu AGI stack — `waefrebeorn/wubuwizard`, `waefrebeorn/slermes`,
`waefrebeorn/WuBuOS`. Built agentically, verified against the originals.*
