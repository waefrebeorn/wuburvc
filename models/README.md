# Models — where the weights live

WuBuRVC carries **no weights** in the repo. Download RVC v2 `.pth` voice
models and extract them to the self-describing binary format the C11 engine
loads. The pitch extractor (`rmvpe.pt`) and content encoder (`hubert`) are
separate extractions.

## Voice conversion models (.pth, RVC v2)

```bash
# 1. put the .pth somewhere, e.g. models/rvc/cartman/EricCartmanV1_v2.pth
# 2. extract to the engine's flat binary format
python tools/extract_rvc_weights.py models/rvc/cartman/EricCartmanV1_v2.pth \
    models/rvc/cartman/EricCartmanV1_v2.pth.weights.bin
# 3. convert
build/wubu_rvc in.wav models/rvc/cartman out.wav --model models/rvc/cartman/EricCartmanV1_v2.pth
```

The engine reads the whole architecture from the config + weight shapes:
sample rate, upsample rates/kernels, MRF resblock kernels/dilations, hidden
channels, conv_post channels/kernel. A 40 kHz Cartman and a 32 kHz Bart load
into the same binary with zero code changes.

## Pitch extractor (RMVPE)

```bash
python tools/extract_rmvpe_weights.py models/rvc/rmvpe.pt
# → models/rvc/rmvpe_weights.bin (742 tensors incl. mel filterbank)
```
The C11 RMVPE port (src/wubu_rmvpe.c) matches PyTorch to corr 0.9999 /
0.19 Hz mean diff. Fallback: YIN (`--f0 yin`).

## Content encoder (HuBERT v2)

RVC v2 uses HuBERT layer-12 (768-dim). Extract its weights once with
`tools/extract_hubert_weights.py`; the C11 port (src/wubu_rvc_hubert.c)
runs the conv encoder + transformer blocks natively.

## Community model sources

- The WuBu voice catalog (organized scraped database of ~9k models with
  ranked download batches) lives under `D:/1aivoice/VoiceModels/` on the
  Windows rig — `tools/download_voice_models.py` in WuBuMedia orchestrates
  the batches.
- RVC model zips typically bundle `*.pth` + `*.index` (FAISS). The engine
  loads the index when present for timbre grounding.
