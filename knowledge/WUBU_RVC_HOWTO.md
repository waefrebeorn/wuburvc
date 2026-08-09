# WuBuRVC — How To Use

WuBuRVC = our own RVC (Retrieval-based Voice Conversion) engine. It takes a
recording of ANYONE talking and re-voices it as a trained character voice —
Cartman, Miku, Wheatley, or any of the 30,878 voices in the voice-models
catalog.

**The pipeline:** `text → TTS (Piper) → RVC conversion → character WAV`

---

## 1. The one-liner (production path, Python)

```python
import sys
sys.path.insert(0, "src")
import wubu_rvc

rvc = wubu_rvc.RVC(device="cuda")        # cuda = RTX, "cpu" also works
out = rvc.convert("my_recording.wav", "cartman")
print(out)   # -> out/speech/rvc_<voice>_<timestamp>.wav
```

Voice names are fuzzy — `"cartman"`, `"Eric Cartman"`, `"miku"`, `"wheatley"`
all work. If the model is local it's used directly; if it's only in the 30k
catalog it prints the download URL.

### Resolve order (how a name finds a voice)
1. exact match in the local bank (out/voices.json) with a real `.pth`
2. fuzzy local-bank hit with a real model file
3. disk scan of `models/rvc/<dir>/` for a downloaded model
4. the 30,878-voice voice-models catalog (tools/voice_repo.py) — prints URL if not downloaded

---

## 2. Text → character voice in three commands

```bash
# a) TTS the line with Piper (amy = female, ryan = male)
printf 'Hey you guys, it is me, Eric Cartman!' | \
  python -m piper -m models/piper/en_US-ryan-medium.onnx \
                  -c models/piper/en_US-ryan-medium.onnx.json \
                  -f outputs/line.wav

# b) convert to the character with the FAST C11 engine
#    (18 s for a 6 s line — not 30 minutes like the Python path!)
./build/wubu_rvc_cli.exe outputs/line.wav \
    models/rvc/cartman outputs/cartman_says_line.wav \
    --model models/rvc/cartman/EricCartmanV1_e650_s10400.pth

# c) listen: outputs/cartman_says_line.wav
```

> Piper CLI note: on this machine the `piper.exe` CLI has a wave-writer bug
> (`# channels not specified`). Use the Python package instead:
> ```python
> import piper, wave
> voice = piper.PiperVoice.load("models/piper/en_US-ryan-medium.onnx",
>                               config_path="models/piper/en_US-ryan-medium.onnx.json")
> with wave.open("outputs/line.wav", "wb") as w:
>     voice.synthesize_wav("Hey you guys!", w)
> ```

### The standalone CLI (`src/wubu_rvc_cli.c`)
The engine as a program, not a test. Pure C11 + OpenMP, no Python at runtime:

```
wubu_rvc_cli <input.wav> <model_dir> <output.wav> [--model model.pth]
```

Pipeline: resample to 16k → C HuBERT (v2, 768) → content ×2 → YIN f0 →
enc_p + flow + GeneratorNSF → 40k PCM_16 wav. It's the same code path the
parity harness verifies, minus the reference comparisons.

---

## 3. C11 engine (SLERM core — no Python at runtime)

The real engine is pure C11: `src/wubu_rvc_real.c` (enc_p + flow +
GeneratorNSF), `src/wubu_rvc_hubert.c` (HuBERT content encoder),
`src/wubu_rvc_f0.c` (YIN pitch). It loads checkpoints via a flat WUBU binary:

```bash
# extract a checkpoint to WUBU format once
python tools/extract_rvc_weights.py models/rvc/cartman/EricCartmanV1_e650_s10400.pth \
    models/rvc/cartman/cartman_weights.bin

# build + run the parity harness (C11 vs PyTorch reference)
# -O3 -march=native -fopenmp are REQUIRED for speed (12 threads on this rig)
export PATH="/c/msys64/mingw64/bin:$PATH"
gcc -O3 -march=native -fopenmp -std=c11 -I src -o build/test_rvc_real.exe \
    src/test_rvc_real.c src/wubu_rvc.c src/wubu_rvc_parity.c \
    src/wubu_rvc_weights.c src/wubu_rvc_kernels_exact.c \
    src/wubu_rvc_real.c src/wubu_rvc_hubert.c src/wubu_rvc_f0.c \
    -lsqlite3 -lm -fopenmp
cp /c/msys64/mingw64/bin/libgomp-1.dll build/   # one-time (runtime dep)
./build/test_rvc_real.exe
```

Current parity (honest numbers, Cartman, 2.8 s input):
- **HuBERT content: PASS** — max abs diff 3e-5 vs PyTorch
- **Synth audio: PASS** — SNR 29.78 dB vs Mangio-RVC reference
- Both checkpoint styles load: `weight`-key (Cartman) and `model`-key (G40k)

### Speed (this machine: 12 logical cores)
| Stage        | Time   | vs realtime |
|--------------|--------|-------------|
| HuBERT       | 0.95 s | **0.34x — faster than realtime** |
| Synth        | 5.31 s | 1.91x |
| **Total**    | **6.3 s** | **2.24x realtime** for 2.8 s audio |

The engine got here via three fixes, in order of impact:
1. **Cache-friendly loop order** (oc → ic → tap → j with j innermost) —
   the old order touched in_ch rows of 11 KB+ per output sample, which was
   catastrophic on the 512-ch convs; this alone took ~530 s → ~61 s.
2. **`-O3 -march=native`** (AVX2/FMA) — ~2x.
3. **OpenMP over output channels/heads** (`-fopenmp`, 12 threads) —
   ~61 s → ~6.3 s. libgomp-1.dll must sit next to the exe at runtime.

The 5.3 s synth is dominated by the HiFi-GAN MRF (~123 G-MACs of conv work
over the 111 k-sample output). Streaming short TTS lines at ~2x realtime is
already usable on-stream; the next big lever is fusing the MRF stacks into
one parallel region and/or SIMD hand-rolling the k=3/7/11 kernels.

---

## 4. Characters available right now (local models)

| Name      | Model dir                        | Notes                        |
|-----------|----------------------------------|------------------------------|
| Cartman   | `models/rvc/cartman/`            | v2, 40k, has FAISS index     |
| Miku      | `models/rvc/miku/`               | v2                           |
| Wheatley  | `D:\Archive\...\RVC3\Mangio-...\weights\` | Portal 2 personality core    |
| GLaDOS    | (30k catalog)                    | run `rvc.resolve('GLaDOS')` for URL |

### Grab any of the 30k catalog voices
```python
rvc = wubu_rvc.RVC(device="cuda")
hit = rvc.resolve("wheatley")        # or "GLaDOS", "obama", "squidward"...
print(hit["download_url"])           # HuggingFace URL — download the .pth,
                                     # drop it in models/rvc/<name>/, done
```

---

## 5. Honest caveats (don't get surprised)

- **Speed:** the C11 engine is ~2.2x realtime total (HuBERT itself 3x
  faster than realtime) on this rig with `-O3 -march=native -fopenmp`.
  The old Python/Mangio subprocess path was ~200x slower — use the C engine.
- **Index:** the FAISS `.index` improves timbre. Without it the pipeline still
  runs (retrieval ratio drops to 0 — never fake-blends).
- **Deterministic parity:** the test zeroes SineGen noise on both sides. Real
  conversion adds the learned excitation noise (`noise_std=0.003`), which is
  fine for listening.
- **Piper CLI bug:** use the Python piper package, not `piper.exe`.
- **Streaming:** for live cohost voice on stream, `src/wubu_voice.py` runs the
  loop: TTS → RVC → Voicemeeter Input (OBS hears it).
