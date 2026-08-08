# WuBu Audio Research — 50-Search Deep Dive (2026-08-07)

The "magical ingestion system" must know audio research: pitch, frequency,
sample-rate, and model differences. Findings from 50 web searches, cited.

## 1. Pitch detection — why Bart goes off-key

- **RMVPE is the best extractor for singing**: 87.2% avg accuracy across 8
  datasets, best on MIR-1K (96.0%) and Vocadito (96.4%) — beats CREPE on
  human singing. CREPE wins overall accuracy but is **77× slower** than FCPE
  and 14× slower than RMVPE (FCPE RTF 0.0062, RMVPE 0.0329, CREPE 0.4775).
  (lars76/pitch-benchmark; FCPE arXiv 2509.15140)
- **RVC itself uses RMVPE + coarse F0** — the prior encoder takes coarse F0
  discretized into fixed integer bins, not the continuous fine F0.
  (gudgud96.github.io/annotated-rvc; RVC extract_f0_print.py)
- **Octave errors are THE cause of "singing off key"**: YIN/cracked-voice
  f0 flips an octave up/down on unstable phonation. Standard fixes: median
  filtering (RVC `filter_radius`, default 3) + optional HMM smoothing.
  (kvraudio YIN thread; PMC9907018; MATLAB pitch-tracking HMM example)
- **SoVITS docs**: "mean filtering of F0 effectively reduces hoarse sound
  caused by predicted fluctuation of pitch." (RVC-Boss/sovits)
- **Vibrato**: singing F0 has 5–8 Hz modulation; aggressive f0 smoothing
  kills vibrato → smoothing must preserve the high-frequency contour.
  (VibE-SVC Interspeech 2025; SVCC 2025 papers)

**Engine action**: implement `filter_radius` median smoothing on the RMVPE
f0 contour before coarse binning (keep the fast contour, kill octave jumps).

## 2. RVC pipeline conditioning (frequency/pitch)

- Pipeline: resample 16k → HuBERT content (v2 = layer 12, 768-dim) →
  ×2 interpolate (nearest) → f0 at 100fps → coarse bins → flow (prior
  encoder + posterior) → NSF vocoder conditioned on fine f0 + noise.
  (annotated-rvc; navan.dev RVC/CoreML pipeline diagram)
- **f0_up_key** shifts f0 in semitones: `f0 * 2^(key/12)`; -12/0/+12 keep
  the key, other values transpose. Community rule: pitch must be -12, 0 or
  +12 relative to the model's trained range or the voice is "out of tune".
  (ultimate-rvc docs; jeromestephan.de)
- **rms_mix_rate (Volume Envelope)**: 0 = match INPUT loudness envelope,
  1 = match model's training loudness. Default 0.25. Missing in our engine.
  (deepwiki RVC inference params; aihub docs)
- **filter_radius** (median filter, default 3) reduces breathiness and
  hoarseness. Missing in our engine.
- **protect (voiceless consonants)**: masks the mel of unvoiced frames to
  prevent artifacts in "s"/"sh"/"f" and breath; 0.5 flatlines pitch, 0 gives
  robotic sibilants; default 0.33. Missing in our engine.
  (alltalk_tts wiki; RVC issue #2180; applio studio guide)
- **index_rate**: FAISS retrieval of training features (top-1 by cosine/IP)
  reduces timbre leakage; higher rate = more timbre but can reduce quality.
  Our engine has `wubu_rvc_load_index`/`wubu_rvc_retrieve` but the CLI never
  uses the .index — a real missing part. (RVC FAQ; voicechanger.live)

## 3. Vocoder/model differences (32k vs 40k vs 48k)

- RVC v2 arch config drives everything: upsample_rates [10,10,2,2],
  initial_channel 512, kernels [16,16,4,4], resblocks [3,7,11]/dil [1,3,5]
  at 40k; Bart-style models differ (rates [10,8,2,2], kernels [20,16,4,4],
  sr 32k). We are now arch-agnostic (config-driven) — verified.
- 40k "sounds better than 48k" (upsampling artifacts); 32k "more life-like
  but limited bandwidth" — the model's trained sr is the target, always.
  (RVC issue #514)
- HiFi-GAN falls short on musical/high-pitched/OOD content; GAN vocoders
  produce metallic artifacts (mitigated by training length); NSF/harmonic
  source-filter vocoders (uSFGAN, HiFTNet) add pitch controllability — our
  NSF sine+noise source follows that line. (EVA-GAN 2402.00892; coqui #1814;
  HiFTNet 2309.09493)
- Upsampling aliasing causes HF artifacts; anti-aliased upsamplers fix it.
  (FA-GAN 2407.04575; Aliasing-Free Neural Audio Synthesis)

## 4. Glitches/clipping physics

- **Inter-sample peaks** are the classic cause of "random clipping" with no
  sample-level clip: the reconstructed analog signal overshoots 0 dBFS.
  Fix: true-peak limiter + ceiling at **-0.3 to -1 dBFS** (pros use -1,
  some -2). (gearspace true-peak threads; mastering.com)
- **Limiter design**: release-only peak limiters let transients through —
  need lookahead/attack handling; oversample 4× for true-peak detection.
- **Chunking artifacts**: RVC real-time changers chunk audio (chunk 0.3 s,
  extra context 2–5 s) and crossfade/overlap-add at boundaries; naive chunk
  cuts cause clicks/pre-echo. Our engine processes whole files (no chunk
  boundaries) — good — but the master limiter must catch transients.
  (voice-changer #1525; Auralis pre-echo issue)
- **DC offset** removal: high-pass at 20–30 Hz, soft slope. (gearspace)

**Engine action**: true-peak (4× oversampled) ceiling + lookahead-ish
limiter + -1 dBFS ceiling in `wubu_master`.

## 5. Sample-rate handling & resampling

- Linear interpolation aliases badly; windowed-sinc/polyphase FIR is the
  standard (Kaiser-windowed sinc). (dsprelated; flyingSand; renoise forum)
- RVC inference uses `resample_sr=0` = keep original; internal content path
  always goes through 16 kHz for HuBERT. (medium rvc-python guide)

**Engine action**: replace linear resampler in `wubu_audioio` with a
windowed-sinc (Kaiser) polyphase resampler.

## 6. TTS for jokes (knock-knock pipeline)

- **Kokoro-82M** is the best open-source TTS (Apache 2.0, 82M params,
  ~44% win rate TTS Arena; runs realtime on CPU; 24 kHz output; voices:
  af_bella, am_adam, am_michael, bm_george...). Needs espeak-ng for
  phonemization. (bentoml, modal, kokoro docs)
- **edge-tts** is a reverse-engineered MS API — NOT licensed for any use.
  (HN thread) → use Kokoro or Windows SAPI for the demo.
- **TTS→RVC workflow confirmed**: generate TTS wav → convert with RVC
  (pitch + index_rate + protect). (tts-with-rvc; substack demo)

## 7. Music production with AI vocals

- Vocal chain order: de-ess → compressor → EQ (de-ess BEFORE EQ, or the EQ
  re-boosts sibilance). (mastering.com; gearspace)
- AI cover post: mix vocals back to instrumental, subtle reverb to blend,
  EQ to match instrumental frequency profile, compression to even levels.
  (musci.io AI cover guide)
- Loudness: streaming ≈ -14 LUFS integrated; masters often -9 RMS ≈ -11
  LUFS; limiter ceiling -1 dBFS. LUFS = K-weighted (high-shelf
  b0=1.5351 b1=-2.6917 b2=1.1984 a1=-1.6907 a2=0.7325 + RLB HP) RMS.
  (mastering.com; essentia LoudnessEBUR128; EBU tech3343)
- Model quality: 10–30 min clean isolated vocals (no reverb/noise), keep
  breaths, trim silence — the community recipe for natural models.
  (voicechanger.live training guide)

## Verdict for the ingestion system

Our RMVPE C11 port (verified 0.9999 vs Python) is the right pitch engine.
The off-key/glitch fixes are: (1) f0 median filter (`filter_radius`),
(2) true-peak limiter at -1 dBFS ceiling, (3) rms_mix_rate envelope match,
(4) protect voiceless consonants, (5) sinc resampler, (6) optional index
retrieval, (7) LUFS metering. All are C11-doable and belong in the engine.

## WordVoice TTS integration (2026-08-08)

- **WordVoice-base-0.5B (XXH333) = the TTS backbone** (Apache-2.0, paper
  arXiv:2607.06461, built on Fun-CosyVoice3-0.5B-2512). Word-level control
  of 5 acoustic dims: duration, boundary (b0-b4), energy, pitch (-1..1),
  tone (flat/rise/rrise/fall/ffall/peak/valley) via "acoustic thinking"
  bound-token. Zero-shot voice cloning from a ~10s prompt.
- **Models** (in `out/WordVoice/checkpoints/`, 15.4 GB total):
  Fun-CosyVoice3-0.5B (9.1G), mms_fa aligner (1.2G), WordVoice-base-0.5B
  (5.1G: wordvoice_llm_en.pt, wordvoice_fm.pt).
- **Windows deps** (installed into .venv_win on top of torch 2.6+cu124):
  einops, conformer, diffusers, inflect, num2words, x_transformers,
  hyperpyyaml, cn2an, torchaudio 2.6.0+cu124 (--no-deps), modelscope,
  openai-whisper, pypinyin, matplotlib, torchcrepe, scipy, onnxruntime,
  lightning==2.2.4 (flow_matching), pyworld, onnx, gdown, wget,
  hydra-core==1.3.2 + omegaconf==2.3.0 (CRITICAL: newer hydra breaks
  CosyVoice's yaml load), python-dateutil (force-reinstall).
- **Repo fixes applied** (reference bugs — we fix and keep parity):
  - `cosyvoice/llm/wordvoice_llm.py`: all `logits[...] = negative_inf`
    must be `negative_inf.to(logits.dtype)` (fp16 inference, 6 sites).
  - `eval()` needs module globals `Aligner_Model` + `wordvoice` set before
    calling (they're only defined under `__main__` in wordvoice_infer.py) —
    driver sets them via `wvi.Aligner_Model = ...`.
  - Init `WordVoice(..., fp16=True)` for GPU (bf16 checkpoint vs fp32
    inputs mismatch otherwise).
- **Driver**: `tools/wubu_wordvoice_jokes.py` — scripted knock-knocks with
  inline `[dur:ms] [eng:x] [pit:x] [bnd:bN] [ton:...]` tags per word →
  WordVoice TTS (reference prompt `demo/prompt_speech_en.mp3` +
  "The team that change what they're doing...") → soundfile PCM16 →
  our C11 RVC character voice. The LLM's default pacing is ~40ms/word —
  explicit [dur:200-240] tags are required for natural speech.
- WordVoice writes float32 wav (format 3) — Python `wave` can't read it;
  use soundfile or our C11 audioio.

## CRASH FIX: rms_mix use-after-free (2026-08-08)

- Symptom: 45s conversions + short TTS inputs crashed (rc=139) right after
  synth, inside the rms_mix resample. Root cause: the CLI freed the input
  `audio` buffer after the 16k resample, then the rms_mix stage read the
  dangling pointer. Deterministic at -O2/-O3, worked at -O0, intermittent
  under gdb — classic freed-heap reuse.
- Fix: `free(audio)` moved to the cleanup block; rms_mix keeps the
  original-rate input. Lesson: when adding a stage that reads the input
  AFTER a free, verify buffer lifetimes (angel coder = no UAF).

## Performance (2026-08-08)

- RMVPE conv2d/convt2d + GRU gate loops now OpenMP-parallelized over
  output channels (was 100% serial). Parity unchanged (0.9999).
- CLI built -O3. 45s conversion ~8-9 min single-threaded-conversion;
  two concurrent conversions contend (avoid running 2 at once).
- HuBERT ~0.7x realtime; synth ~11x realtime (flow+generator, OpenMP'd).
  Next big win: batch GRU over frames (RMVPE) + conv1d tiling.
