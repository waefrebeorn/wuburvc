# BREATH REALISM RESEARCH — SOTA 2024–2026

**Date:** 2026-08-09 | **Status:** Implemented (wubu_breath.c, generator gate)

## Problem (boss-reported)
"Breathiness inconsistency": phantom breaths appear where the source has none;
real breaths (inhalation before phrases, coughs) are never rendered.

## Root causes found (verified against our engine)
1. **RVC preprocessing drops breaths** (RVC-Boss, issue #65): the dataset
   pipeline volume-thresholds away breath sounds because they're quiet, so
   RVC models are *never trained* to render breaths. Hence converted audio
   has no real breath; the model invents artifacts there instead.
2. **Our uv mask is binary and flatness-driven**: spectral-flatness classifies
   SILENCE as "unvoiced" (flat spectrum ≈ noise-like), so the generator's
   unvoiced noise branch (0.0333 Gaussian) fires on silence → phantom breath.
3. **f0 interpolation makes everything "voiced"** (RVC get_f0 semantics):
   `nsff0 > 0` ≈ always true → sine runs through breaths; the real breath's
   noise texture never reaches the vocoder's noise branch.
4. No separate breath-existence detector: no way to know *where* breathing
   actually is in the source → cannot gate/inject correctly.

## SOTA solutions researched (21+ searches, papers extracted)

### 1. Frame-wise breath detection (Yang/Koriyama/Saito, INTERSPEECH 2024, arXiv 2402.00288)
Rule-based breath annotation, precision **0.982**, from pause regions:
- **Duration** > 300 ms (breaths are longer than brief pauses/clicks)
- **Max(ZCR)** > 150 (zero-crossing rate — breath is turbulent noise)
- **Max(VMS)** > 1e-4 (Variance of Mel-Spectrogram — spectral texture)
- **NA-VMS** > 0.6 (normalized average of VMS — distinguishes breath from
  tongue clicks, which have a short VMS peak but low NA-VMS)
Config: 22.05 kHz, 256 mel bands, win 256, hop 128; ZCR same window.
Then a Conformer+BiLSTM detector is self-trained; we use the RULE-BASED
first stage (no ML dependency — fits our C11 engine).

### 2. Breath placement (Sonarworks; Székely et al. ICASSP 2020)
- Breaths occur **50–200 ms before vocal phrases** (inhalation before speech).
- Spontaneous speech: breaths also mid-phrase (disfluency) — but excluding
  "disfluent" breaths improves perceived naturalness (Székely ABB < ABP).
- Breath rendering needs *context* (breath-group bigrams); simple uniform
  breath insertion sounds worse than no breath.

### 3. Breath rendering = noise excitation (source-filter theory)
- PT synthesizer (Camara Largo 2024): breathiness = white noise amplitude
  proportional to **1 − √T** (glottal openness) added to the glottal flow.
- Cantor Digitalis (Feugère 2017): glottal leakage → aspiration/breath noise;
  extreme = whisper (no fold vibration).
- RVC SineGen `noise_amp = uv*0.003 + (1−uv)*0.1/3` — unvoiced frames get
  Gaussian breath texture, but only if uv actually goes 0 (it doesn't after
  f0 interpolation).

### 4. Silence vs breath discrimination (VAD lineage)
- ZCR: voiced low, unvoiced high (Bachu 2010) — breath has HIGH ZCR like
  fricatives but LOW energy like silence. Energy + ZCR together separate
  silence (low/low) from breath (low/high) from fricative (high/high).
- VMS/spectral variance separates breath (sustained textured noise) from
  silence (flat, no variance) and clicks (short spike).

### 5. Community consensus on RVC training data (voicechanger.live, wiki)
"Keep natural breathing — removing all breaths can make the model sound
robotic." Training-side fix is out of scope for inference; our inference
must detect-and-render from the source instead.

## Our engine implementation (wubu_breath.c + generator gate)
1. `wubu_breath_detect()`: per-frame (10 ms, hop 160 @16k) features:
   - RMS energy (silence floor, adaptive)
   - ZCR (breath/fricative signature)
   - spectral variance across FFT bins (VMS analog; mel not needed)
   - spectral flux (click/cough transient vs sustained breath)
   Classifies each frame: silence / breath / consonant / voiced.
   Breath = low energy + high ZCR + sustained (duration ≥ 300 ms) +
   VMS texture; silence = low energy + low ZCR + low VMS; consonant =
   high energy + high ZCR; voiced = harmonic (flatness low).
2. Generator gate (`wubu_generator_nsf`):
   - SILENCE: noise_amp → 0 (kills phantom breath) + sine off.
   - BREATH: noise_amp → breath_gain × 0.1 (renders real inhalation as
     noise texture; model's learned breath behavior is weak because
     training dropped breaths, so we supply the excitation).
   - VOICED: sine + 0.003 dither (unchanged).
   - CONSONANT: sine off + 0.0333 frication (unchanged).
3. CLI: `--breath 0/1` (default 1), `--breath-gain` (default 1.0).
4. Probe: synthetic breath/silence/voiced clip → verify 3-class separation;
   real dry stems → count breath frames before/after.

## Verification numbers (2026-08-09, after mapping fix)
- Probe synthetic: silence 47/48, breath 56/58, voiced 75/80 — 4-class PASS.
- Real stems: rap (clevespooner) 5 breath events at phrase boundaries
  (390–580 ms, zcr 0.59–0.84) — the pre-phrase inhalation pattern; K-pop
  autotune golden 0 events (correct — sustained digital singing).
- Deterministic A/B (--noise 0, same seed path):
  golden clip (0 breaths) breath ON vs OFF: maxdiff=0, 0.00% changed —
  NO phantom breath, singing identical.
  rap (3 breaths) breath ON vs OFF: 0.50% changed (≈75 ms = the breath
  regions only) — singing untouched, inhalations rendered.
- Stochastic (production) stats: rap voiced 0.45 unchanged, pitchCV
  0.328→0.397 (breath breaks monotone pitch), HF5k unchanged;
  golden voiced 0.69, pitchCV 0.348→0.347, HF5k 0.176→0.169.
- CRITICAL PITFALL: the stochastic posterior (noise_scale 0.66666) makes
  EVERY run differ — sample-level A/B diffs are meaningless unless
  --noise 0 is passed. Always use --noise 0 for byte-level comparisons.
- Gain mapping (do not regress): BREATH→1.0 (boost), CONSONANT/VOICED→
  0.25 (leave legacy noise), SILENCE→0.0 (kill phantom). The generator
  treats >0.5 as breath-boost, 0.05..0.5 as leave, <0.05 as kill.

## References
- Yang, Koriyama, Saito, "Frame-Wise Breath Detection with Self-Training",
  INTERSPEECH 2024, arXiv:2402.00288
- Székely, Henter, Beskow, Gustafson, "Breathing and Speech Planning in
  Spontaneous Speech Synthesis", ICASSP 2020
- RVC-Project issue #65 (RVC-Boss: preprocessing drops breaths)
- Sonarworks, "How to add breath sounds and realism to AI vocals"
- Camara Largo, "Generative and parametric models for interactive neural
  synthesizers", 2024 (breathiness ∝ 1−√T)
- Pramono et al., "Automatic Cough Detection in Acoustic Signal using
  Spectral Features" (cough = sharp spectral attack + burst)
- voicechanger.live, "How to Train Your Own RVC Voice Model" (keep breaths)
