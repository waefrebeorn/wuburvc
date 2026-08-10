# TRIPLE DEVIL'S ADVOCATE AUDIT — Remaining Robotic-Speaking Gaps

**Date:** 2026-08-09 | **Method:** 15 web searches (7-steps-to-Kevin-Bacon), every claim
verified against our source, Triple-DA loop (Claim → Verify → Risk → Mitigate) ×3
passes. **Status:** ✅ Verified | 🟡 Partial | ❓ Unchecked | 🔴 Broken

---

## PASS 1 — EXCITATION-SIDE GAPS (source-filter theory)

### GAP 1.1: Zero jitter / zero shimmer — pure periodic sine excitation
- **Claim:** Human voice has cycle-to-cycle f0 (jitter) and amplitude (shimmer)
  variation; our sine excitation is perfectly periodic → robotic timbre on held
  vowels. SOTA: jitter/shimmer are THE discriminators of synthetic voice
  (arXiv:2502.14726 deepfake detection uses them directly).
- **Verify:** `src/wubu_rvc_real.c:1497` — `sinf(2π·phase) · 0.1` pure sine, no
  per-cycle perturbation. The noise dither (0.003) adds broadband noise but does
  NOT modulate the fundamental period. 🟡 PARTIAL — noise exists, period is fixed.
- **Risk:** Adding jitter badly (random period wobble) can sound worse — pitch
  instability, warbling. Confirmation bias: "noise exists so it's natural".
- **Mitigate:** Add *small* per-cycle phase jitter (~0.1–0.3% period modulation)
  and amplitude shimmer (~1–3%) ONLY on voiced frames, scaled by noise_scale so
  `--noise 0` stays bit-exact. Monitoring: pitchCV should RISE on held vowels
  without exceeding ~0.05; revert if warbling.
- **Impact:** 🔴 HIGH — likely the #1 remaining "robotic speaking moment".

### GAP 1.2: No aperiodicity component (mixed excitation missing)
- **Claim:** SOTA vocoders (STRAIGHT/D4C, Kawahara 2001) weight harmonic vs
  noise by a per-band AP (aperiodicity) parameter; single-excitation models
  produce "robotic colorization" (arXiv:2601.10345). Our generator has one
  sine + one global noise branch — no band-dependent AP.
- **Verify:** `wubu_generator_nsf` noise_amp is frame-global (0.003/0.0333);
  no per-frequency-band harmonic/noise ratio. 🟡 PARTIAL — uv gates noise but
  not spectrally.
- **Risk:** Implementing full band-AP requires per-band filtering — heavy for
  realtime. Scope creep vs the jitter gain.
- **Mitigate:** Cheapest SOTA approximation: highpass/lowpass the noise branch
  (breath noise is lowpassed by the model's noise_convs already — verify first
  whether the vocoder already does this). Only add explicit AP if jitter alone
  doesn't move the needle.

### GAP 1.3: Vibrato may be killed by f0 smoothing
- **Claim:** VibE-SVC (arXiv:2605.20794) treats vibrato as a separable
  high-frequency f0 component; smoothing the contour (our `--f0smooth 0.4`)
  suppresses vibrato → flat, robotic singing. Also `wubu_f0_smooth` doc says
  "reduces jitter" — which may strip the natural jitter we want to ADD.
- **Verify:** CLI prints "[5] f0 contour smoothing strength 0.40 applied" and
  default f0_smooth=0.0 (only applied when flag passed). Median filter radius 3
  ALWAYS applied. 🟡 PARTIAL — default is no smoothing; but users pass 0.4.
- **Risk:** Vibrato removal makes sustained notes sound like a synth pad.
- **Mitigate:** Change `wubu_f0_smooth` to vibrato-aware smoothing: extract the
  high-freq vibrato component (DWT or simple band-split), smooth only the slow
  contour, re-add vibrato at the original amplitude. Monitoring: pitchCV on
  sustained golden notes must stay ≥ pre-smoothing.
- **Impact:** 🟡 MEDIUM for singing; LOW for speech.

---

## PASS 2 — CONTENT/PITCH-EXTRACTION GAPS

### GAP 2.1: Content layer is fixed (HuBERT layer 12) — WavLM averaging is SOTA
- **Claim:** SOTA VC averages carefully-selected WavLM+HuBERT layers for content
  (Martín-Cortiñas 2025, arXiv:2505.08278); W2VC uses WavLM. We use HuBERT
  layer 12 only (RVC-faithful).
- **Verify:** `wubu_rvc_hubert.h:18` — "v2: output layer 12 (768 dim)". ✅ VERIFIED.
- **Risk:** Layer-12 HuBERT is speaker-richer than prosody-rich; but changing
  content features REQUIRES retraining the model (enc_p input dims match
  training). This is the backbone milestone, not an inference fix.
- **Mitigate:** Already documented in `knowledge/BACKBONE_RESEARCH.md` — WavLM
  retraining is a training-side milestone. Do NOT swap features at inference
  (breaks the pretrained enc_p). KEEP as is. Monitoring: the SpongeBob retrain
  milestone uses the Vulkan trainer → then WavLM.
- **Impact:** 🟡 MEDIUM — but gated on training milestone; not a quick win.

### GAP 2.2: RMVPE is SOTA — but is it what we actually use by default?
- **Claim:** FCPE paper (arXiv:2509.15140) confirms RMVPE ≥ CREPE ≥ Harvest on
  RPA; RMVPE is the training-time extractor so coarse bins match training.
- **Verify:** CLI default is RMVPE (`wubu_rmvpe_f0`), falls back to YIN.
  ✅ VERIFIED — we use the best extractor.
- **Risk:** None for extraction. But RMVPE weights must be present; if missing,
  YIN fallback is worse. Check model dir.
- **Mitigate:** Verify `models/rvc/rmvpe_weights.bin` exists (it does — used in
  all runs). Monitoring: none needed.

### GAP 2.3: 40k vs 48k — upsampling "duplicate data" robotic claim
- **Claim:** RVC issue #514: 40k models often sound better than 48k because 48k
  upsampling introduces duplicate-sample artifacts that read as robotic.
- **Verify:** Our CLI resamples input to 16k for f0/content, then the generator
  outputs at model sr (40k for v2, 48k for v1). SpongeBob is 48k. ✅/🟡 — the
  resample path uses `wubu_audio_resample` (linear? verify quality).
- **Risk:** If our resampler is linear (not polyphase/sinc), 48k upsampling
  could add the exact duplicate-data artifacts the community reports.
- **Mitigate:** Check `wubu_audio_resample` implementation. If linear, upgrade
  to windowed-sinc (polyphase) — a pure DSP improvement with no model change.
  Monitoring: HF5k on 48k models should drop after fix.

---

## PASS 3 — POST-PROCESSING / PIPELINE GAPS

### GAP 3.1: Artifact filter after synthesis (AF-Vocoder GAFilter)
- **Claim:** AF-Vocoder (Interspeech 2025) adds a global artifact filter after
  generation — M-STFT 0.86→0.78, PESQ 3.6→4.05. We have no artifact filter.
- **Verify:** Post chain = normalize → preset EQ → de-ess. No artifact
  filtering stage. ❓ UNCHECKED — no artifact detector in our pipeline.
- **Risk:** GAFilter is a learned model (112M params) — can't port to C11
  without training. But the CONCEPT (detect & attenuate spectral outliers)
  can be approximated with a spectral-flatness/outlier gate.
- **Mitigate:** Cheap approximation: soft-knee spectral gate on frames whose
  STFT has abnormally low harmonic-to-noise ratio vs neighbors (vocoder
  artifacts are spectrally flat bursts). Gate amplitude, don't filter.
  Monitoring: measure artifact-frame fraction before/after.

### GAP 3.2: Index-rate/protect defaults vs community sweet-spot
- **Claim:** Community: index_rate 0.5–0.75 (not 0.78) reduces artifacts;
  protect 0.75 for voice bleed. We use RVC-faithful 0.78/0.33.
- **Verify:** `--index-rate 0.78`, `--protect 0.33` defaults in CLI. ✅ VERIFIED
  (RVC pipeline.py defaults).
- **Risk:** For SpongeBob the index is EMPTY (ntotal=0) so index_rate is inert —
  no retrieval bias. Protect is inert without an index too. So for our current
  models these knobs do nothing; no artifact risk. But for models WITH indexes
  (Cartman 30k vectors) 0.78 might pull artifacts per community reports.
- **Mitigate:** Keep RVC-faithful defaults (parity), expose `--index-rate`
  already present. Consider lowering default to 0.6 for quality mode when an
  index exists (community sweet spot). Monitoring: Cartman HF5k at 0.78 vs 0.6.

### GAP 3.3: rms_mix_rate 0.25 — implemented but is it active?
- **Claim:** RVC default rms_mix_rate 0.25 applies the INPUT volume envelope to
  the output — without it, output has flat dynamics (robotic energy). We
  implemented it.
- **Verify:** CLI `--rmsmix 0.25` default + `wubu_rms_mix_rate` call at line
  1152–1160. ✅ VERIFIED — implemented and default-on.
- **Risk:** None — but verify it actually runs in chunked mode (whole-track only?).
- **Mitigate:** Check that rms_mix applies in chunked inference too (currently
  it's applied post-stitch — confirm). Monitoring: output dynamics should track
  input RMS contour.

### GAP 3.4: Snake activation fallback — quality loss on pretrained models
- **Claim:** Our generator prefers Snake (BigVGAN) but falls back to LeakyReLU
  when Snake saturates (pretrained RVC weights trained with LReLU). The fallback
  path means pretrained models run LReLU — which is parity-correct but loses
  BigVGAN's aliasing advantage.
- **Verify:** CLI lines 1045–1061 — fallback triggered when sat_frac > 0.5.
  ✅ VERIFIED.
- **Risk:** None — this is CORRECT (using Snake on LReLU weights = square wave).
  Keep the fallback. Monitoring: sat_frac printed on WARNING.

---

## PRIORITY RANKING (impact × effort)

| # | Gap | Impact | Effort | Action |
|---|-----|--------|--------|--------|
| 1 | **Jitter/shimmer injection** (1.1) | 🔴 HIGH | Low | IMPLEMENT — phase jitter + shimmer on voiced frames, scaled by noise_scale |
| 2 | **Vibrato-aware smoothing** (1.3) | 🟡 MED | Low | IMPLEMENT — preserve vibrato HF component in f0_smooth |
| 3 | **Resampler quality** (2.3) | 🟡 MED | Low | CHECK → upgrade to windowed-sinc if linear |
| 4 | **Artifact spectral gate** (3.1) | 🟡 MED | Med | APPROXIMATE — flatness-outlier gate post-synth |
| 5 | **rms_mix chunked** (3.3) | 🟡 MED | Low | VERIFY chunked path |
| 6 | **Index-rate 0.6 for quality** (3.2) | 🟢 LOW | Low | TUNE only for models with real indexes |
| 7 | WavLM backbone (2.1) | 🟡 MED | High | GATED on retrain milestone — keep documented |
| 8 | Band-AP excitation (1.2) | 🔴 HIGH | High | DEFER — vocoder noise_convs may already shape it |

## DECISION
**Implement #1 and #2 now** (both are small, source-filter-correct, and target
the exact "robotic speaking moments" the boss hears). Verify #3 and #5.
Defer #8 (big), #7 (gated), approximate #4 after #1–#2 prove out.

## Open Questions (next loop)
- Does `wubu_audio_resample` use linear or polyphase? (check before #3)
- Does rms_mix run in chunked mode or only whole-track?
- What does the generator's noise_convs actually do to the noise branch —
  does it already band-shape AP?
