# THE FOLDED POLYNOMIAL — deep research (2026-08-08)

Source: **"The Folded Polynomial — N64 Optimization"**, Kaze Emanuar (KazeN64),
YouTube hffgNRfL1XY (Sep 2023, 14:25). Inventor: Silas Lock (posted it in
Kaze's Discord after the "BEST sine function for N64" video).

## The core insight (one sentence)
**Use the function's symmetry to shrink the polynomial's domain, then a
LOWER-order polynomial beats a higher-order one on accuracy — for the same or
fewer cycles.**

Normal polynomial approximation: accuracy comes from raising the order
(3rd-order sin ≈ 0.02 max error, 4th-order cos ≈ 0.003). The folded
polynomial instead cuts the approximation range in half (first EIGHTH of the
sine: [0, π/4]): a 2nd-order polynomial over half the range is **3x more
accurate than a 3rd-order over the full range** — because minimax error
scales with (range)^(order+1), so halving the range gives 2^(order+1) × the
accuracy.

## How it works (the full symmetry chain)
1. Define only cos on [0, π/4] (the first eighth of the sine wave).
2. Even symmetry → cos valid on [−π/4, π/4].
3. sin(x) = cos(x − π/2) → sine on [π/4, 3π/4].
4. Mirror → covers [−π, π] with one tiny polynomial.
5. cos = sqrt(1 − sin²) (and vice versa) — the sqrt fills the other half of
   the pair. Because sin² + cos² = 1 is EXACT, the pair stays normalized
   (this also fixes the "limbs scale" drift of the shift-based method).
6. Quarter-rotation shifts via integer bit-math (90/180/270°), then flip
   signs.

Result on the N64: 3x accuracy at 1.5 cycles *faster* than the previous
poly; the 4th-order cos version: 90x accuracy improvement. (The video is
honest: on the N64 the absolute win is ~50ns/frame — the point is the
TECHNIQUE: "ask what more assumptions can you make, then use every one".)

## General principle (why this is a pivotal optimization class)
- **Range reduction is free accuracy.** Halving the polynomial domain at
  order n multiplies accuracy by 2^(n+1). The libm does range-reduce sin to
  [−π/4, π/4]; the folded polynomial goes one step further to [0, π/4] and
  SHARES the poly across the whole circle via symmetry + the sqrt identity.
- **The sqrt is not the enemy** when it buys a whole half of the graph and
  keeps the pair normalized.
- Applications: sin/cos (sine excitation, snake, STFT windows, pitch),
  exp (softmax — the same trick via 2^(x·log2e) + bit-level range reduction),
  atan2, sqrt itself.

## Why it matters for WuBuRVC (the numbers)
Our engine's sinf call sites (from the profile):
1. **NSF sine excitation** (wubu_rvc_real.c:1102): `sinf(2π·phase)` PER
   SAMPLE — nF × ups_total = 370 × 400 = **148,000 sinf/chunk**; ~1.2M per
   track. libm sinf ≈ 30-40 cycles → ~40M cycles/chunk ≈ 2% of a chunk.
2. **Snake activation** (wubu_rvc_real.c:329, --snake): 2 sinf per element
   across the whole MRF — would be ~2.5s of the 6.6s MRF if enabled.
3. STFT / RMVPE / pitch: window cosines + phase advances.

The folded sin/cos pair (AVX2, 8-wide):
- Range-reduce to [0, π/4] with bit-math (a few AND/shift/sub ops).
- One 2nd/3rd-order minimax poly (5-8 FMA) instead of the libm's branchy
  ~30-40 cycle path.
- sqrt(1−x²) via one `_mm256_sqrt_ps` for the complementary function.
- Expected: **2-4x on every sinf/sinf pair**, at BETTER accuracy than the
  libm (the folded range is smaller than the libm's).

## Implementation plan for our engine
1. `wubu_math.c` (new module): `wubu_sincos_folded(float x, float *s, float *c)`
   + the AVX2 `wubu_sincos8_folded(__m256 x, __m256 *s, __m256 *c)` —
   opaque, C11, no deps.
2. Validate vs libm: max abs error over [−100π, 100π] < 1e-5 (folded should
   beat that), speed bench vs sinf.
3. Wire into the NSF sine excitation (the 148K sinf/chunk) — parity gate:
   output must match the libm version within 1e-4 (the RVC's sine feeds
   tanh → the generator — small diffs are acceptable, the parity bar is the
   mel-corr/SNR, not bit-exact).
4. Wire into the Snake activation (the --snake path) when enabled.
5. Optional: the fast exp for the encoder softmax (same range-reduction
   philosophy; the QuAKE IEEE-754 trick from the 50-fix list).

## Research trail (searches/extractions)
- Video transcript (extracted in full): the technique, the Silas Lock story,
  the honest N64 numbers, the "assume more" takeaway.
- Kaze's previous video: "Finding the BEST sine function for Nintendo 64"
  (xFKFoGiGlXQ) — the baseline 3rd/4th-order polys this improves on.
- Wikipedia: Estrin's scheme — the vectorization-friendly poly evaluation
  (compilers auto-apply it; Boost: unconditionally faster for compile-time
  coefficients).
- Paul Khuong: "The eight useful polynomial approximations of sinf(3)"
  (pvk.ca) — minimax polys over very few points; the folded trick's cousins.
- FABE13-HX (github farukalpay/FABE) — full-range SIMD sin/cos library with
  AVX2 — the reference for the production-grade range reduction.
- Game Math (allenchou.net) — faster sin/cos with poly curves (the fmod +
  odd-poly pattern).
- joelkp.frama.io — modified Taylor polys: why [−π, π] range reduction is
  not enough (confirms the folded [0, π/4] advantage).
- NSF/uSFGAN papers (IEEE, arXiv 2104.04668) — the source-filter vocoder
  structure our sine excitation comes from; confirms the sine is the
  excitation the whole generator filters.
