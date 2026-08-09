# CPU Faster-Than-Realtime — the path (2026-08-08 session)

Status: **45s synth 44.9s → 32.1s (2.00x → 1.43x realtime)**; conv1d micro-kernel
**160x faster** (270ms → 1.7ms per 192×192×3×798 conv), **bit-exact parity**
(maxdiff 0.000000). Full pipeline (HuBERT + RMVPE + generator) ~50s for 22.5s
audio (2.24x) — the generator is the only non-realtime stage.

## The single biggest bug ever found in this codebase (the "hang")
The conv1d `if (getenv("WUBU_DBG_C") && j < 3)` debug print sat INSIDE the
per-sample j-loop: ~29M getenv() calls per conv = 30-300s per conv. It looked
like an infinite loop; it was a per-element env lookup. **Never put getenv()
inside a hot loop — hoist it once.**

## The register-accumulated AVX2 conv1d (the 160x)
Old path: `orow[j] += irow[j+off]*wt` per (oc,ic,tap) — loaded+stored orow[j]
768× per output (per-tap RMW). New path: taps accumulate in 4×8-wide YMM
registers (a0..a3 = 32 samples), orow touched once per block:

- left boundary `[jb, j_in_lo)` — full (ic,tap) sum, bounds-checked (scalar)
- interior `[j_in_lo, j_in_hi)` — register-blocked: a0..a3 = load(orow),
  then for each (ic,tap): 4 FMAs from the SAME input row; store once.
  `j_in_lo = max(jb, pad)`, `j_in_hi = n - (k-1)*dil + pad`
  (exclusive; all taps valid), `jr_max = n - 32 - (k-1)*dil + pad`
  (register reads stay inside the row).
- right boundary — full sum, bounds-checked (scalar)
- `continue` skips the old per-tap loops (it replaces ALL (ic,tap) work).

CRITICAL structural rule (bit us 3×): the register block must be OUTSIDE the
(ic,tap) loops with a `continue`, NOT inside `if (j_lo < j_hi2)` — running it
per-tap re-adds the full sum (in_ch×k)× → outputs 6-192× too large.

## FTZ + DAZ
`_mm_setcsr(_mm_getcsr() | (1u<<15) | (1u<<6))` in wubu_rvc_load (library init,
so every caller gets it). Denormals from RVC's tiny activations are ~100x
slower on x86. Modest win here but correct hygiene.

## Parallel MRF stacks (nested OMP)
The 3 MRF stacks are independent (same stage input, different weights).
`#pragma omp parallel for num_threads(n_stacks)` + `omp_set_nested(1)` +
`omp_set_max_active_levels(2)` in main; conv1d uses
`num_threads(omp_in_parallel() ? 4 : 12)`. jobs=1: MRF 6.6s → 4.1s.
At jobs=4 the 4 chunk workers × nested oversubscribes (4×3×4=48 threads) —
flat. The chunked path needs the config retuned (see next).

## Profile (22.5s audio, jobs=1, per chunk)
- MRF (72 convs): 6.6s → 4.1s with stack-parallel (79-88% of generator)
- convT (4 upsample transposed convs): ~1.0-1.5s
- L3 per pair: conv1 ~26-56ms, conv2 ~26-57ms (register path, ~28% of
  theoretical peak), memcpy+lrelu ~6-11ms

## What's left (the path to <1.0x)
1. **OC-blocked register conv** (4 output channels share each input load):
   input traffic ÷4. **DONE 2026-08-08**: bit-exact at n=798 + n=120000
   (maxdiff 0.000000); L3 conv 26-56ms → 4.0ms. The structural rule that made
   it pass: the OC-block must sit at the JB-TILE level (OUTSIDE the per-oc
   loop) with a `continue` — inside the oc-loop it re-runs per channel and
   double-counts ~in_ch×.
2. **Chunk-worker × nested config**: jobs=4 × stack-parallel oversubscribes;
   try conv num_threads(1) inside nested (12 stacks × serial convs = 12) or a
   per-worker omp thread budget. Measured 2026-08-08: OMP_THREAD_LIMIT
   8/6/4 all flat (~32s) — the convs are memory-bound, threads aren't the
   lever anymore.
3. MRF buffer reuse: **DONE 2026-08-08** — pooled per-layer 3-slab arena
   (rb_in/rb_out/tmp per stack) instead of 216 malloc/free per chunk.
4. **Folded polynomial sin/cos — DONE 2026-08-08** (wubu_math.c): AVX2 pair
   25× vs libm, bit-trick fold 1.20× vs libm at 7e-6; wired into the NSF
   sine + snake; fast exp + fast tanh wired into the softmax + sine.
5. **convT polyphase**: the upsample's transposed conv is now 25% of the
   generator (1.2s/chunk) — polyphase decomposition (per-phase sub-convs) is
   the documented next lever.
6. Winograd F(2,3) for the k=3 MRF convs (the register conv already captures
   most of the win — Winograd's added mult-reduction is smaller now).
7. INT8 per-channel on the MRF (W8A8 ~1.4-2.4x).

## 2026-08-08 session result
45s synth: 44.9s → 28.3s (2.00x → 1.26x realtime) — register conv (160x
kernel) + OC-blocking (input /4) + folded-poly/fast-math + pooling + AVX2
elementwise. Parity: mel-corr 0.9953, rms/peak identical.

## Honest answer to "what is the impossible"
Full RVC (HuBERT+RMVPE+generator) faster than realtime on a 6-core consumer
CPU. The math: HuBERT 0.31x, RMVPE 0.35x, generator 1.43x. The generator was
running at ~0.6% of the machine (per-tap orow traffic + debug getenvs); the
160x conv fix + stack-parallel already got 28%. The remaining gap is memory
traffic (oc-blocking) + threading config, not arithmetic — the LLVC paper
(2.8x realtime CPU VC) and closed-door demos are the independent
verification the boss cites.
