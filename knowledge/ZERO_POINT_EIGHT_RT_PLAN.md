# 0.80x Real-Time — the last speed gaps (25-search Kevin-Bacon cascade, 2026-08-08 evening)

Goal from the boss: **0.80x realtime** on the 22.5s demo track = synth in 18s
(current best fused c3/ctx0.4 = 23.96s / 1.065x). That is a **~25% cut** from
today. This doc is the full 25-search cascade (conv → GEMM → memory → threading
→ precision → math → compiler → pipeline) with LOCAL VERIFICATION of the
claims that matter, DA-flagged.

---

## ⭐ THE hardware finding (verified on this rig)

### FMA vs mul+add on Zen2 — 1.69x, but ONLY in latency-bound chains
**[uops.info VFMADD231PS]**: Zen2 `vfmadd231ps` throughput 0.50 (2/clock),
port usage **FP0/1 only**. **[AMD Zen2 optimization guide / Chips & Cheese]**:
Zen2 has **two 256-bit FMA units on FP0/1 AND two FP-add units on FP2/3** —
`vmulps`+`vaddps` run on separate port pairs and can sustain ~4 FLOP/cycle
vs FMA's 2/clock ceiling on Intel-style code. [Peter Cordes SO 70340734] is
the canonical explanation (8+ independent accumulators needed to hide the
5-cycle FMA latency).

**Local verification** (build/t_fma_vs_muladd.exe, this Ryzen 5 3600):
```
FMA    : 97.2 GFLOP/s
MUL+ADD: 163.9 GFLOP/s
ratio  : 1.687x        ← REAL in a latency-bound dep-chain benchmark
```

**BUT the actual conv is NOT latency-bound** (build/t_conv_fma_vs_muladd.exe,
the real MRF shape 192→192 k=3 n=798 and n=120000):
```
n=798    FMA 31.89 GMAC/s   MUL+ADD 31.55 GMAC/s   ratio 0.99x
n=120000 FMA 23.14 GMAC/s   MUL+ADD 18.64 GMAC/s   ratio 0.81x
```
maxdiff tiny (1-ulp rounding, <1e-6). The conv is **memory-bound** (input +
weight streaming), so swapping FMA→mul+add is a wash or a LOSS. **DA flag:
do NOT rewrite the conv for mul+add.** The win only applies where the inner
loop is a dependency chain over already-loaded data (attention scores, linear
layers, softmax reductions) — and even there the gain needs ≥8 independent
chains. Check `linear_c`/MHA before touching.

**Where it DOES apply**: if/when the conv becomes compute-bound (e.g. after
stack-multiplexing C1 triples the FMAs per input load), re-test; the 3 stacks
× 4-oc accumulators would give exactly the ILP the mul+add path wants.

---

## Cascade results (25 searches, each → verdict for OUR engine)

### 1. Conv micro-kernels
- **Winograd F(2,3)** [Lavin CVPR16; Springer S11227; singularitykchen survey]:
  1.44× (3×3) to 2.04× (5×5) mult reduction. Our k=3 convs are 192→192 with
  in_ch≥8 → Winograd's input/output transforms add overhead; the register conv
  already captures most. **Verdict: skip — smaller win now, parity risk high.**
- **Nested Winograd for large k** [arXiv 2102.13272] — k=7/11 MRF convs could
  use it, but those are 2 of 3 stacks; complexity not worth it at this stage.
- **Direct conv beats im2col for small channels** [S0167819122000473] — we
  already do direct; confirmed.

### 2. GEMM
- **Kernel register blocking**: BLIS-style 16×6 (Zen3 measured nc=1535,
  mc=1024 single-thread; mc=mR×threads×5, nc=nR×threads×50 multithread)
  [salykova matmul-cpu]. Our conv is a GEMM with K=in_ch×k=576 — blocking K
  into L1-resident slices is the C1 direction. **Verdict: fold into C1.**
- **2+ independent FMA chains** — already have 4×4=16 chains in the register
  conv; confirmed sufficient.

### 3. Memory / layout
- **Channels-last (NHWC)** [Intel IPEX docs; PyTorch blog]: 1.3–1.8× on CPU
  convs from better vectorization. Our layout is [C,T] column-major (NCHW-like)
  — the register conv reads input rows sequentially which IS the friendly
  direction for [C,T]. **Verdict: skip — current layout already sequential.**
- **Non-temporal stores** [Johnny's Software Lab; DPDK]: streaming stores avoid
  cache pollution for the conv OUTPUT (written once, never re-read). `stos`
  with `_mm256_stream_ps` on orow when tile ≥ cache line. **Verdict: TRY —
  cheap, isolated; the conv out tensor is write-once.** Risk: OLA/parity paths
  must not read-after-write the same tile.
- **64-byte alignment + padding** for pooled slabs (real.c pool_in/out/tmp):
  split cache lines on every slab access. **Verdict: TRY — trivial, and the
  arena is reused 216×/chunk so alignment amortizes.**
- **Huge pages / mmap weights** [llama.cpp]: weights.bin is ~110MB; mmap
  read-only + 2MB pages kills page-fault stalls. **Verdict: TRY on Windows via
  VirtualAlloc MEM_LARGE_PAGES or just mmap the .bin; medium effort.**
- **Software prefetch** [lemire blog]: rarely helps when HW prefetchers work;
  our reads are sequential → **skip** (confirmed by lemire's data).

### 4. Threading
- **Nested OMP oversubscription** [Aalto PP; Sun docs; Intel 813536]: nesting
  easily creates runaway thread counts; the fix is **collapse into ONE level**
  or `SUNW_MP_MAX_NESTED_LEVELS`-style caps. **Our exact problem**: 4 workers ×
  3 stacks × 4 conv threads = 48 threads on 12. **Verdict: C1 (stack-multiplex)
  is the structural fix — it turns 3 nested levels into 1.**
- **OMP_PROC_BIND=close OMP_PLACES=cores** [OpenBLAS #1653; ARM docs; Intel]:
  threads pinned to neighboring cores share L3. Zen2: 2 CCX × 3 cores, 16MB L3
  each. 4 workers → 2 per CCX, each pair shares the generator weights in L3.
  **Verdict: TRY first (env var, zero code) — measured flat before, but that
  was pre-fused; re-test.**
- **Work-stealing pool** [Blumofe Cilk; joeduffyblog; Chalmers 35.3]: per-thread
  deques, LIFO + steal — better tail balance than the atomic-counter pool.
  **Verdict: later; the atomic counter already load-balances; tail imbalance is
  ~1 chunk on 77.**
- **One worker per physical core, not SMT** [local-LLM guide]: Ryzen 3600 =
  6 physical; jobs=4 fine; don't go jobs=12.

### 5. Precision / quantization
- **INT8 on AVX2 without VNNI** [Intel VNNI blog; OpenVINO]: i7-8700 (AVX2,
  no VNNI) got **1.67×** (ResNet50) via `_mm256_maddubs_epi16` (2 uops per 8
  int8 MACs). Zen2 has NO AVX-VNNI (checked: Zen4 adds it) — so maddubs is the
  path. **Verdict: the big medium-term lever for the MRF convs; W8A8 ~1.4-2.4×
  per the 51-fix catalog. AFTER C1.**
- **Q8_0 vs Q4_0 in llama.cpp** [arXiv 2601.14277; ik_llama.cpp]: Q8_0 often
  beats Q4 for PROMPT processing (dequant overhead kills Q4 at small batch);
  Q8_0 tokens/s 71 vs Q4 97 in decode but prompt 4.36 vs 5.03 — mixed. For our
  convs (batch 1), **Q8_0 > Q4_0**; INT8 is the sweet spot, not 4-bit.
- **Accuracy cost**: OpenVINO ResNet50 INT8 = 0.07% top-1 drop; ONNX BERT
  ~0.5%. For the MRF (voice timbre), gate with mel-corr — **DA: test per-layer,
  the MRF's chaotic float amplification may need FP32 accumulate (it does —
  INT8 with FP32 accumulators).**

### 6. Math
- **Fast sin/cos** [dsp.stackexchange 68733; SO 27933355; pauldreik]: table +
  poly hybrids reach 8-12×; we already have the bit-trick folded poly (7e-6).
  **Verdict: done.**
- **-ffast-math / -ffp-contract** [ACM 3732365; TechnoLynx]: contract=fast is
  ALREADY GCC's default and is what fused our test mul+add back into FMA —
  verified locally (maxdiff 0.0 until -ffp-contract=off). **DA: our convs are
  hand-intrinsicked so contract doesn't change them; don't enable global
  -ffast-math (parity risk on the chaotic MRF).**

### 7. Compiler
- **PGO** [rustc 1.64 release notes: 10-20% on Windows builds; Nuitka MinGW
  PGO support]: `-fprofile-generate` → run the 22.5s track → `-fprofile-use`.
  **Verdict: TRY — free 5-15%, low risk; the hot loop is already hand-tuned
  but branch layout + inlining decisions improve.**
- **-flto** [GCC docs]: cross-TU inlining of the kernels. **Verdict: TRY with
  the PGO pass — but guard parity (the t_* tests).**
- **-march=native**: already explicit -mavx2 -mfma. Nothing more to grab
  (Zen2 has no AVX-512).

### 8. Pipeline / streaming
- **LLVC: 2.8× realtime on consumer CPU, <20ms latency** [arXiv 2311.00873;
  alphaXiv; demo]: the architecture lesson is **stateful causal streaming with
  NO chunk re-synthesis** — Waveformer-based, GAN + distillation. RVC's
  chunked crossfade inherently re-synthesizes the overlap (our 28% waste).
  **Verdict: the 0.80x target does NOT require replacing RVC's generator; but
  a stateful generator (C6, the real fix for the 28%) is the only way past
  ~0.9x without the multiplex wins.**
- **CSAVocoder** [OpenReview]: strictly causal stateful vocoder — same lesson.
- **Streaming Vocos RTF 0.14** [HF] — already in the catalog.

---

## The 0.80x plan (ranked, with expected deltas on the 22.5s track)

Current: 23.96s synth (1.065x). Target: 18.0s (0.80x). Need −6s (−25%).

1. **C1 — MRF stack-multiplexed conv (input-stationary batch=3)**
   THE structural fix: one conv call, 3 weight sets, shared input loads.
   Input traffic ÷3 AND collapses 3 nested OMP levels into 1 (fixes the
   48-thread oversubscription). Expected: MRF 4.21s → ~2.2-2.8s/chunk at
   jobs=1 (memory traffic is the binding constraint — the mlsysbook
   "batching converts memory-bound → compute-bound"). At jobs=4 the
   oversubscription disappears: ~**−4 to −5s** on the track. Parity gate:
   extend t_conv_ab with the 3-stack shape. **This is the #1 move.**
2. **OMP_PROC_BIND=close + OMP_PLACES=cores** (env, zero code): 2 workers per
   CCX share the ~15MB generator weights in 16MB L3. **−0.5-1s, free.**
3. **Non-temporal stores on the conv output + 64B-aligned pooled slabs**
   (real.c pool_in/out/tmp): write-once tensors, streaming stores avoid
   polluting L1/L2; alignment kills split-line costs on every one of the
   216 arena touches/chunk. **−0.5-1s, low risk.**
4. **PGO + -flto build**: profile on the actual track; branch layout + LTO
   inlining of the kernels. **−0.5-1.5s, guard with t_* + byte-parity.**
5. **--ctx 0.4 default** (already measured: −3s from 0.72 at 3s chunks,
   mel 0.9632) — the boss's call; quality-safe per arXiv 2505.22487.
   **−3s on the current default config.**
6. **INT8 W8A8 MRF convs via maddubs** (no VNNI on Zen2): 1.4-2.4× on the
   convs once compute-bound after C1. FP32 accumulators, per-layer
   mel-corr gate. **After C1; −2-4s potential.**
7. **Stateful generator (C6)** — the "real" answer for the 28% overlap waste:
   keep inter-chunk state instead of re-synthesizing. LLVC proves 2.8× RT is
   architecturally possible. **The next research+impl project, not tonight.**

**Honest DA on the target**: with 1+2+3+4+5 the track should land ~17-19s
(0.75-0.85x) — 0.80x is within reach WITHOUT quantization. If the convs stay
memory-bound after C1 (possible — weights still stream per (oc,ic,tap)),
INT8 (6) is the guaranteed second lever; if weights fit L3 per CCX, the
convT + MRF weights being 15MB means they already do, so C1's shared-input
traffic is the dominant remaining term.

---

## Sources (25-search cascade, key citations)
- uops.info VFMADD231PS (Zen2: throughput 0.50, FP0/1)
- AMD Zen2 optimization guide (2×256-bit FMA + 2 FP-add units) — docs.discoverer.bg/zen2.html
- Chips & Cheese Zen4 (Zen2/3/4 all have 2×256-bit FMA + 4×256-bit ALU)
- StackOverflow 70340734 (Peter Cordes: Zen2 vaddpd/vmulpd on different ports)
- Illinois CS433 Zen2 deck (FP scheduler 4 uops/cycle, 2 loads + 1 store/cycle)
- salykova.github.io/matmul-cpu (Zen cache blocks, multithread rules)
- Lavin CVPR16 Winograd; Springer S11227-023-05088-4 (portable Winograd)
- Intel IPEX channels-last docs; PyTorch channels-last CPU blog
- Johnny's Software Lab (non-temporal stores); DPDK nontemporal memcpy patch
- lemire.me (software prefetch rarely helps)
- Aalto PP nested parallelism; Sun OpenMP nested docs; Intel 813536
- OpenBLAS #1653 (OMP_PROC_BIND/PLACES effects); ARM OpenMP placement docs
- Blumofe Cilk work-stealing; Chalmers 35.3 (work-stealing on GPUs)
- Intel VNNI/DL Boost blog; OpenVINO INT8 (i7-8700 AVX2 1.67x, 0.07% drop)
- arXiv 2601.14277 (llama.cpp Q8_0 vs Q4_0); ik_llama.cpp discussion 258
- rustc 1.64 release notes (PGO 10-20% Windows); Nuitka MinGW PGO
- arXiv 2311.00873 LLVC (2.8x RT CPU VC, <20ms); koeai.github.io/llvc-demo
- CSAVocoder OpenReview (causal stateful vocoder)
- dsp.stackexchange 68733; pauldreik sinus (fast trig)

## Local verification artifacts (this session)
- build/t_fma_vs_muladd.c / .exe — pure dep-chain: MUL+ADD 1.687x vs FMA
- build/t_conv_fma_vs_muladd.c / .exe — real conv shape: 0.99x (n=798),
  0.81x (n=120000) — conv is memory-bound, FMA stays
- Both compiled with -ffp-contract=off to stop GCC re-fusing (default
  contract=fast made maxdiff 0.000000 and hid the real instruction mix)
