# Chunking, Sizing & Multiplex Research — the last CPU levers (researched 2026-08-08, 10 web searches + 3 deep extracts)

Question from the boss: *"the last bit of speed ups we can get via optimization of
chunking, sizing strategies, and optimization of Multiplex strategies for CPU."*

Current engine state (chunked path): chunk = 3.0 s, HuBERT context = 0.72 s
(11520 @16k), crossfade = 0.1 s, hop = 2.9 s, jobs = 4 workers, each worker's
MRF runs 3 stacks in parallel (nested OMP). 45 s synth = 26.1 s (1.16x RT).

**The one-line answer**: chunking is currently costing us ~28 % redundant
compute (every chunk synthesizes 3.72 s but only contributes 2.9 s); HuBERT's
effective context is provably ~400 ms (we use 720 ms); and the MRF's 3 stacks
are a free batch dimension we're NOT exploiting — batching them through one
conv (input-stationary multiplex) is the textbook fix for memory-bound convs.

---

## A. Chunking optimization (the 28 % overhead)

### A1. The redundancy math (own, from wubu_rvc_cli.c:559-586)
Per chunk we run HuBERT + the full generator on `len_c = 48000 + 11520 =
59520` samples (3.72 s) but only `hop_16k = 46400` (2.9 s) is NEW audio. The
rest (0.82 s = 22 %) is re-synthesized context that gets crossfaded away.

```
efficiency = hop / len_c = 2.9 / 3.72 = 0.78  →  28 % of every chunk's
HuBERT + generator work is thrown away.
```

HuBERT is ~0.31x RT and the generator ~1.43x RT — the redundancy multiplies
BOTH. Cutting context and growing chunk size attacks the single biggest
remaining inefficiency.

### A2. HuBERT's effective context is short — 400 ms is enough (VERIFIED)
[arXiv 2505.22487 — Effective Context in Neural Speech Models, Meng et al.]
Measured via truncation + Jacobian probing: self-supervised transformers
(HuBERT, wav2vec 2.0, WavLM) have effective context *close to the supervised
phone model* — i.e. short. Streaming at **400 ms history/lookahead** costs
only **0.6–1.5 %** phone-probing degradation vs full utterances. **No
architecture change, no fine-tuning needed.**

Our fixed `extra_16k = 11520` (720 ms) is 1.8× the proven-sufficient context.
Shrinking to 6400 (400 ms): `efficiency = 2.9/3.4 = 0.85` → overhead 28 % → 17 %.
Shrinking to 4800 (300 ms): 0.88 → 14 %.

**Risk/DA flag**: RVC uses HuBERT layer 6 + speaker-invariance tweaks
(ContentVec-style), not the vanilla probing setup — the paper's 1.5 % is a
phone-classification proxy, not RVC output. **Mitigate**: make context a flag
(`--ctx`), A/B at 300/400/720 ms on the mel-corr + SNR gate. Expect no audible
difference at 400 ms; verify, don't trust.

### A3. Chunk size: bigger chunks win on compute, but HuBERT attention is O(T²)
- [Rochester — Streaming Voice Conversion Through Chunk-wise Training (Tian et
  al., NeuralVC, ECE477 2024)] **Table 3** (their frame chunk 3200 vs 320):
  - 3200-frame chunks: RTF 0.18–0.36, similarity 0.75–0.79, WER 0.27–0.43
  - 320-frame chunks: **RTF 2.706 (15× slower!)**, similarity 0.51, WER 1.009
    → tiny chunks are catastrophically slow AND worse quality (discontinuity).
  - Overlap study (3200): 0 % overlap no-buffer = fastest (RTF 0.178, sim
    0.785) but worst WER (0.427); **25 % overlap + ring buffer = best WER
    (0.269)** at RTF 0.358; **12.5 % overlap is the sweet spot** (RTF 0.328,
    WER 0.307).
- [Streaming Vocos (warisqr007)]: chunk = 5 mel frames = 100 ms buffer → RTF
  0.14 on CPU (~7× RT). "Smaller chunks reduce latency at the cost of slightly
  increased computational overhead." → for OFFLINE batch (our case) we want
  the OPPOSITE direction: fewer, larger chunks.

**Our tradeoff**: larger chunk → less % context waste (A1) + fewer HuBERT
re-extractions, BUT HuBERT's transformer attention is O(T²) so per-new-second
attention cost rises. The generator is linear — so the optimum sits where
generator redundancy savings beat the HuBERT quadratic growth. With HuBERT
only 0.31x RT vs generator 1.43x RT, larger chunks win. **Sweep --chunk
3/4.5/6/9 s** at a fixed 0.4 s context, jobs=4, same track; watch `[6] chunked
synth` + tail-imbalance (n_chunks % jobs).

**Crossfade**: 0.1 s on 3 s = 3.3 % overlap; Rochester says 12.5 % is the
quality sweet spot but 0 % is fastest — since we already carry the context
window, raising xfade to ~0.2–0.3 s only costs the hop delta (0.1–0.2 s per
chunk) and buys smoother seams. Test 0.1 vs 0.25 s by ear/mel-corr at seams.

### A4. Two-pass pipeline vs interleaved workers (stage demultiplexing)
Current: each worker does HuBERT → generator serially per chunk (4 workers ×
nested 3-stack = heavy oversubscription; measured flat). Alternative: **phase
separation** — pass 1 runs HuBERT on ALL chunks (independent, pure
transformer, no nested OMP → full 12 threads, no oversubscription), pass 2
runs the generator on ALL chunks (full threads again). This is the LLVC
lesson ([arXiv 2311.00873] — 2.8× RT CPU VC via stage pipelining) and the
vLLM "chunked prefill then decode" pattern: never interleave stages with
different ideal thread counts. Content-feature memory for 16 chunks ≈
256×372×16×4 B ≈ 6 MB — trivial.

---

## B. Sizing strategies (cache-aware tiling numbers for THIS CPU)

### B1. The Ryzen 5 3600 (Zen2) cache geometry
L1d = 32 KB/core, L2 = 512 KB/core, L3 = 32 MB total = **16 MB per CCX**
(2 CCX × 3 cores). DRAM ~ dual-channel DDR4-3200 ≈ 51 GB/s.

### B2. Cache-block sizing that actually works on Zen (BLIS-style)
[salykova.github.io/matmul-cpu (Zen3; Zen2 same hierarchy)] + [Goto/BLIS
anatomy papers]: tile so `mc×kc` fills L2, `kc×nR` fills L1. His measured Zen
block sizes: single-thread `nc=1535, mc=1024`; multithread `mc = mR×threads×5,
nc = nR×threads×50`. Rule: **parallel iterations ≥ threads × block** or cores
idle. For our conv1d (n=798, 32-sample register blocks = 25 j-tiles × OC-blks
= plenty), the OC-blocked register conv already tiles to ~13 KB working set
(4 oc × 192 ic × 3 taps × 4 B = 9.2 KB weights + 3.2 KB input row) — **fits
L1**. That's why it's 160×.

**Sizing levers left**:
1. **64-byte alignment + pad** the pooled arena slabs (pool_in/out/tmp in
   real.c:1273) — split cache lines cost on every slab; trivial to add.
2. **OC_BLK sweep 4/8** — 8 oc halves weight loads again but working set
   doubles past L1 (18.4 KB + input) — likely loses; measure.
3. **j-tile 32 → 64** (4 YMM accumulators → 8): halves the number of stores to
   orow and doubles weight reuse per load; accumulator register pressure is
   the limit (8 oc × 64 j × 8 floats = too many regs — so only with OC_BLK=4:
   4×64×8 = 32 YMM = overflow). 32 is likely optimal; the win now is
   multiplex, not bigger tiles.

### B3. Batching is THE lever for memory-bound kernels
[mlsysbook.ai/vol1/hw_acceleration — dataflow]: *"Batching is the primary
lever for converting a bandwidth-bound inference kernel into a compute-bound
one. At batch 1 every weight is read once and discarded; at batch 256 the same
weight row multiplies 256 inputs before being discarded."* Our convs are
measured memory-bound (threading flat at 8/6/4 — skill notes). More threads
won't help; **more FLOPs per byte loaded** will. That's the multiplex section.

---

## C. Multiplex strategies for CPU (the real prize)

### C1. MRF stack-multiplexed conv — input-stationary batch = 3 (OUR #1)
real.c:1253-1330: the 3 MRF stacks are independent convs with the **SAME
stage input, DIFFERENT weights**, run as 3 OMP threads each spawning 4 conv
threads (12 total, oversubscribed at jobs=4). Each stack's conv re-loads the
same input row from memory 3×.

**Fix — stack-batched conv1d**: one call, same kernel, `w[3]` weight pointers;
per (ic,tap) load `irow[j+off]` ONCE and FMA it against all 3 stacks' weights
(input-stationary dataflow, mlsysbook fig). Input traffic ÷3; the 3× FMAs
per load turn the arithmetic intensity up 3× — precisely the
memory-bound → compute-bound conversion. Implementation: extend the existing
register/OC-blocked conv with an outer `for s in 0..n_stacks` on the weight
pointer + accumulator array `a0s[3][...]`; the L1 working set becomes 3 × 9.2 KB
weights + 3.2 KB input ≈ 31 KB — **just fits L1** (keep OC_BLK=4).

**Verification (triple-DA)**: parity test vs the current 3-thread path —
extend t_conv_ab with the 3-stack shape (192→192, k=3, dil=1, n=798); require
maxdiff 0.000000 (same accumulation order per stack as today). Then jobs=4
chunked timing. Risk: register pressure (3× accumulators) — mitigation:
process stacks in pairs of 2 if 3 spills.

### C2. Weight-stationary worker pinning — share the generator weights
4 chunk workers all run the SAME generator weights. If the generator weight
set fits the 16 MB CCX L3 (it does — MRF ~12×3 convs × 192²×3×4 B ≈ 15 MB,
borderline), then **2 workers per CCX** (pin via
`SetThreadAffinityMask`/OMP_PROC_BIND) let both workers hit the same L3 copy
of the weights → DRAM weight traffic ÷2. Today with no pinning, workers drift
across CCX and may thrash both copies. The measured "threading flat" may be
partly this. Cheap, reversible: test `OMP_PROC_BIND=close OMP_PLACES=cores`.

### C3. Chunk-worker pool with work-stealing + tail fill (time multiplex)
Current: static pthread pool, atomic counter (already good). Gaps:
- **Tail imbalance**: 45 s / 2.9 s hop = 16 chunks / 4 workers = clean; but a
  3:47 track = 77 chunks → last wave idles 1-3 workers. Fix: split the tail
  chunk (the OLA writer already handles partial overlap) or let the last
  worker take the leftover hop.
- **Worker multiplex on the SAME chunk**: two workers split one chunk's
  generator (stack 1-2 vs 3) — not worth it; keep 1 chunk = 1 worker.

### C4. Realtime/multi-voice multiplex (the VST pattern — for the cohost)
The vendored RVCRealtimeVST (knowledge/rvc_original/RVCRealtimeVST) IS the
reference multiplex: **N worker processes, each owns one model instance;
shared-memory transport (RVCP protocol, 4096-byte header + double-buffered
float maps) + SPSC float rings** (SpscRing.hpp, power-of-2 capacity, relaxed
loads/release stores). When WuBuMedia needs 3 live voices (Cleveland, Peter,
Seth) simultaneously: one worker process per model + one ring per stream =
CPU time-multiplex across models, memory-buffered latency hiding. The
architecture doc for that: wubudesk-cohost-runtime skill. Batch-multiplexing
across DIFFERENT models is NOT worth it (no weight sharing) — only
time-multiplex.

### C5. Batched HuBERT (space multiplex across chunks)
The HuBERT transformer (12 layers, C=768) on 16 chunks: each chunk's attention
is independent — **batched attention amortizes the Q/K/V + FFN weight loads
across chunks** (same mlsysbook argument). This is a bigger code change
(batch dim through the MHA) but HuBERT is 0.31x RT — after A2 (context cut)
and C1 the generator stops dominating; batched HuBERT becomes the next
bottleneck. Document now, implement after C1.

---

## Recommended execution order (value / effort)
1. **`--ctx` flag + default 0.4 s** (A2): ~10 % pipeline, 1-line change + A/B. ✅ cheap
2. **MRF stack-multiplexed conv** (C1): input traffic ÷3 on the dominant
   stage. Medium; parity-gated. **THE win.**
3. **--chunk sweep 3/4.5/6/9 + xfade 0.25** (A3): pick the real optimum on the
   boss's 45 s track; also fixes the 28 % overhead. Cheap.
4. **Worker pinning OMP_PROC_BIND=close** (C2): free win if L3-resident; test.
5. **Two-pass pipeline** (A4): kills the nested-OMP oversubscription entirely.
   Medium; big architectural cleanliness win.
6. **Batched HuBERT** (C5): after C1 changes the bottleneck. Later.

## Sources
- arXiv 2505.22487 (Effective Context in Neural Speech Models) — HuBERT 400 ms streaming, 0.6-1.5 % degradation
- Tian et al., Rochester ECE477 2024 PDF (Streaming VC chunk-wise) — Table 3 chunk/overlap/RTF
- warisqr007/StreamingVocos (HF) — chunk=5 mel frames → RTF 0.14 CPU
- salykova.github.io/matmul-cpu — Zen cache-block sizes, multithread rules
- mlsysbook.ai/vol1/hw_acceleration — weight/input-stationary, batching converts memory-bound → compute-bound
- arXiv 2311.00873 (LLVC) — stage pipelining 2.8× RT CPU VC
- RVCRealtimeVST (vendored in knowledge/rvc_original/) — worker-process multiplex, SPSC rings, RVCP shared memory
- vLLM docs (chunked prefill) — stage separation precedent
