# 50 CPU Speed Fixes — RVC Engine (researched 2026-08-08, 30 web searches)

The boss's "seven steps to Kevin Bacon" research cascade: conv micro-kernels →
GEMM → memory layout → threading → precision → math → compiler → pipeline.
Every fix cites its source. Items marked **DONE** are already in wuburvc.

## A. Conv micro-kernel (10)
1. **im2col + GEMM** — conv as matmul; high-perf impls reach 60 GFLOP/s on
   consumer cores. [sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution]
2. **Register-accumulated microkernel** — taps accumulate in YMM, output
   touched once per block. **DONE**: 160x on our conv1d, bit-exact.
3. **Winograd F(2,3)** — 1.5-2x fewer multiplies on 3-tap convs; LoWino
   F(4,3) up to 2.04x over oneDNN. [dl.acm.org 10.1145/3472456.3472464]
4. **FFT convolution** — faster than direct once the impulse response
   exceeds ~50-128 samples (our upsample kernels are 16 taps — not yet;
   the MRF k=3/7/11 — no). [dsp.stackexchange 71278]
5. **Output-channel blocking** — 4+ output channels share each input load
   (input traffic ÷4). First attempt failed parity; the register math
   needs the a0[OC_BLK] check. [NVIDIA CUTLASS efficient_gemm]
6. **Multi-level cache tiling** — block for L1/L2/L3 like CUTLASS's
   threadblock→warp→instruction hierarchy. [docs.nvidia.com cutlass]
7. **Direct conv beats im2col for small channels** — direct is competitive
   when in_ch·k is small; pick per-layer. [sciencedirect S1383762122002910]
8. **Software prefetch** — prefetch the next input tile while FMAs run.
   [oneDNN conv docs; CUTLASS mainloop]
9. **2+ independent FMA chains** — hide FMA latency (each chain is
   dependent; 2 chains = 2x ILP). [deep-kondah high-performance GEMM]
10. **Kernel specialization per (k, dil, stride)** — one hot path per shape
    instead of generic bounds. [XNNPACK microkernel decomposition]

## B. Memory / layout (8)
11. **Channels-last (NHWC)** — convs prefer channel-contiguous activations
    on CPU; oneDNN/TF default. [intel-extension-for-pytorch nhwc]
12. **Weight repacking once** — XNNPACK's "weights cache" repacks static
    weights at load; never during inference. [blog.tensorflow 2022-06]
13. **Buffer pooling** — replace per-(stack,pair) malloc/free with a
    preallocated arena (MemoMalloc). [mailman.cs.uchicago MemoMalloc]
14. **64-byte cache-line alignment** for all AVX buffers. [XNNPACK]
15. **Huge pages (2MB)** for the 110MB weights.bin.
16. **mmap read-only weights** — llama.cpp loads GGUF via mmap; no page
    faults in the hot loop. [medium llama.cpp CPU inference]
17. **Fuse activations into the conv epilogue** — one pass instead of
    conv-write + separate lrelu read/write. [oneDNN fused primitives]
18. **Kill redundant memcpy** — MRF's rb_in copy + copy-back are 2 full
    tensor copies per pair; reuse buffers in place. [own profile]

## C. Threading (8)
19. **Thread-count tuning** — llama.cpp: 4 threads beat 32 on consumer
    chips; sweep 1/2/4/6. [reddit r/LocalLLaMA 190v426]
20. **OMP_THREAD_LIMIT + OMP_MAX_ACTIVE_LEVELS** to stop nested
    oversubscription (4 workers × 3 stacks × 4 = 48 on a 12-thread CPU).
    [docs.oracle.com nested parallelism]
21. **Undersubscription** — Intel: undersubscribe each nested level for
    better overall throughput. [community.intel 813536]
22. **Work-stealing pool** for the chunk workers instead of raw pthreads.
    [joeduffyblog work-stealing queue; wikipedia]
23. **Core pinning / NUMA affinity** — NUMA pinning 340→412 inf/s (+21%).
    [oxmaint numa-aware scheduling]
24. **Static schedule for balanced conv tiles** (identical tile costs) vs
    dynamic. [OpenMP schedule semantics]
25. **Spin/busy-wait** for short per-op critical sections (avoid futex
    sleeps at chunk boundaries). [llama.cpp spin locks]
26. **One worker per physical core, not SMT thread**, for memory-bound
    stages. [local-LLM optimization guide]

## D. Precision / quantization (7)
27. **INT8 per-channel conv** — W8A8 typical 1.4-2.4x on AVX2 (our prior
    research table). [openreview SklzIjActX]
28. **W4A8 mixed** — up to 3x over W4A16, 1.4x over W8A8. [CVPRW 2025
    Dual Precision Quantization]
29. **BF16/FP16 weights with FP32 accumulate** — TF reports 2x on-device;
    F16C converts cheaply. [blog.tensorflow 2023-11 half precision]
30. **Weight-only quant (GGUF-style)** — 4-bit weights + dequant in the
    kernel; memory-bound stages win. [medium llama.cpp]
31. **Groupwise quantization** — groups of 32/256 share a scale (Q4_K
    superblocks). [emergentmind llama.cpp]
32. **Block floating point** — per-block scales, 2.6-3.5 bits/weight.
    [emergentmind llama.cpp]
33. **Quantized intermediates** (like KV-cache q8) for the MRF's residual
    chains. [emergentmind llama.cpp]

## E. Math / activation (6)
34. **Fast exp/softmax via IEEE-754 bit tricks** — 10-35% total inference
    speedup (QuAKE). [github kkokosa/dotLLM#55]
35. **Vectorized sinf via polynomial** — 50x atan2f-style wins on
    transcendental-heavy code (our Snake activation when enabled).
    [mazzo.li/posts/vectorized-atan2]
36. **FTZ + DAZ flush denormals** — Intel: set MXCSR bits; no FP-assist
    slowdowns. **DONE** in wubu_rvc_load. [intel onemkl denormals]
37. **FMA contraction** (`-ffp-contract=fast`) — fuse mul+add into one
    instruction. [gcc docs]
38. **Vectorized softmax with AVX** — Intel Xeon softmax optimizations
    (max-subtract, exp via bit tricks). [simplecore.intel softmax]
39. **Approximate GELU/sigmoid** (tanh form) for the encoder FFN — the
    codebase already uses the exact tanh form; a poly is 5-10x cheaper.

## F. Compiler / runtime (6)
40. **-march=native / -mavx2 -mfma** — **DONE** (explicit -mavx2 -mfma).
41. **PGO** — typical 5-30%; MSVC /LTCG:PGI, Clang AutoFDO. [learn.microsoft PGO]
42. **-flto** — cross-TU inlining of the hot kernels. [gcc docs]
43. **-ffast-math** — careful with parity (the MRF's chaotic float
    amplification); test per-layer, not globally.
44. **-funroll-loops** for the tile loops (the compiler already unrolls
    the register blocks).
45. **mimalloc/jemalloc** — cut the 72×/track malloc/free overhead.
    [mimalloc repo]

## G. Pipeline / streaming (5)
46. **Chunked inference with overlap-fade** — 16-32 frame chunks with
    crossfade; **DONE** (3s chunks + 0.1s crossfade). [emergentmind TTS]
47. **Causal streaming vocoder** — Streaming Vocos RTF 0.14 (~7x realtime
    on CPU) via stateful causal decoding. [huggingface StreamingVocos]
48. **Stage pipelining** — overlap HuBERT/f0/generator; LLVC runs 2.8x
    faster-than-realtime on a consumer CPU this way. [arxiv 2311.00873]
49. **Smaller chunk buffers for latency** — latency is buffering-dominated,
    not compute-dominated (Streaming Vocos: 3.4ms/chunk). [StreamingVocos]
50. **Async I/O + double buffering** — read the next chunk while the
    workers process the current one. [switchboard.audio voice-ai-latency]

## The RVC-specific answer (from the community)
RVC 10.6's "secret sauce" (issue #1434, yxlllc): the speed was always
there on GPU; CPU users saw 1:1 because of **pitch extractor choice**
(RMVPE is fast, CREPE/Harvest are slow) and config, not magic kernels.
Our CPU path already beats the stock RVC: RMVPE + the register conv.
[github RVC-Project issue 1434]
