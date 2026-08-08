# Engine Optimization Research — making WuBuRVC the better engine

7-step online research (2026-08-08) on making the C11 RVC engine faster AND
higher quality. Mandate: "we made our own engine, now we make it the better
engine."

## Step 1 — RVC real-time deployment (Wikipedia/RVC + Advanced-RVC-Inference)
- The reference achieves **90 ms end-to-end real-time** via ONNX/TensorRT GPU
  exports. GPU inference is THE canonical speedup for RVC.
- Advanced-RVC-Inference: real-time live conversion with VAD + low-latency
  processing.

## Step 2 — Low-latency/streaming VC (StreamVC, Google 2024)
- StreamVC: streaming voice conversion — process audio in small frames, match
  timbre while preserving content/prosody. The architecture for live use.
- LLVC (2023): voice conversion with just nearest neighbors — the FAISS path
  is the efficient retrieval core (we already use it).

## Step 3 — HiFi-GAN/NSF vocoder acceleration
- The RVC generator IS a HiFi-GAN-variant (MRF resblocks). Its convs dominate
  CPU inference. Known accelerations: fused conv+act, weight packing,
  streaming chunking (done — 3s chunks).

## Step 4 — Attention acceleration (flash attention)
- Flash attention: 2-4x + 10-20x memory for O(T^2) transformer attention.
- Applies to HuBERT (12 layers) and the flow's enc_p. Our chunking already
  reduces the sequence from 10k frames to ~186 per chunk (the O(T^2) is
  effectively neutralized); flash-style tiling is a further win for the
  non-chunked path.

## Step 5 — CPU GEMM/conv optimization (SIMD + cache blocking)
- Cache-blocked GEMM with packed panels (L1/L2/L3 aware) + AVX2/FMA kernels
  = 10-50x for matmul-bound code (salykova.github.io/matmul-cpu).
- im2col+GEMM beats naive SIMD conv; direct conv with loop reordering is close
  for small kernels.
- IMPLEMENTED: input-stationary tiled conv1d (sequence tiles, all output
  channels per tile) + AVX2/FMA micro-kernels in conv1d_c (stride=1) and
  linear_c. Parity preserved (HUBERT + SYNTH PASS). Synth 2.39x → 1.81x
  realtime on the 45s benchmark; total 45s conversion 2:20 → 1:54.

## Step 6 — INT8 quantization
- INT8: 1.4-2.4x (up to 10x packed) with small quality loss; needs per-channel
  scales + calibration (TensorRT/QAT for accuracy).
- NEXT: int8 weights for the generator convs (the biggest model component) —
  the convs are tolerance-heavy; per-channel scale keeps quality.

## Step 7 — Quality techniques (RMVPE/FAISS/annotated RVC)
- RVC uses COARSE f0 in the prior encoder (discretized bins) + fine nsff0 in
  the NSF generator — we match this exactly (verified).
- FAISS retrieval for timbre consistency (index_rate blend) — implemented.
- RMVPE is the state-of-the-art singing f0 (87.2% accuracy) — implemented.
- The annotated RVC walkthrough confirms: HiFi-GAN-style adversarial training
  for the decoder = the quality ceiling; our LeakyReLU-vs-Snake handling
  preserves the pretrained behavior.

## Decision — the faster-engine roadmap
1. DONE: chunked inference (3s + 0.72s context + crossfade, pthread pool).
2. DONE: AVX2/FMA conv+linear kernels (-mavx2 -mfma).
3. NEXT: int8 quantized generator weights (per-channel).
4. NEXT: CUDA generator + flow kernels (RTX 2080 SUPER sm_75) — the 10-50x
   jump to faster-than-realtime. The WuBuRVC Makefile.win already has the
   sm_75 CUDA targets.
5. NEXT: streaming mode (StreamVC-style) for live conversion.
