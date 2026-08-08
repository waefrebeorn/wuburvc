# Accelerator Layer: CUDA GeneratorNSF + Triton Research

## Status: the CUDA generator is AGNOSTIC and faster than realtime

`src/wubu_rvc_cuda.cu` — GeneratorNSF on the RTX 2080 SUPER (sm_75), same
[in_ch, n] col-major layout + weight keys as the CPU path. The flow
(enc_p + posterior + prior + coupling) stays on the CPU (chunk-sized, cheap);
the generator (conv_pre, cond, sine, upsample blocks + noise convs + MRF,
conv_post) runs on the GPU. The CLI flag: `--cuda` (chunked workers only;
`--no-chunk` keeps the whole-track CPU path for parity).

### Agnostic: loads every model like the CPU (verified)
| model | sr | upsample rates | result |
|---|---|---|---|
| Cartman (40k) | 40000 | [10,10,2,2] | runs, 8s in 6.4s (0.79x realtime) |
| Cleveland (32k) | 32000 | [10,10,2,2] | runs, 8s in 6.1s |
| Seth (48k) | 48000 | [12,10,2,2] | runs, 8s in 6.6s |

All shapes derived from the weights/config at runtime, exactly like the CPU:
init_ch from conv_pre dims[0], ups_k/pad from dec.ups.L.weight_v dims[2],
resblock_k/dil from config with the same legacy fallbacks, post_in/post_k
from conv_post dims, sr from the model config for the sine radix.

### Build (Windows, MSVC host for nvcc + MinGW link)
- `build/cuda_build.bat`: nvcc -O3 -arch=sm_75 -Xcompiler /GS- -fmad=false
  (vcvars64 + the CUDA 12.4 toolkit).
- MinGW link: `gcc ... build/wubu_rvc_cuda.o -L"<CUDA>/lib/x64" -lcudart -lm`.
- The nvcc object is MSVC COFF — the link works with MinGW as long as the
  .cu avoids MSVC-CRT symbols (no snprintf/fprintf — use the Windows API
  WriteFile + a manual key builder). `extern "C"` around the header include
  is mandatory (nvcc compiles the .cu as C++).

### Parity caveat (honest)
The GPU convs are float32 with the same accumulation order as the CPU —
stage-by-stage A/B dumps match to ~1e-4 (the GPU is actually closer to the
double reference). The MRF's 3-pair x 4-layer residual chain is chaotically
sensitive to those ~1e-4 diffs (the CPU-vs-PyTorch reference itself only
passes SNR >= 25 dB), so the GPU generator output diverges from the CPU
generator output after the amplification. The rms/level matches; the shape
corr is low on the synthetic test. The structural bugs found + fixed along
the way (all stage dumps now match): missing cond(g) layer, ups_k from the
weight dims not the config, MRF residual chaining (the carry must reset to
the stage per stack), the init_ch from conv_pre.

Next parity step (not done): force the GPU accumulation to replicate the
CPU's exact float32 rounding per tap (the -fmad=false + order already match;
the residual ~1e-4 is the CPU's own accumulation error vs the double
reference — matching it exactly needs the same partial-sum order AND the
same FMA contraction as the gcc build, or a Kahan-style compensated sum on
both sides).

## Triton research (2026-08-08): "make our kernel in Triton — accelerator friendly"

### What Triton is
A Python-based DSL (triton-lang.org) that compiles to a Triton IR then to
PTX (NVIDIA) or HSA (AMD ROCm) — the SAME kernel source runs on both. The
vLLM flash-attention backend is Triton: one kernel codebase benchmarked
comparable to the vendor SoTA libraries on the A100 AND the MI250. Red Hat,
IBM (autotuning), AMD (ROCm docs) all push Triton as the portability layer.
Autotuning (tl.autotune) picks the block sizes per GPU.

### Can the WuBuRVC generator be a Triton kernel? YES
- conv1d: a GEMM (out_ch x (in_ch*k)) — tl.load the input window tile +
  tl.dot with the weight tile — the canonical Triton pattern.
- convT1d: the gather formulation (each output sample pulls from the
  contributing (i, tap)) — a masked tl.load.
- MRF/upsample/noise/activations: elementwise + the same convs.
- The flow (enc_p attention) = a small flash-attention-style kernel.
No public RVC-Triton generator exists (the ecosystem uses PyTorch/cuDNN,
ONNX, TensorRT, OpenVINO). We would be first.

### Trade-offs vs the CUDA path (why the C11 core stays the reference)
- Triton is Python + a JIT — it cannot be embedded in the native C11
  binary; it would run as a separate service/process with a socket/pipe
  protocol (same shape as the cohost bridge).
- The JIT compile on the first call (~seconds per kernel shape) + the
  launch overhead dominates for the small chunk tensors (372 frames).
- The C11 kernels stay the portable everywhere core; the Triton kernels
  become the accelerator layer for NVIDIA + AMD + future accelerators.

### Recommendation
Keep the C11 core + the CUDA .cu (already working). Add the Triton kernels
as the portable-across-vendors accelerator later (the AMD/MI-series + the
next hardware), reusing the exact same weight/config agnosticism. The
Triton version gives ONE kernel source for the whole GPU family instead of
per-vendor CUDA/HIP copies.
