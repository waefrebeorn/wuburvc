#!/usr/bin/env bash
# build_fast.sh — WuBuRVC fast build (AVX2+FMA+OpenMP, CUDA+Vulkan linked).
# Usage: bash build_fast.sh [output_name]   (default wubu_rvc_fast.exe)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
GCC=/c/msys64/mingw64/bin/gcc.exe
OUT="${1:-wubu_rvc_fast}"
CUDART=$(cygpath -m "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64/cudart.lib")
export TMP="$(pwd)/build/tmp" TEMP="$(pwd)/build/tmp" TMPDIR="$(pwd)/build/tmp"
mkdir -p build/tmp
RVC_SRCS="src/wubu_rvc.c src/wubu_rvc_parity.c src/wubu_rvc_weights.c \
          src/wubu_rvc_kernels_exact.c src/wubu_rvc_real.c src/wubu_rvc_hubert.c \
          src/wubu_rvc_f0.c src/wubu_postproc.c src/wubu_rmvpe.c \
          src/wubu_stft.c src/wubu_gru.c src/wubu_audioio.c \
          src/wubu_pitch.c src/wubu_autokey.c src/wubu_fft.c src/wubu_math.c src/wubu_vk.c"
"$GCC" -std=c11 -O3 -mavx2 -mfma -fopenmp -pthread -I src \
    src/wubu_rvc_cli.c $RVC_SRCS build/wubu_rvc_cuda.o \
    -lm "$CUDART" -lvulkan-1 -o "build/$OUT.exe" 2>&1 | grep -iE "error|undefined" | head -5 || true
echo "BUILD_DONE: build/$OUT.exe"
ls -la "build/$OUT.exe"
