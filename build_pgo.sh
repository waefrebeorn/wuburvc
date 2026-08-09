#!/usr/bin/env bash
# build_pgo.sh — PGO build: profile-generate → run → profile-use.
# Usage: bash build_pgo.sh   (writes build/wubu_rvc_pgo.exe)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PATH="/c/msys64/mingw64/bin:/c/msys64/usr/bin:$PATH"
GCC=/c/msys64/mingw64/bin/gcc.exe
CUDART=$(cygpath -m "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64/cudart.lib")
export TMP="$(pwd)/build/tmp" TEMP="$(pwd)/build/tmp" TMPDIR="$(pwd)/build/tmp"
mkdir -p build/tmp
RVC_SRCS="src/wubu_rvc.c src/wubu_rvc_parity.c src/wubu_rvc_weights.c \
          src/wubu_rvc_kernels_exact.c src/wubu_rvc_real.c src/wubu_rvc_hubert.c \
          src/wubu_rvc_f0.c src/wubu_postproc.c src/wubu_rmvpe.c \
          src/wubu_stft.c src/wubu_gru.c src/wubu_audioio.c \
          src/wubu_pitch.c src/wubu_autokey.c src/wubu_fft.c src/wubu_math.c src/wubu_vk.c"
COMMON="-std=c11 -O3 -mavx2 -mfma -fopenmp -pthread -I src"

echo "=== step 1: profile-generate ==="
rm -f build/*.gcda build/*.gcno
"$GCC" $COMMON -fprofile-generate src/wubu_rvc_cli.c $RVC_SRCS build/wubu_rvc_cuda.o \
    -lm "$CUDART" -lvulkan-1 -o build/wubu_rvc_pg.exe

echo "=== step 2: profile run (WuBuMedia track) ==="
WM=/c/Users/eman5/WuBuMedia
PTH=$(ls "$WM/models/rvc/cartman/"*.pth | head -1)
"$WM/build/wubu_rvc_pg.exe" "$WM/out/demo/album_cart_45s.wav" "$WM/models/rvc/cartman" \
    "$WM/out/demo/pgo_prof.wav" --model "$PTH" --noise 0.33333 --jobs 4 2>&1 | grep -E "\[6\] chunked"
ls build/*.gcda >/dev/null 2>&1 && echo "gcda collected" || echo "NO GCDA"

echo "=== step 3: profile-use ==="
"$GCC" $COMMON -fprofile-use -fprofile-correction src/wubu_rvc_cli.c $RVC_SRCS build/wubu_rvc_cuda.o \
    -lm "$CUDART" -lvulkan-1 -o build/wubu_rvc_pgo.exe 2>&1 | grep -iE "error|warning: profile" | head -5 || true
echo "PGO_DONE: build/wubu_rvc_pgo.exe"
ls -la build/wubu_rvc_pgo.exe
