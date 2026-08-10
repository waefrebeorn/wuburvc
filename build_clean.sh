#!/usr/bin/env bash
# build_clean.sh — STANDARDIZED clean build for WuBuRVC (wuburvc repo).
#
# Why this exists: manual `bash build_fast.sh` runs left ~40 stale .exe/.o/.gcda
# artifacts in build/ and the SPIR-V headers could go stale relative to the
# .comp sources — making verification ambiguous (two different shader versions
# produced a bit-identical exe size). This script makes the build
# deterministic:
#
#   1. DELETE all build artifacts (exe/o/gcda/tmp/spv/logs) — nothing survives
#      from a previous build except cuda_build.bat (a source script).
#   2. REGENERATE the SPIR-V headers from src/*.comp (glslangValidator).
#   3. REBUILD the CUDA generator object (nvcc via cuda_build.bat).
#   4. LINK fresh (gcc, AVX2/FMA, OpenMP, cudart, vulkan) with a clean TMP.
#   5. COPY the exe to WuBuMedia/build/ (the runtime install location).
#
# Usage (from wuburvc root or anywhere):
#   bash build_clean.sh [target] [flags]
#     target  wubu_rvc_vk | wubu_rvc_fast | wubu_rvc_fix   (default wubu_rvc_vk)
#     flags   --no-spv   skip SPIR-V regeneration (only if .comp unchanged)
#             --no-cuda  skip nvcc (use existing build/wubu_rvc_cuda.o)
#             --no-copy  do not install (keeps exe in build/)
#             --bench    also build bench_conv3.exe (conv correctness harness)
#   bash build_clean.sh clean          # delete ALL build artifacts, keep sources
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TARGET="${1:-wubu_rvc_vk}"
FLAGS=""
[ $# -gt 1 ] && FLAGS="${*:2}"

GCC=/c/msys64/mingw64/bin/gcc.exe
CUDART=$(cygpath -m "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64/cudart.lib")
VULKAN_LNK=-lvulkan-1
ARCH="-mavx2 -mfma -mbmi2 -march=znver2"

RVC_SRCS="src/wubu_rvc.c src/wubu_rvc_parity.c src/wubu_rvc_weights.c \
          src/wubu_rvc_kernels_exact.c src/wubu_rvc_real.c src/wubu_rvc_hubert.c \
          src/wubu_rvc_f0.c src/wubu_postproc.c src/wubu_rmvpe.c \
          src/wubu_stft.c src/wubu_gru.c src/wubu_audioio.c \
          src/wubu_pitch.c src/wubu_autokey.c src/wubu_fft.c src/wubu_math.c src/wubu_vk.c \
          src/wubu_harmony.c src/wubu_consonant.c src/wubu_breath.c \
          src/wubu_train.c src/wubu_train_vk.c"

log() { printf '\033[1;36m[build]\033[0m %s\n' "$*"; }

# nvcc (cudafe) needs a writable TMP dir or it fails with NVCC_RC=2 silently.
# Export BEFORE the nvcc step — the earlier version exported TMP only before the
# gcc link (step 4), so nvcc ran with empty TMP and died.
mkdir -p build/tmp
export TMP="$(pwd)/build/tmp" TEMP="$(pwd)/build/tmp" TMPDIR="$(pwd)/build/tmp"

if [ "$TARGET" = "clean" ]; then
    log "clean: deleting ALL build artifacts (keeping sources + cuda_build.bat)"
    rm -rf build/tmp build/spv
    rm -f build/*.exe build/*.o build/*.gcda build/*.gcno build/*.spv
    rm -f build/*.log build/err*.log build/gcc_err.txt build/v.log
    log "clean done — build/ now holds only sources/scripts:"
    ls build/ 2>/dev/null | grep -v cuda_build.bat || true
    exit 0
fi

# ── 1. DELETE stale artifacts ────────────────────────────────────────────────
log "step 1/5: deleting stale build artifacts"
mkdir -p build/tmp build/spv
rm -f build/*.exe build/*.o build/*.gcda build/*.gcno build/*.spv
rm -f build/err*.log build/gcc_err.txt build/v.log
# keep cuda_build.bat (source script), bench sources live at repo root

# ── 2. REGENERATE SPIR-V ─────────────────────────────────────────────────────
if [[ " $FLAGS " != *" --no-spv "* ]]; then
    log "step 2/5: regenerating SPIR-V from src/*.comp"
    bash tools/gen_spv.sh
else
    log "step 2/5: SKIP SPIR-V regeneration (--no-spv)"
fi

# ── 3. REBUILD CUDA object ───────────────────────────────────────────────────
if [[ " $FLAGS " != *" --no-cuda "* ]]; then
    log "step 3/5: rebuilding CUDA generator object (nvcc sm_75)"
    # MSYS_NO_PATHCONV=1 stops git-bash mangling cmd.exe /c into an interactive
    # session (MSYS converts "/c" to "C:\" — cmd then prints a prompt banner and
    # never runs the bat; NVCC_RC stays unknown and no .o is produced).
    MSYS_NO_PATHCONV=1 cmd.exe /c "build\\cuda_build.bat" > build/cuda_build.log 2>&1 || true
    NVCC_RC=$(grep -o "NVCC_RC=[0-9]*" build/cuda_build.log | tail -1 | cut -d= -f2)
    if [ ! -f build/wubu_rvc_cuda.o ] || [ "${NVCC_RC:-1}" != "0" ]; then
        echo "[build] ERROR: nvcc failed (NVCC_RC=${NVCC_RC:-unknown})" >&2
        tail -20 build/cuda_build.log >&2
        exit 1
    fi
    log "cuda object OK (NVCC_RC=0, $(stat -c%s build/wubu_rvc_cuda.o) bytes)"
else
    log "step 3/5: SKIP nvcc (--no-cuda, reusing build/wubu_rvc_cuda.o)"
    [ -f build/wubu_rvc_cuda.o ] || { echo "[build] ERROR: no cuda .o (run without --no-cuda once)" >&2; exit 1; }
fi

# ── 4. LINK FRESH ────────────────────────────────────────────────────────────
log "step 4/5: linking ${TARGET}.exe (gcc $ARCH)"
export TMP="$(pwd)/build/tmp" TEMP="$(pwd)/build/tmp" TMPDIR="$(pwd)/build/tmp"
"$GCC" -std=c11 -O3 -fopenmp -pthread -I src $ARCH \
    src/wubu_rvc_cli.c $RVC_SRCS build/wubu_rvc_cuda.o \
    -lm "$CUDART" "$VULKAN_LNK" -o "build/${TARGET}.exe"
log "linked: $(ls -la build/${TARGET}.exe | awk '{print $5}') bytes"
# fail loudly if any referenced symbol is missing (gcc reports at link time)

# ── 4b. REBUILD MIXMASTER (album pipeline dependency) ────────────────────────
# Step 1's `rm -f build/*.exe` deletes wubu_mixmaster.exe, and the album
# pipeline (tools/album_build.py / wubu_album_pipeline.py) hard-fails without
# it — so a clean build MUST recreate it or the next album run breaks. The
# Makefile.win recipe is the canonical build (C11, O2, no OpenMP needed).
if [[ " $FLAGS " != *" --no-mixmaster "* ]]; then
    log "step 4b: rebuilding wubu_mixmaster.exe (album mix dependency)"
    "$GCC" -std=c11 -O2 -I src tools/wubu_mixmaster.c \
        src/wubu_master.c src/wubu_audioio.c -lm \
        -o build/wubu_mixmaster.exe
    log "mixmaster linked: $(ls -la build/wubu_mixmaster.exe | awk '{print $5}') bytes"
else
    log "step 4b: SKIP mixmaster rebuild (--no-mixmaster)"
fi

# ── 5. INSTALL ───────────────────────────────────────────────────────────────
if [[ " $FLAGS " != *" --no-copy "* ]]; then
    log "step 5/5: installing to build/install/"
    mkdir -p build/install
    cp "build/${TARGET}.exe" build/install/
    log "installed: build/install/${TARGET}.exe"
else
    log "step 5/5: SKIP install (--no-copy)"
fi

# optional bench
if [[ " $FLAGS " == *" --bench "* ]]; then
    log "extra: building bench_conv3.exe (conv correctness + timing)"
    "$GCC" -std=c11 -O2 -fopenmp -I src bench_conv3.c src/wubu_vk.c \
        src/wubu_rvc_real.c src/wubu_rvc_weights.c src/wubu_math.c src/wubu_fft.c \
        build/wubu_rvc_cuda.o -lm "$CUDART" "$VULKAN_LNK" -o build/bench_conv3.exe
    log "extra: bench_conv3.exe built"
fi

log "DONE: build/${TARGET}.exe is fresh (no stale artifacts survive)."
