#!/usr/bin/env bash
# time_rmvpe.sh — coarse RMVPE stage timing via WUBU_RMVPE_DUMP + wall clock.
# Builds a tiny instrumented driver? No — reuse the CLI: RMVPE runs before [6].
# We time the whole pre-[6] wall and subtract model load.
set -e
export PATH="/c/msys64/mingw64/bin:/c/msys64/usr/bin:$PATH"
cd /c/Users/eman5/WuBuMedia
EXE=./build/wubu_rvc_zn.exe
echo "=== RMVPE with 12 threads (current) ==="
for i in 1 2; do
  t0=$(date +%s.%N)
  $EXE out/demo/album_cart_45s.wav models/rvc/cartman out/demo/r$i.wav --model models/rvc/cartman/*.pth --noise 0.33333 --jobs 4 --mode speed >/dev/null 2>&1
  t1=$(date +%s.%N)
  echo "run $i wall: $(echo "$t1 - $t0" | bc)s"
done
echo "=== RMVPE with 1 thread (scaling check) ==="
t0=$(date +%s.%N)
OMP_NUM_THREADS=1 $EXE out/demo/album_cart_45s.wav models/rvc/cartman out/demo/r1t.wav --model models/rvc/cartman/*.pth --noise 0.33333 --jobs 4 --mode speed >/dev/null 2>&1
t1=$(date +%s.%N)
echo "1-thread wall: $(echo "$t1 - $t0" | bc)s"
