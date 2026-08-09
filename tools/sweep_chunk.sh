#!/usr/bin/env bash
# Sweep chunk/ctx/xfade/jobs on the standard 45s demo track.
# Saves every output WAV under out/demo/sweep/ for the quality gate.
set -u
cd "$(dirname "$0")/.."
BIN=./build/wubu_rvc_fast.exe
IN=out/demo/album_cart_45s.wav
MD=models/rvc/cartman
PTH=$(ls "$MD"/*.pth | head -1)
mkdir -p out/demo/sweep

run() { # name chunk ctx xfade jobs
  local name=$1 chunk=$2 ctx=$3 xfade=$4 jobs=$5
  local out="out/demo/sweep/${name}.wav"
  local line
  line=$("$BIN" "$IN" "$MD" "$out" --model "$PTH" --noise 0.33333 \
         --chunk "$chunk" --ctx "$ctx" --xfade "$xfade" --jobs "$jobs" 2>&1 \
         | grep -E '\[6\] chunked')
  echo "$name | $line"
}

echo "=== SWEEP (chunk x ctx) jobs=4 xfade=0.1 ==="
run base_3s_072  3.0 0.72 0.10 4
run c3_ctx03     3.0 0.30 0.10 4
run c3_ctx04     3.0 0.40 0.10 4
run c3_ctx05     3.0 0.50 0.10 4
run c45_ctx03    4.5 0.30 0.10 4
run c45_ctx04    4.5 0.40 0.10 4
run c45_ctx072   4.5 0.72 0.10 4
run c6_ctx03     6.0 0.30 0.10 4
run c6_ctx04     6.0 0.40 0.10 4
run c6_ctx072    6.0 0.72 0.10 4
run c9_ctx04     9.0 0.40 0.10 4
echo "=== XFADE + JOBS (best candidates) ==="
run c6_ctx04_xf25 6.0 0.40 0.25 4
run c45_ctx04_xf25 4.5 0.40 0.25 4
run c3_ctx04_xf25  3.0 0.40 0.25 4
run c6_ctx04_j6    6.0 0.40 0.10 6
run c3_ctx04_j6    3.0 0.40 0.10 6
echo "=== DONE ==="
