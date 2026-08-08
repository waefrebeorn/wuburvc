#!/usr/bin/env bash
# WuBuRVC quickstart — convert a vocal through any RVC v2 model.
# Usage: bash examples/convert.sh input.wav model_dir output.wav [model.pth]
set -e
cd "$(dirname "$0")/.."

IN="${1:?input.wav}"
MDIR="${2:?model dir, e.g. models/rvc/cartman}"
OUT="${3:?output.wav}"
PTH="${4:-$(ls "$MDIR"/*.pth 2>/dev/null | head -1)}"

echo "WuBuRVC: $IN -> $OUT  (model $PTH)"
./build/wubu_rvc "$IN" "$MDIR" "$OUT" --model "$PTH" --noise 0.33333
echo "done: $OUT"
