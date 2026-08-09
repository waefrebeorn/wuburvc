#!/usr/bin/env bash
# ab_5models_fixed.sh — regenerate the 5-clip A/B with the FIXED median filter.
set -e
export PATH="/c/msys64/mingw64/bin:/c/msys64/usr/bin:$PATH"
cd /c/Users/eman5/WuBuMedia
EXE=./build/wubu_rvc_f0fix.exe
OUTDIR=out/demo/ab
mkdir -p "$OUTDIR"

run() {
  local clip="$1" name="$2" dir="$3" pth="$4" mode="$5" out="$6"
  "$EXE" "$OUTDIR/$clip" "$dir" "$out" --model "$pth" --noise 0.33333 --jobs 4 ${mode} 2>&1 | grep -E "\[6\] chunked" || echo "FAILED $name $mode"
}

run clip1_5s.wav cartman   models/rvc/cartman   "models/rvc/cartman/EricCartmanV1_e650_s10400.pth"   ""              "$OUTDIR/cartman_quality_fixed.wav"
run clip1_5s.wav cartman   models/rvc/cartman   "models/rvc/cartman/EricCartmanV1_e650_s10400.pth"   "--mode speed"  "$OUTDIR/cartman_speed_fixed.wav"
run clip2_5s.wav cleveland models/rvc/cleveland "models/rvc/cleveland/Cleveland_Brown_220e_7920s.pth" ""              "$OUTDIR/cleveland_quality_fixed.wav"
run clip2_5s.wav cleveland models/rvc/cleveland "models/rvc/cleveland/Cleveland_Brown_220e_7920s.pth" "--mode speed"  "$OUTDIR/cleveland_speed_fixed.wav"
run clip3_5s.wav peter     models/rvc/peter     "models/rvc/peter/Peter_Griffin_Fake_e220_s2200.pth"  ""              "$OUTDIR/peter_quality_fixed.wav"
run clip3_5s.wav peter     models/rvc/peter     "models/rvc/peter/Peter_Griffin_Fake_e220_s2200.pth"  "--mode speed"  "$OUTDIR/peter_speed_fixed.wav"
run clip4_5s.wav seth      models/rvc/seth      "models/rvc/seth/sethmacfarlene.pth"                  ""              "$OUTDIR/seth_quality_fixed.wav"
run clip4_5s.wav seth      models/rvc/seth      "models/rvc/seth/sethmacfarlene.pth"                  "--mode speed"  "$OUTDIR/seth_speed_fixed.wav"
run clip5_5s.wav jackblack models/rvc/jackblack "models/rvc/jackblack/jackblackrvc_e1860_s87420.pth"  ""              "$OUTDIR/jackblack_quality_fixed.wav"
run clip5_5s.wav jackblack models/rvc/jackblack "models/rvc/jackblack/jackblackrvc_e1860_s87420.pth"  "--mode speed"  "$OUTDIR/jackblack_speed_fixed.wav"
echo "=== DONE ==="
ls -la "$OUTDIR"/*_fixed.wav | awk '{print $5, $9}'
