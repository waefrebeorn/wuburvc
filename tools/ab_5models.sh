#!/usr/bin/env bash
# ab_5models.sh — A/B: 5 DIFFERENT 5s clips, each through ONE model,
# quality vs speed mode → 10 files x 5s = 50s.
# clip1→cartman, clip2→cleveland, clip3→peter, clip4→seth, clip5→jackblack
set -e
export PATH="/c/msys64/mingw64/bin:/c/msys64/usr/bin:$PATH"
cd /c/Users/eman5/WuBuMedia
EXE=./build/wubu_rvc_prof.exe
OUTDIR=out/demo/ab
mkdir -p "$OUTDIR"

run() {
  local clip="$1" name="$2" dir="$3" pth="$4" mode="$5" out="$6"
  echo "=== $name <- $clip ($mode) ==="
  "$EXE" "$OUTDIR/$clip" "$dir" "$out" --model "$pth" --noise 0.33333 --jobs 4 ${mode} 2>&1 | grep -E "\[0\] mode|\[6\] chunked" || echo "FAILED $name $mode"
}

run clip1_5s.wav cartman   models/rvc/cartman   "models/rvc/cartman/EricCartmanV1_e650_s10400.pth"   ""              "$OUTDIR/cartman_quality.wav"
run clip1_5s.wav cartman   models/rvc/cartman   "models/rvc/cartman/EricCartmanV1_e650_s10400.pth"   "--mode speed"  "$OUTDIR/cartman_speed.wav"
run clip2_5s.wav cleveland models/rvc/cleveland "models/rvc/cleveland/Cleveland_Brown_220e_7920s.pth" ""              "$OUTDIR/cleveland_quality.wav"
run clip2_5s.wav cleveland models/rvc/cleveland "models/rvc/cleveland/Cleveland_Brown_220e_7920s.pth" "--mode speed"  "$OUTDIR/cleveland_speed.wav"
run clip3_5s.wav peter     models/rvc/peter     "models/rvc/peter/Peter_Griffin_Fake_e220_s2200.pth"  ""              "$OUTDIR/peter_quality.wav"
run clip3_5s.wav peter     models/rvc/peter     "models/rvc/peter/Peter_Griffin_Fake_e220_s2200.pth"  "--mode speed"  "$OUTDIR/peter_speed.wav"
run clip4_5s.wav seth      models/rvc/seth      "models/rvc/seth/sethmacfarlene.pth"                  ""              "$OUTDIR/seth_quality.wav"
run clip4_5s.wav seth      models/rvc/seth      "models/rvc/seth/sethmacfarlene.pth"                  "--mode speed"  "$OUTDIR/seth_speed.wav"
run clip5_5s.wav jackblack models/rvc/jackblack "models/rvc/jackblack/jackblackrvc_e1860_s87420.pth"  ""              "$OUTDIR/jackblack_quality.wav"
run clip5_5s.wav jackblack models/rvc/jackblack "models/rvc/jackblack/jackblackrvc_e1860_s87420.pth"  "--mode speed"  "$OUTDIR/jackblack_speed.wav"
echo "=== DONE ==="
ls -la "$OUTDIR"/*.wav | awk '{print $5, $9}'
