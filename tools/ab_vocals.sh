#!/usr/bin/env bash
# ab_vocals.sh — A/B on REAL VOCAL STEMS (NOT mixes — mixes break phonemes).
# Each vocal stem is cut at its loudest 5s window, run through the engine
# in quality + speed mode.
# Usage: bash tools/ab_vocals.sh
set -e
export PATH="/c/msys64/mingw64/bin:/c/msys64/usr/bin:$PATH"
cd /c/Users/eman5/WuBuMedia
EXE=./build/wubu_rvc_f0fix.exe
OUTDIR=out/demo/ab
mkdir -p "$OUTDIR"

# (stem_wav model_dir model_pth out_name)
run_vocal() {
  local stem="$1" dir="$2" pth="$3" name="$4"
  local clip="$OUTDIR/${name}_clip.wav"
  # cut loudest 5s from the stem
  ./.venv_win/Scripts/python.exe -c "
import wave, numpy as np
w=wave.open('$stem','rb'); sr=w.getframerate(); ch=w.getnchannels(); d=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16); w.close()
a=d.astype(np.float32)/32768
if ch==2: a=a.reshape(-1,2).mean(axis=1)
wl=sr*5; n=len(a)//wl
if n<1: raise SystemExit('stem too short')
rms=[np.sqrt(np.mean(a[i*wl:(i+1)*wl]**2)) for i in range(n)]
b=int(np.argmax(rms)); seg=a[b*wl:(b+1)*wl]
wo=wave.open('$clip','wb'); wo.setnchannels(1); wo.setsampwidth(2); wo.setframerate(sr); wo.writeframes((seg*32767).astype(np.int16).tobytes()); wo.close()
print(f'cut {len(seg)/sr:.1f}s from {b*5:.0f}s')
" 2>&1 | grep -v Warning
  echo "=== $name (quality) ==="
  "$EXE" "$clip" "$dir" "$OUTDIR/${name}_quality.wav" --model "$pth" --noise 0.33333 --jobs 4 2>&1 | grep "\[6\] chunked" || echo FAIL
  echo "=== $name (speed) ==="
  "$EXE" "$clip" "$dir" "$OUTDIR/${name}_speed.wav" --model "$pth" --noise 0.33333 --jobs 4 --mode speed 2>&1 | grep "\[6\] chunked" || echo FAIL
}

# cartman on the cartman vocal stem
run_vocal out/demo/album_cartman_vocal.wav models/rvc/cartman \
  "models/rvc/cartman/EricCartmanV1_e650_s10400.pth" "cartman_vocal"

# cleveland on the cleveland vocal stem (track1 gold)
run_vocal out/album/track1_gold_vocal.wav models/rvc/cleveland \
  "models/rvc/cleveland/Cleveland_Brown_220e_7920s.pth" "cleveland_gold"

# peter on the peter vocal stem (track6)
run_vocal out/album/track6_peter_vocal.wav models/rvc/peter \
  "models/rvc/peter/Peter_Griffin_Fake_e220_s2200.pth" "peter_sings"

# seth on a vocals stem
run_vocal out/album/track2_driving_vocal.wav models/rvc/seth \
  "models/rvc/seth/sethmacfarlene.pth" "seth_driving"

# jackblack on a vocal stem
run_vocal out/album/track9_quahoghour_vocal.wav models/rvc/jackblack \
  "models/rvc/jackblack/jackblackrvc_e1860_s87420.pth" "jackblack_hour"

echo "=== DONE ==="
ls -la "$OUTDIR"/*_vocal*.wav "$OUTDIR"/*_gold*.wav "$OUTDIR"/*_sings*.wav "$OUTDIR"/*_driving*.wav "$OUTDIR"/*_hour*.wav 2>/dev/null | awk '{print $5, $9}'
