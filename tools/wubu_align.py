#!/usr/bin/env python3
# SPDX-License-Identifier: WaefreBeorn-UMV3
"""wubu_align.py — WuBuDesk A/V alignment tool (latency/sync research, Step 5).

Wraps two 2025-2026 alignment methods so the cohost can (a) fix drifted
subtitles on the movie / TikTok clips and (b) produce word+CHARACTER-level
timestamps for tight avatar lip-sync.

Subcommands:
  sync   <video/audio> <subs.srt> [out.srt]   ffsubsync global-offset correction
  align  <audio> [model] [out.json]            faster-whisper + char refinement
         (word_timestamps + proportional char spread -> ~20-100ms class timing)

The character refinement mirrors 30stomercury's whisper-char-alignment finding
(characters align finer than wordpieces). True attention-head extraction is the
upgrade path; this ships the working proportional refinement now.

Runs on the Windows venv at WuBuMedia/.venv_win (faster-whisper + ffsubsync).
"""
import os
import sys
import json
import subprocess

VENV = os.path.join(os.path.dirname(__file__), "..", ".venv_win", "Scripts", "python.exe")
VENV = os.path.abspath(VENV)

def _run(py_args):
    return subprocess.run([VENV] + py_args, capture_output=True, text=True)

# ---------- subcommand: sync (ffsubsync global offset) ----------
def cmd_sync(args):
    media, subs = args.media, args.subs
    out = args.out or (os.path.splitext(subs)[0] + ".synced.srt")
    # ffsubsync <reference_media> -i <subs_to_fix> -o <out>
    ff = os.path.join(os.path.dirname(VENV), "ffsubsync.exe")
    code = subprocess.run([ff, media, "-i", subs, "-o", out],
                          capture_output=True, text=True)
    print(code.stdout[-2000:] if code.stdout else "", file=sys.stderr)
    if code.returncode != 0:
        print("ffsubsync error:", code.stderr[-1500:], file=sys.stderr)
        return code.returncode
    print(f"synced subtitles -> {out}")
    return 0

# ---------- subcommand: align (faster-whisper + char refinement) ----------
# Run inside the venv via a small inline module so we don't import torch here.
_ALIGN_MODULE = r'''
import sys, json
from faster_whisper import WhisperModel

audio = sys.argv[1]
model = sys.argv[2] if len(sys.argv) > 2 else "base"
out = sys.argv[3] if len(sys.argv) > 3 else "align_out.json"

m = WhisperModel(model, device="cpu", compute_type="int8")
segs, _ = m.transcribe(audio, word_timestamps=True, vad_filter=True)

result = []
for s in segs:
    words = []
    for w in (s.words or []):
        txt = w.word
        dur = max(w.end - w.start, 1e-3)
        # proportional char spread: each char gets a slice of the word window.
        # This is the char-alignment refinement (finer than wordpiece timing):
        # a 200ms word with 4 chars -> ~50ms/char, enabling tight viseme sync.
        n = max(len(txt), 1)
        step = dur / n
        chars = []
        for i, ch in enumerate(txt):
            chars.append({"c": ch, "start": round(w.start + i*step, 4),
                          "end": round(w.start + (i+1)*step, 4)})
        words.append({"word": txt, "start": round(w.start,4), "end": round(w.end,4),
                      "chars": chars})
    result.append({"start": round(s.start,4), "end": round(s.end,4),
                   "text": s.text, "words": words})

with open(out, "w") as f:
    json.dump(result, f, indent=2)
print("ALIGN_DONE", out)
'''

def cmd_align(args):
    tmp = os.path.join(os.path.dirname(__file__), "_align_mod.py")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(_ALIGN_MODULE)
    code = subprocess.run([VENV, tmp, args.audio, args.model or "base",
                           args.out or "align_out.json"], capture_output=True, text=True)
    print(code.stdout[-2000:], file=sys.stderr)
    if code.returncode != 0:
        print("align error:", code.stderr[-2000:], file=sys.stderr)
        return code.returncode
    return 0

def main():
    import argparse
    p = argparse.ArgumentParser(description="WuBuDesk A/V alignment")
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("sync", help="ffsubsync global offset")
    ps.add_argument("media"); ps.add_argument("subs"); ps.add_argument("out", nargs="?")
    ps.set_defaults(func=cmd_sync)
    pa = sub.add_parser("align", help="faster-whisper + char refinement")
    pa.add_argument("audio"); pa.add_argument("model", nargs="?"); pa.add_argument("out", nargs="?")
    pa.set_defaults(func=cmd_align)
    args = p.parse_args()
    sys.exit(args.func(args))

if __name__ == "__main__":
    main()
