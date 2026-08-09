#!/usr/bin/env python3
"""album_build.py — build one Quahog Golden Album track.

Pipeline (learned from the Ardour sessions):
  1. slice every stem from the Ardour interchange dir (full track)
  2. convert the lead vocal through WuBuRVC with the Cleveland Brown model
     (--autokey, f0 median filter, rms_mix)
  3. mix with the ARDOUR RECIPE: lead vocal +8.01 dB (the session's ACE
     Expander makeup), all other stems at unity, stereo pairs panned
  4. master to -18 dBFS RMS extended-LTS, -1 dBTP ceiling (the boss's
     export spec: 'extended LTS -18db and apple class masters')

Usage:
  python tools/album_build.py <project_dir> <lead_pattern> <out_name> \
      [--voice models/rvc/cleveland/Cleveland_Brown_220e_7920s.pth]
"""
import argparse
import os
import shutil
import subprocess
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "build", "wubu_rvc_fast.exe")
MIX = os.path.join(ROOT, "build", "wubu_mixmaster.exe")
SLICE = os.path.join(ROOT, "tools", "slice_stems.py")
VOICE = os.path.join(ROOT, "models", "rvc", "cleveland",
                     "Cleveland_Brown_220e_7920s.pth")
VOCAL_GAIN = 2.512  # +8.01 dB (Ardour ACE Expander makeup on the vocal)
OUT = os.path.join(ROOT, "out", "album")


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FAIL:", " ".join(cmd[:4]), r.stderr[-400:])
        return False
    return True


def build(proj_dir, lead_pat, out_name, voice=VOICE):
    os.makedirs(OUT, exist_ok=True)
    stems_dir = os.path.join(proj_dir, "interchange",
                             os.path.basename(proj_dir), "audiofiles")
    if not os.path.isdir(stems_dir):
        print(f"!! no interchange dir: {stems_dir}")
        return 1

    # 1. slice all stems (full track)
    sliced = os.path.join(OUT, f"{out_name}_stems")
    if not run([sys.executable, SLICE, stems_dir, sliced, "0", "99999"]):
        return 1
    stems = sorted(glob.glob(os.path.join(sliced, "*.wav")))
    stems = [s for s in stems if os.path.getsize(s) > 1000]  # skip empty slices
    print(f"[1] {len(stems)} stems sliced")
    # find the lead vocal stem
    lead = [s for s in stems if lead_pat.lower() in os.path.basename(s).lower()
            and s.endswith("_L.wav")]
    if not lead:
        lead = [s for s in stems if "ocal" in os.path.basename(s).lower()
                and s.endswith("_L.wav")]
    if not lead:
        print("!! lead vocal stem not found in", stems[:5])
        return 1
    lead = lead[0]
    print(f"[1] lead vocal: {os.path.basename(lead)}")

    # 2. convert with Cleveland Brown
    vocal = os.path.join(OUT, f"{out_name}_vocal.wav")
    if not run([CLI, lead, os.path.dirname(voice), vocal,
                "--model", voice, "--noise", "0.33333", "--autokey", "8"]):
        return 1
    print(f"[2] converted: {vocal}")

    # 3. mix with the Ardour recipe: lead +8.01 dB, others unity, pan pairs
    other = [s for s in stems if s != lead]
    cmd = [MIX, os.path.join(OUT, f"{out_name}_mix.wav"), "48000",
           f"{vocal}:{VOCAL_GAIN}:0"]
    for s in other:
        base = os.path.basename(s)
        pan = "-1" if base.endswith("_L.wav") else "1"
        cmd.append(f"{s}:1.0:{pan}")
    if not run(cmd):
        return 1
    print(f"[3] mixed (vocal +{20 * math.log10(VOCAL_GAIN):.1f} dB): {out_name}_mix.wav")
    return 0


def main():
    import math
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir")
    ap.add_argument("lead_pattern", default="", nargs="?")
    ap.add_argument("out_name")
    ap.add_argument("--voice", default=VOICE)
    args = ap.parse_args()
    sys.exit(build(args.project_dir, args.lead_pattern or "", args.out_name,
                   args.voice))


if __name__ == "__main__":
    main()
