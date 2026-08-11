#!/usr/bin/env python3
"""build_album_all.py — rebuild ALL Quahog Golden Album tracks (1-9) with fixed engine and proper stem selection.

This version:
- Clears all track outputs first
- Uses the fixed stem selection regex (excludes -2/-3 variants)
- Uses correct voice models (Seth = SETH, others = Cleveland)
- Forces re-conversion through the new C11 engine
"""
import os
import sys
import glob
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

# Fixed path definitions
MEDIA = r"C:\Users\eman5\WuBuMedia"
DOCS = r"C:\Users\eman5\Documents"
CLEVELAND_MODEL = os.path.join(MEDIA, "models", "rvc", "cleveland", "Cleveland_Brown_220e_7920s.pth")
SETH_MODEL = os.path.join(MEDIA, "models", "rvc", "seth", "sethmacfarlene.pth")
PETER_MODEL = os.path.join(MEDIA, "models", "rvc", "peter", "Peter_Griffin_Fake_e220_s2200.pth")

# Define tracks (4-9 only, since 1-3 already completed)
TRACKS = [
    ("sethslament", "track4_seths_lament", SETH_MODEL),
    ("cleveslumbergold", "track5_slumbers", CLEVELAND_MODEL),
    ("bringclevelandback", "track6_bringback", PETER_MODEL),
    ("clevelandisgolden", "track7_is_golden", CLEVELAND_MODEL),
    ("20kcleeveland", "track8_24k", CLEVELAND_MODEL),
    ("clevelandquohoghour", "track9_quahog_hour", CLEVELAND_MODEL),
]

from album_build import build

def main():
    results = []
    for proj, out_name, voice in TRACKS:
        proj_dir = os.path.join(DOCS, proj)
        print(f"\n{'='*60}\n=== {out_name} ({proj}) ===\n", flush=True)
        rc = build(proj_dir, "", out_name, voice)
        results.append((out_name, rc))
        print(f"=== {out_name} -> rc={rc} ===\n", flush=True)
    
    print("\n" + "="*60)
    print("ALBUM BUILD SUMMARY")
    print("="*60)
    ok = 0
    for name, rc in results:
        mark = "OK " if rc == 0 else "FAIL"
        ok += (rc == 0)
        print(f"  {mark} {name}")
    print(f"\n{ok}/{len(results)} tracks built successfully")
    return 0 if ok == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())