#!/usr/bin/env python3
"""build_album_all.py — rebuild ALL Quahog Golden Album tracks in the new
C11 VK RVC engine from the DRY Ardour stems (the old RVC audio is dead).

Track map (TRACKLIST.txt + the boss's voice rules):
 1  G.O.L.D.                    cleveGOLDhope       Cleveland
 2  Freestyle Driving Lesson    clevespooner        Cleveland
 3  Cleveland Sings Golden      clevegoldenkpop     Cleveland
 4  Seth's Lament               sethslament         SETH (his own voice model)
 5  Cleveland Slumbers in Gold  cleveslumbergold    Cleveland
 6  Bring My Cleveland Back     bringclevelandback  PETER GRIFFIN
 7  Cleveland Is Golden         clevelandisgolden   Cleveland
 8  24K Cleveland               20kcleeveland       Cleveland
 9  Cleveland's Quahog Hour     clevelandquohoghour Cleveland
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

MEDIA = r"C:\Users\eman5\WuBuMedia"
DOCS = r"C:\Users\eman5\Documents"
CLEVELAND = os.path.join(MEDIA, "models", "rvc", "cleveland",
                         "Cleveland_Brown_220e_7920s.pth")
PETER = os.path.join(MEDIA, "models", "rvc", "peter",
                     "Peter_Griffin_Fake_e220_s2200.pth")
SETH = os.path.join(MEDIA, "models", "rvc", "seth", "sethmacfarlene.pth")

TRACKS = [
    ("cleveGOLDhope",       "track1_gold",           CLEVELAND),
    ("clevespooner",        "track2_driving",        CLEVELAND),
    ("clevegoldenkpop",     "track3_sings_golden",   CLEVELAND),
    ("sethslament",         "track4_seths_lament",   SETH),
    ("cleveslumbergold",    "track5_slumbers",       CLEVELAND),
    ("bringclevelandback",  "track6_bringback",      PETER),
    ("clevelandisgolden",   "track7_is_golden",      CLEVELAND),
    ("20kcleeveland",       "track8_24k",            CLEVELAND),
    ("clevelandquohoghour", "track9_quahog_hour",    CLEVELAND),
]

from album_build import build

def main():
    results = []
    for proj, out_name, voice in TRACKS:
        proj_dir = os.path.join(DOCS, proj)
        print(f"\n{'='*60}\n=== {out_name}  ({proj}, {os.path.basename(os.path.dirname(voice))}) ===", flush=True)
        rc = build(proj_dir, "", out_name, voice)
        results.append((out_name, rc))
        print(f"=== {out_name} -> rc={rc} ===", flush=True)
    print("\n\n=== ALBUM SUMMARY ===")
    ok = 0
    for name, rc in results:
        mark = "OK " if rc == 0 else "FAIL"
        ok += (rc == 0)
        print(f"  {mark} {name}")
    print(f"{ok}/{len(results)} tracks built")
    return 0 if ok == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
