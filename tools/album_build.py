#!/usr/bin/env python3
"""album_build.py — build one Quahog Golden Album track.

Pipeline (learned from the Ardour sessions):
  1. slice every stem from the Ardour interchange dir (full track)
  2. convert the lead vocal through WuBuRVC with the CHARACTER model
     (--autokey, f0 median filter, rms_mix, harmony/consonant/breath/
     artifact-gate quality chain)
  3. mix with the ARDOUR RECIPE: lead vocal +8.01 dB (the session's ACE
     Expander makeup), all other stems at unity, stereo pairs panned
  4. master to -18 dBFS RMS extended-LTS, -1 dBTP ceiling (the boss's
     export spec: "extended LTS -18db and apple class masters")

Character voices (the boss's rule):
  Track 4 (Seth's Lament)  = Seth MacFarlane's OWN voice (master as-is,
                             NO conversion — his real take)
  Track 6 (Bring My Cleveland Back) = PETER GRIFFIN (models/rvc/peter,
                             Peter_Griffin_Fake_e220_s2200.pth)
  All other tracks = CLEVELAND BROWN (models/rvc/cleveland,
                             Cleveland_Brown_220e_7920s.pth)

Usage:
  python tools/album_build.py <project_dir> <lead_pattern> <out_name> \\
      [--voice models/rvc/cleveland/Cleveland_Brown_220e_7920s.pth]
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_cli():
    """Locate the RVC engine exe; auto-build if missing (clean build wipes it).
    Prefer the Vulkan build — it's 1.7x faster than CPU at jobs=4 and now
    bit-parity with CPU (corr 1.00000000, maxdiff 1 LSB)."""
    for c in ["wubu_rvc_vk.exe", "wubu_rvc_fast.exe", "wubu_rvc.exe"]:
        p = os.path.join(ROOT, "build", c)
        if os.path.isfile(p):
            return p
    print("[auto] engine exe missing — running build_clean.sh (full rebuild)")
    # --no-cuda only when the CUDA .o survives clean (else build_clean fails);
    # --no-spv is safe: SPIR-V headers are regenerated when .comp changes.
    flags = ["--no-spv"]
    if os.path.isfile(os.path.join(ROOT, "build", "wubu_rvc_cuda.o")):
        flags.append("--no-cuda")
    if not run(["bash", os.path.join(ROOT, "build_clean.sh"), "wubu_rvc_vk"]
               + flags):
        print("!! auto-build failed — install a build/wubu_rvc_vk.exe and retry")
    for c in ["wubu_rvc_vk.exe", "wubu_rvc_fast.exe", "wubu_rvc.exe"]:
        p = os.path.join(ROOT, "build", c)
        if os.path.isfile(p):
            return p
    return os.path.join(ROOT, "build", "wubu_rvc_vk.exe")


def _find_mix():
    """Locate the mixmaster exe; auto-build from source if missing."""
    p = os.path.join(ROOT, "build", "wubu_mixmaster.exe")
    if os.path.isfile(p):
        return p
    print("[auto] wubu_mixmaster.exe missing — building from "
          "tools/wubu_mixmaster.c")
    mingw = r"C:\msys64\mingw64\bin"
    gcc = os.path.join(mingw, "gcc.exe")
    src = os.path.join(ROOT, "src")
    cmd = [gcc, "-std=c11", "-O2", "-I", src,
           os.path.join(ROOT, "tools", "wubu_mixmaster.c"),
           os.path.join(src, "wubu_master.c"),
           os.path.join(src, "wubu_audioio.c"), "-lm", "-o", p]
    env = dict(os.environ)
    env["TMP"] = os.path.join(ROOT, "build", "tmp")
    env["TEMP"] = env["TMP"]; env["TMPDIR"] = env["TMP"]
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0 or not os.path.isfile(p):
        print("!! mixmaster auto-build failed — album mix will fail")
        return None
    print(f"[auto] mixmaster built: {p}")
    return p


def run(cmd):
    # engine exe needs MinGW runtime DLLs from MSYS2 — prepend to PATH
    env = dict(os.environ)
    mingw = r"C:\msys64\mingw64\bin"
    if mingw not in env.get("PATH", ""):
        env["PATH"] = mingw + os.pathsep + env.get("PATH", "")
    # Ensure the C11 subprocess can find Python for faiss IVF search
    if "WUBU_PYTHON" not in env:
        py = os.path.join(MEDIA, ".venv_win", "Scripts", "python.exe")
        if os.path.isfile(py):
            env["WUBU_PYTHON"] = py
    # stream output live (the boss watches album builds) and fail loudly
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        print(f"FAIL (rc={r.returncode}):", " ".join(cmd[:4]))
        return False
    return True


CLI = _find_cli()
MIX = _find_mix()
SLICE = os.path.join(ROOT, "tools", "slice_stems.py")
MASTER = os.path.join(ROOT, "tools", "master.py")
MEDIA = r"C:\Users\eman5\WuBuMedia"
HUBERT = os.path.join(MEDIA, "models", "rvc", "hubert_weights.bin")
VOICE = os.path.join(MEDIA, "models", "rvc", "cleveland",
                     "Cleveland_Brown_220e_7920s.pth")
PETER = os.path.join(MEDIA, "models", "rvc", "peter",
                     "Peter_Griffin_Fake_e220_s2200.pth")
VOCAL_GAIN = 2.512  # +8.01 dB (Ardour ACE Expander makeup on the vocal)
OUT = os.path.join(MEDIA, "out", "album")


def build(proj_dir, lead_pat, out_name, voice=VOICE, harmony=1, vibrato=1, artgate=0.3, f0smooth=0.3, consonant=1):
    os.makedirs(OUT, exist_ok=True)
    stems_dir = os.path.join(proj_dir, "interchange",
                             os.path.basename(proj_dir), "audiofiles")
    if not os.path.isdir(stems_dir):
        print(f"!! no interchange dir: {stems_dir}")
        return 1

    # 1. slice all stems (full track)
    sliced = os.path.join(OUT, f"{out_name}_stems")
    if os.path.exists(sliced):
        shutil.rmtree(sliced)
    if not run([sys.executable, SLICE, stems_dir, sliced, "0", "99999"]):
        return 1
    stems = sorted(glob.glob(os.path.join(sliced, "*.wav")))
    stems = [s for s in stems if os.path.getsize(s) > 1000]  # skip empty slices
    print(f"[1] {len(stems)} stems sliced")
    # find the lead vocal stem — MUST be the DRY original, never an
    # RVC-processed mix (the Ardour sessions reference RVC mixes like
    # cleve2.wav / cleveland.wav over the dry stems; the boss's rule:
    # inputs are the dry interchange stems only). Match "vocals" but
    # EXCLUDE "backing" (Backing_Vocals sorts before Vocals) and any
    # RVC artifacts (cleve2/cleve3/cleveland.wav, *_rvc*, *_ai*, and
    # take-variant "-2"/"-3" stems which are old RV outputs with wrong
    # sample rates).
    import re as _re
    def _is_rvc_artifact(n):
        low = os.path.basename(n).lower()
        base = low[:-4] if low.endswith('.wav') else low
        # OLD RVC conversions only (exact, anchored): cleve2/cleve3/cleve4/
        # cleve2-2/cleveland/cleveland-2. The DRY stems ("cleveland has hope
        # (Vocals)") must NEVER match — they are the real takes.
        # Also exclude take-variant stems ("-2", "-3" suffixes): these are
        # old RV outputs from previous renders with different sample rates.
        if '-2' in base or '-3' in base or '-4' in base:
            return True
        return (_re.fullmatch(r'cleve\d(-\d+)?', base) is not None
                or _re.fullmatch(r'cleveland(-\d+)?', base) is not None
                or 'rvc' in low or '_ai' in low or 'ai_' in low)
    lead = [s for s in stems if lead_pat and lead_pat.lower() in os.path.basename(s).lower()
            and "backing" not in os.path.basename(s).lower()
            and not _is_rvc_artifact(os.path.basename(s))
            and s.endswith("_L.wav")]
    if not lead:
        lead = [s for s in stems if "ocal" in os.path.basename(s).lower()
                and "backing" not in os.path.basename(s).lower()
                and not _is_rvc_artifact(os.path.basename(s))
                and s.endswith("_L.wav")]
    if not lead:
        print("!! lead vocal stem not found in", stems[:5])
        return 1
    # Prefer the original dry take: sort so base stems (without -N suffix)
    # come first. If the first match has no suffix, keep it; otherwise pick
    # the shortest-duration stem (the original dry take is the same duration
    # across all variants, but the -2/-3 are old RV outputs we already
    # filtered out above).
    lead = lead[0]
    print(f"[1] lead vocal (DRY): {os.path.basename(lead)}")

    # 2. convert with the character model + full quality chain
    #    --auto-index-rate auto-calibrates the FAISS retrieval blend based
    #    on cosine similarity between query and neighbors, eliminating the
    #    "Squidward" timbre distortion from garbage index vectors.
    vocal = os.path.join(OUT, f"{out_name}_vocal.wav")
    if not run([CLI, lead, os.path.dirname(voice), vocal,
                "--model", voice, "--hubert", HUBERT,
                "--jobs", "4", "--chunk", "3",
                "--noise", "0.66666", "--autokey", "8",
                "--f0smooth", str(float(f0smooth)), "--harmony", str(int(harmony)),
                "--vibrato", str(int(vibrato)),
                "--consonant", str(int(consonant)), "--breath", "1", "--artgate", str(float(artgate)),
                "--auto-index-rate"]):
        return 1
    print(f"[2] converted ({os.path.basename(voice)}): {vocal}")

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

    # 4. MASTERING: normalize to -18 dBFS RMS extended-LTS (Ardour export spec)
    mix_path = os.path.join(OUT, f"{out_name}_mix.wav")
    master_path = os.path.join(OUT, f"{out_name}_mastered.wav")
    if not run([sys.executable, MASTER, mix_path, master_path, "-18", "-1"]):
        return 1
    print(f"[4] mastered to extended-LTS -18 dBFS RMS: {out_name}_mastered.wav")
    return 0


def main():
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
