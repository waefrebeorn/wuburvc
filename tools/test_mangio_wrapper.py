#!/usr/bin/env python3
"""Direct test of the Mangio wrapper with the real HuBERT — captures the
full subprocess stderr so we can see the actual failure."""
import os, sys, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import wubu_rvc

MANGIO = wubu_rvc.MANGIO
ROOT = wubu_rvc.ROOT
input_dir = os.path.join(ROOT, "out", "speech", "_rvc_in")
output_dir = os.path.join(ROOT, "out", "speech", "_rvc_out")
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
wav_in = os.path.join(ROOT, "outputs", "cartman_base.wav")
wav_basename = os.path.basename(wav_in)
import shutil
shutil.copy2(wav_in, os.path.join(input_dir, wav_basename))
staged_out = os.path.join(output_dir, wav_basename)
if os.path.exists(staged_out):
    os.remove(staged_out)

argv = repr([
    "infer_batch_rvc.py", "0", input_dir, "default", "rmvpe",
    output_dir, os.path.join(ROOT, "models", "rvc", "cartman", "EricCartmanV1_e650_s10400.pth"),
    "0.0", "cuda:0", "True", "3", "0", "0.25", "0.5",
])
source = wubu_rvc._WRAPPER.format(
    argv=argv,
    mangio_path=repr(MANGIO),
    script_path=repr(os.path.join(MANGIO, "infer_batch_rvc.py")),
    wubu_tools_path=repr(os.path.join(ROOT, "tools")),
    hubert_path=repr(os.path.join(ROOT, "models", "rvc", "hubert_base.pt")),
    use_half=repr(True),
)
with tempfile.TemporaryDirectory() as td:
    wp = os.path.join(td, "_rvc_wrap_test.py")
    with open(wp, "w") as wf:
        wf.write(source)
    cmd = [sys.executable, "-u", wp]
    print("RUNNING wrapper (max 1500s)...")
    with open(os.path.join(ROOT, "out", "speech", "_wrap_test.log"), "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                           timeout=1500, cwd=MANGIO)
    print("RC:", r.returncode)
    logp = os.path.join(ROOT, "out", "speech", "_wrap_test.log")
    with open(logp, encoding="utf-8", errors="replace") as lf:
        lines = lf.read().splitlines()
    print("=== log tail (%d lines) ===" % len(lines))
    print("\n".join(lines[-30:]))
    print("out exists:", os.path.exists(staged_out))
