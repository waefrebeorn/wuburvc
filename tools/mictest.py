#!/usr/bin/env python3
"""mictest.py -- prove the ears work before building on them.

Listens on the given device for N seconds, reports RMS level, and transcribes
with faster-whisper. Temporary probe; the real ears live in wubu_ears.py.
"""
import sys
import os

sys.path.insert(0, r"C:\Users\eman5\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages")
os.environ.setdefault("HOME", r"C:\Users\eman5")
os.environ.setdefault("USERPROFILE", r"C:\Users\eman5")

import numpy as np
import sounddevice as sd

DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 22
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
SR = 16000

print(f"recording {SECS}s on device {DEV} ({sd.query_devices(DEV)['name']}) ...")
audio = sd.rec(int(SECS * SR), samplerate=SR, channels=1, dtype="float32", device=DEV)
sd.wait()
a = audio.flatten()
rms = float(np.sqrt(np.mean(a ** 2)))
peak = float(np.max(np.abs(a)))
print(f"RMS={rms:.5f}  PEAK={peak:.5f}  {'SILENT' if rms < 0.0005 else 'SIGNAL PRESENT'}")

if rms >= 0.0005:
    from faster_whisper import WhisperModel
    m = WhisperModel("base.en", device="cpu", compute_type="int8")
    segs, _ = m.transcribe(a, language="en", vad_filter=True)
    text = " ".join(s.text.strip() for s in segs).strip()
    print("HEARD:", repr(text) if text else "(no speech decoded)")
