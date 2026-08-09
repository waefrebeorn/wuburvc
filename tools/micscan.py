#!/usr/bin/env python3
"""micscan.py -- find which input actually carries the boss's voice.

Boss runs Voicemeeter Banana + NVIDIA Broadcast + OBS holding the webcams, so
the hardware endpoints are contested and several report pure silence to a third
listener. Scan every input, report RMS/peak, rank them.

  python tools/micscan.py [seconds]
"""
import sys
import os

sys.path.insert(0, r"C:\Users\eman5\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages")
os.environ.setdefault("HOME", r"C:\Users\eman5")
os.environ.setdefault("USERPROFILE", r"C:\Users\eman5")

import numpy as np
import sounddevice as sd

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 2.5
SR = 16000

# Prefer the virtual buses: they can be opened alongside OBS/Broadcast.
INTERESTING = ("voicemeeter", "nvidia broadcast", "ag06", "usb audio codec",
               "rtx-audio", "brio", "digital audio")

seen = {}
results = []
print(f"scanning inputs, {SECS}s each (talk continuously!)\n")
for i, d in enumerate(sd.query_devices()):
    if d.get("max_input_channels", 0) < 1:
        continue
    name = d["name"]
    low = name.lower()
    if not any(k in low for k in INTERESTING):
        continue
    if low in seen:          # same endpoint exposed via MME/WASAPI/DS
        continue
    seen[low] = True
    try:
        a = sd.rec(int(SECS * SR), samplerate=SR, channels=1,
                   dtype="float32", device=i)
        sd.wait()
        a = a.flatten()
        rms = float(np.sqrt(np.mean(a ** 2)))
        peak = float(np.max(np.abs(a)))
        results.append((rms, peak, i, name))
        flag = "SILENT" if rms < 0.0005 else ("quiet" if rms < 0.005 else "GOOD")
        print(f"  dev {i:3d}  {name[:44]:46s} rms={rms:.5f} peak={peak:.4f}  {flag}")
    except Exception as e:
        print(f"  dev {i:3d}  {name[:44]:46s} ERR {str(e)[:40]}")

print("\n--- ranked ---")
for rms, peak, i, name in sorted(results, reverse=True)[:6]:
    print(f"  {rms:.5f}  dev {i:3d}  {name}")
if results:
    best = sorted(results, reverse=True)[0]
    print(f"\nUSE: --device {best[2]}   ({best[3]})")
