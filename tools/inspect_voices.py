#!/usr/bin/env python3
"""Inspect the 30k voice-models.com catalog + local voice bank."""
import json, os, sys

BASE = r"C:\Users\eman5\WuBuMedia"
idx_path = os.path.join(BASE, "models", "vm_full_index.json")
bank_path = os.path.join(BASE, "out", "voices.json")

idx = json.load(open(idx_path, encoding="utf-8"))
print("vm_full_index keys:", list(idx.keys()) if isinstance(idx, dict) else type(idx))
if isinstance(idx, dict):
    for k, v in idx.items():
        if isinstance(v, list):
            print(f"  {k}: len={len(v)} sample={str(v[0])[:120]}")
        else:
            print(f"  {k}: {type(v).__name__} {str(v)[:120]}")

print()
if os.path.exists(bank_path):
    bank = json.load(open(bank_path, encoding="utf-8"))
    voices = bank.get("voices", bank)
    print(f"local bank: {len(voices)} voices")
    names = list(voices.keys())[:10]
    for n in names:
        v = voices[n]
        print("  ", n, "->", {k: str(v[k])[:80] for k in list(v)[:4]})
    withpth = [n for n, v in voices.items() if v.get("pth") and os.path.exists(v["pth"])]
    print("voices with .pth on disk:", len(withpth))
else:
    print("no out/voices.json")
