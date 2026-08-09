#!/usr/bin/env python3
"""Inspect voice repo catalogs + hubert keys."""
import json, os

BASE = r"C:\Users\eman5\WuBuMedia"

# 1. vm_full_index.json structure
p = os.path.join(BASE, "models", "vm_full_index.json")
if os.path.exists(p):
    idx = json.load(open(p))
    print("vm_full_index:", type(idx).__name__, "len:", len(idx) if hasattr(idx, "__len__") else "?")
    if isinstance(idx, dict):
        for k in list(idx.keys()):
            v = idx[k]
            if isinstance(v, (list, tuple)):
                print("  ", k, "len:", len(v), "sample:", str(v[0])[:160] if v else "")
            elif isinstance(v, dict):
                print("  ", k, "dict keys:", list(v.keys())[:5])
            else:
                print("  ", k, "=", repr(v)[:160])
    elif isinstance(idx, list):
        print("  sample:", str(idx[0])[:200] if idx else "empty")

# 2. voice_catalog.json
p = os.path.join(BASE, "models", "voice_catalog.json")
if os.path.exists(p):
    vc = json.load(open(p))
    print("\nvoice_catalog:", type(vc).__name__, "len:", len(vc) if hasattr(vc, "__len__") else "?")
    if isinstance(vc, dict):
        k0 = list(vc.keys())[:3]
        for k in k0:
            print("  sample:", repr(k), "->", str(vc[k])[:150])

# 3. hubert keys dump (write to repo so it persists)
p = os.path.join(BASE, "models", "rvc", "hubert_keys.json")
if os.path.exists(p):
    ks = json.load(open(p))
    print("\nhubert_keys total:", len(ks))
    for k in ks:
        if any(t in k for t in ("conv", "feature", "pos", "embed", "final")):
            print("  ", k)
