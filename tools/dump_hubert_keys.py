#!/usr/bin/env python3
"""Dump hubert_base.pt tensor keys+shapes to a text file for the C forward design."""
import sys, os, types, importlib.abc, importlib.machinery

class Dummy:
    def __init__(self, *a, **k): pass
    def __setattr__(self, n, v): pass

class StubMod(types.ModuleType):
    def __getattr__(self, name):
        d = Dummy; d.__name__ = name; return d

class StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = StubMod(spec.name); mod.__path__ = []; sys.modules[spec.name] = mod; return mod
    def exec_module(self, module): pass

class FairseqBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "fairseq" or fullname.startswith("fairseq."):
            return importlib.machinery.ModuleSpec(fullname, loader=StubLoader(), is_package=True)
        return None

import torch
sys.meta_path.insert(0, FairseqBlocker())
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = c.get("model", c) if isinstance(c, dict) else c
out = []
for k in sorted(sd.keys()):
    v = sd[k]
    if hasattr(v, "shape"):
        out.append(f"{k} {tuple(v.shape)} {v.dtype}")
with open(sys.argv[2], "w") as f:
    f.write("\n".join(out))
print(f"wrote {len(out)} keys -> {sys.argv[2]}")
