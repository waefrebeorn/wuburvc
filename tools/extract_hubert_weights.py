#!/usr/bin/env python3
"""Extract hubert_base.pt (fairseq checkpoint) into a WUBU flat binary that
wubu_rvc_weights.c can load — WITHOUT importing real fairseq (it crashes on
Python 3.11 dataclass/hydra/omegaconf).

Usage: python extract_hubert_weights.py models/rvc/hubert_base.pt models/rvc/hubert_weights.bin
"""
import sys, os, struct, types, importlib.abc, importlib.machinery


class Dummy:
    def __init__(self, *a, **k):
        pass

    def __setattr__(self, n, v):
        pass


class StubMod(types.ModuleType):
    """Module whose missing attributes are Dummy classes (pickle only needs
    the classes to exist so find_class() succeeds; the returned objects are
    never used for the state dict)."""

    def __getattr__(self, name):
        d = Dummy
        d.__name__ = name
        return d


class StubLoader(importlib.abc.Loader):
    """Loader that builds the stub module (keeps attribute-access fallback
    alive; a loader=None spec degenerates into a namespace package which
    loses __getattr__)."""

    def create_module(self, spec):
        mod = StubMod(spec.name)
        mod.__path__ = []
        sys.modules[spec.name] = mod
        return mod

    def exec_module(self, module):
        pass


class FairseqBlocker(importlib.abc.MetaPathFinder):
    """Intercept any fairseq[.*] import and serve a stub module instead of
    executing the real fairseq package (which crashes on py3.11)."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "fairseq" or fullname.startswith("fairseq."):
            return importlib.machinery.ModuleSpec(
                fullname, loader=StubLoader(), is_package=True
            )
        return None


def extract(src_pth, dst_bin):
    import numpy as np
    import torch

    sys.meta_path.insert(0, FairseqBlocker())

    ckpt = torch.load(src_pth, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    # Strip any non-tensor entries
    items = []
    for key in sorted(sd.keys()):
        val = sd[key]
        if hasattr(val, "shape") and hasattr(val, "numpy"):
            items.append((key, val.detach().cpu().float().numpy()))

    with open(dst_bin, "wb") as f:
        f.write(b"WUBU")
        f.write(struct.pack("<I", len(items)))
        for name, arr in items:
            nb = name.encode("ascii")
            f.write(struct.pack("B", len(nb)))
            f.write(nb)
            shape = list(arr.shape) if arr.ndim > 0 else [len(arr)]
            ndim = len(shape) if arr.ndim > 0 else 1
            data = arr.astype(np.float32).tobytes() if arr.dtype != np.float32 else arr.tobytes()
            f.write(struct.pack("<I", ndim))
            for d in shape:
                f.write(struct.pack("<I", d))
            f.write(struct.pack("<I", len(data)))
            f.write(data)

    print(f"Extracted {len(items)} tensors -> {dst_bin} ({os.path.getsize(dst_bin)} bytes)")
    for name, arr in items[:12]:
        print(f"  {name}: shape={list(arr.shape)} size={arr.size}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <hubert_base.pt> <output.bin>")
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2])
