#!/usr/bin/env python3
"""extract_rmvpe_weights.py — rmvpe.pt → WUBU flat binary (C11 runtime loader).

Dumps every tensor in the RMVPE state_dict to a self-describing flat binary:
  [u32 count]
  per tensor: [u32 name_len][name bytes][u32 ndims][u32 dims[ndims]][f32 data...]

Also precomputes the mel filterbank basis (librosa.filters.mel, htk=True,
n_mels=128, fmin=30, fmax=8000, n_fft=1024 → 128×513) and stores it as a
tensor named "mel_basis" so the C11 engine needs no librosa math at runtime.

Usage: .venv_win/Scripts/python.exe tools/extract_rmvpe_weights.py
Output: models/rvc/rmvpe_weights.bin

License: WaefreBeorn-UMV3
"""
import os
import struct
import numpy as np
import torch
from librosa.filters import mel as librosa_mel

SRC = "models/rvc/rmvpe.pt"
DST = "models/rvc/rmvpe_weights.bin"


def main():
    sd = torch.load(SRC, map_location="cpu")
    if "model" in sd:
        sd = sd["model"]
    items = list(sd.items())

    # mel basis: (128, 513) float32
    mel_basis = librosa_mel(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)
    mel_basis = np.ascontiguousarray(mel_basis, dtype=np.float32)

    with open(DST, "wb") as f:
        f.write(struct.pack("<I", len(items) + 1))
        for name, t in items + [("mel_basis", torch.from_numpy(mel_basis))]:
            t = t.detach().cpu().float().numpy()
            nb = name.encode("utf-8")
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", t.ndim))
            for d in t.shape:
                f.write(struct.pack("<I", d))
            f.write(np.ascontiguousarray(t, dtype=np.float32).tobytes())
    print(f"OK: {DST} ({os.path.getsize(DST)} bytes, {len(items)+1} tensors incl mel_basis)")


if __name__ == "__main__":
    main()
