#!/usr/bin/env python3
"""Verify hubert pos_conv weight_norm shape semantics from the WUBU bin."""
import struct, numpy as np

def load_bin(path):
    f = open(path, 'rb')
    magic = f.read(4)
    n = struct.unpack('<I', f.read(4))[0]
    tensors = {}
    for _ in range(n):
        nl = struct.unpack('B', f.read(1))[0]
        name = f.read(nl).decode()
        nd = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<I', f.read(4))[0] for _ in range(nd)]
        dl = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(dl), dtype=np.float32).reshape(dims)
        tensors[name] = data
    f.close()
    return tensors

t = load_bin(r'C:\Users\eman5\WuBuMedia\models\rvc\hubert_weights.bin')
print('n tensors:', len(t))
g = t.get('encoder.pos_conv.0.weight_g')
v = t.get('encoder.pos_conv.0.weight_v')
b = t.get('encoder.pos_conv.0.bias')
print('weight_g', None if g is None else g.shape, g.dtype if g is not None else '')
print('weight_v', None if v is None else v.shape)
print('bias', None if b is None else b.shape)
if g is not None and v is not None:
    # weight_norm dim=2: W[...,k] = g[...,k]?? unusual; test broadcast
    print('g sample:', g.flatten()[:8])
    print('v[0,0,:5]:', v[0,0,:5])
    # Try: W = g * v / ||v|| over last dim, g broadcastable (1,1,128)?
    norm = np.sqrt((v**2).sum(-1, keepdims=True)) + 1e-8
    W1 = g * v / norm
    print('W1 shape:', W1.shape, 'W1[0,0,:3]:', W1[0,0,:3])
    # Try: W = g * v / ||v|| over dim 0? g (1,1,128) can't
    # Check if g broadcasts over channels: g shape (1,1,128) -> per-kernel scale?
    print('v[0,0,:3]:', v[0,0,:3])
    # effective weight per (out,in) with per-k gain:
    W2 = v * g / norm
    print('W2[0,0,:3]:', W2[0,0,:3])
