#!/usr/bin/env python3
"""slice_stems.py — extract a section from Ardour float-WAV album stems.

Ardour exports 32-bit float wavs (format 3, possibly extensible). Reads them
with a proper chunk walker, slices [start_s, start_s+len_s), and writes clean
pcm_s16le mono/stereo wavs for the C11 RVC + mixmaster pipeline.
"""
import os, struct, sys
import numpy as np

def read_float_wav(path):
    d = open(path, 'rb').read()
    pos = 12
    fmt_tag = ch = sr = bits = 0
    data = None
    while pos < len(d) - 8:
        cid = d[pos:pos + 4]
        sz = struct.unpack('<I', d[pos + 4:pos + 8])[0]
        if cid == b'fmt ':
            fmt_tag = struct.unpack('<H', d[pos + 8:pos + 10])[0]
            ch = struct.unpack('<H', d[pos + 10:pos + 12])[0]
            sr = struct.unpack('<I', d[pos + 12:pos + 16])[0]
            bits = struct.unpack('<H', d[pos + 22:pos + 24])[0] if sz >= 24 else 16
        elif cid == b'data':
            raw = d[pos + 8:pos + 8 + sz]
            if fmt_tag == 3 or (fmt_tag == 0xFFFE and bits == 32):
                data = np.frombuffer(raw, dtype=np.float32).copy()
            elif bits == 16:
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif bits == 24:
                b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
                v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
                v = np.where(v & 0x800000, v - 0x1000000, v)
                data = v.astype(np.float32) / 8388608.0
            break
        pos += 8 + sz + (sz & 1)
    return fmt_tag, ch, sr, bits, data

def write_pcm16(path, data, sr, mono=True):
    if mono and data.ndim == 2:
        data = data[:, 0]
    d = np.clip(data, -1.0, 1.0)
    pcm = (d * 32767.0).astype('<i2')
    n = len(pcm)
    ch = 1 if mono else 2
    with open(path, 'wb') as f:
        f.write(b'RIFF' + struct.pack('<I', 36 + n * ch * 2) + b'WAVE')
        f.write(b'fmt ' + struct.pack('<IHHIIHH', 16, 1, ch, sr, sr * ch * 2, ch * 2, 16))
        f.write(b'data' + struct.pack('<I', n * ch * 2))
        pcm.tofile(f)

def main():
    src_dir = sys.argv[1]
    out_dir = sys.argv[2]
    start_s = float(sys.argv[3])
    len_s = float(sys.argv[4])
    os.makedirs(out_dir, exist_ok=True)
    # auto-discover stems: any "%L.wav" file in the session audiofiles dir
    stems = []
    for fn in sorted(os.listdir(src_dir)):
        if fn.endswith('%L.wav') or fn.endswith('%L.WAV'):
            stems.append(fn[:-len('%L.wav')])
    if not stems:
        print(f'!! no stems found in {src_dir}')
        return 1
    for stem in stems:
        for side in ('L', 'R'):
            p = os.path.join(src_dir, f'{stem}%{side}.wav')
            fmt, ch, sr, bits, data = read_float_wav(p)
            if data is None:
                print(f'!! cannot read {p}'); continue
            n0 = int(start_s * sr)
            n1 = min(len(data), int((start_s + len_s) * sr))
            seg = data[n0:n1]
            tag = stem.replace(' ', '_')[:40]
            write_pcm16(os.path.join(out_dir, f'{tag}_{side}.wav'), seg, sr)
            print(f'{tag}_{side}: {len(seg)/sr:.1f}s from {p}')
    return 0

if __name__ == '__main__':
    main()
