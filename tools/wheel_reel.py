#!/usr/bin/env python3
"""wheel_reel.py — stitch the wheel outputs into one demo reel with voice labels.

Each voice clip is followed by a short silence gap; a spoken/beep-free simple
assembly. Writes wheel_reel.wav at the input clip's sample rate.

Usage: python tools/wheel_reel.py [wheel_dir] [out_wav]
"""
import os, sys, wave, struct

BASE = r'C:\Users\eman5\WuBuMedia'
WHEEL = os.path.join(BASE, 'out', 'album', 'pipeline', 'wheel')


def read_wav(p):
    with wave.open(p, 'rb') as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        data = w.readframes(w.getnframes())
    if ch == 1:
        data = b''.join(data[i:i + 2] + data[i:i + 2] for i in range(0, len(data), 2))
    return sr, data


def main():
    wheel_dir = sys.argv[1] if len(sys.argv) > 1 else WHEEL
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(wheel_dir, 'wheel_reel.wav')
    files = sorted(f for f in os.listdir(wheel_dir) if f.startswith('wheel_') and f.endswith('.wav') and 'reel' not in f)
    # reorder: cleveland, peter, seth, cartman, bart, freddie, mj, jackblack, miku, mj83k
    order = {'cleveland': 0, 'peter': 1, 'seth': 2, 'cartman': 3, 'bart': 4,
             'freddie': 5, 'mj': 6, 'jackblack': 7, 'miku': 8, 'mj83k': 9}
    files.sort(key=lambda f: order.get(f.replace('wheel_', '').replace('.wav', ''), 99))
    sr = None
    chunks = []
    for f in files:
        p = os.path.join(wheel_dir, f)
        try:
            s, d = read_wav(p)
        except Exception as e:
            print(f'!! {f}: {e}')
            continue
        if sr is None:
            sr = s
        # voice label via a 220Hz tone burst (100ms) so segments are audible
        tone = b''.join(struct.pack('<h', int(9000 * __import__('math').sin(2 * 3.14159 * 220 * (i / s)) * (1 if i < int(s * 0.1) else 0)))
                       for i in range(int(s * 0.15)))
        chunks.append(tone)
        chunks.append(d)
        chunks.append(b'\x00\x00' * int(s * 0.25) * 2)  # 0.25s silence
    if not chunks or sr is None:
        print('no clips'); return 1
    with wave.open(out, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b''.join(chunks))
    print(f'reel: {len(files)} voices -> {out} ({sr} Hz)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
