#!/usr/bin/env python3
"""voice_wheel.py — run one clean-vocal clip through the whole voice wheel.

Every converted model in WuBuMedia/models/rvc/ (9 voices) + optional archive
dirs. Writes one output per voice; prints a summary table.

Usage: python voice_wheel.py <input.wav> [out_dir]
"""
import os, sys, glob, subprocess, wave
import numpy as np

BASE = r'C:\Users\eman5\WuBuMedia'
RVC = os.path.join(BASE, 'build', 'wubu_rvc_f0fix.exe')
PY = os.path.join(BASE, '.venv_win', 'Scripts', 'python.exe')

def voices():
    out = []
    root = os.path.join(BASE, 'models', 'rvc')
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        if not os.path.isdir(d): continue
        pth = glob.glob(os.path.join(d, '*.pth'))
        if pth:
            out.append((os.path.basename(d), d, pth[0]))
    return out

def run(in_wav, name, model_dir, pth, out_wav):
    r = subprocess.run([RVC, in_wav, model_dir, out_wav, '--model', pth,
                        '--noise', '0.33333', '--jobs', '4'],
                       capture_output=True, text=True)
    t = None
    for line in r.stdout.splitlines():
        if '[6] chunked' in line: t = line.strip()
    if not os.path.exists(out_wav):
        return None, r.stdout[-500:] + r.stderr[-500:]
    return t, None

def main():
    if len(sys.argv) < 2:
        print(__doc__); return 1
    inp = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, 'out', 'album', 'pipeline', 'wheel')
    os.makedirs(outdir, exist_ok=True)
    vs = voices()
    print(f'wheel: {len(vs)} voices on {inp}')
    results = []
    for name, d, pth in vs:
        out = os.path.join(outdir, f'wheel_{name}.wav')
        t, err = run(inp, name, d, pth, out)
        if t:
            results.append((name, out, t))
            print(f'  {name:<12} OK  {t}')
        else:
            print(f'  {name:<12} FAIL {(err or "")[:120]}')
    print('\n=== WHEEL DONE ===')
    for name, out, t in results:
        print(f'{name}: {out}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
