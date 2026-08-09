#!/usr/bin/env python3
"""voice_wheel.py — run one clean-vocal clip through the whole voice wheel.

Every converted model in WuBuMedia/models/rvc/ (11 voices) + optional archive
dirs. Writes one output per voice; prints a summary table.

Usage: python voice_wheel.py <input.wav> [out_dir]
"""
import os, sys, glob, subprocess, wave
import numpy as np

BASE = r'C:\Users\eman5\WuBuMedia'
WUBU = r'C:\Users\eman5\wuburvc'
PY = os.path.join(BASE, '.venv_win', 'Scripts', 'python.exe')


def _find_cli():
    for c in ['wubu_rvc_vk.exe.exe', 'wubu_rvc_vk.exe']:
        p = os.path.join(WUBU, 'build', c)
        if os.path.isfile(p):
            return p
    return os.path.join(WUBU, 'build', 'wubu_rvc_vk.exe')


RVC = _find_cli()
HUBERT = os.path.join(BASE, 'models', 'rvc', 'hubert_weights.bin')


def voices():
    out = []
    root = os.path.join(BASE, 'models', 'rvc')
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        if not os.path.isdir(d):
            continue
        pth = glob.glob(os.path.join(d, '*.pth'))
        if pth:
            out.append((os.path.basename(d), d, pth[0]))
    return out


def run(in_wav, name, model_dir, pth, out_wav):
    # The engine exe needs MinGW runtime DLLs (libgomp-1.dll,
    # libwinpthread-1.dll) from MSYS2 — prepend to PATH for the subprocess.
    env = dict(os.environ)
    mingw = r'C:\msys64\mingw64\bin'
    if mingw not in env.get('PATH', ''):
        env['PATH'] = mingw + os.pathsep + env.get('PATH', '')
    cmd = [RVC, in_wav, model_dir, out_wav, '--model', pth,
           '--hubert', HUBERT, '--noise', '0.33333', '--jobs', '4']
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    t = None
    for line in r.stdout.splitlines():
        if '[6] chunked' in line:
            t = line.strip()
    if not os.path.exists(out_wav):
        return None, r.stdout[-500:] + r.stderr[-500:]
    # sanity: a converted clip must have audible energy (model collapsed to
    # silence is a model problem, surface it instead of shipping silence)
    try:
        import wave as _w
        with _w.open(out_wav, 'rb') as wf:
            raw = wf.readframes(min(wf.getnframes(), wf.getframerate() * 2))
        vals = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(vals ** 2))) if len(vals) else 0.0
        if rms < 1e-4:
            return None, f'{name}: converted output is SILENT (rms {rms:.2e}) — model likely overtrained/collapsed'
    except Exception as e:
        pass
    return t, None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
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
        else:
            print(f'  !! {name}: {err}')
    print()
    print(f'{"voice":<12} {"out":<55} {"time":<28}')
    for name, out, t in results:
        print(f'{name:<12} {out:<55} {t}')
    print(f'\n{len(results)}/{len(vs)} voices converted')
    return 0 if len(results) == len(vs) else 2


if __name__ == '__main__':
    sys.exit(main())
