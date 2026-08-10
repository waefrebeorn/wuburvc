#!/usr/bin/env python3
"""Fast word-dense section finder (no autocorrelation — spectral only).
Scores 15s windows by: voiced/unvoiced transitions via RMS envelope dips
(consonant gaps), syllable rate via envelope modulation, and HF energy
(zero-crossing rate) — words have consonant bursts between vowels."""
import wave, struct, math, sys, os

def load(path):
    w = wave.open(path, 'rb')
    n = w.getnframes(); d = w.readframes(n)
    sr = w.getframerate()
    return sr, [x / 32768.0 for x in struct.unpack('<%dh' % n, d)]

def analyze(path):
    sr, x = load(path)
    hop = int(sr * 0.010)   # 10ms frames
    win = int(sr * 0.030)   # 30ms analysis window
    n = len(x)
    rms = []
    zcr = []
    for s in range(0, n - win, hop):
        seg = x[s:s + win]
        e = sum(v * v for v in seg) / win
        rms.append(math.sqrt(e))
        zc = sum(1 for i in range(1, win) if (seg[i] >= 0) != (seg[i-1] >= 0))
        zcr.append(zc / win)
    nf = len(rms)
    # voiced flag: rms above adaptive threshold (0.25 * max local)
    # use global noise floor + adaptive: 0.02 absolute OR 0.2 of local max
    vmax = max(rms) if rms else 1.0
    voiced = [1 if r > 0.02 and r > 0.15 * vmax else 0 for r in rms]
    trans = [0] * nf
    for i in range(1, nf):
        if voiced[i] != voiced[i-1]:
            trans[i] = 1
    # 15s window = 1500 frames
    wframes = int(15.0 / 0.010)
    half = wframes // 2
    best = None
    for s in range(0, nf - wframes, half):
        t = sum(trans[s:s+wframes])
        v = sum(voiced[s:s+wframes])
        hf = sum(zcr[s:s+wframes]) / wframes
        rmsavg = sum(rms[s:s+wframes]) / wframes
        tps = t / 15.0
        vr = v / wframes
        # words: many transitions, voiced balance not all-sustain, HF bursts
        score = tps * (1.0 + 6.0 * hf) * (0.5 + rmsavg * 20.0) * (1.0 - abs(vr - 0.5))
        if best is None or score > best[0]:
            best = (score, s * 0.010, tps, vr, hf, rmsavg)
    return best

def main():
    files = sorted(sys.argv[1:])
    results = []
    for f in files:
        if not os.path.exists(f):
            print('MISS', f); continue
        b = analyze(f)
        if not b:
            print('NO-SCORE', f); continue
        score, start_s, tps, vr, hf, ra = b
        results.append((score, f, start_s, tps, vr, hf, ra))
        print('%-30s score=%7.2f start=%7.1fs trans/s=%5.2f voiced=%5.2f HF=%5.3f rms=%5.3f' %
              (os.path.basename(f), score, start_s, tps, vr, hf, ra))
    results.sort(reverse=True)
    if results:
        score, f, start_s, tps, vr, hf, ra = results[0]
        print('\nBEST: %s @ %.1fs (score %.2f, %.2f trans/s)' % (f, start_s, score, tps))

if __name__ == '__main__':
    main()
