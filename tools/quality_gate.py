#!/usr/bin/env python3
"""Quality gate: compare every sweep WAV vs the baseline (base_3s_072.wav).
Metrics: global corr, RMS/peak ratio, mel-spectral corr, and boundary-seam
corr (correlation within +/-0.25s around each chunk seam vs the baseline) —
the place chunking artifacts show first.
"""
import numpy as np, wave, glob, os, sys

def wav(path):
    w = wave.open(path, 'rb')
    d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)/32768.0
    w.close()
    return d

def mel_corr(a, b, sr=40000, n_fft=1024, hop=256, n_mels=64, fmin=80, fmax=12000):
    """Log-mel magnitude correlation — catches spectral/timbre drift."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    def spec(x):
        # simple mel-ish filterbank via FFT magnitude bins
        win = np.hanning(n_fft)
        nf = (len(x) - n_fft) // hop + 1
        S = np.zeros((n_fft//2+1, nf))
        for i in range(nf):
            seg = x[i*hop:i*hop+n_fft] * win
            S[:, i] = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(n_fft, 1/sr)
        # mel bin edges (slaney-ish)
        mel_lo = 2595*np.log10(1+fmin/700); mel_hi = 2595*np.log10(1+fmax/700)
        edges = 700*(10**((mel_lo + (mel_hi-mel_lo)*np.arange(n_mels+1)/n_mels)/2595)-1)
        M = np.zeros((n_mels, nf))
        for m in range(n_mels):
            lo, hi = edges[m], edges[m+1]
            mask = (freqs >= lo) & (freqs < hi)
            if mask.any():
                M[m] = np.sqrt(np.mean(S[mask]**2, axis=0))
        return np.log1p(M)
    A, B = spec(a), spec(b)
    A = (A - A.mean()) / (A.std()+1e-9)
    B = (B - B.mean()) / (B.std()+1e-9)
    return float(np.mean(A*B))

def seam_corr(a, b, n_seams, sr=40000, win=0.25):
    """Correlation inside +/-win around each seam — boundary artifact probe."""
    if n_seams <= 0: return float('nan')
    n = min(len(a), len(b))
    segs = []
    for s in range(n_seams):
        c = (s+1) * n // (n_seams+1)  # approximate seam positions
        i0, i1 = max(0, c-int(win*sr)), min(n, c+int(win*sr))
        if i1 - i0 >= sr//2:
            segs.append((i0, i1))
    cs = []
    for (i0, i1) in segs:
        x, y = a[i0:i1], b[i0:i1]
        x = (x-x.mean())/(x.std()+1e-9); y = (y-y.mean())/(y.std()+1e-9)
        cs.append(float(np.mean(x*y)))
    return float(np.mean(cs))

BASE = 'out/demo/sweep/base_3s_072.wav'
if not os.path.exists(BASE):
    sys.exit(f'missing baseline: {BASE}')
base = wav(BASE)
files = sorted(glob.glob('out/demo/sweep/*.wav'))
print(f"{'file':<22} {'corr':>6} {'mel':>6} {'seam':>6} {'rmsR':>6} {'pkR':>6}  verdict")
print('-'*66)
results = []
for f in files:
    name = os.path.basename(f)[:-4]
    if name == 'base_3s_072': continue
    x = wav(f)
    n = min(len(base), len(x))
    a, b = base[:n], x[:n]
    a = (a-a.mean())/(a.std()+1e-9); b = (b-b.mean())/(b.std()+1e-9)
    corr = float(np.mean(a*b))
    mc = mel_corr(base, x)
    # seam positions: track 22.5s, chunk sizes known from sweep names
    nch = {'c3':8, 'c45':6, 'c6':4, 'c9':3, 'base':8}.get(name.split('_')[0], 4)
    sc = seam_corr(base, x, nch-1)
    rmsR = float(np.sqrt(np.mean(x**2))/np.sqrt(np.mean(base**2)))
    pkR = float(np.max(np.abs(x))/np.max(np.abs(base)))
    ok = corr >= 0.99 and mc >= 0.95 and sc >= 0.90 and 0.97 <= rmsR <= 1.03 and 0.97 <= pkR <= 1.03
    verdict = 'OK' if ok else 'CHECK'
    results.append((name, corr, mc, sc, rmsR, pkR, verdict))
    print(f"{name:<22} {corr:>6.4f} {mc:>6.4f} {sc:>6.4f} {rmsR:>6.3f} {pkR:>6.3f}  {verdict}")
print('\nPASS = corr>=0.99, mel>=0.95, seam>=0.90, rms/pk within 3%')
