import numpy as np, wave

def load_f0(p):
    return np.fromfile(p, dtype=np.float32)

def load_wav(p):
    w = wave.open(p, 'rb')
    return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)/32768

fw = load_f0(r'C:/Users/eman5/wuburvc/build/f0_with.bin')
fn = load_f0(r'C:/Users/eman5/wuburvc/build/f0_no.bin')
print(f"frames: with={len(fw)} no={len(fn)}  (100 fps -> {len(fw)/100:.2f}s)")

def vibrato_stats(f0, sr=100):
    """Measure vibrato rate on voiced segments of the f0 contour."""
    voiced = f0 > 50
    # find longest voiced run
    runs = []
    start = None
    for i, v in enumerate(voiced):
        if v and start is None: start = i
        if not v and start is not None:
            runs.append((start, i)); start = None
    if start is not None: runs.append((start, len(voiced)))
    runs.sort(key=lambda r: r[1]-r[0], reverse=True)
    if not runs: return {}
    s, e = runs[0]
    seg = f0[s:e]
    n = len(seg)
    if n < 30: return {}
    # detrend: subtract moving average (25 frames = 0.25s)
    ma = np.convolve(seg, np.ones(25)/25, mode='same')
    vib = seg - ma
    # dominant modulation frequency via FFT of the vibrato component
    win = np.hanning(n)
    X = np.abs(np.fft.rfft(vib * win))
    freqs = np.fft.rfftfreq(n, 1/sr)
    X[0] = 0
    peak = int(np.argmax(X))
    # vibrato depth: std of vib relative to mean f0 (cents)
    cents = 1200 * np.log2(seg / np.mean(seg))
    return {
        'longest_voiced_s': round(n/sr, 2),
        'mean_f0_hz': round(float(seg.mean()), 1),
        'vibrato_rate_hz': round(float(freqs[peak]), 2),
        'vibrato_depth_cents': round(float(np.std(vib) / np.mean(seg) * 1200), 1),
        'pitch_std_cents': round(float(cents.std()), 1),
    }

print("WITH-INDEX :", vibrato_stats(fw))
print("NO-INDEX   :", vibrato_stats(fn))
