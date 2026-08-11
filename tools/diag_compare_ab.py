import wave, numpy as np, sys

def load(p):
    w = wave.open(p, 'rb')
    sr = w.getframerate()
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)/32768
    return pcm, sr

a, sr = load(r'C:/Users/eman5/wuburvc/build/ab_withindex.wav')
b, _ = load(r'C:/Users/eman5/wuburvc/build/ab_noindex.wav')

n = min(len(a), len(b))
a, b = a[:n], b[:n]
print(f"samples: {n}, sr={sr}, dur={n/sr:.2f}s")
print(f"with-index   rms={np.sqrt((a**2).mean()):.4f} peak={np.abs(a).max():.4f}")
print(f"without-index rms={np.sqrt((b**2).mean()):.4f} peak={np.abs(b).max():.4f}")

# correlation
c = np.corrcoef(a, b)[0,1]
print(f"corr with-vs-without = {c:.4f}")

# spectral difference: FFT-based HF energy
def hf5k(x, sr):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1/sr)
    return X[freqs > 5000].sum() / (X.sum()+1e-12)

print(f"HF5k with={hf5k(a, sr):.4f} without={hf5k(b, sr):.4f}")

# how different are they actually? maxdiff
print(f"maxdiff={np.abs(a-b).max():.4f} meandiff={np.abs(a-b).mean():.4f}")

# voiced ratio estimate via simple energy threshold
def voiced_frac(x, sr, win=0.03):
    w = int(sr*win); nw = len(x)//w
    rms = np.array([np.sqrt((x[i*w:(i+1)*w]**2).mean()) for i in range(nw)])
    return (rms > rms.max()*0.15).mean()

print(f"voiced frac with={voiced_frac(a,sr):.3f} without={voiced_frac(b,sr):.3f}")
