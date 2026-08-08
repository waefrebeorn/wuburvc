#!/usr/bin/env python3
"""Compare C11 CLI output vs PyTorch reference vs C11-with-ref-f0."""
import numpy as np, wave, sys

def npy(path):
    return np.fromfile(path, dtype=np.float32)

def wav(path):
    w = wave.open(path, 'rb')
    d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)/32768.0
    w.close()
    return d

def stats(x, sr=40000):
    peak = np.max(np.abs(x))
    rms = np.sqrt(np.mean(x**2))
    # crude f0 via autocorr
    frame, hop = 2048, 512
    f0s = []
    for st in range(0, len(x)-frame, hop):
        seg = x[st:st+frame]*np.hanning(frame)
        if np.std(seg) < 1e-4: continue
        ac = np.correlate(seg, seg, 'full')[frame-1:]
        ac /= ac[0]+1e-9
        lo, hi = int(sr/400.0), int(sr/60.0)
        if hi >= len(ac): continue
        pk = lo + int(np.argmax(ac[lo:hi]))
        f0s.append(sr/pk)
    f0s = np.array(f0s)
    cv = float(np.std(f0s)/np.mean(f0s)) if len(f0s) and np.mean(f0s) > 0 else 0.0
    return peak, rms, cv, len(f0s)

def corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    return float(np.mean(a*b))

def save_wav(x, path, sr=40000):
    x16 = np.clip(x, -1, 1)
    x16 = (x16*32767).astype(np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(x16.tobytes())

files = {
    'ref_pytorch (Mangio)': npy('outputs/rvc_ref/output_audio.npy'),
    'ref_c11 (ref f0)':     npy('outputs/rvc_ref/c_gen_output.npy'),
    'cli_ab (YIN f0)':      wav('outputs/ab_nosnake_det.wav'),
}
print(f"{'file':<28} {'peak':>7} {'rms':>7} {'pitchCV':>8} {'nF0':>5}")
for name, x in files.items():
    p, r, cv, nf = stats(x)
    print(f"{name:<28} {p:>7.3f} {r:>7.4f} {cv:>8.3f} {nf:>5}")

print("\ncorrelations:")
names = list(files)
for i in range(len(names)):
    for j in range(i+1, len(names)):
        print(f"  {names[i]} vs {names[j]}: {corr(files[names[i]], files[names[j]]):.4f}")

save_wav(files['ref_pytorch (Mangio)'], 'outputs/rvc_ref/ref_pytorch.wav')
save_wav(files['ref_c11 (ref f0)'], 'outputs/rvc_ref/ref_c11.wav')
print("\nsaved: outputs/rvc_ref/ref_pytorch.wav, ref_c11.wav")
