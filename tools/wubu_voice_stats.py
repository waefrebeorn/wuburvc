#!/usr/bin/env python3
"""wubu_voice_stats.py — voice-quality stats for WuBuRVC A/B verification.

Prints objective metrics for a WAV file so we can prove (Triple-DA) that an
output is NOT a square wave / robotic:
  - pitch CV        : std/mean of voiced F0 (0.30+ = natural prosody;
                      ~0.00 = flat/square wave artifact)
  - HF energy       : fraction of spectral energy above 5 kHz
  - peak / RMS      : clipping check (peak < 0.99 = no clip)
  - zero rate       : fraction of exact zeros (high = dead samples)
  - voiced ratio    : fraction of frames with detected pitch
  - harmonicity     : mean HNR-ish measure via autocorrelation (0-1)

Usage: python wubu_voice_stats.py file1.wav [file2.wav ...]
Run with .venv_win/Scripts/python.exe (needs numpy; librosa optional).
"""
import sys
import numpy as np
import wave


def load_wav(path):
    w = wave.open(path, "rb")
    n = w.getnframes()
    sr = w.getframerate()
    ch = w.getnchannels()
    d = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    w.close()
    if ch > 1:
        d = d.reshape(-1, ch).mean(axis=1)
    return d, sr


def pitch_cv(x, sr):
    """PYIN-free F0: autocorrelation in 60-400 Hz band on 50ms frames."""
    try:
        import librosa
        f0, _, _ = librosa.pyin(x, fmin=50, fmax=1100, sr=sr)
    except Exception:
        f0 = None
    if f0 is not None and np.any(~np.isnan(f0)):
        vf = f0[~np.isnan(f0)]
        cv = float(np.std(vf) / np.mean(vf)) if np.mean(vf) > 0 else 0.0
        voiced = float(len(vf) / len(f0))
        return cv, voiced, float(np.mean(vf))
    # fallback: crude autocorrelation f0
    frame = 2048
    hop = 512
    f0s = []
    for st in range(0, len(x) - frame, hop):
        seg = x[st:st + frame] * np.hanning(frame)
        if np.std(seg) < 1e-4:
            continue
        ac = np.correlate(seg, seg, "full")[frame - 1:]
        ac /= ac[0] + 1e-9
        lo, hi = int(sr / 400.0), int(sr / 60.0)
        if hi >= len(ac):
            continue
        pk = lo + int(np.argmax(ac[lo:hi]))
        f0s.append(sr / pk)
    if not f0s:
        return 0.0, 0.0, 0.0
    f0s = np.array(f0s)
    return float(np.std(f0s) / np.mean(f0s)), float(len(f0s)), float(np.mean(f0s))


def hf_energy(x, sr):
    spec = np.fft.rfft(x * np.hanning(len(x)))
    mag = np.abs(spec)
    n = len(mag)
    hi = mag[int(n * 5000.0 / (sr / 2)):]
    return float(hi.sum() / (mag.sum() + 1e-12))


def harmonicity(x, sr):
    """Mean normalized autocorrelation at the local pitch lag (0..1)."""
    frame, hop = 2048, 512
    h = []
    for st in range(0, len(x) - frame, hop):
        seg = x[st:st + frame] * np.hanning(frame)
        if np.std(seg) < 1e-4:
            continue
        ac = np.correlate(seg, seg, "full")[frame - 1:]
        ac /= ac[0] + 1e-9
        lo, hi = int(sr / 400.0), int(sr / 60.0)
        if hi >= len(ac):
            continue
        h.append(float(np.max(ac[lo:hi])))
    return float(np.mean(h)) if h else 0.0


def stats(path):
    x, sr = load_wav(path)
    cv, voiced, mean_f0 = pitch_cv(x, sr)
    hfe = hf_energy(x, sr)
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    zr = float(np.mean(np.abs(x) < 1e-6))
    harm = harmonicity(x, sr)
    n = len(x)
    return {
        "file": path,
        "samples": n,
        "dur_s": n / sr,
        "peak": peak,
        "rms": rms,
        "pitch_cv": cv,
        "mean_f0": mean_f0,
        "voiced": voiced,
        "hf_5k": hfe,
        "zero_rate": zr,
        "harm": harm,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    rows = [stats(p) for p in sys.argv[1:]]
    print(f"{'file':<28} {'peak':>6} {'rms':>6} {'pitchCV':>8} {'f0':>6} "
          f"{'voiced':>7} {'HF5k':>6} {'zeros':>6} {'harm':>6}")
    for r in rows:
        print(f"{r['file']:<28} {r['peak']:>6.3f} {r['rms']:>6.3f} "
              f"{r['pitch_cv']:>8.3f} {r['mean_f0']:>6.1f} "
              f"{r['voiced']:>7.2f} {r['hf_5k']:>6.3f} {r['zero_rate']:>6.3f} "
              f"{r['harm']:>6.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
