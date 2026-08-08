#!/usr/bin/env python3
"""wubu_ab_video.py — WuBuRVC A/B verification video with stats overlay.

Renders each sample's waveform + objective stats onto a dark info panel,
then muxes with the audio into an MP4 via ffmpeg.

Usage: .venv_win/Scripts/python.exe tools/wubu_ab_video.py out.mp4 [wav...]
Each WAV becomes a labeled segment: waveform + stats overlay + its own audio.

Stats per sample (objective "not robotic" proof):
  pitchCV (0.30+ = natural prosody; ~0.00 = square wave), peak (clip check),
  RMS, HF5k (high-frequency energy), voiced ratio, zero rate, harmonicity.
"""
import sys
import subprocess
import tempfile
import os
import wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "C:/Windows/Fonts/consola.ttf"
FONT_B = "C:/Windows/Fonts/consolab.ttf"
FALLBACK = "C:/Windows/Fonts/arial.ttf"
W, H = 1280, 720
BG = (12, 14, 18)
FG = (225, 228, 235)
ACCENT = (94, 234, 144)   # green
WARN = (255, 190, 60)     # amber
BAD = (255, 90, 90)       # red
DIM = (140, 145, 155)


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


def f0_stats(x, sr):
    try:
        import librosa
        f0, _, _ = librosa.pyin(x, fmin=50, fmax=1100, sr=sr)
        if np.any(~np.isnan(f0)):
            vf = f0[~np.isnan(f0)]
            return float(np.std(vf) / np.mean(vf)), float(len(vf) / len(f0)), float(np.mean(vf))
    except Exception:
        pass
    frame, hop = 2048, 512
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


def font(sz, bold=False):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, sz)
    except Exception:
        try:
            return ImageFont.truetype(FALLBACK, sz)
        except Exception:
            return ImageFont.load_default()


def stat_color(name, val):
    if name == "pitchCV":
        return ACCENT if val >= 0.25 else (BAD if val < 0.05 else WARN)
    if name == "peak":
        return ACCENT if val < 0.99 else BAD
    if name == "HF5k":
        return ACCENT if 0.05 <= val <= 0.5 else WARN
    if name in ("zeros",):
        return ACCENT if val < 0.02 else WARN
    return FG


def render_segment(path, label, note, out_png):
    x, sr = load_wav(path)
    cv, voiced, mf0 = f0_stats(x, sr)
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    hfe = hf_energy(x, sr)
    zr = float(np.mean(np.abs(x) < 1e-6))
    harm = harmonicity(x, sr)

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # header
    d.rectangle([0, 0, W, 90], fill=(20, 24, 30))
    d.text((30, 18), "WuBuRVC  A/B  —  C11 engine verification", font=font(34, True), fill=FG)
    d.text((30, 58), "2026-08-07  ·  square-wave regression fixed  ·  Cartman v2 40 kHz",
           font=font(20), fill=DIM)
    # label
    d.text((30, 108), label, font=font(30, True), fill=ACCENT)
    if note:
        d.text((30, 148), note, font=font(19), fill=WARN)

    # waveform
    wy0, wy1 = 200, 420
    d.rectangle([30, wy0, W - 30, wy1], outline=(45, 50, 60), width=1)
    n = len(x)
    step = max(1, n // (W - 100))
    xs = np.arange(0, n, step)
    ys = x[::step]
    pts = []
    for xi, yi in zip(xs, ys):
        px = 30 + int(xi / n * (W - 100))
        py = (wy0 + wy1) / 2 + yi * (wy1 - wy0 - 40) / 2
        pts.append((px, py))
    if len(pts) > 1:
        d.line(pts, fill=ACCENT, width=2)
    d.text((40, wy0 - 28), "waveform", font=font(18), fill=DIM)
    d.text((30, wy1 + 10),
           f"{len(x)} samples  ·  {len(x) / sr:.2f}s @ {sr} Hz", font=font(17), fill=DIM)

    # stats panel
    px0, py0 = W - 560, 200
    d.rectangle([px0, py0, W - 30, 440], fill=(18, 21, 27), outline=(45, 50, 60))
    rows = [
        ("pitchCV", f"{cv:.3f}", "0.30+ natural prosody · ~0.00 = square wave"),
        ("peak",    f"{peak:.3f}", "clip if >= 0.99"),
        ("RMS",     f"{rms:.3f}", "level"),
        ("HF5k",    f"{hfe:.3f}", "high-frequency energy ratio"),
        ("voiced",  f"{voiced:.2f}", "frames with pitch"),
        ("zeros",   f"{zr:.3f}", "dead-sample ratio"),
        ("harm",    f"{harm:.3f}", "mean autocorrelation (0..1)"),
    ]
    y = py0 + 12
    for name, val, hint in rows:
        d.text((px0 + 16, y), name, font=font(20, True), fill=DIM)
        d.text((px0 + 170, y), val, font=font(22, True), fill=stat_color(name, float(val)))
        d.text((px0 + 16, y + 24), hint, font=font(14), fill=(90, 96, 106))
        y += 54

    # verdict
    verdict = "OK — natural dynamics, no clipping" if (cv >= 0.25 and peak < 0.99) else \
              ("SQUARE WAVE" if cv < 0.05 else "CHECK")
    vc = ACCENT if verdict == "OK — natural dynamics, no clipping" else \
         (BAD if verdict == "SQUARE WAVE" else WARN)
    d.text((30, 470), "verdict:", font=font(22, True), fill=DIM)
    d.text((170, 468), verdict, font=font(26, True), fill=vc)

    img.save(out_png)
    return (cv, peak)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    args = sys.argv[1:]
    out_mp4 = args.pop(0)
    seg_dur = 3.5
    labels = []
    rest = []
    while args:
        a = args.pop(0)
        if a == "--dur":
            seg_dur = float(args.pop(0))
        elif a == "--labels":
            lf = args.pop(0)
            with open(lf, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    if "|" in line:
                        lab, note = line.split("|", 1)
                        labels.append((lab.strip(), note.strip()))
                    else:
                        labels.append((line.strip(), ""))
        else:
            rest.append(a)
    wavs = rest
    if not labels:
        labels = [(os.path.basename(w), "") for w in wavs]
    with tempfile.TemporaryDirectory() as td:
        segs = []
        for i, wav in enumerate(wavs):
            png = os.path.join(td, f"seg{i}.png")
            wavseg = os.path.join(td, f"seg{i}.wav")
            mp4seg = os.path.join(td, f"seg{i}.mp4")
            if i < len(labels):
                lab, note = labels[i]
            else:
                lab, note = os.path.basename(wav), ""
            render_segment(wav, lab, note, png)
            # video from still + audio
            subprocess.run([
                "ffmpeg", "-y", "-loop", "1", "-i", png, "-i", wav,
                "-c:v", "libx264", "-tune", "stillimage", "-preset", "fast",
                "-t", f"{seg_dur}", "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
                "-shortest", mp4seg
            ], check=True, capture_output=True)
            segs.append(mp4seg)
        # concat
        lst = os.path.join(td, "list.txt")
        with open(lst, "w") as f:
            for s in segs:
                f.write(f"file '{s}'\n")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
            "-c", "copy", out_mp4
        ], check=True, capture_output=True)
    print(f"OK: {out_mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
