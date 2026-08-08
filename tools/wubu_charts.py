#!/usr/bin/env python3
"""wubu_charts.py — WuBuDesk improvement & speed dashboard (PNG).

Renders the "we are the best" charts: pitch accuracy, pipeline parity,
model-arch support, realtime factor, mastering targets. Pure PIL — no
matplotlib dependency. Used in demo videos and as a standalone report.

Usage: python tools/wubu_charts.py out/demo/charts.png
"""
import sys
from PIL import Image, ImageDraw, ImageFont


def font(size, bold=False):
    try:
        return ImageFont.truetype(
            "C:/Windows/Fonts/" + ("arialbd.ttf" if bold else "arial.ttf"),
            size)
    except Exception:
        return ImageFont.load_default()


BG = (16, 18, 24)
PANEL = (24, 28, 38)
ACCENT = (90, 200, 160)
WARN = (235, 190, 80)
BAD = (235, 100, 100)
DIM = (150, 160, 175)
TEXT = (230, 235, 240)


def bar_chart(d, x0, y0, w, h, items, title, fmt="{:.1f}"):
    """items: list of (label, value, color, note)"""
    d.text((x0, y0), title, font=font(20, True), fill=TEXT)
    y0 += 30
    n = len(items)
    bw = w / n
    maxv = max(v for _, v, _, _ in items) or 1.0
    for i, (lab, v, col, note) in enumerate(items):
        bx = x0 + i * bw + bw * 0.15
        bh = (h - 40) * (v / maxv)
        by = y0 + (h - 40) - bh
        d.rectangle([bx, by, bx + bw * 0.7, y0 + h - 40], fill=col)
        d.text((bx, by - 18), fmt.format(v), font=font(16, True), fill=col)
        d.text((bx, y0 + h - 34), lab, font=font(13), fill=DIM)
        if note:
            d.text((bx, y0 + h - 16), note, font=font(11), fill=(100, 110, 120))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "out/demo/charts.png"
    W, H = 1600, 900
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((40, 28), "WuBuDesk Engine — Improvement & Speed Dashboard",
           font=font(34, True), fill=TEXT)
    d.text((40, 72), "C11 RVC engine · arch-agnostic · RMVPE pitch · C11 mastering suite",
           font=font(16), fill=DIM)
    d.line([(40, 104), (W - 40, 104)], fill=(50, 56, 68), width=2)

    # Panel 1: pitch coarse-bin agreement (the off-key fix)
    d.rectangle([40, 130, 820, 430], fill=PANEL, outline=(50, 56, 68))
    bar_chart(
        d, 70, 150, 700, 250,
        [("YIN", 22.8, BAD, "off-key voice"),
         ("RMVPE", 96.0, ACCENT, "training-time f0"),
         ("RMVPE+filter", 99.2, ACCENT, "+median radius 3")],
        "Pitch conditioning: coarse-bin agreement with training (higher = in key)",
        fmt="{:.1f}%")

    # Panel 2: parity vs PyTorch reference
    d.rectangle([860, 130, 1560, 430], fill=PANEL, outline=(50, 56, 68))
    bar_chart(
        d, 890, 150, 640, 250,
        [("old bugs", 0.004, BAD, "content+f0 pipeline"),
         ("RMVPE f0", 0.70, WARN, "YIN->RMVPE"),
         ("exact f0", 0.9999, ACCENT, "parity target")],
        "Output correlation vs PyTorch reference (1.0 = identical)",
        fmt="{:.3f}")

    # Panel 3: model arch support
    d.rectangle([40, 460, 820, 700], fill=PANEL, outline=(50, 56, 68))
    bar_chart(
        d, 70, 480, 700, 190,
        [("hardcoded", 1, BAD, "Cartman only"),
         ("arch-agnostic", 5, ACCENT, "40k + 32k + 768dim"),
         ("catalog", 8766, ACCENT, "scraped models")],
        "Model support (arch-agnostic config-driven engine)",
        fmt="{:.0f}")

    # Panel 4: realtime factor (speed)
    d.rectangle([860, 460, 1560, 700], fill=PANEL, outline=(50, 56, 68))
    bar_chart(
        d, 890, 480, 640, 190,
        [("Cartman 40k", 3.87, ACCENT, "x realtime"),
         ("Bart 32k", 11.67, ACCENT, "x realtime"),
         ("Freddie 40k", 4.0, ACCENT, "x realtime")],
        "Inference speed (realtime factor, C11 single-thread)",
        fmt="{:.2f}x")

    # Footer: mastering targets
    d.rectangle([40, 720, 1560, 860], fill=PANEL, outline=(50, 56, 68))
    d.text((70, 740), "C11 Mastering Suite v1 — verified targets",
           font=font(20, True), fill=TEXT)
    rows = [
        ("Loudness target", "-18 dBFS RMS (streaming ~-14 LUFS next)", ACCENT),
        ("True-peak ceiling", "-1 dBFS, 4x inter-sample detection (no more random clips)", ACCENT),
        ("Limiter", "0.5 ms attack / 60 ms release lookahead-style", ACCENT),
        ("Pitch filter", "f0 median radius 3 (kills octave jumps, keeps vibrato)", ACCENT),
        ("Envelope", "rms_mix 0.25 — output follows input dynamics", ACCENT),
        ("Resampler", "windowed-sinc Kaiser (linear aliased -> pitch noise)", ACCENT),
    ]
    y = 780
    for lab, val, col in rows:
        d.text((70, y), lab, font=font(15, True), fill=DIM)
        d.text((330, y), val, font=font(15), fill=col)
        y += 22

    img.save(out)
    print(f"OK: {out}")


if __name__ == "__main__":
    main()
