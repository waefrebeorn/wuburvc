#!/usr/bin/env python3
"""wubu_tts_jokes.py — TTS knock-knock jokes with WuBu character voices.

Pipeline: text -> Windows SAPI TTS (native, instant, no downloads) -> WuBuRVC
character conversion (--voice selects the RVC model dir). The character then
tells the joke. WordVoice/CosyVoice3 (0.5B, word-level pitch/tone/energy
control) is the planned emotional backbone — this is the fast local path.

Usage:
  python tools/wubu_tts_jokes.py --voice cartman --out out/demo/joke_cartman.wav
"""
import argparse
import os
import subprocess
import sys
import tempfile

JOKES = [
    ("Knock knock.", "Who's there?", "Orange.", "Orange who?",
     "Orange you glad I didn't say banana?"),
    ("Knock knock.", "Who's there?", "Doris.", "Doris who?",
     "Doris locked, that's why I'm knocking!"),
    ("Knock knock.", "Who's there?", "Tank.", "Tank who?",
     "You're welcome!"),
    ("Knock knock.", "Who's there?", "Interrupting cow.", "Interrupting cow wh—",
     "MOO!"),
    ("Knock knock.", "Who's there?", "Atch.", "Atch who?",
     "Bless you!"),
]

VOICE_ALIASES = {
    "cartman": "models/rvc/cartman/EricCartmanV1_e650_s10400.pth",
    "bart": "models/rvc/bart/BartSimpsonKLM41.pth",
    "freddie": "models/rvc/freddie/FM_FALSETTOS_400e_7200s.pth",
    "mj": "models/rvc/mj/MJInvincibleEra.pth",
    "jackblack": "models/rvc/jackblack/jackblackrvc_e1860_s87420.pth",
}


def sapi_speak(text, out_wav, rate=0):
    """Windows SAPI TTS -> 16-bit wav via PowerShell."""
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.Rate = {rate};"
        f"$s.SetOutputToWaveFile('{out_wav}');"
        f"$s.Speak({text!r});"
        "$s.Dispose()"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("SAPI error:", r.stderr)
        return False
    return os.path.exists(out_wav)


def convert(cli, joke_wav, model_pth, out_wav):
    model_dir = os.path.dirname(model_pth)
    cmd = [cli, joke_wav, model_dir, out_wav,
           "--model", model_pth, "--noise", "0.25"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("RVC error:", r.stderr[-500:])
        return False
    return os.path.exists(out_wav)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="cartman", choices=list(VOICE_ALIASES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--joke", type=int, default=0)
    ap.add_argument("--cli", default="build/wubu_rvc_cli_fixed.exe")
    ap.add_argument("--rate", type=int, default=0, help="SAPI rate -10..10")
    args = ap.parse_args()

    script = JOKES[args.joke % len(JOKES)]
    text = " ".join(script)
    model_pth = VOICE_ALIASES[args.voice]
    out = args.out or f"out/demo/joke_{args.voice}_{args.joke}.wav"
    os.makedirs("out/demo", exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tts_wav = tf.name

    if not sapi_speak(text, tts_wav, rate=args.rate):
        sys.exit(1)
    print(f"[1] SAPI TTS: {tts_wav} ({os.path.getsize(tts_wav)} bytes)")

    if not convert(args.cli, tts_wav, model_pth, out):
        sys.exit(1)
    print(f"[2] {args.voice} converted: {out}")
    os.unlink(tts_wav)
    print("Joke:", " | ".join(script))


if __name__ == "__main__":
    main()
