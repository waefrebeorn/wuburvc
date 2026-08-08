#!/usr/bin/env python3
"""wubu_wordvoice_jokes.py — WordVoice emotional TTS knock-knock jokes -> WuBuRVC.

WordVoice (CosyVoice3 0.5B) is our TTS backbone: word-level duration,
energy, pitch, boundary and tone control (the emotional engine). Each joke
is scripted with inline control tags for comedic timing, synthesized with a
reference voice (zero-shot), then converted to a character voice by the C11
WuBuRVC engine. We make our own; WordVoice's Python output is the reference
for parity.

Usage:
  python tools/wubu_wordvoice_jokes.py --voice cartman --joke 0 --out out/demo/joke_cartman.wav
"""
import argparse
import os
import subprocess
import sys

WV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "WordVoice")
sys.path.insert(0, WV_DIR)
sys.path.insert(0, os.path.join(WV_DIR, "CosyVoice"))
sys.path.insert(0, os.path.join(WV_DIR, "CosyVoice", "third_party", "Matcha-TTS"))
sys.path.insert(0, os.path.join(WV_DIR, "xxh_tools"))

from wordvoice_infer import eval as wordvoice_eval, boundary_class, tone_class  # noqa: E402
import wordvoice_infer as wvi
from get_timestamp_mmsfa import MMSFA_Aligner  # noqa: E402
from cosyvoice.cli.cosyvoice import WordVoice  # noqa: E402

# eval() reads module-level globals that wordvoice_infer only defines under
# __main__. Initialize them here (once) — the model init is the slow part.
_INIT_DONE = [False]


def _init_models():
    if _INIT_DONE[0]:
        return
    aligner_path = os.path.join(WV_DIR, "checkpoints", "mms_fa")
    cosyvoice_path = os.path.join(WV_DIR, "checkpoints", "Fun-CosyVoice3-0.5B")
    llm_path = os.path.join(WV_DIR, "checkpoints", "WordVoice-base-0.5B",
                            "wordvoice_llm_en.pt")
    flow_path = os.path.join(WV_DIR, "checkpoints", "WordVoice-base-0.5B",
                             "wordvoice_fm.pt")
    hyper_yaml_path = os.path.join(WV_DIR, "config", "wordvoice.yaml")
    wvi.Aligner_Model = MMSFA_Aligner(model_path=aligner_path)
    wvi.wordvoice = WordVoice(model_dir=cosyvoice_path, llm_path=llm_path,
                              flow_path=flow_path,
                              hyper_yaml_path=hyper_yaml_path,
                              fp16=True)
    _INIT_DONE[0] = True

VOICE_ALIASES = {
    "cartman": "models/rvc/cartman/EricCartmanV1_e650_s10400.pth",
    "bart": "models/rvc/bart/BartSimpsonKLM41.pth",
    "freddie": "models/rvc/freddie/FM_FALSETTOS_400e_7200s.pth",
    "mj": "models/rvc/mj/MJInvincibleEra.pth",
    "jackblack": "models/rvc/jackblack/jackblackrvc_e1860_s87420.pth",
}

# Each joke: list of (text_with_tags, control_dict) — inline tags carry the
# comedy: rise on questions, peak on the punchline, long boundary pause
# before the reveal. Explicit [dur:ms] keeps natural pacing (the LLM's
# default 40ms/word is far too fast).
JOKES = [
    ("[dur:240]Knock [dur:240]knock. [bnd:b4] Who's [dur:200]there? [ton:rise] "
     "Orange. [bnd:b3] Orange [dur:180]who? [ton:rrise] Orange you [dur:220]glad "
     "I didn't [dur:200]say [dur:240]banana? [ton:ffall]", {}),
    ("[dur:240]Knock [dur:240]knock. [bnd:b4] Who's [dur:200]there? [ton:rise] "
     "Doris. [bnd:b3] Doris [dur:180]who? [ton:peak] Doris locked, that's why "
     "I'm [dur:200]knocking! [ton:ffall]", {}),
    ("[dur:240]Knock [dur:240]knock. [bnd:b4] Who's [dur:200]there? [ton:rise] "
     "Tank. [bnd:b3] Tank [dur:180]who? [ton:peak] You're [dur:220]welcome! [ton:fall]", {}),
    ("[dur:240]Knock [dur:240]knock. [bnd:b4] Who's [dur:200]there? [ton:rise] "
     "Interrupting cow. [ton:rrise] Interrupting cow [dur:180]wh— [eng:0.9] "
     "MOO! [ton:ffall]", {}),
    ("[dur:240]Knock [dur:240]knock. [bnd:b4] Who's [dur:200]there? [ton:rise] "
     "Atch. [bnd:b3] Atch [dur:180]who? [ton:valley] Bless [dur:220]you! [ton:rise]", {}),
]

PROMPT_TEXT = "The team that change what they're doing. If you don't change some of the coaches or perhaps change."
PROMPT_SPEECH = os.path.join(WV_DIR, "demo", "prompt_speech_en.mp3")


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
    ap.add_argument("--joke", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cli", default="build/wubu_rvc_cli_fixed.exe")
    ap.add_argument("--no-rvc", action="store_true", help="skip RVC conversion")
    args = ap.parse_args()

    tts_text, control_dict = JOKES[args.joke % len(JOKES)]
    model_pth = VOICE_ALIASES[args.voice]
    out = args.out or f"out/demo/joke_{args.voice}_{args.joke}.wav"
    os.makedirs("out/demo", exist_ok=True)

    tts_wav = os.path.abspath("out/demo/_wordvoice_raw.wav")
    tts_p16 = os.path.abspath("out/demo/_wordvoice_p16.wav")
    os.chdir(WV_DIR)  # model paths + prompt speech are relative to the repo
    print("[1] WordVoice synthesizing with emotional control...")
    _init_models()
    wordvoice_eval(PROMPT_TEXT, PROMPT_SPEECH, tts_text, control_dict,
                   tts_wav, lan="en")
    os.chdir(os.path.join("..", ".."))
    print(f"[1] TTS raw: {tts_wav} ({os.path.getsize(tts_wav)} bytes)")

    # WordVoice writes float32 wav (format 3) — normalize to PCM16 for the CLI
    import soundfile as sf
    data, sr = sf.read(tts_wav, dtype="float32")
    sf.write(tts_p16, data, sr, subtype="PCM_16")
    print(f"[1] TTS pcm16: {len(data)/sr:.2f}s @ {sr}")

    if not args.no_rvc:
        if not convert(args.cli, tts_p16, model_pth, out):
            sys.exit(1)
        print(f"[2] {args.voice} converted: {out}")
    print("Joke text:", tts_text)


if __name__ == "__main__":
    main()
