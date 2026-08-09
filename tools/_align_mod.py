
import sys, json
from faster_whisper import WhisperModel

audio = sys.argv[1]
model = sys.argv[2] if len(sys.argv) > 2 else "base"
out = sys.argv[3] if len(sys.argv) > 3 else "align_out.json"

m = WhisperModel(model, device="cpu", compute_type="int8")
segs, _ = m.transcribe(audio, word_timestamps=True, vad_filter=True)

result = []
for s in segs:
    words = []
    for w in (s.words or []):
        txt = w.word
        dur = max(w.end - w.start, 1e-3)
        # proportional char spread: each char gets a slice of the word window.
        # This is the char-alignment refinement (finer than wordpiece timing):
        # a 200ms word with 4 chars -> ~50ms/char, enabling tight viseme sync.
        n = max(len(txt), 1)
        step = dur / n
        chars = []
        for i, ch in enumerate(txt):
            chars.append({"c": ch, "start": round(w.start + i*step, 4),
                          "end": round(w.start + (i+1)*step, 4)})
        words.append({"word": txt, "start": round(w.start,4), "end": round(w.end,4),
                      "chars": chars})
    result.append({"start": round(s.start,4), "end": round(s.end,4),
                   "text": s.text, "words": words})

with open(out, "w") as f:
    json.dump(result, f, indent=2)
print("ALIGN_DONE", out)
