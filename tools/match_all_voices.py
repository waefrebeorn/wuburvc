#!/usr/bin/env python3
"""Match all voice directories on the hard drive against the full index cache.

Reads the voice dir list (models/all_voice_dirs.txt + D: drive scan), matches
each against models/vm_full_index.json (the one-time full scrape cache), and
produces:
  - models/voice_match_report.json : full match report
  - models/voice_download_batch.json : ready-to-download matches

Matching strategy (no more scraping):
  1. Normalize both sides: lowercase, strip (RVC)/(RVC-2)/(SVC)/epoch/k/step noise
  2. Split voice dir name into tokens, check for token overlap with index names
  3. Score by matched token count; require >=2 tokens or a strong single token
  4. Prefer English models: drop names containing espanol/latino/francais/etc.

Usage:
  python3 tools/match_all_voices.py            # full run
  python3 tools/match_all_voices.py --limit 50 # first 50 dirs only

License: WaefreBeorn-UMV3
"""
import os, re, json, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INDEX = os.path.join(ROOT, "models", "vm_full_index.json")
VOICE_LIST = os.path.join(ROOT, "models", "all_voice_dirs.txt")
REPORT = os.path.join(ROOT, "models", "voice_match_report.json")
BATCH = os.path.join(ROOT, "models", "voice_download_batch.json")

NON_ENGLISH = [
    'espanol', 'latino', 'latina', 'spanish', 'francais', 'francés', 'francais',
    'alemán', 'allemand', 'deutsch', 'italiano', 'japanese', 'japonés', 'portuguese',
    'brasil', 'portugues', 'russian', 'русский', 'korean', 'jap', 'jp dub', 'jap dub',
    'chinese', 'arabic', 'turkey', 'turkish', 'polski', 'polish',
    'hindi', 'indian', 'thai', 'vietnamese', 'indonesian', 'tagalog',
]

# Languages we should still KEEP for character dubs (many popular anime/game chars
# have Japanese VA models which are the canonical voice). But the user wants
# ENGLISH-focused popular characters — so default is to prefer English, but keep
# JP VA models for characters where the JP voice IS the famous one (anime).
JP_OK = ['japanese', 'jp', 'jap']


def normalize(s):
    s = s.lower()
    s = re.sub(r'\[.*?\]', ' ', s)
    s = re.sub(r'\(.*?\)', ' ', s)
    s = re.sub(r'\b(rvc|rvc2|rvc v2|svc|gpt|sovits|gpthsovits|rmvpe|crepe|ov2|titan|pretrain|epochs?|steps?|k|unknown|version|v\d+)\b', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def tokens(s):
    return set(normalize(s).split())


def score(voice, index_name):
    """Return (score, matched_tokens) between voice dir and index model name."""
    vt = tokens(voice)
    nt = tokens(index_name)
    if not vt or not nt:
        return 0, set()
    matched = vt & nt
    # Ignore super-generic tokens
    generic = {'the', 'from', 'of', 'and', 'a', 'an', 'model', 'voice', 'rvc', 'character'}
    matched = {t for t in matched if t not in generic}
    score = len(matched)
    # Boost: if the whole voice name appears in index name
    vn = normalize(voice)
    nn = normalize(index_name)
    if vn and vn in nn:
        score += 5
    elif nn and nn in vn:
        score += 5
    return score, matched


def is_non_english(name):
    nl = name.lower()
    hits = [k for k in NON_ENGLISH if k in nl]
    # Only flag as non-english if NOT also an english dub marker
    if 'english' in nl or 'eng dub' in nl or 'eng ' in nl or ' (eng)' in nl:
        return False
    return len(hits) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-score", type=int, default=2)
    args = ap.parse_args()

    with open(INDEX) as f:
        data = json.load(f)
    index = data["index"]

    # Build name-list per voice dir
    if os.path.exists(VOICE_LIST):
        with open(VOICE_LIST) as f:
            voices = [l.strip() for l in f if l.strip()]
    else:
        voices = []

    # Also add 1v1 model dir voices
    voices = list(dict.fromkeys(voices))  # dedupe

    if args.limit:
        voices = voices[:args.limit]

    print(f"Matching {len(voices)} voice dirs against {len(index)} cached models...")

    report = {}
    batch = {}

    for i, voice in enumerate(voices):
        best = None
        best_score = 0
        candidates = []

        vn = normalize(voice)
        vt = tokens(voice)

        # Fast pass: only check index names that share at least one meaningful token
        for mid, meta in index.items():
            name = meta.get("name", "")
            if not name:
                continue
            if is_non_english(name):
                continue
            nt = tokens(name)
            overlap = vt & nt
            generic = {'the', 'from', 'of', 'and', 'a', 'an', 'model', 'voice', 'rvc', 'character'}
            overlap = {t for t in overlap if t not in generic}
            if not overlap:
                continue
            s, matched = score(voice, name)
            if s > best_score:
                best_score = s
                best = (mid, name, meta.get("download_url"), meta.get("size"))
                candidates = [(mid, name, s, matched)]

        entry = {
            "voice": voice,
            "normalized": vn,
            "best_score": best_score,
            "match": best,
        }
        report[voice] = entry

        if best_score >= args.min_score and best and best[2]:
            batch[voice] = {
                "model_id": best[0],
                "name": best[1],
                "download_url": best[2],
                "size": best[3] or "??",
            }

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(voices)}... matched so far: {len(batch)}")

    with open(REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    with open(BATCH, 'w') as f:
        json.dump(batch, f, indent=2)

    print(f"\n=== Match Report ===")
    print(f"Voice dirs:      {len(voices)}")
    print(f"Matched (DL):    {len(batch)}")
    print(f"Unmatched:       {len(voices) - len(batch)}")
    print(f"\nSaved: {REPORT}")
    print(f"Saved: {BATCH}")

    print("\n--- Matched voices ---")
    for v in sorted(batch.keys()):
        print(f"  + {v}")
    print("\n--- Top unmatched (first 30) ---")
    unmatched = [v for v in report if report[v]['best_score'] < args.min_score]
    for v in unmatched[:30]:
        print(f"  - {v}")


if __name__ == "__main__":
    main()
