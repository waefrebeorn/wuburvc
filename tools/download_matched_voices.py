#!/usr/bin/env python3
"""Batch download matched RVC voices from the local cache.

Reads models/voice_download_matches.json (built from the one-time full site
scrape — no re-scraping) and downloads each voice's model zip into the
pre-organized voice directory on D: (D:/1aivoice/Music-AI-Voices/<voice>/).

Uses the directory structure as the download list: each voice dir name maps
to a matched download URL from the cache.

Usage:
  python3 tools/download_matched_voices.py
  python3 tools/download_matched_voices.py --voice "Cartman"
  python3 tools/download_matched_voices.py --dry-run
  python3 tools/download_matched_voices.py --target D:/1aivoice/Music-AI-Voices

License: WaefreBeorn-UMV3
"""
import os, sys, re, json, urllib.request, urllib.parse, time, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MATCHES = os.path.join(ROOT, "models", "voice_download_matches.json")
DEFAULT_TARGET = "D:/1aivoice/Music-AI-Voices"

def normalize_path(path):
    if path.startswith("/d/"):
        return "D:" + path[2:]
    elif path.startswith("/e/"):
        return "E:" + path[2:]
    elif path.startswith("/c/"):
        return "C:" + path[2:]
    elif path.startswith("/"):
        if len(path) > 2 and path[2] == '/':
            return path[1].upper() + ":" + path[2:]
    return path

def sanitize_dirname(name):
    safe = re.sub(r'[^\w\-\s]', '', name)
    safe = re.sub(r'\s+', '_', safe.strip()).lower()
    return safe or "unknown"

def download_with_curl(url, dest_path):
    """Download using curl (better TLS + redirect handling on Windows)."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"  DL: {url[:120]}...")
    try:
        result = subprocess.run(
            ['curl', '-sS', '-L', '--max-time', '600', '--retry', '3',
             '-o', dest_path, url],
            capture_output=True, text=True, timeout=650
        )
        if result.returncode != 0:
            print(f"    [FAIL] curl rc={result.returncode}: {result.stderr[:200]}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return False
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"    [OK] {dest_path} ({size_mb:.1f}MB)")
        return True
    except Exception as e:
        print(f"    [FAIL] {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download matched RVC voices from local cache")
    parser.add_argument("--target", default=DEFAULT_TARGET,
                        help=f"Voice directory root (default: {DEFAULT_TARGET})")
    parser.add_argument("--voice", help="Download only voices matching this name")
    parser.add_argument("--dry-run", action="store_true", help="List without downloading")
    parser.add_argument("--workers", type=int, default=3, help="Parallel downloads (default 3)")
    args = parser.parse_args()

    target = normalize_path(args.target)

    with open(MATCHES) as f:
        matches = json.load(f)

    print(f"=== WuBuVoice Batch Downloader ===")
    print(f"Matches file: {MATCHES}")
    print(f"Target root:  {target}")
    print(f"Matched:      {len(matches)} voices")
    print()

    # Group by voice dir
    jobs = []
    for voice, data in matches.items():
        if args.voice and args.voice.lower() not in voice.lower():
            continue
        mid, name, dl_url, size = (data[0], data[1], data[2], data[3]) if isinstance(data, (list, tuple)) else (data.get('model_id'), data.get('name'), data.get('download_url'), '??')
        if not dl_url or dl_url == 'None':
            print(f"[SKIP] {voice}: no download URL in cache")
            continue

        # Destination: the pre-organized voice dir on the target drive
        out_dir = os.path.join(target, voice)
        os.makedirs(out_dir, exist_ok=True)

        # Filename from URL
        fname = dl_url.split('/')[-1].split('?')[0]
        fname = urllib.parse.unquote(fname)
        if not fname or not fname.endswith(('.zip', '.pth', '.index')):
            fname = sanitize_dirname(voice) + ".zip"
        dest = os.path.join(out_dir, fname)

        jobs.append((voice, name, dl_url, dest))

    if args.dry_run:
        print("=== DRY RUN ===")
        for voice, name, dl_url, dest in jobs:
            print(f"  {voice}: {name[:60]}")
            print(f"    -> {dest}")
            print(f"    {dl_url[:100]}")
        print(f"\n({len(jobs)} downloads ready)")
        return 0

    if not jobs:
        print("No downloads to do (all cached or filtered out).")
        return 0

    print(f"Downloading {len(jobs)} voices to {target}...\n")
    results = {"ok": [], "fail": []}

    # Run with limited parallelism
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def do_job(job):
        voice, name, dl_url, dest = job
        if os.path.exists(dest) and os.path.getsize(dest) > 100000:
            print(f"[SKIP] {voice}: already exists ({os.path.getsize(dest)//1024//1024}MB)")
            return voice, True, "exists"
        print(f"\n--- {voice} ---")
        print(f"  Match: {name[:70]}")
        ok = download_with_curl(dl_url, dest)
        return voice, ok, dest

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(do_job, j) for j in jobs]
        for f in as_completed(futures):
            voice, ok, info = f.result()
            if ok:
                results["ok"].append(voice)
            else:
                results["fail"].append((voice, info))

    print(f"\n=== Summary ===")
    print(f"OK:   {len(results['ok'])}")
    for v in results["ok"]:
        print(f"  + {v}")
    print(f"FAIL: {len(results['fail'])}")
    for v, why in results["fail"]:
        print(f"  - {v}: {why}")

    return 0 if not results["fail"] else 1

if __name__ == "__main__":
    sys.exit(main())
