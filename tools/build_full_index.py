#!/usr/bin/env python3
"""One-time full index build from the saved sitemap list.

Reads models/vm_sitemap_urls.txt (the full 30,878-URL scrape we already did)
and fetches each model page ONCE to extract name + download URL, saving the
complete index to models/vm_full_index.json. Afterwards, every lookup uses
the cache — no more scraping.

This is the final full pass. Subsequent runs of download_matched_voices.py
or download_voice_models.py should load vm_full_index.json instead of
re-fetching model pages.

Usage:
  python3 tools/build_full_index.py            # full 30,878 (hours)
  python3 tools/build_full_index.py --max 5000 # first 5000 (quick test)
  python3 tools/build_full_index.py --workers 40
  python3 tools/build_full_index.py --check "ArianaGrande"  # search cached index

License: WaefreBeorn-UMV3
"""
import os, sys, re, json, time, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SITEMAP = os.path.join(ROOT, "models", "vm_sitemap_urls.txt")
INDEX_OUT = os.path.join(ROOT, "models", "vm_full_index.json")


def fetch_model_page(url):
    """Fetch a model page with curl, extract mid/name/download_url/size."""
    mid = url.split('/')[-1]
    try:
        result = subprocess.run(
            ['curl', '-s', '-L', '--max-time', '12',
             '-H', 'User-Agent: Mozilla/5.0', url],
            capture_output=True, text=True, timeout=20
        )
        html = result.stdout
        h1 = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.DOTALL)
        if not h1:
            return mid, None, None, None
        name = re.sub(r'<[^>]+>', '', h1.group(1)).strip()
        name = re.sub(r'&quot;', '"', name)
        name = re.sub(r'&#039;', "'", name)
        name = re.sub(r'&amp;', '&', name)
        name = re.sub(r'\s+', ' ', name).strip()

        dl_url = None
        ea = re.search(r'easyaivoice\.com/run\?url=([^&\s"<]+)', html)
        if ea:
            import urllib.parse
            dl_url = urllib.parse.unquote(ea.group(1))
            dl_url = re.sub(r'["<>]+', '', dl_url).strip()
        if not dl_url:
            direct = re.findall(r'href=["\']?(https?://(?:huggingface\.co|drive\.google\.com|mega\.nz)[^"\'<>]+)', html)
            if direct:
                dl_url = re.sub(r'["<>]+', '', direct[-1]).strip()

        sizes = re.findall(r'(\d+(?:\.\d+)?)\s*(MB|GB)', html)
        size_str = f"{sizes[0][0]} {sizes[0][1]}" if sizes else "??"
        return mid, name, dl_url, size_str
    except Exception:
        return mid, None, None, None


def build_index(max_pages=None, workers=30):
    with open(SITEMAP) as f:
        urls = [line.strip() for line in f if line.strip()]
    if max_pages:
        urls = urls[:max_pages]

    print(f"Building full index from {len(urls)} sitemap URLs ({workers} workers)...")
    index = {}  # mid -> {name, download_url, size}
    t0 = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_model_page, u) for u in urls]
        for f in as_completed(futures):
            mid, name, dl_url, size = f.result()
            done += 1
            if done % 500 == 0:
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                eta = (len(urls) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(urls)} ({rate:.1f}/s, ETA {eta/60:.0f}min)")
            if mid and name:
                index[mid] = {"name": name, "download_url": dl_url, "size": size}

    # Also build a name->[mids] reverse lookup
    by_name = {}
    for mid, meta in index.items():
        key = meta["name"].lower().replace('(', '').replace(')', '').strip()
        by_name.setdefault(key, []).append(mid)

    out = {"index": index, "by_name": by_name,
           "built_at": time.time(), "total": len(index)}
    with open(INDEX_OUT, 'w') as f:
        json.dump(out, f)
    print(f"\nIndex saved: {INDEX_OUT}")
    print(f"  {len(index)} models indexed from {len(urls)} URLs")
    return index, by_name


def check_index(query):
    """Search the cached full index for a name query."""
    if not os.path.exists(INDEX_OUT):
        print("No index yet. Run without --check first.")
        return
    with open(INDEX_OUT) as f:
        data = json.load(f)
    by_name = data["by_name"]
    index = data["index"]
    q = query.lower().replace('(', '').replace(')', '').strip()
    hits = []
    for key, mids in by_name.items():
        if q in key or key in q:
            for mid in mids[:3]:
                m = index[mid]
                hits.append((key, mid, m["name"], m["download_url"]))
    print(f"Hits for '{query}': {len(hits)}")
    for key, mid, name, dl in hits[:10]:
        print(f"  {mid}: {name[:80]}")
        if dl:
            print(f"    {dl[:120]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the one-time full index from saved sitemap")
    ap.add_argument("--max", type=int, default=0, help="Max URLs to process (0=all)")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--check", help="Search cached index for this name")
    args = ap.parse_args()

    if args.check:
        check_index(args.check)
    else:
        build_index(max_pages=args.max or None, workers=args.workers)
