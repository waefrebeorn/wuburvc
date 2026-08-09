#!/usr/bin/env python3
"""Download RVC voice models from voice-models.com using directory names as the download list.

The directory names ARE the list of what to download. Each subdirectory name under the voice root
(e.g. D:/1aivoice/Music-AI-Voices/) is a voice to download. The script:

  1. Reads directory names from the voice root (and 1v1 model dir if present)
  2. Searches voice-models.com for matching models by name
  3. Downloads .pth/.index/.zip files to models/rvc/<voice_name>/ subdirectory
  6. Records what was downloaded/skipped/failed in models/training_assets_manifest.json

The directory names ARE the list — no external file list needed.

Multi-drive support: --root reads from one drive (D:), --out writes to another (E:)
or the same drive. The script normalizes paths for both Windows and POSIX.

Usage:
  python3 tools/download_voice_models.py
  python3 tools/download_voice_models.py --dry-run
  python3 tools/download_voice_models.py --voice "Cartman"  (single voice)
  python3 tools/download_voice_models.py --root D:/1aivoice/Music-AI-Voices --out E:/rvc_models
  python3 tools/download_voice_models.py --limit 50  (download first N voices only)

License: WaefreBeorn-UMV3
"""
import os, sys, re, json, urllib.request, urllib.parse, time, hashlib
from pathlib import Path

# Multi-drive support: D: and E: drives. D: holds the voice directory list.
DEFAULT_ROOT = "D:/1aivoice/Music-AI-Voices"
DEFAULT_OUT = "models/rvc"  # relative to WuBuMedia root

VM_URL = "https://voice-models.com"
SEARCH_URL = "https://voice-models.com/search?q="

def normalize_path(path):
    """Normalize path for multi-drive environment."""
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

def normalize_voice_name(dirname):
    """Normalize a directory name for matching against voice-models.com listings."""
    s = dirname.lower()
    s = re.sub(r'\s*\(rvc\)\s*', '', s)
    s = re.sub(r'\s*\(rvc-?2\)\s*', '', s)
    s = re.sub(r'\s+\(?\d+[.,]?\d*\s*(?:epoch|epochs|k|steps)\s*', '', s)
    s = re.sub(r'\s+\\d+k\\s*$', '', s)
    s = re.sub(r'\s+v\d+\s*$', '', s)
    s = re.sub(r'\s+$', '', s)
    return s.strip()

def get_voice_dirs(root, extra_dirs=None):
    """Get all voice directory names from the root and extra directories."""
    voices = set()
    root = normalize_path(root)
    
    if os.path.isdir(root):
        for d in os.listdir(root):
            full = os.path.join(root, d)
            if os.path.isdir(full) and len(d) > 2 and not d.startswith('.'):
                voices.add(d)
                
    if extra_dirs:
        for extra in extra_dirs:
            ed = normalize_path(ed)
            if os.path.isdir(ed):
                for d in os.listdir(ed):
                    full = os.path.join(ed, d)
                    if os.path.isdir(full) and len(d) > 2 and not d.startswith('.'):
                        voices.add(d)
                        
    return sorted(voices)

def find_matching_models(voice_name, vm_index):
    """Find matching voice-models.com entries for a given voice directory name."""
    norm = normalize_voice_name(voice_name)
    matches = []
    
    for index_name, entries in vm_index.items():
        norm_idx = normalize_voice_name(index_key)
        # Match if normalized names overlap significantly
        if norm in norm_idx or norm_idx in norm:
            for entry in entries:
                mid, url, size_str, orig_name = entry
                if norm in name or norm_idx in name:
                    matches.append((name, mid, url, size_str))
                    
    return matches

def download_file(url, dest_path):
    """Download a file with progress. Handles HuggingFace redirect URLs."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if 'drive.google.com' in url:
        # Handle Google Drive confirmation
        file_id_match = re.search(r'/file/d/([^/]+)/', url)
        if file_id_match:
            url = f"https://drive.google.com/uc?id={file_id_match.group(1)}&export=download"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (total // 10 or 65536) == 0:
                        pct = downloaded * 100 / total
                        sys.stdout.write(f"\r    {pct:.0f}% ({downloaded//1024}KB/{total//1024}KB)")
                        sys.stdout.flush()
            sys.stdout.write("\n")
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"    ✓ Saved: {dest_path} ({size_mb:.1f}MB)")
        return True
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def main():
    parser = argparse.ArgumentParser(description="Download RVC voice models from voice-models.com using directory names as the list")
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help=f"Voice directory root (default: {DEFAULT_ROOT})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"Output directory for downloaded models (default: {DEFAULT_OUT})")
    parser.add_argument("--extra-root", default=DEFAULT_EXTRA_ROOT,
                        help=f"Additional voice directory root (default: {DEFAULT_EXTRA_ROOT})")
    parser.add_argument("--voice", help="Download only this specific voice")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be downloaded without fetching")
    parser.add_argument("--pages", type=int, default=5,
                        help="Number of voice-models.com browse pages to scrape (default 5)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Download at most N voices (0 = no limit)")
    args = parser.parse_args()
    
    # Normalize paths
    root = normalize_path(args.root)
    out_base = args.out
    extra_dir = normalize_path(args.extra_root) if args.extra_root else None
    
    print(f"=== WuBuVoice Model Downloader ===")
    print(f"Voice root:  {root}")
    print(f"Output dir:  {out_base}")
    print(f"Extra root:  {extra}")
    print(f"Dry run:     {args.dry_run}")
    print(f"Pages:       {args.pages}")
    print(f"Limit:       {args.limit or 'none'}")
    print()
    
    # Get voice list from directory names
    voices = get_voice_dirs(root, extra_dirs=[extra] if extra else None)
    if not voices:
        print(f"ERROR: No voice directories found in {root}")
        return 1
        
    print(f"Found {len(voices)} voice directories to process:")
    for v in voices[:20]:
        print(f"  - {v}")
    if len(voices) > 20:
        print(f"  ... and {len(voices) - 20} more")
    print()
    
    # Apply voice filter if specified
    if args.voice:
        voices = [v for v in voices if args.voice.lower() in v.lower()]
        if not voices:
            print(f"No voices matching '{args.voice}' found in {root}")
            return 1
        print(f"Filtered to: {voices}")
        print()
        
    if args.limit and args.limit > 0:
        voices = voices[:args.limit]
        print(f"Limited to first {args.limit} voices: {voices}")
        print()
        
    if args.dry_run:
        print("=== DRY RUN — would search and download ===")
        for v in voices:
            print(f"  Would fetch: {v}")
        return 0
        
    # Scrape voice-models.com index
    print(f"Scraping voice-models.com ({args.pages} pages)...")
    vm_index = scrape_voice_models(pages=args.pages)
    print(f"Indexed {sum(len(v) for v in vm_index.values())} models across {len(vm_index)} names\n")
    
    # Process each voice
    results = {"downloaded": [], "skipped": [], "failed": []}
    out_base = normalize_path(out_base)
    
    for voice in voices:
        print(f"--- {voice} ---")
        matches = find_matching_models(voice, vm_index)
        
        if not matches:
            print(f"  ⚠ No model found on voice-models.com for '{voice}'")
            results["failed"].append(voice)
            continue
            
        # Use the first match
        model_name, mid, dl_url, size_str = matches[0]
        out_dir = os.path.join(out_base_resolved, voice.replace(" ", "_").lower())
        os.makedirs(out_dir, exist_ok=True)
        
        # Determine filename
        if dl_url and 'huggingface.co' in dl_url:
            fname = urllib.parse.unquote(dl_url.split('/')[-1].split('?')[0])
        elif dl_url and ('google' in dl_url or 'mega' in dl_url):
            fname = f"{voice.replace(' ', '_')}.zip"
        else:
            fname = dl_url.split('/')[-1].split('?')[0] if dl_url else f"{voice.replace(' ', '_')}.bin"
            
        dest = os.path.join(out_dir, fname)
        
        # Skip if already exists and is valid
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"  ✓ Already exists: {dest} ({os.path.getsize(dest)//1024}KB)")
            results["skipped"].append(f"{voice} ({fname})")
            continue
            
        print(f"  Match: {model_name} ({size_str})")
        print(f"  URL: {dl_url[:120]}...")
        print(f"  Downloading: {dest}")
        
        if download_file(dl_url, dest):
            results["downloaded"].append(f"{voice} ({fname})")
        else:
            results["failed"].append(voice)
            
        time.sleep(0.3)  # Be polite to the server
        
    # Save manifest
    manifest_path = os.path.join(out_base_resolved, "..", "training_assets_manifest.json")
    manifest_path = os.path.normpath(manifest_path)
    
    # Load existing manifest
    existing = {"downloaded": [], "skipped": [], "failed": []}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            existing = json.load(f)
            
    # Merge new results
    for d in results["downloaded"]:
        if d not in existing.get("downloaded", []):
            existing.setdefault("downloaded", []).append(d)
    for s in results["skipped"]:
        if s not in existing.get("skipped", []):
            existing.setdefault("skipped", []).append(s)
    for fail in results["failed"]:
        if fail not in existing.get("failed", []):
            existing.setdefault("failed", []).append(fail)
            
    with open(manifest_path, "w") as f:
        json.dump(existing, f, indent=2)
        
    print(f"\n=== Summary ===")
    print(f"Downloaded: {len(results['downloaded'])}")
    for d in results["downloaded"]:
        print(f"  + {d}")
    print(f"Skipped (already exists): {len(results['skipped'])}")
    for s in results["skipped"]:
        print(f"  = {s}")
    print(f"Failed/missing: {len(results['failed'])}")
    for fail in results["failed"][:30]:
        print(f"  - {fail}")
    if len(results["failed"]) > 30:
        print(f"  ... and {len(results['failed']) - 30} more")
    print(f"\nManifest: {manifest_path}")
    return 0 if not results["failed"] else 1