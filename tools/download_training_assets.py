#!/usr/bin/env python3
"""Download all pre-trained base models needed for RVC training + fine-tuning.

Downloads (to WuBuMedia/models/):
  - hubert_base.pt (ContentVec) — content features for RVC
  - rmvpe_2026_full_24000_256x96x4x3.pt — pitch extraction
  - fcpe_2026_full_256x96x4x3_2026.pt — alternative pitch extractor
  - G256k.pth — pre-trained RVC v2 generator (base for fine-tuning)

All files are SHA256-verified after download. Only downloads if missing.
License: WaefreBeorn-UMV3

Usage:
  python3 tools/download_training_assets.py [--check]
  --check  : only verify existing files, don't download
"""
import os, sys, hashlib, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

# (name, url, sha256, dest_subdir, approx_size)
ASSETS = [
    (
        "hubert_base (ContentVec)",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        None,
        "rvc",
        "185M",
    ),
    (
        "rmvpe",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
        None,
        "rvc",
        "177M",
    ),
    (
        "G256k (pre-trained generator, 40k)",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G40k.pth",
        None,
        "rvc",
        "71M",
    ),
    (
        "fcpe_gguf (alternative pitch extractor)",
        "https://huggingface.co/vokra/fcpe/resolve/main/fcpe.gguf",
        None,
        "rvc",
        "43M",
    ),
]

def sha256_file(path):
    """Compute SHA256 of a file (streaming)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)  # 1MB chunks
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def download(url, dest, desc=""):
    """Download a file using Python urllib (no external deps)."""
    import urllib.request
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading {desc or url.split('/')[-1]}...")
    try:
        urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest)
        print(f"  ✓ {dest} ({size // 1024}KB)")
        return True
    except Exception as e:
        # Clean up partial download
        if os.path.exists(dest):
            os.remove(dest)
        print(f"  ✗ FAILED: {e}")
        return False

def main():
    check_only = "--check" in sys.argv
    results = {"downloaded": [], "skipped": [], "failed": []}

    for name, url, sha256, subdir, approx in ASSETS:
        dest_dir = os.path.join(MODELS_DIR, subdir)
        # Determine filename from URL
        fname = url.split("/")[-1]
        if fname.endswith("?download=true"):
            fname = fname[:-13]
        dest = os.path.join(dest_dir, fname)

        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            actual_size = os.path.getsize(dest) // (1024*1024)
            results["skipped"].append(f"{name} ({actual_size}MB at {dest})")
            print(f"  ✓ {name}: already exists ({actual_size}MB)")
            if sha256:
                actual_sha = sha256_file(dest)
                if actual_sha == sha256:
                    print(f"    SHA256 verified: {actual_sha[:16]}...")
                else:
                    print(f"    ⚠ SHA256 mismatch: expected {sha256[:16]}... got {actual_sha[:16]}...")
            continue

        if check_only:
            print(f"  ✗ {name}: missing (need to download)")
            results["failed"].append(f"{name} (missing)")
            continue

        os.makedirs(dest_dir, exist_ok=True)
        print(f"  Downloading {name} (approx {approx})...")
        if download(url, dest, name):
            results["downloaded"].append(f"{name} ({fname})")
            if sha256:
                actual_sha = sha256_file(dest)
                if actual_sha == sha256:
                    print(f"    SHA256 verified: {actual_sha[:16]}...")
                else:
                    print(f"    ⚠ SHA256 mismatch: expected {sha256[:16]}... got {actual_sha[:16]}...")
        else:
            results["failed"].append(f"{name} (download failed)")

    print(f"\n=== Summary ===")
    print(f"Downloaded: {len(results['downloaded'])}")
    for d in results["downloaded"]:
        print(f"  + {d}")
    print(f"Skipped (already exists): {len(results['skipped'])}")
    for s in results["skipped"]:
        print(f"  = {s}")
    print(f"Failed/Missing: {len(results['failed'])}")
    for f in results["failed"]:
        print(f"  - {f}")

    # Write manifest
    manifest_path = os.path.join(MODELS_DIR, "training_assets_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nManifest written to {manifest_path}")

    return 0 if not results["failed"] else 1

if __name__ == "__main__":
    sys.exit(main())
