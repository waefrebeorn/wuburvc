#!/usr/bin/env python3
"""Download VM models in batches of 1000, from D: drive batch manifests.

Reads:
  - D:/1aivoice/VoiceModels/ranked_XXXX/batch_manifest.json

Downloads models to:
  - target_dir from manifest (translated to D_DOWNLOAD_ROOT)
  - Or D_DOWNLOAD_ROOT/VoiceModels/ranked_XXXX/models/ if no target_dir

Environment variables:
  - WUBU_DOWNLOAD_ROOT: Override the download root (default: D:/1aivoice)
    Allows reading manifests from one drive and writing downloads to another.
  - WUBU_CONCURRENCY: Max concurrent downloads (default: 8)

Usage:
  python acquire_models.py [batch_number]   # Download specific batch (1-8)
  python acquire_models.py                  # Download all batches
"""
import json
import os
import re
import sys
import asyncio
import aiohttp
import time
import html

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D_ROOT = os.environ.get('WUBU_DOWNLOAD_ROOT', 'D:/1aivoice')
D_DOWNLOAD_ROOT = os.environ.get('WUBU_DOWNLOAD_ROOT', D_ROOT)
CONCURRENCY = int(os.environ.get('WUBU_CONCURRENCY', '8'))
BASE_URL = 'https://huggingface.co'
GOOGLE_DRIVE_PATTERN = r'https://drive\.google\.com/(?:drive/folders/|file/d/|open\?id=)([a-zA-Z0-9_-]+)'

def sanitize_filename(name):
    """Create a safe filename from a model name."""
    safe = re.sub(r'[^\w\s.-]', '', name)
    safe = re.sub(r'[\s]+', '_', safe.strip())
    return safe[:100] if safe else 'model'

def sanitize_url(url):
    """Fix common URL issues in manifests: HTML entities, blob→resolve,
    double-encoding, trailing #/? fragments, trailing punctuation."""
    if not url:
        return None

    # Decode HTML entities (e.g., &#039; -> ', &amp; -> &)
    url = html.unescape(url)

    # Replace /blob/ with /resolve/ on HuggingFace URLs
    url = re.sub(
        r'huggingface\.co/(.+?)/(.+?)/blob/(.+)',
        r'huggingface.co/\1/\2/resolve/\3',
        url
    )

    # Strip trailing '#' fragments
    url = url.rstrip('#')

    # Remove any remaining fragment identifiers after query string
    if '#' in url:
        url = url.split('#')[0]

    # Fix URLs that end with ':' or '**' (trailing punctuation from parsing)
    url = re.sub(r'[:*]+$', '', url)

    # Strip encoded trailing characters like %3E (>) at end of URL
    url = re.sub(r'%3E+$', '', url)

    # Fix double-encoded percent signs: %2520 -> %20, %252520 -> %20, etc.
    url = re.sub(r'%2525(?=\d)', '%25', url)
    url = re.sub(r'%25(?=[0-9A-Fa-f]{2})', '%', url)

    # After double-encoding fix, %3F -> ? and %3D -> = in query strings
    url = re.sub(r'%3Fdownload%3Dtrue', '?download=true', url)

    # Fix duplicate ?download=true
    url = re.sub(r'(\?download=true)\?download=true', r'\1', url)
    url = re.sub(r'(\?download=true)&download=true', r'\1', url)
    url = re.sub(r'\?download=true\?download=true', '?download=true', url)

    return url

def get_target_path(m, batch_num):
    """Determine the download target path from manifest's target_dir or default.
    
    The manifest's target_dir may use various path formats:
    - Windows drive-letter: D:/1aivoice\Path or D:\1aivoice\Path
    - Unix mount: /d/1aivoice/Path or /e/1aivoice/Path
    
    D_DOWNLOAD_ROOT (default: D:/1aivoice) replaces the drive/mount prefix.
    """
    target_dir = m.get('target_dir', '')
    if not target_dir:
        return os.path.join(D_DOWNLOAD_ROOT, 'VoiceModels', f'ranked_{batch_num}000', 'models')

    # Normalize backslashes to forward slashes
    target_normalized = target_dir.replace('\\', '/')

    # Strip the drive/mount prefix and everything up to D_DOWNLOAD_ROOT's basename
    # e.g., /d/1aivoice/ModelWeights/... -> strip /d/1aivoice -> ModelWeights/...
    # e.g., D:/1aivoice\ModelWeights/... -> strip D:/1aivoice -> ModelWeights/...
    download_root_norm = D_DOWNLOAD_ROOT.replace('\\', '/')
    download_root_basename = os.path.basename(download_root_norm)  # e.g., "1aivoice"
    
    if re.match(r'^[A-Za-z]:/?', target_normalized):
        # Windows drive-letter path: D:/1aivoice/ModelWeights/...
        rel_path = re.sub(r'^[A-Za-z]:/?', '', target_normalized)
    elif re.match(r'^/[a-z]/', target_normalized):
        # Unix mount path: /d/1aivoice/ModelWeights/...
        rel_path = re.sub(r'^/[a-z]/', '', target_normalized)
    else:
        rel_path = target_normalized.lstrip('/')

    # Strip the download root basename if it's part of the relative path
    # (target_dir includes the full path like "1aivoice/ModelWeights/...")
    if rel_path.startswith(download_root_basename + '/'):
        rel_path = rel_path[len(download_root_basename) + 1:]

    # Join with download root
    target_path = os.path.join(D_DOWNLOAD_ROOT, rel_path)
    # Normalize path separators for cross-platform consistency
    target_path = target_path.replace('\\', '/')
    return target_path

def get_extension(url, name):
    """Determine file extension from URL or name."""
    url_no_query = url.split('?')[0].rstrip('/')
    if url_no_query.endswith('.pth') or url_no_query.endswith('.bin'):
        return '.pth'
    elif url_no_query.endswith('.zip'):
        return '.zip'
    elif url_no_query.endswith('.rar'):
        return '.rar'
    elif 'google.com' in url:
        return '.zip'
    else:
        return '.pth'

async def download_single(session, url, dest_path, name, idx, total, stats):
    """Download a single file asynchronously."""
    try:
        async with session.get(url) as response:
            if response.status != 200:
                print(f'\n  [{idx+1}/{total}] HTTP {response.status}: {name[:60]}')
                stats['failed'] += 1
                return

            total_size = int(response.headers.get('content-length', 0)) if response.headers.get('content-length') else None
            downloaded = 0
            with open(dest_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = downloaded / total_size * 100
                        print(f'\r  [{idx+1}/{total}] {downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB ({pct:.0f}%) {name[:40]}', end='', flush=True)

            # Verify download
            if os.path.getsize(dest_path) < 100:
                print(f'\n  [{idx+1}/{total}] WARNING: File too small ({os.path.getsize(dest_path)} bytes): {name[:50]}')
                os.remove(dest_path)
                stats['failed'] += 1
            else:
                size_mb = os.path.getsize(dest_path) / 1024 / 1024
                print(f'\r  [{idx+1}/{total}] ✓ {size_mb:.1f}MB: {name[:60]}')
                stats['success'] += 1
    except Exception as e:
        print(f'\r  [{idx+1}/{total}] ERROR: {e}: {name[:50]}')
        stats['failed'] += 1

async def download_batch_async(batch_num, download_items):
    """Download all items in a batch with concurrency control."""
    stats = {'success': 0, 'failed': 0, 'skipped': 0}
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=CONCURRENCY * 2),
        headers={'User-Agent': 'Mozilla/5.0'}
    ) as session:

        async def bounded_download(url, dest, name, idx, total, stats):
            async with semaphore:
                await download_single(session, url, dest, name, idx, total, stats)

        tasks = []
        for item in download_items:
            task = bounded_download(
                item['url'], item['dest'], item['name'],
                item['idx'], item['total'], stats
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

    return stats

def download_batch(batch_num):
    """Download all models in a batch."""
    batch_dir = os.path.join(D_ROOT, 'VoiceModels', f'ranked_{batch_num}000')
    manifest_path = os.path.join(batch_dir, 'batch_manifest.json')

    if not os.path.exists(manifest_path):
        print(f'ERROR: Manifest not found: {manifest_path}')
        return 0, 0, 0

    with open(manifest_path) as f:
        manifest = json.load(f)

    total = manifest['total_models']
    training_data = 0
    weights_only = 0
    bundles = 0
    skipped = 0

    print(f'\n=== Batch {batch_num}: {total} models ===')
    print(f'  Size dist: {manifest["size_distribution"]}')
    print(f'  Content types: {manifest["content_type_distribution"]}')
    print(f'  With URLs: {len(manifest["download_urls"])}')
    print(f'  Download root: {D_DOWNLOAD_ROOT}')
    print(f'  Concurrency: {CONCURRENCY}')
    print()

    # Build list of download items
    download_items = []
    for i, m in enumerate(manifest['models']):
        url = m.get('download_url')
        name = m.get('name', 'unknown')
        content_type = m.get('content_type', 'unknown')
        pop_score = m.get('popularity_score', 0)

        url = sanitize_url(url)

        if content_type == 'training_data':
            training_data += 1
        elif content_type == 'bundle':
            bundles += 1
        else:
            weights_only += 1

        if not url:
            skipped += 1
            continue

        if 'drive.google.com/drive/folders' in url:
            skipped += 1
            continue

        if 'huggingface.co' in url and '/resolve/' not in url:
            skipped += 1
            continue

        if 'mega.nz' in url:
            skipped += 1
            continue

        safe_name = sanitize_filename(name)
        ext = get_extension(url, name)

        target_path = get_target_path(m, batch_num)
        os.makedirs(target_path, exist_ok=True)
        dest = os.path.join(target_path, f'{safe_name}{ext}')

        # Check if already downloaded
        if os.path.exists(dest) and os.path.getsize(dest) > 100:
            skipped += 1
            continue

        download_items.append({
            'url': url,
            'dest': dest,
            'name': name,
            'idx': i,
            'total': total,
        })

    print(f'  To download: {len(download_items)} | Skipped: {skipped}')
    print(f'  Content: training_data={training_data}, bundles={bundles}, weights={weights_only}')
    print()

    if not download_items:
        print('  Nothing to download.')
        results = {
            'batch': batch_num,
            'total': total,
            'success': 0,
            'failed': 0,
            'skipped': skipped,
            'training_data_models': training_data,
            'bundle_models': bundles,
            'weights_only_models': weights_only,
            'models_dir': batch_dir,
        }
        results_path = os.path.join(batch_dir, 'download_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        return 0, 0, skipped

    start_time = time.time()
    stats = asyncio.run(download_batch_async(batch_num, download_items))
    elapsed = time.time() - start_time

    success = stats['success']
    failed = stats['failed']

    print(f'\n=== Batch {batch_num} Complete ===')
    print(f'  Success: {success}')
    print(f'  Failed:  {failed}')
    print(f'  Skipped: {skipped}')
    print(f'  Training data models: {training_data}')
    print(f'  Bundle models: {bundles}')
    print(f'  Weights-only models: {weights_only}')
    print(f'  Time: {elapsed:.0f}s ({success / max(elapsed, 1) * 3600:.0f} files/hr)')

    results = {
        'batch': batch_num,
        'total': total,
        'success': success,
        'failed': failed,
        'skipped': skipped,
        'training_data_models': training_data,
        'bundle_models': bundles,
        'weights_only_models': weights_only,
        'models_dir': batch_dir,
    }
    results_path = os.path.join(batch_dir, 'download_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Results saved: {results_path}')

    return success, failed, skipped

def main():
    batch_nums = list(range(1, 9))
    if len(sys.argv) > 1:
        batch_nums = [int(sys.argv[1])]

    total_success = 0
    total_failed = 0
    total_skipped = 0

    for batch_num in batch_nums:
        s, f, sk = download_batch(batch_num)
        total_success += s
        total_failed += f
        total_skipped += sk

    print(f'\n{"=" * 60}')
    print(f'TOTAL ACROSS ALL BATCHES:')
    print(f'  Success: {total_success}')
    print(f'  Failed:  {total_failed}')
    print(f'  Skipped: {total_skipped}')
    print(f'{"=" * 60}')

if __name__ == '__main__':
    main()
