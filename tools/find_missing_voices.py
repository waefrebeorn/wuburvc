#!/usr/bin/env python3
"""Parallel search for remaining missing voices using curl subprocess calls."""
import subprocess, re, json, sys, os, time

# Read the sitemap URLs
with open('models/vm_sitemap_urls.txt') as f:
    all_urls = [line.strip() for line in f if line.strip()]

# Sitemap structure:
# sitemap-1: URLs 0-9986 (newer, longer IDs)
# sitemap-2: URLs 9987-19984 (newer, longer IDs)  
# sitemap-3: URLs 19985-29983 (older, shorter IDs)
# sitemap-4: URLs 29984-30877 (oldest, shortest IDs like 9BW)

# The 9 voices we found were in sitemap-3 (URLs 19985-29983)
# The remaining 5 may be in sitemap-2 or sitemap-4

# Search sitemap-4 (shortest IDs) first, then sitemap-2
search_ranges = [
    ('sitemap-4 (short IDs)', all_urls[-894:]),
    ('sitemap-3 part 2', all_urls[25000:]),  # rest of sitemap-3
]

missing_kws = {
    'ArianaGrande': 'arianagrande',
    'Fase Yoda 50k': 'fase yoda',
    'Samuel L Jackson 30k': 'samuel l jackson',
    'Moonman 120k': 'moonman',
    'SonicDarkEraV2': 'sonicdarkera',
}

def fetch_and_extract(url):
    """Use curl to fetch model page, extract name and download URL."""
    mid = url.split('/')[-1]
    try:
        result = subprocess.run(
            ['curl', '-s', '-L', '--max-time', '10',
             '-H', 'User-Agent: Mozilla/5.0', url],
            capture_output=True, text=True, timeout=15
        )
        html = result.stdout
        h1 = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.DOTALL)
        if not h1:
            return mid, None, None
        name = re.sub(r'<[^>]+>', '', h1.group(1)).strip()
        name = re.sub(r'&quot;', chr(34), name)
        name = re.sub(r'&#039;', chr(39), name)
        name = re.sub(r'&amp;', '&', name)
        
        # Get download URL
        dl_url = None
        ea_url = re.search(r'easyaivoice\.com/run\?url=([^&\s"<]+)', html)
        if ea_url:
            import urllib.parse
            dl_url = urllib.parse.unquote(ea_url.group(1))
            dl_url = re.sub(r'["<>]+', '', dl_url).strip()
        if not dl_url:
            direct = re.findall(r'href=["\']?(https?://(?:huggingface\.co|drive\.google\.com|mega\.nz)[^"\'<>]+)', html)
            if direct:
                dl_url = re.sub(r'["<>]+', '', direct[-1]).strip()
        
        return mid, name, dl_url
    except Exception:
        return mid, None, None

def parallel_search(urls, keywords, batch_size=50, workers=10):
    """Search model pages in parallel using curl subprocesses."""
    import subprocess
    found = {}
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i+batch_size]
        # Launch parallel curl processes
        procs = []
        for url in batch:
            p = subprocess.Popen(
                ['curl', '-s', '-L', '--max-time', '10',
                 '-H', 'User-Agent: Mozilla/5.0', url],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            procs.append((p, url))
        
        # Collect results
        for p, url in procs:
            mid = url.split('/')[-1]
            try:
                stdout, _ = p.communicate(timeout=12)
                html = stdout.decode('utf-8', errors='replace')
                h1 = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.DOTALL)
                if h1:
                    name = re.sub(r'<[^>]+>', '', h1.group(1)).strip()
                    name = re.sub(r'&quot;', chr(34), name)
                    name = re.sub(r'&#039;', chr(39), name)
                    name = re.sub(r'&amp;', '&', name)
                    for voice, kw in keywords.items():
                        if voice in found:
                            continue
                        if kw in name.lower():
                            dl_url = None
                            ea_url = re.search(r'easyaivoice\.com/run\?url=([^&\s"<]+)', html)
                            if ea_url:
                                import urllib.parse
                                dl_url = urllib.parse.unquote(ea_url.group(1))
                                dl_url = re.sub(r'["<>]+', '', dl_url).strip()
                            found[voice] = (mid, name, dl_url)
                            print(f'FOUND: {voice} -> model/{mid}: {name[:100]}')
                            if dl_url:
                                print(f'  DL: {dl_url[:120]}')
            except:
                pass
        
        if len(found) == len(keywords):
            break
        if i % 2000 == 0 and i > 0:
            print(f'  Scanned {i}/{len(urls)}...')
    
    return found

# Load existing matches
with open('models/voice_download_matches.json') as f:
    matches = json.load(f)

# Filter to only missing voices
missing = {k: v for k, v in missing_kws.items() if k not in matches}
print(f"Missing voices: {list(missing.keys())}")

found = {}
for label, urls in search_ranges:
    if not missing:
        break
    print(f"\nSearching {label} ({len(urls)} URLs)...")
    new_found = parallel_search(urls, missing)
    found.update(new_found)
    # Remove found voices from missing
    for k in new_found:
        del missing[k]

# Add found voices to matches
for voice, (mid, name, dl_url) in found.items():
    matches[voice] = [mid, name, dl_url, '??']

with open('models/voice_download_matches.json', 'w') as f:
    json.dump(matches, f, indent=2)

print(f"\n=== Total matches: {len(matches)} ===")
for k in sorted(matches.keys()):
    data = matches[k]
    if isinstance(data, (list, tuple)):
        dl = "YES" if data[2] else "NO"
        print(f"  {k}: model/{data[0]} | DL={dl} | {data[1][:60]}")
