#!/usr/bin/env python3
"""Check sitemap-3 models for English voice models."""
import urllib.request, re, urllib.parse, json

with open('models/vm_sitemap_urls.txt') as f:
    urls = [line.strip() for line in f if line.strip()]

# sitemap-3 is URLs 19985..29983 (middle of the file)
sitemap3 = urls[19985:29984]
print(f'sitemap-3: {len(sitemap3)} URLs')

def fetch_model(url):
    mid = url.split('/')[-1]
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8', errors='replace')
        h1 = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.DOTALL)
        if h1:
            name = re.sub(r'<[^>]+>', '', h1.group(1)).strip()
            name = re.sub(r'&quot;', chr(34), name)
            name = re.sub(r'&#039;', chr(39), name)
            name = re.sub(r'&amp;', '&', name)
            name = re.sub(r'\s+', ' ', name).strip()
        else:
            name = 'NO H1'
        return mid, name
    except:
        return mid, 'ERROR'

# Check first 30 and last 30
for label, batch in [('first 30', sitemap3[:30]), ('last 30', sitemap3[-30:])]:
    print(f'\n=== {label} ===')
    for url in batch:
        mid, name = fetch_model(url)
        print(f'  {mid}: {name[:100]}')
