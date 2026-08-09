#!/usr/bin/env python3
"""Analyze model sizes in the ranked VM models to find anomalies and categorize.

Reads:
  - out/vm_models_ranked.json (8,766 ranked models)

Writes:
  - out/model_size_analysis.json (size categories + anomalies)
"""
import json
import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKED_PATH = os.path.join(BASE, 'out', 'vm_models_ranked.json')
OUT_PATH = os.path.join(BASE, 'out', 'model_size_analysis.json')

def parse_size(size_str):
    """Parse a size string like '??', '55.2', '1.2 GB', '500 MB', etc. into MB."""
    if not size_str or size_str == '??' or size_str == '':
        return None

    s = str(size_str).strip().lower()

    # Handle patterns like "55.2", "1.2g", "500m", "1.2gb", "500mb"
    m = re.match(r'^([\d.]+)\s*(gb|mb|g|m|k|kb|kib|mb|gb)?$', s)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        if unit in ('gb', 'g'):
            return val * 1024
        elif unit in ('mb', 'm'):
            return val
        elif unit in ('k', 'kb', 'kib'):
            return val / 1024
        else:
            # Assume MB if no unit (RVC models are typically reported in MB)
            return val

    # Try extracting a number from the string
    nums = re.findall(r'[\d.]+', s)
    if nums:
        return float(nums[0])

    return None

def categorize_size(size_mb):
    """Categorize model size into tiers."""
    if size_mb is None:
        return 'unknown'
    elif size_mb < 50:
        return 'tiny'
    elif size_mb < 100:
        return 'small'
    elif size_mb < 200:
        return 'medium'
    elif size_mb < 500:
        return 'large'
    elif size_mb < 1000:
        return 'xlarge'
    elif size_mb < 5000:
        return 'xxlarge'
    else:
        return 'anomaly_giant'  # Possibly includes training data

def categorize_by_content(name):
    """Try to detect if a model includes training data based on its name."""
    name_lower = name.lower()

    indicators = {
        'training_data': ['full package', 'training data', 'dataset', 'training files',
                         'complete set', 'with data', 'all files', 'bundle',
                         'source audio', 'wav files', 'training wavs'],
        'pure_model': ['rvc', 'svc', 'v2', 'crepe', 'rmvpe', 'titan', 'pretrain',
                      '300 epochs', '400 epochs', '500 epochs', '600 epochs',
                      '1000 epochs', '1500 epochs', '2000 epochs', '2500 epochs'],
        'weights_only': ['.pth', 'checkpoint', 'weights', 'model.pth'],
    }

    text = name_lower
    if any(kw in text for kw in indicators['training_data']):
        return 'training_data'
    elif any(kw in text for kw in indicators['weights_only']):
        return 'weights_only'
    else:
        return 'unknown'

def main():
    with open(RANKED_PATH) as f:
        data = json.load(f)

    models = data['models']
    print(f"Analyzing {len(models)} ranked models...")

    # Parse sizes and find anomalies
    size_stats = {}
    categories = {}
    anomalies = []

    for m in models:
        size_mb = parse_size(m.get('size'))
        size_cat = categorize_size(size_mb)
        content_cat = categorize_by_content(m.get('name', ''))

        m['_size_mb'] = size_mb
        m['_size_category'] = size_cat
        m['_content_category'] = content_cat

        size_stats[size_cat] = size_stats.get(size_cat, 0) + 1
        categories[content_cat] = categories.get(content_cat, 0) + 1

        # Flag potential anomalies: large models that might contain training data
        if size_cat in ('xlarge', 'xxlarge', 'anomaly_giant') and content_cat == 'unknown':
            anomalies.append(m)
        # Also flag models with size in the "training data" text range
        if size_mb and size_mb > 1000 and 'size' in str(m.get('name', '')).lower():
            anomalies.append(m)

    print(f"\nSize distribution:")
    for cat in ['tiny', 'small', 'medium', 'large', 'xlarge', 'xxlarge', 'anomaly_giant', 'unknown']:
        if cat in size_stats:
            print(f"  {cat:15s}: {size_stats[cat]:5d} models")

    print(f"\nContent distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat:20s}: {count:5d} models")

    print(f"\nPotential anomalies (large models without clear content indicator): {len(anomalies)}")
    for a in anomalies[:10]:
        print(f"  {a['popularity_score']:4d}  {a['name'][:60]}  size={a.get('size')}  url={str(a.get('download_url', ''))[:50]}")

    # Show size ranges for ranked vs N/A models
    print(f"\nSize ranges:")
    sizes = [m['_size_mb'] for m in models if m['_size_mb'] is not None]
    if sizes:
        sizes.sort()
        print(f"  Min: {sizes[0]:.1f} MB")
        print(f"  Max: {sizes[-1]:.1f} MB")
        print(f"  Median: {sizes[len(sizes)//2]:.1f} MB")
        print(f"  90th pct: {sizes[int(len(sizes)*0.9)]:.1f} MB")
        print(f"  95th pct: {sizes[int(len(sizes)*0.95)]:.1f} MB")
        print(f"  99th pct: {sizes[int(len(sizes)*0.99)]:.1f} MB")

    # Save updated models with size analysis
    output = {
        'total': len(models),
        'size_distribution': size_stats,
        'content_distribution': categories,
        'anomalies': [
            {
                'name': a['name'],
                'popularity_score': a['popularity_score'],
                'size': a.get('size'),
                'size_mb': a['_size_mb'],
                'size_category': a['_size_category'],
                'content_category': a['_content_category'],
                'download_url': a.get('download_url'),
                'vm_id': a.get('vm_id'),
            }
            for a in anomalies
        ],
        'models': models,  # Include the full list with annotated fields
    }

    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_PATH}")

if __name__ == '__main__':
    main()
