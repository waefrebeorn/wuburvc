#!/usr/bin/env python3
"""Acquire and organize VM models by popularity, 1000 at a time, on D: drive.

Reads:
  - out/vm_models_ranked.json (8,766 ranked models with download URLs)

Writes:
  - D:/1aivoice/model_organization_plan.json  (full organization plan)
  - D:/1aivoice/VoiceModels/ranked_XXXX/batch_manifest.json  (per-batch manifests)
  - D:/1aivoice/SizeCategories/<category>/  (directory structure)
  - D:/1aivoice/TrainingData/               (models with training data)
  - D:/1aivoice/ModelWeights/               (pure .pth weights)
  - D:/1aivoice/ArchiveBundles/             (large bundles with training data)

Acquisition strategy:
  - Batch 1 (ranks 1-1000): Top 1000 most popular models
  - Batch 2 (ranks 1001-2000): Next 1000
  - ...up to Batch 8 (ranks 7001-8000)
  - Remaining 766 models go to overflow
"""
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKED_PATH = os.path.join(BASE, 'out', 'vm_models_ranked.json')

# D: drive storage root (use Windows-style path for consistency)
D_ROOT = r'D:/1aivoice'
BATCH_DIRS = [os.path.join(D_ROOT, 'VoiceModels', f'ranked_{i}000') for i in range(1, 9)]

def parse_size(size_str):
    if not size_str or size_str == '??' or size_str == '':
        return None
    s = str(size_str).strip().lower()
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
            return val
    nums = re.findall(r'[\d.]+', s)
    if nums:
        return float(nums[0])
    return None

def categorize_size(size_mb):
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
        return 'anomaly_giant'

def detect_content_type(name, download_url):
    """Detect if a model includes training data, is pure weights, or is a bundle."""
    name_lower = name.lower()
    url_lower = (download_url or '').lower()

    # Training data indicators
    training_indicators = [
        'full package', 'training data', 'dataset', 'training files',
        'complete set', 'with data', 'all files', 'bundle',
        'source audio', 'wav files', 'training wavs', 'audio dataset',
    ]

    # Large downloads (>100MB) often include training data or are bundles
    # .pth files are pure weights

    for kw in training_indicators:
        if kw in name_lower or kw in url_lower:
            return 'training_data'

    # Based on URL extension clues
    if url_lower.endswith('.pth') or url_lower.endswith('.bin'):
        return 'weights_only'
    if url_lower.endswith('.zip') or url_lower.endswith('.rar'):
        return 'bundle'

    # Default: weights-only for RVC .pth models
    return 'weights_only'

def main():
    with open(RANKED_PATH) as f:
        data = json.load(f)
    models = data['models']

    print(f"Loaded {len(models)} ranked models")
    print(f"D: storage root: {D_ROOT}")

    # Ensure directories exist
    for d in BATCH_DIRS:
        os.makedirs(d, exist_ok=True)
    for cat in ['tiny', 'small', 'medium', 'large', 'xlarge', 'xxlarge', 'anomaly_giant', 'unknown']:
        os.makedirs(os.path.join(D_ROOT, 'SizeCategories', cat), exist_ok=True)
    os.makedirs(os.path.join(D_ROOT, 'TrainingData'), exist_ok=True)
    os.makedirs(os.path.join(D_ROOT, 'ModelWeights'), exist_ok=True)
    os.makedirs(os.path.join(D_ROOT, 'ArchiveBundles'), exist_ok=True)

    # Create batches of 1000 by popularity rank
    organization_plan = {
        'total_models': len(models),
        'batch_size': 1000,
        'batch_count': 8,
        'overflow_count': len(models) - 8000,
        'storage_root': D_ROOT,
        'batches': [],
        'size_categories': {},
        'content_types': {},
    }

    size_cat_counts = {}
    content_type_counts = {}

    for i, m in enumerate(models):
        size_mb = parse_size(m.get('size'))
        size_cat = categorize_size(size_mb)
        content_type = detect_content_type(m.get('name', ''), m.get('download_url', ''))

        # Determine batch
        batch_idx = i // 1000  # 0-7 for batches 1-8, 8+ for overflow
        if batch_idx < 8:
            batch_dir = BATCH_DIRS[batch_idx]
            batch_name = f'ranked_{batch_idx * 1000 + 1000}'
        else:
            batch_dir = os.path.join(D_ROOT, 'VoiceModels', 'overflow')
            os.makedirs(batch_dir, exist_ok=True)
            batch_name = 'overflow'

        # Determine target storage location based on content type
        if content_type == 'training_data':
            target_dir = os.path.join(D_ROOT, 'TrainingData', batch_name)
        elif content_type == 'bundle':
            target_dir = os.path.join(D_ROOT, 'ArchiveBundles', batch_name)
        else:
            target_dir = os.path.join(D_ROOT, 'ModelWeights', batch_name)
        os.makedirs(target_dir, exist_ok=True)

        # Also create size category symlinks/dirs
        size_dir = os.path.join(D_ROOT, 'SizeCategories', size_cat, batch_name)
        os.makedirs(size_dir, exist_ok=True)

        # Record in plan
        m_info = {
            'vm_id': m.get('vm_id'),
            'name': m.get('name'),
            'popularity_score': m.get('popularity_score'),
            'size': m.get('size'),
            'size_mb': size_mb,
            'size_category': size_cat,
            'content_type': content_type,
            'download_url': m.get('download_url'),
            'target_dir': target_dir,
            'batch': batch_name,
            'batch_dir': batch_dir,
        }

        # Track counts
        size_cat_counts[size_cat] = size_cat_counts.get(size_cat, 0) + 1
        content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1

        # Add to batch
        organization_plan['batches'].append(m_info)

    organization_plan['size_categories'] = size_cat_counts
    organization_plan['content_types'] = content_type_counts

    # Save organization plan
    plan_path = os.path.join(D_ROOT, 'model_organization_plan.json')
    with open(plan_path, 'w') as f:
        json.dump(organization_plan, f, indent=2, ensure_ascii=False)
    print(f"\nSaved organization plan: {plan_path}")

    # Save per-batch manifests
    all_batches = organization_plan['batches']
    for batch_idx in range(8):
        batch_start = batch_idx * 1000
        batch_end = min(batch_start + 1000, len(all_batches))
        batch = all_batches[batch_start:batch_end]
        batch_name = f'ranked_{batch_idx * 1000 + 1000}'
        batch_dir = BATCH_DIRS[batch_idx]
        manifest = {
            'batch_name': batch_name,
            'rank_range': f'{batch_idx * 1000 + 1}-{batch_idx * 1000 + len(batch)}',
            'total_models': len(batch),
            'size_distribution': {},
            'content_type_distribution': {},
            'download_urls': [m['download_url'] for m in batch if m['download_url']],
            'models': [{
                'vm_id': m['vm_id'],
                'name': m['name'],
                'popularity_score': m['popularity_score'],
                'size': m['size'],
                'size_category': m['size_category'],
                'content_type': m['content_type'],
                'download_url': m['download_url'],
                'target_dir': m['target_dir'],
            } for m in batch],
        }

        # Count distributions
        for m in batch:
            manifest['size_distribution'][m['size_category']] = manifest['size_distribution'].get(m['size_category'], 0) + 1
            manifest['content_type_distribution'][m['content_type']] = manifest['content_type_distribution'].get(m['content_type'], 0) + 1

        manifest_path = os.path.join(batch_dir, 'batch_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"  Batch {batch_idx + 1}: {len(batch)} models -> {manifest_path}")
        print(f"    Size dist: {manifest['size_distribution']}")
        print(f"    Content types: {manifest['content_type_distribution']}")
        print(f"    Models with download URLs: {len(manifest['download_urls'])}")

    # Summary
    print(f"\n=== Organization Summary ===")
    print(f"Total models: {len(models)}")
    print(f"Batches: 8 x 1000 = {min(8000, len(models))}, overflow: {len(models) - 8000}")
    print(f"\nSize distribution:")
    for cat, count in sorted(size_cat_counts.items()):
        print(f"  {cat:15s}: {count:5d} models")
    print(f"\nContent type distribution:")
    for ct, count in sorted(content_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ct:20s}: {count:5d} models")

    # Show anomaly models
    anomalies = [m for m in models if categorize_size(parse_size(m.get('size'))) == 'anomaly_giant']
    if anomalies:
        print(f"\nPotential anomalies (giant models that may contain training data):")
        for a in anomalies[:10]:
            print(f"  {a['popularity_score']:4d}  {a['name'][:60]}  size={a.get('size')}")

if __name__ == '__main__':
    main()
