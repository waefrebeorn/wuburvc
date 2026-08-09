#!/usr/bin/env python3
"""Set up the proper WuBuMedia voice model repository structure.

Creates a clean, organized structure for:
- Curated 500-voice catalog with popularity scores
- 8766 ranked VM models with download URLs
- Local RVC models (21 voices from Mangio-RVC archive)
- D: drive storage for downloaded models (8 batches of 1000)
- C11 RVC inference engine (wubu_rvc.c + CUDA kernels)
- PyTorch reference engine (gen_reference_pytorch3.py)
- A/B test results
"""
import os
import json
import shutil

BASE = r'C:\Users\eman5\WuBuMedia'
D_ROOT = r'D:\1aivoice'

def create_repo_structure():
    """Create the proper repository structure."""

    # --- 1. Voice models directory structure (links to D: drive) ---
    dirs_to_create = [
        'voice_models',
        'voice_models/catalog',           # Curated 500-voice catalog
        'voice_models/catalog/local',     # 21 local RVC models
        'voice_models/catalog/vm_index',  # 8766 ranked VM models
        'voice_models/downloaded',        # Symlinks to D: drive batches
        'voice_models/research',          # Research data (BAFTA, MAL, etc.)
        'voice_models/tests',             # Test outputs (A/B test results)
    ]

    for d in dirs_to_create:
        path = os.path.join(BASE, d)
        os.makedirs(path, exist_ok=True)

    # --- 2. Document the repository structure ---
    readme = """# WuBuMedia Voice Models

AI voice model catalog and inference engine — organized by cultural popularity.

## Structure

```
WuBuMedia/
├── voice_models/          # Voice model catalog and downloads
│   ├── catalog/           # Curated voice catalog
│   │   ├── local/         # 21 local RVC models (.pth + .index)
│   │   ├── vm_index/      # 8,766 ranked VM models with download URLs
│   │   └── voice_catalog.json  # Full 500-voice ranked catalog
│   ├── downloaded/        # Downloaded models (symlinks to D:\\1aivoice)
│   │   ├── ranked_1000/   # Top 1000 most popular models
│   │   ├── ranked_2000/   # Next 1000 most popular
│   │   ├── ranked_3000/   # ...
│   │   └── overflow/      # 766 models beyond 8000
│   ├── research/          # Research data (BAFTA, MyAnimeList, SocialBlade)
│   └── tests/             # A/B test results and verification
├── src/                   # Source code
│   ├── wubu_rvc.py        # WuBuMedia RVC integration
│   └── wubu_rvc.c         # C11 RVC inference engine
├── build/                 # Compiled binaries
│   ├── test_pipeline.exe
│   ├── test_rvc_compare.exe
│   ├── test_speed_real.exe
│   └── test_quality.exe
├── tools/                 # CLI tools
│   ├── curate_voice_catalog.py  # VOCAB + popularity scorer
│   ├── crossref_vm.py          # Cross-reference with 30K VM index
│   ├── organize_models.py      # Organize into D: drive batches
│   ├── acquire_models.py       # Download models 1000 at a time
│   ├── ab_test_rvc.py          # A/B test runner
│   └── generate_ab_video.py    # A/B test video generator
├── out/                   # Generated outputs
│   ├── voices.json          # 500 curated voices with URLs
│   └── vm_models_ranked.json  # 8766 ranked VM models
└── outputs/               # Test outputs
    ├── ab_test_stats.json      # A/B test results
    ├── ab_test_stats.png       # Stats overlay image
    └── ab_test_video.mp4       # A/B test video
```

## Storage

All downloaded models are stored on the **D: drive** (1.9TB), not C:.
- D:\\1aivoice\\VoiceModels\\ranked_1000/  → Top 1000 most popular models
- D:\\1aivoice\\VoiceModels\\ranked_2000/  → Next 1000
- D:\\1aivoice\\TrainingData/             → Models with training data (93)
- D:\\1aivoice\\ModelWeights/             → Pure .pth weights (4713)
- D:\\1aivoice\\ArchiveBundles/           → Zip/rar bundles (3960)

## Engine

**WuBuRVC** — our custom RVC inference engine in C11 with CUDA kernels (sm_75).
- `src/wubu_rvc.c`         — Main engine (HiFi-GAN generator)
- `src/wubu_rvc_kernels.cu` — CUDA kernels for fused operations
- `src/wubu_rvc_mono.cu`    — Monophonic RVC conversion kernel
- Build: `cc -Wall -std=c11 -g -I src src/*.c -lm -o build/engine`

**PyTorch Reference** — Mangio-RVC fork reference implementation.
- `tools/gen_reference_pytorch3.py` — HiFiGAN generator in PyTorch

## Usage

### Run A/B test
```bash
python tools/ab_test_rvc.py
python tools/generate_ab_video.py
```

### Download models
```bash
python tools/acquire_models.py 1   # Top 1000 models
python tools/acquire_models.py     # All 8 batches
```

### Run inference
```bash
build/test_pipeline.exe    # Full pipeline test
build/test_rvc_compare.exe # Accuracy comparison
build/test_speed_real.exe  # Speed benchmark
```

License: WaefreBeorn-UMV3
"""

    readme_path = os.path.join(BASE, 'voice_models', 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme)
    print(f"Created: {readme_path}")

    # --- 3. Copy catalog files ---
    catalog_files = [
        ('out/voices.json', 'voice_models/catalog/voices.json'),
        ('models/voice_catalog.json', 'voice_models/catalog/voice_catalog.json'),
        ('models/all_voice_dirs.txt', 'voice_models/catalog/all_voice_dirs.txt'),
        ('models/vm_full_index.json', 'voice_models/catalog/vm_index/vm_full_index.json'),
        ('out/vm_models_ranked.json', 'voice_models/catalog/vm_index/vm_models_ranked.json'),
        ('out/vm_sorted_by_popularity.json', 'voice_models/catalog/vm_index/vm_sorted_by_popularity.json'),
        ('models/voice_download_matches.json', 'voice_models/catalog/vm_index/voice_download_matches.json'),
        ('D:/1aivoice/model_organization_plan.json', 'voice_models/catalog/vm_index/model_organization_plan.json'),
    ]

    for src, dst in catalog_files:
        src_path = os.path.join(BASE, src) if not src.startswith('D:') else src
        dst_path = os.path.join(BASE, dst)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  Copied: {src} -> {dst}")
        elif os.path.exists(src):
            shutil.copy2(src, dst_path)
            print(f"  Copied: {src} -> {dst}")
        else:
            print(f"  SKIP (not found): {src}")

    # --- 4. Copy research data ---
    research_files = [
        'models/popular_english_voices.json',
        'models/training_assets_manifest.json',
    ]
    for rf in research_files:
        src_path = os.path.join(BASE, rf)
        dst_path = os.path.join(BASE, 'voice_models', 'research', os.path.basename(rf))
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  Copied: {rf} -> voice_models/research/")

    # --- 5. Copy test outputs ---
    test_files = [
        'outputs/ab_test_stats.json',
        'outputs/ab_test_stats.png',
        'outputs/ab_test_video.mp4',
        'outputs/ab_test_wuburvc.wav',
        'outputs/ab_test_pytorch.wav',
        'outputs/rvc_inference_test.wav',
    ]
    for tf in test_files:
        src_path = os.path.join(BASE, tf)
        dst_path = os.path.join(BASE, 'voice_models', 'tests', os.path.basename(tf))
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  Copied: {tf} -> voice_models/tests/")

    # --- 6. Create batch manifest symlinks to D: drive ---
    print("\n  Batch manifests (pointing to D: drive):")
    for i in range(1, 9):
        batch_name = f'ranked_{i}000'
        batch_dir = os.path.join(BASE, 'voice_models', 'downloaded', batch_name)
        os.makedirs(batch_dir, exist_ok=True)

        # Create a manifest that points to D: drive
        manifest = {
            'batch_name': batch_name,
            'storage_path': f'D:\\1aivoice\\VoiceModels\\{batch_name}',
            'local_models_dir': os.path.join(batch_dir, 'models'),
            'note': 'Models downloaded to D: drive. See D:\\1aivoice\\VoiceModels\\{batch_name}\\batch_manifest.json',
        }
        manifest_path = os.path.join(batch_dir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"    {batch_name}: {manifest_path}")

    # --- 7. Create .gitignore ---
    gitignore = """# Voice model binaries (stored on D: drive)
voice_models/downloaded/*/models/

# Large data files
models/vm_full_index.json
models/voice_catalog.json
out/vm_models_ranked.json
out/vm_sorted_by_popularity.json

# Build artifacts
build/*.o
build/*.exe
build/*.log

# Python
.venv/
__pycache__/
*.pyc

# Output audio
outputs/*.wav
outputs/*.npy

# Temp
tools/__pycache__/
"""
    gitignore_path = os.path.join(BASE, '.gitignore')
    with open(gitignore_path, 'w') as f:
        f.write(gitignore)
    print(f"\n  Created: .gitignore")

    print("\n=== Repository structure created ===")
    print(f"Base: {BASE}")
    print(f"Storage: {D_ROOT}")

if __name__ == '__main__':
    create_repo_structure()
