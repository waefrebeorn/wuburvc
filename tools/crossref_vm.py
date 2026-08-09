#!/usr/bin/env python3
"""Cross-reference and sort voices by cultural popularity (optimized).

Reads:
  - out/voices.json         (curated 500 voices with popularity scores)
  - models/vm_full_index.json  (30,878 VM models)

Writes:
  - out/voices.json         (updated with vm_download_url, vm_model_id)
  - out/vm_sorted_by_popularity.json  (all 30K+ VM models ranked by cultural popularity or N/A)
  - out/vm_models_ranked.json         (full list of ranked models)
"""
import json
import re
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_VOICES = os.path.join(BASE, 'out', 'voices.json')
VM_INDEX = os.path.join(BASE, 'models', 'vm_full_index.json')
OUT_VM_SORTED = os.path.join(BASE, 'out', 'vm_sorted_by_popularity.json')
OUT_RANKED = os.path.join(BASE, 'out', 'vm_models_ranked.json')

sys.path.insert(0, os.path.join(BASE, 'tools'))
from curate_voice_catalog import normalize, popularity_score, category_from_name

# Import all char lists to pre-build a lookup
from curate_voice_catalog import (
    CARTOON_CHARS, ANIME_CHARS, MOVIE_CHARS, MUSIC_CHARS,
    YT_CHARS, GAME_CHARS, POLITICAL_CHARS, COMEDY_CHARS, TV_CHARS
)

def word_match(key, norm):
    if not key or not norm:
        return False
    key_words = key.split()
    if len(key_words) <= 1:
        return bool(re.search(r'\b' + re.escape(key) + r'\b', norm))
    norm_words = set(norm.split())
    key_words_set = set(key_words)
    return len(key_words_set & norm_words) >= max(1, len(key_words_set) // 2)

def build_vocab_lookup():
    """Pre-build a normalized name -> (pop_score, category) lookup from all char lists."""
    cat_bonus = {'cartoon': 10, 'anime': 5, 'game': 5, 'movie': 0, 'music': 0, 'personality': 0}
    lookup = {}  # normalized_name -> (best_score, category)

    all_lists = [CARTOON_CHARS, ANIME_CHARS, MOVIE_CHARS, MUSIC_CHARS,
                 YT_CHARS, GAME_CHARS, POLITICAL_CHARS, COMEDY_CHARS, TV_CHARS]

    for char_list in all_lists:
        for name, rank, cat in char_list:
            key = normalize(name)
            base = max(10, 1000 - (rank * 10))
            score = base + cat_bonus.get(cat, 0)
            if key not in lookup or score > lookup[key][0]:
                lookup[key] = (score, cat)

    return lookup

def fast_popularity_score(norm_name, vocab_lookup):
    """Fast popularity score using pre-built lookup."""
    # 1. Exact match
    if norm_name in vocab_lookup:
        return vocab_lookup[norm_name][0]

    # 2. Word match (token overlap)
    best = 0
    for vkey, (score, cat) in vocab_lookup.items():
        if word_match(vkey, norm_name) or word_match(norm_name, vkey):
            if score > best:
                best = score

    return best

def main():
    with open(OUT_VOICES) as f:
        data = json.load(f)
    voices = data['voices']

    with open(VM_INDEX) as f:
        vm_data = json.load(f)

    vm_by_name = vm_data['by_name']
    vm_index = vm_data['index']
    total_models = vm_data['total']

    print(f"Loaded {len(voices)} curated voices and {total_models} VM models")
    print("Building VOCAB lookup (fast scoring)...")
    vocab_lookup = build_vocab_lookup()
    print(f"  VOCAB entries: {len(vocab_lookup)}")

    # --- Step 1: Match 500 curated voices to VM index ---
    matched = 0
    for key, v in voices.items():
        vnorm = normalize(v['name'])
        best_entry = None
        best_score = 0

        # 1a. Exact match in by_name
        if vnorm in vm_by_name:
            for vm_id in vm_by_name[vnorm]:
                entry = vm_index.get(vm_id)
                if entry:
                    score = 100
                    if entry.get('download_url'):
                        score += 10
                    if score > best_score:
                        best_score = score
                        best_entry = {
                            'vm_id': vm_id,
                            'name': entry.get('name'),
                            'download_url': entry.get('download_url'),
                            'size': entry.get('size'),
                        }

        # 1b. Word match (token overlap)
        if not best_entry:
            for vm_name_norm, vm_ids in vm_by_name.items():
                if word_match(vnorm, vm_name_norm) or word_match(vm_name_norm, vnorm):
                    for vm_id in vm_ids:
                        entry = vm_index.get(vm_id)
                        if entry and entry.get('download_url'):
                            best_entry = {
                                'vm_id': vm_id,
                                'name': entry.get('name'),
                                'download_url': entry.get('download_url'),
                                'size': entry.get('size'),
                            }
                            best_score = 75
                            break
                        elif entry and not best_entry:
                            best_entry = {
                                'vm_id': vm_id,
                                'name': entry.get('name'),
                                'download_url': entry.get('download_url'),
                                'size': entry.get('size'),
                            }
                            best_score = 50
                    if best_entry:
                        break

        if best_entry:
            v['vm_model_id'] = best_entry['vm_id']
            v['vm_name'] = best_entry['name']
            v['vm_download_url'] = best_entry.get('download_url')
            matched += 1

    print(f"  Matched {matched} voices with VM models")

    with open(OUT_VOICES, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {OUT_VOICES}")

    # --- Sync voice_catalog.json with VM download URLs ---
    CATALOG_PATH = os.path.join(BASE, 'models', 'voice_catalog.json')
    with open(CATALOG_PATH) as f:
        catalog_data = json.load(f)

    # Build a lookup from voices.json (by name) to get VM URLs
    voices_by_name = {}
    for v in voices.values():
        voices_by_name[v['name']] = v

    # Update catalog entries
    for entry in catalog_data['voices']:
        vname = entry['name']
        if vname in voices_by_name:
            v = voices_by_name[vname]
            if v.get('vm_download_url'):
                entry['vm_model_id'] = v['vm_model_id']
                entry['vm_name'] = v.get('vm_name')
                entry['download_url'] = v['vm_download_url']

    with open(CATALOG_PATH, 'w') as f:
        json.dump(catalog_data, f, indent=2, ensure_ascii=False)
    cat_matched = sum(1 for e in catalog_data['voices'] if e.get('download_url'))
    print(f"  Updated: {CATALOG_PATH} ({cat_matched} voices with VM URLs)")

    # --- Step 2: Sort ALL 30K VM models by cultural popularity ---
    print(f"\nSorting {total_models} VM models by cultural popularity...")

    sorted_models = []
    count = 0
    for vm_id, entry in vm_index.items():
        vm_name = entry.get('name', '')
        vnorm = normalize(vm_name)

        # Calculate cultural popularity score using fast lookup
        pop_score = fast_popularity_score(vnorm, vocab_lookup)

        if pop_score > 0:
            status = 'ranked'
        else:
            pop_score = 0
            status = 'n/a'

        sorted_models.append({
            'vm_id': vm_id,
            'name': vm_name,
            'download_url': entry.get('download_url'),
            'size': entry.get('size'),
            'popularity_score': pop_score,
            'status': status,
            'normalized_name': vnorm,
        })

        count += 1
        if count % 5000 == 0:
            print(f"  Processed {count}/{total_models}...")

    # Sort: ranked first (by popularity desc, then name), then N/A (by name)
    sorted_models.sort(key=lambda x: (-x['popularity_score'], x['name']) if x['status'] == 'ranked' else (1, x['name']))

    ranked_count = sum(1 for m in sorted_models if m['status'] == 'ranked')
    na_count = total_models - ranked_count
    print(f"  Done! Ranked: {ranked_count}, N/A: {na_count}")

    # Save full ranked list
    ranked_models = [m for m in sorted_models if m['status'] == 'ranked']
    na_models = [m for m in sorted_models if m['status'] == 'n/a']

    with open(OUT_RANKED, 'w') as f:
        json.dump({
            'total': len(ranked_models),
            'description': f'VM models with cultural popularity scores > 0, sorted by score',
            'models': ranked_models,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {OUT_RANKED} ({len(ranked_models)} ranked models)")

    # Save combined output (top 5000 ranked + first 500 N/A)
    output = {
        'total': len(sorted_models),
        'ranked_count': len(ranked_models),
        'na_count': len(na_models),
        'description': f'All {total_models} VM models sorted by cultural popularity, unranked entries marked as n/a',
        'sources': ['vm_full_index'],
        'models': ranked_models[:5000] + na_models[:500],
        'all_models_count': len(sorted_models),
        'truncated_note': f'Showing top {min(5000, len(ranked_models))} ranked + 500 N/A samples. Full ranked list in vm_models_ranked.json',
    }

    with open(OUT_VM_SORTED, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {OUT_VM_SORTED}")

    # Print top 20 ranked models
    print("\nTop 20 VM models by cultural popularity:")
    for m in ranked_models[:20]:
        url = m['download_url'] or 'N/A'
        print(f"  {m['popularity_score']:4d}  {m['name']}  URL={url[:60]}")

    print("\nFirst 5 N/A VM models:")
    for m in na_models[:5]:
        print(f"  {m['name']}")

if __name__ == '__main__':
    main()
