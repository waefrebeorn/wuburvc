#!/usr/bin/env python3
"""Search the 30K VM index for popular character names."""
import json, re

INDEX = r'C:\Users\eman5\WuBuMedia\models\vm_full_index.json'

with open(INDEX) as f:
    data = json.load(f)

items = list(data['index'].items())
print(f'Total VM models: {len(items)}')

search_terms = [
    'mickey', 'homer', 'simpsons', 'cartman', 'spongebob', 'bugs bunny', 'daffy', 'shrek',
    'donald duck', 'batman', 'superman', 'spider', 'star wars', 'darth', 'yoda', 'sonic',
    'pikachu', 'mario', 'link', 'zelda', 'pokemon', 'naruto', 'goku', 'luffy', 'levi', 'eren',
    'iron man', 'captain america', 'thor', 'joker', 'harry potter', 'voldemort',
    'gandalf', 'wolverine', 'deadpool', 'harley quinn', 'rick sanchez',
    'michael jackson', 'freddie mercury', 'elvis', 'elton john', 'taylor swift',
    'ariana', 'beyonce', 'eminem', 'drake', 'kanye', 'lady gaga',
    'mrbeast', 'pewdiepie', 'ninja', 'shroud', 'xqc', 'markiplier', 'jacksepticeye',
    'donald trump', 'joe biden', 'barack obama', 'elon musk', 'joe rogan', 'alex jones',
    'jack black', 'gabe newell', 'cleveland', 'jack', 'john stewart', 'jimmy fallon',
    'glados', 'wheatley', 'gordon freeman', 'cloud strife', 'sephiroth', 'samus',
    'kratos', 'nathan drake', 'geralt', 'arthur morgan', 'sonic', 'pacman',
    'forrest gump', 'joker', 'jack sparrow', 'indiana jones', 'rocky', 'godfather',
    'scarface', 'hannibal', 'terminator', 'alien', 'predator', 'robocop',
    'spider-man', 'spiderman', 'wonder woman', 'black panther',
    'thanos', 'star lord', 'doctor strange', 'deadpool',
]

NON_ENGLISH = [
    'espanol', 'latino', 'latina', 'spanish', 'francais', 'francés', 'alemán', 'allemand',
    'deutsch', 'italiano', 'japanese', 'japonés', 'portuguese', 'brasil', 'portugues',
    'russian', 'русский', 'korean', 'chinese', 'arabic', 'turkey', 'turkish', 'polski',
    'polish', 'हिंदी', 'indi', 'thai', 'vietnamese', 'indonesian', 'tagalog', 'українська',
]

for term in search_terms:
    matches = []
    for mid, meta in items:
        if not isinstance(meta, dict):
            continue
        name = meta.get('name', '')
        if term.lower() not in name.lower():
            continue
        nl = name.lower()
        non_eng = any(k in nl for k in NON_ENGLISH)
        if non_eng and not any(k in nl for k in ['english', 'eng dub', 'eng ']):
            continue
        dl = meta.get('download_url', '')
        matches.append((name[:80], bool(dl), mid))

    if matches:
        print(f'\n{term} ({len(matches)} matches):')
        for n, has_dl, mid in matches[:8]:
            dl = 'DL' if has_dl else 'NO-DL'
            print(f'  [{dl:4s}] {mid[:8]} {n}')
