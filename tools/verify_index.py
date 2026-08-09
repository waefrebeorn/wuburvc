#!/usr/bin/env python3
"""Verify the full index cache and show stats."""
import json

with open('models/vm_full_index.json') as f:
    data = json.load(f)

print(f'Total indexed models: {data["total"]}')
print(f'Built at: {data["built_at"]}')

items = list(data['index'].items())[:5]
for mid, meta in items:
    print(f'  {mid}: {meta["name"][:70]}')

with_dl = sum(1 for m in data['index'].values() if m.get('download_url'))
print(f'Models with download URLs: {with_dl}')
