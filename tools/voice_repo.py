#!/usr/bin/env python3
"""voice_repo.py — the 30k voice catalog as a first-class WuBuRVC citizen.

voice-models.com (vm_full_index.json, 30,878 voices) is the "magic 30k repo
of voices the world already put the work in". This module:

  * resolves ANY voice by casual name (fuzzy, case/space-insensitive)
  * prefers a local .pth/.index when already downloaded
  * else returns the canonical HuggingFace download_url
  * knows the RVC version (v1/v2) per voice when recorded

Local bank (out/voices.json) stays the fast path; the full catalog is the
long tail. Both share the same resolve() interface.
"""
import json
import os
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FULL_INDEX = os.environ.get("WUBU_VOICE_FULL_INDEX") or \
    os.path.join(ROOT, "models", "vm_full_index.json")
LOCAL_BANK = os.environ.get("WUBU_VOICE_JSON") or \
    os.path.join(ROOT, "out", "voices.json")


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


class VoiceRepo:
    def __init__(self, full_index=FULL_INDEX, local_bank=LOCAL_BANK):
        self.full = None
        self.full_by_name = {}
        self.local = {}
        if os.path.exists(full_index):
            try:
                with open(full_index, encoding="utf-8") as f:
                    idx = json.load(f)
                self.full = idx.get("index", {})
                self.full_by_name = idx.get("by_name", {})
                print(f"[voicerepo] full catalog: {len(self.full)} voices "
                      f"(vm_full_index.json)")
            except Exception as e:
                print(f"[voicerepo] full index failed: {e}")
        if os.path.exists(local_bank):
            try:
                with open(local_bank, encoding="utf-8") as f:
                    self.local = json.load(f).get("voices", {})
                print(f"[voicerepo] local bank: {len(self.local)} voices")
            except Exception as e:
                print(f"[voicerepo] local bank failed: {e}")

    def resolve(self, name):
        """Return a voice record for `name`, or None.
        Record fields: name, pth, index, download_url, version, source."""
        n = _norm(name)
        if not n:
            return None
        # 1. exact local bank hit
        for k, v in self.local.items():
            if _norm(k) == n:
                return self._local_record(k, v)
        # 2. fuzzy local hit (substring)
        hits = [(k, v) for k, v in self.local.items() if n in _norm(k)]
        if hits:
            hits.sort(key=lambda kv: -int(kv[1].get("popularity_score", 0) or 0))
            return self._local_record(hits[0][0], hits[0][1])
        # 3. full catalog: normalized name exact
        ids = self.full_by_name.get(n)
        if not ids:
            # try contains
            ids = [i for k, i in self.full_by_name.items() if n in k][:1]
            ids = ids[0] if ids else None
        if ids:
            rid = ids[0] if isinstance(ids, list) else ids
            rec = self.full.get(rid, {})
            return self._full_record(rec)
        return None

    def _local_record(self, name, v):
        rec = {"name": v.get("name", name), "source": v.get("source", "local")}
        pth = v.get("pth") or v.get("path") or ""
        idx = v.get("index") or ""
        rec["pth"] = pth if os.path.exists(pth) else ""
        rec["index"] = idx if os.path.exists(idx) else ""
        rec["version"] = v.get("version", "v2")
        if not rec["pth"] and v.get("download_url"):
            rec["download_url"] = v["download_url"]
        return rec

    def _full_record(self, rec):
        r = {"name": rec.get("name", "?"), "source": "voicemodels-30k"}
        r["download_url"] = rec.get("download_url", "")
        ver = (rec.get("name") or "").lower()
        r["version"] = "v1" if "rvc v1" in ver or "rvcv1" in ver else "v2"
        # index URLs appear as separate entries in the index dict
        if rec.get("index_url"):
            r["index_url"] = rec["index_url"]
        return r

    def count(self):
        return len(self.full) if self.full else 0

    def search(self, query, limit=10):
        """Return up to `limit` matching names (catalog only)."""
        q = _norm(query)
        if not q:
            return []
        out = []
        for k in self.full_by_name:
            if q in k:
                out.append(k)
                if len(out) >= limit:
                    break
        return out


if __name__ == "__main__":
    repo = VoiceRepo()
    print(f"catalog voices: {repo.count()}")
    for q in ("cartman", "wheatley", "GLaDOS", "vinny", "obama", "squidward"):
        hit = repo.resolve(q)
        print(f"  {q!r:12} -> {(hit['name'] if hit else None)!r}"
              f" v={hit.get('version') if hit else '-'}"
              f" src={hit.get('source') if hit else '-'}")
    print("sample search 'squid':", repo.search("squid", 3))
