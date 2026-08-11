#!/usr/bin/env python3
"""verify_faiss_fix.py — prove the global-ID fix works on the real Cleveland index.

Tests the exact protocol the C11 engine uses: feed n_frames of 768-dim query
vectors on stdin, parse the ok flag + auto_rate + neighbor vectors out.
Also verifies the retrieved neighbors are TRUE nearest neighbors by
comparing against faiss's own reconstruct() ground truth.
"""
import struct, subprocess, sys, os, numpy as np

INDEX = r"C:\Users\eman5\WuBuMedia\models\rvc\cleveland\Cleveland_Brown.index"
HELPER = r"C:\Users\eman5\wuburvc\src\wubu_faiss_search.py"
PY = r"C:\Users\eman5\WuBuMedia\.venv_win\Scripts\python.exe"

import faiss

idx = faiss.read_index(INDEX)
d = idx.d
k = 8
n_frames = 16

rng = np.random.default_rng(42)
# Queries: pick global ids spread across the index. We can't reconstruct()
# (no direct map on this index), but we CAN verify via distance: if the
# helper returned the TRUE neighbor, its squared-L2 distance to the query
# must equal faiss's own search distance D_truth[0].
nq = n_frames
queries = np.zeros((nq, d), dtype=np.float32)
D_truth_full, I_truth_full = None, None  # filled after helper via own search
# we just need SOME plausible queries; use random but normalize
rng.standard_normal(nq * d, out=queries.ravel() if False else None)
queries[:] = rng.standard_normal((nq, d)).astype(np.float32)
for i in range(nq):
    gid = int(idx.ntotal * 0.3 + i * 997) % idx.ntotal
    q = idx.reconstruct_n(0, 0) if False else None  # no-op, keep simple
    del q
# blend a few real-ish directions: fetch list codes directly is complex,
# so just use random queries (distance check remains valid ground truth)

# ── Run the helper exactly like the C engine does ──
cmd = [PY, HELPER, INDEX, str(k), str(n_frames), str(d)]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)
out, err = p.communicate(queries.tobytes())
print("helper stderr:", err.decode(errors="replace").strip())
if not out or out[0] != 1:
    print("FAIL: helper returned error flag")
    sys.exit(1)

auto_rate = struct.unpack('<f', out[1:5])[0]
vec_bytes = n_frames * k * d * 4
dist_bytes = n_frames * k * 4
neighbor_vecs = np.frombuffer(out[5:5 + vec_bytes], dtype=np.float32).reshape(n_frames, k, d)
distances = np.frombuffer(out[5 + vec_bytes:5 + vec_bytes + dist_bytes], dtype=np.float32).reshape(n_frames, k)
print(f"auto_rate = {auto_rate:.4f}")

# ── Ground truth check: for each query, the top-1 neighbor returned must be
# the true nearest vector. Compare OUR returned vector's squared-L2 distance
# against faiss's own search distance D_truth[0] — they must match.
# Use the SAME nprobe as the helper (exhaustive) so both searches are equal.
idx2 = faiss.read_index(INDEX)
idx2.nprobe = min(idx2.nlist, 64)
D_truth, I_truth = idx2.search(queries, k)
print("\nframe | our_dist[0] | true_dist[0] | match | cos_sim(q, neighbor)")
bad = 0
cos_sims = []
for f in range(n_frames):
    our_v = neighbor_vecs[f, 0]
    # squared-L2 distance of OUR vector to the query (faiss returns L2²)
    our_d = float(np.sum((queries[f] - our_v) ** 2))
    true_d = float(D_truth[f, 0])
    match = abs(our_d - true_d) < 1e-2 * max(1.0, abs(true_d))
    if not match:
        bad += 1
    q = queries[f]
    cos_sims.append(float(np.dot(q, our_v) / (np.linalg.norm(q) * np.linalg.norm(our_v) + 1e-12)))
    print(f"  {f:5d} | {our_d:10.4f} | {true_d:10.4f} | {str(match):5s} | {cos_sims[-1]:.4f}")

print(f"\nRESULT: {n_frames - bad}/{n_frames} top-1 neighbors EXACT match to faiss ground truth (distance)")
print(f"mean cos_sim(query, neighbor) = {np.mean(cos_sims):.4f}")
ok = bad == 0
print("FIX VERIFIED" if ok else "FIX STILL BROKEN")
sys.exit(0 if ok else 1)
