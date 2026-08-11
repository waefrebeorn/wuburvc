import struct, subprocess, sys, os, numpy as np

INDEX = r"C:\Users\eman5\WuBuMedia\models\rvc\cleveland\Cleveland_Brown.index"
HELPER = r"C:\Users\eman5\wuburvc\src\wubu_faiss_search.py"
PY = r"C:\Users\eman5\WuBuMedia\.venv_win\Scripts\python.exe"

import faiss
idx = faiss.read_index(INDEX)
d = idx.d
k = 8
n_frames = 16
print(f"index: d={d} ntotal={idx.ntotal} type={type(idx).__name__} nprobe={getattr(idx,'nprobe','n/a')} nlist={getattr(idx,'nlist','n/a')}", flush=True)

# count how many vectors are actually extractable from inverted lists
if hasattr(idx, 'invlists'):
    import ctypes
    inv = idx.invlists
    nb = idx.ntotal
    filled = np.zeros(nb, dtype=bool)
    total_codes = 0
    for il in range(inv.nlist):
        sz = inv.list_size(il)
        if sz == 0: continue
        total_codes += sz
        ids_ptr = inv.get_ids(il)
        ids_buf = (ctypes.c_int64 * sz).from_address(int(ids_ptr))
        gids = np.frombuffer(ids_buf, dtype=np.int64)
        for g in gids:
            gid = int(g)
            if 0 <= gid < nb:
                filled[gid] = True
    print(f"list vectors: {total_codes}, filled unique ids: {filled.sum()} / nb={nb}", flush=True)

# Run helper exactly like the C engine
rng = np.random.default_rng(42)
queries = rng.standard_normal((n_frames, d)).astype(np.float32)
cmd = [PY, HELPER, INDEX, str(k), str(n_frames), str(d)]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = p.communicate(queries.tobytes())
print("rc:", p.returncode, flush=True)
print("helper stderr:", err.decode(errors="replace").strip(), flush=True)
if not out or out[0] != 1:
    print("FAIL: helper returned error flag")
    sys.exit(1)
auto_rate = struct.unpack('<f', out[1:5])[0]
vec_bytes = n_frames * k * d * 4
dist_bytes = n_frames * k * 4
neighbor_vecs = np.frombuffer(out[5:5+vec_bytes], dtype=np.float32).reshape(n_frames, k, d)
distances = np.frombuffer(out[5+vec_bytes:5+vec_bytes+dist_bytes], dtype=np.float32).reshape(n_frames, k)
print(f"auto_rate = {auto_rate:.4f}", flush=True)
print(f"neighbor_vecs zero rows: {(np.abs(neighbor_vecs).sum(axis=2) == 0).sum()} / {n_frames*k}", flush=True)
print(f"distances range: [{distances.min():.4f}, {distances.max():.4f}], negative: {(distances<0).sum()}", flush=True)

# Ground truth: for each query, is the top-1 neighbor the true nearest?
idx.nprobe = min(idx.nlist, 64)
D_truth, I_truth = idx.search(queries, 1)
diff = neighbor_vecs[:, 0, :] - queries
d2 = (diff * diff).sum(axis=1)
print("top-1 distance from helper vec vs faiss truth:", flush=True)
for i in range(min(5, n_frames)):
    print(f"  q{i}: helper_d2={d2[i]:.4f} truth_d2={D_truth[i,0]:.4f} match={abs(d2[i]-D_truth[i,0])<1e-2}", flush=True)
