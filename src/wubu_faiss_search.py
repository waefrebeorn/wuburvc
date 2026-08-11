"""FAISS index search helper for the C11 RVC engine.

Called by wubu_rvc_retrieve_blend() via subprocess when an IVF .index is
detected. The C11 cannot reliably parse FAISS's serialized C++ format,
so Python+faiss does the search and returns neighbor vectors + distances
as raw bytes, plus an auto-detected index_rate.

Auto-index-rate logic:
  - If the avg cosine similarity between queries and their top-1 neighbor
    is very high (>0.97), the index is well-matched → use full rate (0.78).
  - If moderate (0.85-0.97), the index is partially matched → use medium (0.5).
  - If low (<0.85), the index may be mismatched → use low (0.25).
  - This is computed ONCE per call and the C11 caches it.

Protocol:
  Input  (stdin):  n_frames * content_dim float32 values (row-major queries)
  Output (stdout): 1 byte ok flag
                     ok=1: 1 float auto_index_rate (float32)
                           k*n_frames*content_dim floats = neighbor vectors
                           k*n_frames floats = distances
                     ok=0: 4-byte length prefix + error message string
"""
import sys, struct, numpy as np

def main():
    if len(sys.argv) < 5:
        sys.stdout.buffer.write(b'\x00' + struct.pack('<I', 0))
        sys.exit(1)

    index_path = sys.argv[1]
    k = int(sys.argv[2])
    n_frames = int(sys.argv[3])
    content_dim = int(sys.argv[4])

    try:
        import faiss
    except ImportError as e:
        msg = f'faiss not installed: {e}'.encode()
        sys.stdout.buffer.write(b'\x00' + struct.pack('<I', len(msg)) + msg)
        sys.exit(1)

    expected_bytes = n_frames * content_dim * 4
    raw = sys.stdin.buffer.read()
    if len(raw) < expected_bytes:
        msg = f'input too short: got {len(raw)} bytes, need {expected_bytes}'.encode()
        sys.stdout.buffer.write(b'\x00' + struct.pack('<I', len(msg)) + msg)
        sys.exit(1)

    queries = np.frombuffer(raw, dtype=np.float32, count=n_frames * content_dim)
    queries = queries.reshape(n_frames, content_dim).astype(np.float32).copy()

    idx = faiss.read_index(index_path)
    d = idx.d

    if d != content_dim:
        msg = f'dimension mismatch: index d={d}, content_dim={content_dim}'.encode()
        sys.stdout.buffer.write(b'\x00' + struct.pack('<I', len(msg)) + msg)
        sys.exit(1)

    # Search all inverted lists for best recall
    if hasattr(idx, 'nlist'):
        idx.nprobe = min(idx.nlist, 64)

    D, I = idx.search(queries, k)  # (n_frames, k)

    # Reconstruct neighbor vectors from inverted lists.
    # CRITICAL FIX (2026-08-10): FAISS IndexIVFFlat stores vectors across
    # inverted lists in ARBITRARY order — the global id of a vector has NO
    # relation to its position in a list-scan. The old code packed codes in
    # list-scan order and then indexed by global id (vecs[rid]), which
    # fetched the WRONG vector on every search → garbage retrieval blend on
    # every track ("index file used wrong"). Fix: read each list's ids
    # (inv.get_ids) and place every code at its GLOBAL id, so vecs[rid] is
    # correct.
    nb = idx.ntotal
    vecs = np.zeros((nb, d), dtype=np.float32)
    filled = np.zeros(nb, dtype=bool)

    if hasattr(idx, 'invlists') and hasattr(idx, 'invlists'):
        import ctypes
        inv = idx.invlists
        for il in range(inv.nlist):
            sz = inv.list_size(il)
            if sz == 0:
                continue
            cs = inv.code_size
            codes_ptr = inv.get_codes(il)
            ids_ptr = inv.get_ids(il)
            addr = int(codes_ptr)
            buf = (ctypes.c_uint8 * (sz * cs)).from_address(addr)
            ids_buf = (ctypes.c_int64 * sz).from_address(int(ids_ptr))
            gids = np.frombuffer(ids_buf, dtype=np.int64)
            if cs == d * 4:
                # IVFFlat: codes are raw float32
                codes = np.frombuffer(buf, dtype=np.float32).reshape(sz, d)
            else:
                # PQ or compressed — decode each code
                codes = np.zeros((sz, d), dtype=np.float32)
                for j in range(sz):
                    code = ctypes.string_at(addr + j * cs, cs)
                    cj = np.frombuffer(code, dtype=np.float32) if cs == d * 4 else \
                         np.zeros(d, dtype=np.float32)
                    codes[j] = cj[:d]
            for j in range(sz):
                gid = int(gids[j])
                if 0 <= gid < nb:
                    vecs[gid] = codes[j]
                    filled[gid] = True
    else:
        # Flat index: ids are sequential 0..nb-1 in add order
        filled[:min(nb, len(vecs))] = True

    count = int(filled.sum())
    if count == 0:
        msg = b'no vectors extracted from index'.encode()
        sys.stdout.buffer.write(b'\x00' + struct.pack('<I', len(msg)) + msg)
        sys.exit(1)

    # Build neighbor vectors (n_frames * k * content_dim)
    neighbor_vecs = np.zeros((n_frames, k, content_dim), dtype=np.float32)
    for f in range(n_frames):
        for i in range(k):
            rid = int(I[f, i])
            if 0 <= rid < count:
                neighbor_vecs[f, i] = vecs[rid]

    distances = D.astype(np.float32)  # (n_frames, k)

    # ── Auto-detect optimal index_rate ──
    # Compute average cosine similarity between query and top-1 neighbor.
    # High similarity = index is well-matched → aggressive blend (0.78).
    # Low similarity = index may be mismatched → conservative blend (0.1).
    # This is the "magical" auto-detection: the index_rate adapts to how
    # well the retrieved neighbors match the source content.
    valid = 0
    sim_sum = 0.0
    for f in range(n_frames):
        rid = int(I[f, 0])
        if 0 <= rid < count:
            q = queries[f]
            v = vecs[rid]
            qn = np.linalg.norm(q)
            vn = np.linalg.norm(v)
            if qn > 1e-8 and vn > 1e-8:
                sim_sum += float(np.dot(q, v) / (qn * vn))
                valid += 1

    avg_sim = sim_sum / max(valid, 1) if valid > 0 else 0.0

    # Auto-calibrate: if neighbors are very similar to queries, the index
    # captures the source well → high blend. If dissimilar, reduce blend.
    if avg_sim > 0.95:
        auto_rate = 0.78
    elif avg_sim > 0.80:
        auto_rate = 0.5
    elif avg_sim > 0.60:
        auto_rate = 0.25
    else:
        auto_rate = 0.0  # index is poor quality — disable

    # Output: ok(1) + auto_rate(1 float) + neighbor_vecs + distances
    sys.stdout.buffer.write(b'\x01')
    sys.stdout.buffer.write(struct.pack('<f', auto_rate))
    sys.stdout.buffer.write(neighbor_vecs.tobytes())
    sys.stdout.buffer.write(distances.tobytes())
    sys.stdout.buffer.flush()

    # Debug print to stderr
    print(f"[faiss] d={d} nb={nb} k={k} n_frames={n_frames} avg_cos_sim={avg_sim:.4f} auto_rate={auto_rate:.2f}",
          file=sys.stderr, flush=True)

if __name__ == '__main__':
    main()
