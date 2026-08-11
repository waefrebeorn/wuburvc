"""Triple-DA: compare native C11 wubu_ivf search vs faiss ground truth.
Generates queries, runs test_wubu_ivf.exe, and asserts 16/16 exact ID+distance
match (same nprobe the old Python helper used: min(nlist,64)=64... but the
engine's default is the file's nprobe=1; we test both nprobe values).
"""
import faiss, numpy as np, subprocess, sys, os

INDEX = r'C:/Users/eman5/WuBuMedia/models/rvc/cleveland/Cleveland_Brown.index'
EXE   = r'C:/Users/eman5/wuburvc/build/test_wubu_ivf.exe'
TMP   = r'C:/Users/eman5/wuburvc/build/tmp'

idx = faiss.read_index(INDEX)
d = idx.d
rng = np.random.RandomState(42)
queries = rng.randn(16, d).astype('<f4')

qpath = os.path.join(TMP, 'ivf_queries.bin')
queries.tofile(qpath)

for nprobe in (1, 64):
    k = 8
    p = subprocess.run([EXE, INDEX, qpath, '16', str(k), str(nprobe)],
                       capture_output=True, text=True, timeout=300)
    lines = [l for l in p.stdout.splitlines() if l.startswith('r ')]
    if len(lines) != 16 * k:
        print('nprobe=%d: expected %d result lines, got %d' % (nprobe, 16*k, len(lines)))
        print(p.stderr[-2000:])
        sys.exit(1)
    got_ids = np.array([[int(l.split()[3]) for l in lines[f*k:(f+1)*k]] for f in range(16)])
    got_dst = np.array([[float(l.split()[4]) for l in lines[f*k:(f+1)*k]] for f in range(16)])

    # faiss ground truth with the same nprobe
    idx.nprobe = nprobe
    Dt, It = idx.search(queries, k)

    id_match = (got_ids == It).sum()
    dist_ok = np.allclose(got_dst, Dt, rtol=1e-2, atol=1e-2)  # float accumulation order tolerance
    print('nprobe=%-3d id_match=%d/128  dist_match=%s' % (nprobe, id_match, dist_ok))
    if id_match != 128 or not dist_ok:
        bad = np.where(got_ids != It)
        for f, i in list(zip(*bad))[:10]:
            print('  f=%d i=%d got_id=%d faiss_id=%d got_d=%.4f faiss_d=%.4f' % (
                f, i, got_ids[f,i], It[f,i], got_dst[f,i], Dt[f,i]))
        sys.exit(1)

print('C11 IVF vs faiss: 16/16 x 2 nprobe PASS')
