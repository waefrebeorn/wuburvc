"""Standalone Triple-DA test: tiny IVF index, known vectors, compare
C11 wubu_ivf_search against faiss ground truth."""
import faiss, numpy as np, subprocess, struct, os, sys

np.random.seed(7)
d = 16
nlist = 5
ntotal = 25
k = 3
nq = 4

# Build + save
idx = faiss.index_factory(d, 'IVF%d,Flat' % nlist, faiss.METRIC_L2)
vecs = np.random.randn(ntotal, d).astype('float32')
idx.train(vecs)
idx.add(vecs)
faiss.write_index(idx, '/tmp/tiny.index')

# Deterministic queries
queries = np.random.randn(nq, d).astype('float32')
queries.tofile('/tmp/tiny_q.bin')

# Ground truth from faiss (single list scan like the real engine)
idx.nprobe = 1
Df, If = idx.search(queries, k)

# Run C11
EXE = '/tmp/test_wubu_ivf'
r = subprocess.run(['gcc', '-std=c11', '-O2', '-I/c/Users/eman5/wuburvc',
                    '-o', EXE,
                    '/c/Users/eman5/wuburvc/tools/test_wubu_ivf.c',
                    '/c/Users/eman5/wuburvc/src/wubu_ivf.c', '-lm'], capture_output=True)
if r.returncode != 0:
    print('BUILD FAIL:', r.stderr.decode()[-1000:]); sys.exit(1)

out = subprocess.run([EXE, '/tmp/tiny.index', '/tmp/tiny_q.bin',
                      str(nq), str(k), '1'], capture_output=True, text=True)
lines = [l for l in out.stdout.splitlines() if l.startswith('r ')]
if len(lines) != nq * k:
    print('BAD output lines:', len(lines)); print(out.stdout); print('ERR', out.stderr[-1000:]); sys.exit(1)

g_ids = [[int(l.split()[2]) for l in lines[f*k:(f+1)*k]] for f in range(nq)]
g_dst = [[float(l.split()[3]) for l in lines[f*k:(f+1)*k]] for f in range(nq)]

ok = True
for f in range(nq):
    for i in range(k):
        match_id = g_ids[f][i] == If[f][i]
        match_d = abs(g_dst[f][i] - Df[f][i]) < 1e-4
        if not (match_id and match_d):
            ok = False
            print('MISMATCH f=%d i=%d c11=(%d, %.6f) faiss=(%d, %.6f)' %
                  (f, i, g_ids[f][i], g_dst[f][i], If[f][i], Df[f][i]))

print('tiny IVF (nprobe=1):', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
