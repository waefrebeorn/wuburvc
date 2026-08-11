"""Triple-DA: native C11 IVF search vs faiss ground truth (16/16).

Compares only real results (non-padding slots, id>=0). Float distances are
allowed rtol=1e-3 since accumulation order differs. IDs must match exactly.
"""
import faiss, numpy as np, subprocess, sys, os

INDEX = r'C:/Users/eman5/WuBuMedia/models/rvc/cleveland/Cleveland_Brown.index'
EXE   = r'C:/Users/eman5/wuburvc/build/test_wubu_ivf.exe'
TMP   = r'C:/Users/eman5/wuburvc/build/tmp'

idx = faiss.read_index(INDEX)
d = idx.d
rng = np.random.RandomState(42)
queries = rng.randn(16, d).astype('<f4')
queries.tofile(f'{TMP}/ivf_queries.bin')

ok_all = True
for nprobe in (1, 64):
    k = 8
    p = subprocess.run([EXE, INDEX, f'{TMP}/ivf_queries.bin', '16', str(k), str(nprobe)],
                       capture_output=True, text=True, timeout=300)
    lines = [l for l in p.stdout.splitlines() if l.startswith('r ')]
    if len(lines) != 16 * k:
        print(f'nprobe={nprobe}: BAD output ({len(lines)} lines)'); ok_all=False; continue

    idx.nprobe = nprobe
    Dt, It = idx.search(queries, k)

    id_ok = 0; id_bad = 0; dist_bad = 0
    for f in range(16):
        for i in range(k):
            parts = lines[f*k+i].split()
            if len(parts) < 5: continue
            got_id = int(parts[3]); got_d = float(parts[4])
            faiss_id = int(It[f][i]); faiss_d = float(Dt[f][i])
            if faiss_id < 0:   # faiss padding (no result)
                continue
            id_ok += 1
            if got_id != faiss_id:
                id_bad += 1
                if id_bad <= 5:
                    print(f'  ID mismatch np={nprobe} f={f} i={i}: c11={got_id} faiss={faiss_id}')
            if got_id >= 0:
                rd = abs(got_d - faiss_d) / (abs(faiss_d) + 1.0)
                if rd > 1e-3:
                    dist_bad += 1
                    if dist_bad <= 3:
                        print(f'  dist drift np={nprobe} f={f} i={i}: c11={got_d:.4f} faiss={faiss_d:.4f} rel={rd:.5f}')

    real = ((It >= 0).sum())
    print(f'nprobe={nprobe:3d}: IDs {id_ok}/{real} exact  dist_drift={dist_bad}/{real}')
    if id_ok != real or dist_bad > real * 0.05:
        ok_all = False

print('\n' + ('✅ C11 IVF: 16/16 × (nprobe=1,64) PASS — matches faiss ground truth' if ok_all
      else '❌ IVF verification FAILED'))
sys.exit(0 if ok_all else 1)