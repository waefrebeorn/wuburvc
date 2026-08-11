"""Decode the IndexIVFFlat binary layout iteratively, checking every field
against faiss ground truth. When all checks pass, this IS the spec for the
C11 parser."""
import faiss, struct, numpy as np, sys

INDEX = r'C:/Users/eman5/WuBuMedia/models/rvc/cleveland/Cleveland_Brown.index'

ref = faiss.read_index(INDEX)
inv = ref.invlists
GT = dict(
    d=ref.d, ntotal=ref.ntotal, nlist=ref.nlist,
    code_size=ref.code_size, metric=ref.metric_type,
    trained=ref.is_trained, nprobe=ref.nprobe,
)

with open(INDEX, 'rb') as f:
    data = f.read()

pos = [0]
def u8():
    b = data[pos[0]]; pos[0] += 1; return b
def u32():
    v = struct.unpack_from('<I', data, pos[0])[0]; pos[0] += 4; return v
def i32():
    v = struct.unpack_from('<i', data, pos[0])[0]; pos[0] += 4; return v
def u64():
    v = struct.unpack_from('<Q', data, pos[0])[0]; pos[0] += 8; return v
def i64():
    v = struct.unpack_from('<q', data, pos[0])[0]; pos[0] += 8; return v
def fourcc():
    s = data[pos[0]:pos[0]+4].decode('latin1')
    pos[0] += 4
    return s

def check(name, got, want):
    ok = got == want
    print('%s %-28s got=%-12s want=%-12s %s' % ('✅' if ok else '❌', name, got, want, '' if ok else '<-- MISMATCH'))
    return ok

def vec_u64():
    n = u64(); return [u64() for _ in range(n)]

ok = True
# magic
m = fourcc()
print('fourcc:', m)
ok &= m == 'IwFl'
ok &= check('d', u32(), GT['d'])
ok &= check('ntotal', u64(), GT['ntotal'])
d1 = u64(); d2 = u64()
print('dummy1=%d dummy2=%d (expect 1048576 each)' % (d1, d2))
ok &= d1 == 1048576 and d2 == 1048576
ok &= check('is_trained (u8)', u8(), 1 if GT['trained'] else 0)
ok &= check('metric_type (i32)', i32(), GT['metric'])
ok &= check('nlist (i64)', i64(), GT['nlist'])
# NOTE: NO code_size stored in the IVF header — recomputed as d*4 on load
ok &= check('nprobe (i32)', i32(), GT['nprobe'])
max_codes = i32()
print('max_codes =', max_codes)

# quantizer: IndexFlat (fourcc IxF2)
qm = fourcc()
print('quantizer fourcc:', qm)
ok &= qm == 'IxF2'
ok &= check('quantizer d', i32(), GT['d'])
ok &= check('quantizer ntotal (=nlist)', i64(), GT['nlist'])
qd1 = u64(); qd2 = u64()
print('q dummy1=%d q dummy2=%d (expect 1048576)' % (qd1, qd2))
ok &= qd1 == 1048576 and qd2 == 1048576
ok &= check('q is_trained (u8)', u8(), 1)
ok &= check('q metric (i32)', i32(), 1)  # L2
# xb vector: u64 count + count*d floats
xc = u64()
print('quantizer xb count =', xc, '(expect nlist*d =', GT['nlist']*GT['d'], ')')
ok &= xc == GT['nlist'] * GT['d']
centroids = np.frombuffer(data, dtype='<f4', count=xc, offset=pos[0]).reshape(GT['nlist'], GT['d']).copy()
pos[0] += xc * 4
# compare with faiss quantizer reconstruct
qref = np.zeros((GT['nlist'], GT['d']), dtype='<f4')
ref.quantizer.reconstruct_n(0, GT['nlist'], qref)
maxdiff = float(np.abs(centroids - qref).max())
print('centroid maxdiff vs faiss:', maxdiff)
ok &= maxdiff < 1e-6

# direct_map: WRITE1(type as char) + WRITEVECTOR(array: u64 count + count*i64)
dm_type = u8()
dm_count = u64()
print('direct_map: type=%d array_len=%d (expect type 0=NoMap, len 0)' % (dm_type, dm_count))
ok &= dm_type == 0 and dm_count == 0
for _ in range(dm_count):
    i64()

# invlists: fourcc ilar
im = fourcc()
print('invlists fourcc:', im)
ok &= im == 'ilar'
ok &= check('inv nlist (u64)', u64(), GT['nlist'])
ok &= check('inv code_size (u64)', u64(), GT['code_size'])
lt = fourcc()
print('list_type fourcc:', lt, '(expect full/sprs)')
ok &= lt in ('full', 'sprs')
sizes = vec_u64()
print('sizes vector len =', len(sizes), '(expect nlist =', GT['nlist'], ')')
ok &= len(sizes) == GT['nlist']
if lt == 'sprs':
    full = [0]*GT['nlist']
    for j in range(0, len(sizes), 2):
        full[sizes[j]] = sizes[j+1]
    sizes = full

ref_sizes = [inv.list_size(i) for i in range(inv.nlist)]
ok &= sizes == ref_sizes
print('list sizes match faiss:', sizes == ref_sizes, ' (sum=%d)' % sum(sizes))

# codes + ids per list
codes = []
ids = []
for i in range(GT['nlist']):
    n = sizes[i]
    if n == 0: continue
    c = np.frombuffer(data, dtype='<f4', count=n*GT['d'], offset=pos[0]).reshape(n, GT['d']).copy()
    pos[0] += n * GT['code_size']
    idv = np.frombuffer(data, dtype='<i8', count=n, offset=pos[0]).copy()
    pos[0] += n * 8
    codes.append(c); ids.append(idv)

# Verify codes via faiss search: for each query, our extracted vector at the
# faiss-returned ID must reproduce faiss's distance (proves ids+codes layout).
rng = np.random.RandomState(1234)
q = rng.randn(4, GT['d']).astype('<f4')
Dq, Iq = ref.search(q, 8)
# build global id -> (list, offset)
id_to_l = {}
for i in range(GT['nlist']):
    for j, gid in enumerate(ids[i]):
        id_to_l[int(gid)] = (i, j)
ok_search = True
for f in range(4):
    for t in range(8):
        gid = int(Iq[f, t])
        li, oj = id_to_l[gid]
        v = codes[li][oj]
        d = float(np.sum((q[f] - v) ** 2))
        dd = float(Dq[f, t])
        if abs(d - dd) > 1e-2 * max(1.0, abs(dd)):
            ok_search = False
            print('  query %d nn %d id %d: our_d=%.4f faiss_d=%.4f MISMATCH' % (f, t, gid, d, dd))
print('search-based code verification (16/16):', 'PASS' if ok_search else 'FAIL')
ok &= ok_search

# direct_map
tail = data[pos[0]:]
print('remaining bytes after invlists:', len(tail))
dm = tail[:4].decode('latin1', 'replace')
print('direct_map fourcc:', repr(dm))
if dm == 'dmap':
    pos[0] += 4
    # DirectMap has type + array
    dm_type = u8()
    print('dmap type:', dm_type, '(0=NoMap 1=Array 2=Hashtable)')
    if dm_type == 1:
        narr = u64()
        arr = [i64() for _ in range(narr)]
        print('dmap array len:', len(arr), '(expect ntotal =', GT['ntotal'], ')')
        ok &= len(arr) == GT['ntotal']

print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')
print('total bytes consumed:', pos[0], 'of', len(data))
sys.exit(0 if ok else 1)
