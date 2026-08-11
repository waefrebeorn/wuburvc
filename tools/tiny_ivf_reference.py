"""Create a tiny known IVF index with the same faiss that read the real one,
then hexdump it so we can decode the exact byte layout for the C11 parser."""
import faiss, numpy as np, struct

np.random.seed(42)
d = 4
nlist = 3
ntotal = 7

# Build a small IVFFlat with known vectors
idx = faiss.index_factory(d, 'IVF%d,Flat' % nlist, faiss.METRIC_L2)
vecs = np.arange(ntotal * d, dtype=np.float32).reshape(ntotal, d) * 0.5
idx.train(vecs)
idx.add(vecs)

print('d=%d nlist=%d ntotal=%d code_size=%d' % (idx.d, idx.nlist, idx.ntotal, idx.code_size))
print('is_trained=%s metric=%d nprobe=%d' % (idx.is_trained, idx.metric_type, idx.nprobe))
print('direct_map:', hasattr(idx, 'direct_map') and idx.direct_map is not None)

faiss.write_index(idx, r'C:/Users/eman5/wuburvc/build/tmp/tiny_ivf.index')

with open(r'C:/Users/eman5/wuburvc/build/tmp/tiny_ivf.index', 'rb') as f:
    raw = f.read()

print('\nfile size:', len(raw))
print('total data expected (ntotal*(code_size+8)):', ntotal * (d*4 + 8))

for off in range(0, min(len(raw), 160), 4):
    chunk = raw[off:off+4]
    val_u32 = struct.unpack('<I', chunk)[0] if len(chunk) == 4 else None
    print('0x%03x: %s  u32=%d' % (off, chunk.hex(), val_u32))

# dump the invlist section around the expected data start
print('\n=== tail dump (last 96 bytes) ===')
print(raw[-96:].hex())
