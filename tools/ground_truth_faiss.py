"""Ground-truth the FAISS IndexIVFFlat binary layout for the C11 parser."""
import faiss, struct, sys, os

INDEX = r'C:/Users/eman5/WuBuMedia/models/rvc/cleveland/Cleveland_Brown.index'

idx = faiss.read_index(INDEX)
print('=== faiss attributes ===')
print('d        =', idx.d)
print('ntotal   =', idx.ntotal)
print('nlist    =', idx.nlist)
print('code_size=', idx.code_size)
print('metric   =', idx.metric_type)
print('is_trained=', idx.is_trained)
print('nprobe   =', idx.nprobe)
inv = idx.invlists
print('invlists.nlist =', inv.nlist)
print('invlists.code_size =', inv.code_size)
print('invlists has direct_map:', hasattr(idx, 'direct_map') and idx.direct_map is not None)

# sample list sizes
sizes = [inv.list_size(i) for i in range(inv.nlist)]
print('list sizes: min=%d max=%d nonzero=%d/%d' % (min(sizes), max(sizes), sum(1 for s in sizes if s), len(sizes)))

# Dump raw header bytes for layout mapping
with open(INDEX, 'rb') as f:
    raw = f.read(64)

print('\n=== raw header bytes ===')
for off in range(0, 64, 8):
    chunk = raw[off:off+8]
    print('0x%02x: %s  u32=%d u64=%d' % (off, chunk.hex(), struct.unpack('<I', chunk[:4])[0], struct.unpack('<Q', chunk)[0]))

# After the header: nlist * (list_size + codes + ids). Find where invlists start by
# scanning: total data = ntotal*(code_size+8). Walk forward from a candidate header size.
# The header = quantizer (IndexFlatL2: its own header + nlist*d floats) + IVF fields.
print('\n=== computed layout ===')
print('quantizer bytes (nlist*d*4) =', idx.nlist * idx.d * 4)
print('invlist data bytes =', idx.ntotal * (idx.code_size + 8))

# Search a real query to confirm search semantics
q = faiss.randn(1, idx.d).astype('float32')
D, I = idx.search(q, 8)
print('\n=== sample search ===')
print('D =', D[0])
print('I =', I[0])
