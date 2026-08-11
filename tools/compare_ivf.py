"""Test C11 IVF vs faiss using the same test file."""
import faiss, numpy as np, struct, sys

# Create a small IVF index with known vectors
np.random.seed(42)
d = 32
nlist = 10
ntotal = 100

idx = faiss.index_factory(d, 'IVF%d,Flat' % nlist, faiss.METRIC_L2)
vecs = np.random.randn(ntotal, d).astype('float32')
idx.train(vecs)
idx.add(vecs)

faiss.write_index(idx, '/tmp/small_ivf.index')

# Test with known query
query = np.random.randn(d).astype('float32')
D, I = idx.search(query, 10)

# Print
print('d=%d nlist=%d ntotal=%d nprobe=%d' % (idx.d, idx.nlist, idx.ntotal, idx.nprobe))
print('faiss distances (top-5):', [float(D[0,i]) for i in range(5)])
print('faiss ids (top-5):', I[0,:5])

# Verify the small index can be read by our C11 code
with open('/tmp/small_ivf.index', 'rb') as f:
    raw = f.read()
print('file size:', len(raw))

# Now run our C11 code to compare
import subprocess
cmd = ['gcc', '-std=c11', '-O2', '-I.', '-o', '/tmp/test_ivf', 'tools/test_wubu_ivf.c', 'src/wubu_ivf.c', '-lm']
subprocess.run(cmd, check=True)

# Build a test that just prints the top-1 of the IVFFlat
# For a proper cross-comparison, we'd run the C code with nprobe=1
# and compare. But let's just print the results.
print('\n=== Cross-check ===')
print('Query vector norm:', np.linalg.norm(query))

# Run C11 search
import sys
sys.path.insert(0, '/tmp')
# Actually let's just do it via the exe
result = subprocess.run(['/tmp/test_ivf', '/tmp/small_ivf.index', '/tmp/query.bin', '1', '1', '1'],
                        capture_output=True, text=True, timeout=60)
print('C11 output:')
print(result.stdout)
if result.stderr:
    print('stderr:', result.stderr[-500:])

# Verify against C11 output
# Let's also run our native code directly by building a quick test
print('\n=== Comparing distances ===')
# Build a simple query in C code

# Check if our code's distance for the nearest vector matches faiss
# Actually let me just print the actual results from the C code
# by adding debug output to the C code

# Run:
# Our C code (test_wubu_ivf.c) outputs top-1 for the first query,
# and we compare against faiss.
# For now, let's just print what the C code finds for this case.
print('Done. Compare: nprobe=1 search vs faiss reference')
