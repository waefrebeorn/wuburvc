import struct, sys
path = sys.argv[1]
f = open(path, 'rb')
magic = f.read(4)
if magic != b'GGUF':
    print("not GGUF"); sys.exit(1)
ver = struct.unpack('<I', f.read(4))[0]
f.read(8)  # total_size
nkv = struct.unpack('<Q', f.read(8))[0]
for _ in range(nkv):
    klen = struct.unpack('<Q', f.read(8))[0]
    f.read(klen)
    vlen = struct.unpack('<Q', f.read(8))[0]
    f.read(vlen)
nt = struct.unpack('<Q', f.read(8))[0]
print(f"version={ver} tensors={nt}")
seen = set()
for _ in range(nt):
    nlen = struct.unpack('<Q', f.read(8))[0]
    name = f.read(nlen).decode('utf-8', 'replace')
    typ = struct.unpack('<I', f.read(4))[0]
    nd = struct.unpack('<Q', f.read(8))[0]
    f.read(8 * nd)  # dims
    f.read(8)       # offset
    if any(k in name for k in ('ffn', 'gate', 'moe', 'expert', 'up', 'down', 'shared')):
        if name not in seen:
            seen.add(name)
            print(name)
