import struct, sys

def read_u64(f): return struct.unpack('<Q', f.read(8))[0]
def read_u32(f): return struct.unpack('<I', f.read(4))[0]

def read_kv_value(f, kind, fsize, pos):
    # returns (value, new_pos); kind per GGUF spec
    if kind == 0: return struct.unpack('<B', f.read(1))[0], pos+1
    if kind == 1: return struct.unpack('<b', f.read(1))[0], pos+1
    if kind == 2: return struct.unpack('<H', f.read(2))[0], pos+2
    if kind == 3: return struct.unpack('<h', f.read(2))[0], pos+2
    if kind == 4: return struct.unpack('<I', f.read(4))[0], pos+4
    if kind == 5: return struct.unpack('<i', f.read(4))[0], pos+4
    if kind == 6: return struct.unpack('<Q', f.read(8))[0], pos+8
    if kind == 7: return struct.unpack('<q', f.read(8))[0], pos+8
    if kind == 8: return struct.unpack('<f', f.read(4))[0], pos+4
    if kind == 9: return struct.unpack('<d', f.read(8))[0], pos+8
    if kind == 10: return struct.unpack('<B', f.read(1))[0], pos+1
    if kind == 11:  # string
        slen = read_u64(f); pos += 8
        s = f.read(slen).decode('utf-8','replace'); pos += slen
        return s, pos
    if kind == 12:  # array
        atype = read_u32(f); pos += 4
        alen = read_u64(f); pos += 8
        esz = {0:1,1:1,2:2,3:2,4:4,5:4,6:8,7:8,8:4,9:8,10:1,11:8}.get(atype,8)
        f.read(alen*esz); pos += alen*esz
        return f"<arr type={atype} len={alen}>", pos
    raise ValueError(f"unknown kv kind {kind} at {pos}")

def read_header(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GGUF', magic
        version = read_u32(f)
        tensor_count = read_u64(f)
        kv_count = read_u64(f)
        pos = f.tell()
        kv = {}
        for i in range(kv_count):
            klen = read_u64(f); pos += 8
            if klen > 10_000_000:
                return version, kv, [], pos, f"KV desync at entry {i}, klen={klen}"
            key = f.read(klen).decode('utf-8','replace'); pos += klen
            vkind = read_u32(f); pos += 4
            try:
                val, pos = read_kv_value(f, vkind, 0, pos)
            except Exception as e:
                return version, kv, [], pos, f"KV value err entry {i} key={key!r} kind={vkind}: {e}"
            kv[key] = val
        # tensor infos
        tensors = []
        for _ in range(tensor_count):
            nlen = read_u64(f); pos += 8
            name = f.read(nlen).decode('utf-8','replace'); pos += nlen
            gtype = read_u32(f); pos += 4
            nd = read_u64(f); pos += 8
            dims = struct.unpack('<'+'Q'*nd, f.read(8*nd)); pos += 8*nd
            doff = read_u64(f); pos += 8
            tensors.append((name, gtype, dims, doff))
        data_start = f.tell()
    return version, kv, tensors, data_start, None

def human(n):
    n2=float(n)
    for u in ['B','KB','MB','GB','TB']:
        if n2<1024: return f"{n2:.1f}{u}"
        n2/=1024
    return f"{n2:.1f}PB"

for p in sys.argv[1:]:
    try:
        v,kv,tensors,dstart,err = read_header(p)
    except Exception as e:
        print(f"\n=== {p} === ERROR {e}"); continue
    print(f"\n=== {p} ===")
    if err: print(f"  PARSE WARN: {err}")
    print(f"version={v} tensors={len(tensors)} data_start={human(dstart)}")
    for k in ('split.count','split.no','split.tensors.count','general.architecture','ssm_value_dim','ssm_conv_dim','ssm_key_dim','ssm_state_dim','n_experts','hidden_size','num_hidden_layers','ssm_head_dim'):
        if k in kv: print(f"  {k} = {kv[k]}")
    types={}
    for n,gt,d,o in tensors: types[gt]=types.get(gt,0)+1
    print(f"  ggml_types: {types}")
    if tensors:
        print(f"  first: {tensors[0][0]} off={tensors[0][3]}")
        print(f"  last:  {tensors[-1][0]} off={tensors[-1][3]} dims={tensors[-1][2]}")
