import struct, sys

def read_tensors(path):
    with open(path,'rb') as f:
        f.read(4); ver=struct.unpack('<I',f.read(4))[0]
        nt, nkv = struct.unpack('<Q Q', f.read(16))
        for _ in range(nkv):
            kl=struct.unpack('<Q',f.read(8))[0]; f.read(kl)
            vk=struct.unpack('<I',f.read(4))[0]
            if vk==11: sl=struct.unpack('<Q',f.read(8))[0]; f.read(sl)
            elif vk in (4,5): f.read(4)
            elif vk in (6,7): f.read(8)
            elif vk in (0,1): f.read(1)
            elif vk in (2,3): f.read(2)
            elif vk==8: f.read(4)
            elif vk==9: f.read(8)
            elif vk==10: f.read(1)
            elif vk==12:
                at=struct.unpack('<I',f.read(4))[0]; al=struct.unpack('<Q',f.read(8))[0]
                f.read(al*{0:1,1:1,2:2,3:2,4:4,5:4,6:8,7:8,8:4,9:8,10:1,11:8}.get(at,8))
        tensors=[]
        for _ in range(nt):
            nl=struct.unpack('<Q',f.read(8))[0]; name=f.read(nl).decode('utf-8','replace')
            gt=struct.unpack('<I',f.read(4))[0]; nd=struct.unpack('<Q',f.read(8))[0]
            dims=struct.unpack('<'+'Q'*nd, f.read(8*nd)); off=struct.unpack('<Q',f.read(8))[0]
            tensors.append((name,gt,list(dims),off))
        data_start=f.tell()
    return tensors, data_start

for p in sys.argv[1:]:
    ts, ds = read_tensors(p)
    print(f"\n=== {p} ===  tensors={len(ts)} data_start={ds}")
    bad=0
    for i,t in enumerate(ts):
        name,gt,dims,off=t
        ne=1
        for d in dims: ne*=d
        # actual span from next tensor in this file
        nxt = ts[i+1][3] if i+1<len(ts) else ds
        actual = nxt - off
        # F32 expected size
        exp_f32 = ne*4
        if gt==0 and actual != exp_f32:
            # declared F32 but size mismatch -> corrupted label
            print(f"  CORRUPT F32? {name}: ne={ne} declared_f32={exp_f32} actual={actual} (ratio={actual/exp_f32:.2f})")
            bad+=1
        if bad>=15: 
            print("  ... (more)"); break
    if bad==0: print("  no F32-size mismatches found")
