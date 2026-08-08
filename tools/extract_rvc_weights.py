#!/usr/bin/env python3
"""Extract RVC .pth model weights into a flat binary format that our C11
   engine can load without PyTorch at runtime.

Usage: python3 extract_rvc_weights.py model.pth output.bin

Output format (binary):
  [4 bytes] magic = "WUBU"
  [4 bytes] n_tensors (uint32)
  For each tensor:
    [1 byte]  name_len
    [name_len bytes] name (ASCII)
    [4 bytes] n_dims
    [4*n_dims bytes] dims
    [4 bytes] data_len (bytes)
    [data_len bytes] raw float32 data (weights converted from fp16→fp32)
  [4 bytes] config_len (uint32) — may be 0 if no config
  [config_len bytes] config data:
    For each config field:
      [1 byte] field_id
      [4 bytes] n_values
      For each value:
        [1 byte] type (0=int32, 1=float32)
        [4 bytes] value

Config field IDs:
  1 = upsample_rates (list of int32)
  2 = sample_rate (int32)
  3 = hidden_channels (int32)
  4 = mel_channels (int32)
  5 = version (int32, 1 or 2)
"""
import sys, struct, os

def extract(src_pth, dst_bin):
    import numpy as np

    # Try torch first
    try:
        import torch
        ckpt = torch.load(src_pth, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'weight' in ckpt:
            sd = ckpt['weight']
        elif isinstance(ckpt, dict) and 'model' in ckpt:
            # RVC-Project-style checkpoints (G40k etc.) keep tensors under 'model'
            sd = ckpt['model']
        else:
            sd = ckpt
    except Exception:
        # Fallback: zipimport the raw tensors
        import zipfile, pickle, io
        zf = zipfile.ZipFile(src_pth)
        pkl_data = zf.read([n for n in zf.namelist() if 'data.pkl' in n][0])
        # This is complex pickle — torch is needed for full fidelity
        print("ERROR: PyTorch required to parse .pth", file=sys.stderr)
        sys.exit(1)

    tensors = []
    for key in sorted(sd.keys()):
        val = sd[key]
        if hasattr(val, 'shape') and hasattr(val, 'numpy'):
            # Convert fp16→fp32, preserve original shape (do NOT flatten)
            np_val = val.detach().cpu().float().numpy()
            tensors.append((key, np_val))

    with open(dst_bin, 'wb') as f:
        f.write(b'WUBU')
        f.write(struct.pack('<I', len(tensors)))
        for name, arr in tensors:
            nb = name.encode('ascii')
            f.write(struct.pack('B', len(nb)))
            f.write(nb)
            ndim = arr.ndim if arr.ndim > 0 else 1
            shape = list(arr.shape) if arr.ndim > 0 else [len(arr)]
            data = arr.astype(np.float32).tobytes() if arr.dtype != np.float32 else arr.tobytes()
            f.write(struct.pack('<I', len(shape)))
            for d in shape:
                f.write(struct.pack('<I', d))
            f.write(struct.pack('<I', len(data)))
            f.write(data)

    # Extract config from checkpoint
    config = ckpt.get('config', [])
    if isinstance(config, str):
        config = eval(config) if config.startswith('[') else []
    elif not isinstance(config, (list, tuple)):
        config = []
    config = list(config)

    # Write config section
    config_data = b''
    if len(config) >= 13:
        # config[12] = upsample_rates
        rates = config[12] if isinstance(config[12], (list, tuple)) else [10,10,2,2]
        config_data += struct.pack('B', 1)  # field_id=1 (upsample_rates)
        config_data += struct.pack('<I', len(rates))
        for r in rates:
            config_data += struct.pack('B', 0)  # type=int32
            config_data += struct.pack('<i', int(r))
    if len(config) >= 18:
        config_data += struct.pack('B', 2)  # field_id=2 (sample_rate)
        config_data += struct.pack('<I', 1)
        config_data += struct.pack('B', 0)
        config_data += struct.pack('<i', int(config[17]))
    if len(config) >= 17:
        config_data += struct.pack('B', 3)  # field_id=3 (hidden_channels)
        config_data += struct.pack('<I', 1)
        config_data += struct.pack('B', 0)
        config_data += struct.pack('<i', int(config[16]))
    if len(config) >= 6:
        config_data += struct.pack('B', 4)  # field_id=4 (mel_channels)
        config_data += struct.pack('<I', 1)
        config_data += struct.pack('B', 0)
        config_data += struct.pack('<i', int(config[5]))
    # resblock kernel sizes (config[10], e.g. [3,7,11])
    if len(config) >= 11 and isinstance(config[10], (list, tuple)):
        ks = [int(x) for x in config[10]]
        config_data += struct.pack('B', 6)
        config_data += struct.pack('<I', len(ks))
        for x in ks:
            config_data += struct.pack('B', 0)
            config_data += struct.pack('<i', x)
    # resblock dilation sizes (config[11], e.g. [[1,3,5],[1,3,5],[1,3,5]])
    if len(config) >= 12 and isinstance(config[11], (list, tuple)):
        dl = config[11]
        flat = [int(x) for row in dl for x in row] if all(isinstance(r, (list, tuple)) for r in dl) else [int(x) for x in dl]
        config_data += struct.pack('B', 7)
        config_data += struct.pack('<I', len(flat))
        for x in flat:
            config_data += struct.pack('B', 0)
            config_data += struct.pack('<i', x)
    # version
    version_str = ckpt.get('version', 'v2') if isinstance(ckpt, dict) else 'v2'
    if isinstance(ckpt, dict) and 'model' not in ckpt:
        version_str = ckpt.get('version', 'v2')
    config_data += struct.pack('B', 5)  # field_id=5 (version)
    config_data += struct.pack('<I', 1)
    config_data += struct.pack('B', 0)
    v = 2 if 'v2' in str(version_str) else (1 if 'v1' in str(version_str) else 2)
    config_data += struct.pack('<i', v)

    with open(dst_bin, 'ab') as f:
        f.write(config_data)
        f.write(struct.pack('<I', len(config_data)))

    print(f"Extracted {len(tensors)} tensors → {dst_bin} ({os.path.getsize(dst_bin)} bytes)")
    # Print a summary
    for name, arr in tensors[:30]:
        print(f"  {name}: shape={list(arr.shape)} dtype=fp32 size={arr.size}")
    if len(tensors) > 30:
        print(f"  ... and {len(tensors) - 30} more")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <model.pth> <output.bin>")
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2])
