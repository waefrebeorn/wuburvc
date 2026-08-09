"""Check what dim PyTorch's weight_norm uses for ConvTranspose1d."""
import torch, torch.nn as nn, sys

ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
sd = ckpt['weight']

# ConvTranspose1d weight is (in_ch, out_ch, k)
for i in range(4):
    wv = sd[f'dec.ups.{i}.weight_v']
    wg = sd[f'dec.ups.{i}.weight_g']
    dim = '1(out_ch)' if wg.shape[0] == wv.shape[1] else '0(in_ch)'
    print(f"ups.{i}: weight_v={tuple(wv.shape)} weight_g={tuple(wg.shape)} -> dim={dim}")

# Conv1d weight is (out_ch, in_ch, k)
wv0 = sd['dec.resblocks.0.convs1.0.weight_v']
wg0 = sd['dec.resblocks.0.convs1.0.weight_g']
print(f"\nresblock 0 convs1.0: weight_v={tuple(wv0.shape)} weight_g={tuple(wg0.shape)} -> dim=0(out_ch)")
