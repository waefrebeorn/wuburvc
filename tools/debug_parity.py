"""Deep debug: run PyTorch generator step by step to find where C11 diverges."""
import sys, os, numpy as np, torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from collections import OrderedDict

sys.path.insert(0, '.')
from tools.gen_reference_pytorch3 import HiFiGANGenerator, ResBlock

pth_path = sys.argv[1]
ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
sd = ckpt['weight']
config = ckpt['config']
h = [x.item() if hasattr(x, 'item') else x for x in config]

gen = HiFiGANGenerator(
    initial_channel=h[3],
    resblock_kernel_sizes=h[10],
    resblock_dilation_sizes=h[11],
    upsample_rates=h[12],
    upsample_initial_channel=h[13],
    upsample_kernel_sizes=h[14],
)
gen.load_state_dict(sd, strict=False)
gen.eval()
for m in gen.resblocks:
    m.remove_weight_norm()
for i in range(len(gen.ups)):
    if hasattr(gen.ups[i], 'remove_weight_norm'):
        gen.ups[i].remove_weight_norm()

mel_np = np.load('pytorch_ref_mel.npy')
n_frames = 4
inter_ch = h[3]

gen_input = torch.zeros(1, inter_ch, n_frames)
for c in range(min(80, inter_ch)):
    gen_input[0, c, :] = torch.from_numpy(mel_np[:, c])

print(f"gen_input: shape={gen_input.shape}, mean={gen_input.mean():.6f}")

with torch.no_grad():
    x = gen.conv_pre(gen_input)
    print(f"conv_pre: shape={x.shape}, mean={x.mean():.8f}, std={x.std():.8f}")

    for i in range(4):
        x = F.leaky_relu(x, 0.1)
        x = gen.ups[i](x)
        print(f"  ups.{i}: shape={x.shape}, mean={x.mean():.8f}, std={x.std():.8f}")
        xs = None
        for j in range(3):
            if xs is None:
                xs = gen.resblocks[i * 3 + j](x)
            else:
                xs += gen.resblocks[i * 3 + j](x)
        x = xs / 3
        print(f"  MRF.{i}: shape={x.shape}, mean={x.mean():.8f}, std={x.std():.8f}")

    x = F.leaky_relu(x, 0.1)
    print(f"final lrelu: mean={x.mean():.8f}, std={x.std():.8f}")
    x = gen.conv_post(x)
    print(f"conv_post: mean={x.mean():.8f}, std={x.std():.8f}")
    x = torch.tanh(x)
    print(f"tanh: mean={x.mean():.8f}, std={x.std():.8f}, min={x.min():.8f}, max={x.max():.8f}")

out = x.squeeze().numpy()
print(f"\nOutput: {len(out)} samples")
print(f"Stats: mean={out.mean():.6f} std={out.std():.6f} min={out.min():.6f} max={out.max():.6f}")
print(f"First 5: {out[:5]}")
print(f"Nonzero: {(out != 0).sum()}/{len(out)}")
