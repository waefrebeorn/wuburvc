#!/usr/bin/env python3
"""Generate PyTorch reference output for Cartman v2 model.
Runs the FULL HiFi-GAN generator forward pass using the actual .pth weights.
This is the ground-truth reference for verifying our C11 kernel."""
import sys, os, struct, numpy as np, torch
import torch.nn.functional as F
torch.manual_seed(42)

# ── ResBlock (Multi-receptive-field from HiFi-GAN v2) ──
# Each ResBlock has 3 Conv1d layers with the SAME kernel but DIFFERENT dilations
# (from resblock_dilation_sizes). All at the same channel count (depthwise).
# Input → convs[0] → convs[1] → convs[2], each + LeakyReLU + residual

class ResBlock(torch.nn.Module):
    """MRF ResBlock: 3 depthwise Conv1d with different dilations, residual sum."""
    def __init__(self, channels, kernel, dilations):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for d in dilations:
            self.convs.append(torch.nn.Conv1d(channels, channels, kernel, stride=1,
                               padding=d * (kernel // 2), dilation=d,
                               groups=channels))

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, 0.01)
            xt = c(xt)
            x = x + xt  # residual
        return x

# ── Generator (exact copy of Applio/hifigan.py HiFiGANGenerator.__init__ + forward) ──
class HiFiGANGenerator(torch.nn.Module):
    def __init__(self, initial_channel, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)  # 3
        self.num_upsamples = len(upsample_rates)  # 4
        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        self.ups = torch.nn.ModuleList()
        self.resblocks = torch.nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(torch.nn.ConvTranspose1d(
                upsample_initial_channel // (2**i),                    # in: 512, 256, 128, 64
                upsample_initial_channel // (2 ** (i + 1)),             # out: 256, 128, 64, 32
                k, u, padding=(k - u) // 2
            ))
            ch = upsample_initial_channel // (2 ** (i + 1))  # 256, 128, 64, 32
            for j, (kk, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, kk, d))

        self.conv_post = torch.nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(self._init_weights)
        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.normal_(m.weight, 0.0, 0.001)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.01)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, 0.01)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

# Load model
pth_path = sys.argv[1] if len(sys.argv) > 1 else "models/rvc/cartman/EricCartmanV1_e650_s10400.pth"
ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
sd = ckpt['weight']
config = ckpt['config']
h = [x.item() if hasattr(x, 'item') else x for x in config]

print(f"Config: inter={h[3]}, filter={h[4]}, upsample_init={h[13]}, rates={h[12]}, kernels={h[14]}")
print(f"resblock_kernels={h[10]} dilations={h[11]}")

gen = HiFiGANGenerator(
    initial_channel=h[3],                    # 192
    resblock_kernel_sizes=h[10],             # [3,7,11]
    resblock_dilation_sizes=h[11],            # [[1,3,5],[1,3,5],[1,3,5]]
    upsample_rates=h[12],                     # [10,10,2,2]
    upsample_initial_channel=h[13],           # 512
    upsample_kernel_sizes=h[14],              # [16,16,4,4]
)

# Load weights — key names match our model
gen.load_state_dict(sd, strict=False)
print(f"Loaded {len(sd)} tensors into generator (missing: {[k for k in gen.state_dict() if k not in sd]})")

# Generate mel input (matching our C test: seed=42, shape (n_frames, 80), scale 2.0)
rng = np.random.RandomState(42)
n_frames = 4
mel_np = rng.randn(n_frames, 80).astype(np.float32) * 2.0

# Our C pipeline produces 256-channel flow output. For reference,
# we need to feed the generator's expected input (192 channels).
# We'll test BOTH paths:
# 1. Full pipeline: 80 mel → 192 ch (pad) → conv_pre → 512 → ups chain → conv_post
# 2. Direct to ups.0: 512 ch (pad from 80) → ups chain

# Path 1: Full pipeline through conv_pre
inter_ch = h[3]  # 192
gen_input = torch.zeros(1, inter_ch, n_frames)
for c in range(min(80, inter_ch)):
    gen_input[0, c, :] = torch.from_numpy(mel_np[:, c])

with torch.no_grad():
    out_full = gen(gen_input)  # (1, 1, n_audio)

out_full_np = out_full.squeeze().numpy()
print(f"\n=== Full Pipeline (conv_pre → ups → MRF → conv_post) ===")
print(f"Output: {len(out_full_np)} samples")
print(f"Stats: mean={out_full_np.mean():.6f} std={out_full_np.std():.6f} min={out_full_np.min():.6f} max={out_full_np.max():.6f} rms={np.sqrt(np.mean(out_full_np**2)):.6f}")

# Path 2: Skip conv_pre, feed 512 channels directly to ups.0
gen_input2 = torch.zeros(1, 512, n_frames)
for c in range(min(80, 512)):
    gen_input2[0, c, :] = torch.from_numpy(mel_np[:, c])

# Manually run ups chain (skip conv_pre, MRF, conv_post to match our C kernel)
x = F.leaky_relu(gen_input2, 0.01)
for i in range(4):
    x = F.leaky_relu(gen.ups[i](x), 0.01)
print(f"\nAfter ups chain: shape={x.shape}")

# Full through generator but using 512-channel input to ups directly
# (simulates what our C code does: bypass conv_pre)
class HiFiGANNoPre(torch.nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.ups = gen.ups
        self.resblocks = gen.resblocks
        self.conv_post = gen.conv_post
        self.num_kernels = gen.num_kernels
        self.num_upsamples = gen.num_upsamples

    def forward(self, x):
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.01)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, 0.01)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

gen_nopre = HiFiGANNoPre(gen)
# Feed 512 channels → ups.0 expects 512 input
# But ups.0 weight is (512, 256, 16) in PyTorch = in=512, out=256
# Wait — our ups.0 in_channels = 512 // 1 = 512, out = 512 // 2 = 256
# Our input is 512 channels → ups.0 takes 512 → outputs 256
# Then resblocks at 256 channels ✓
# Then ups.1 takes 256 → outputs 128 ✓
# etc.

# But our C code passes 256 channels (not 512) to ups.0!
# ups.0 weight is (512, 256, 16) = in=512, out=256 in PyTorch.
# Our code passes 512 (conv_pre_in) with only 256 actual values (zero-padded).

with torch.no_grad():
    # Feed 512 channels (first 256 from mel, rest zero) to ups chain directly
    out_nopre = gen_nopre(gen_input2)

out_nopre_np = out_nopre.squeeze().numpy()
print(f"\n=== No-conv_pre path (512 ch → ups chain) ===")
print(f"Output: {len(out_nopre_np)} samples")
print(f"Stats: mean={out_nopre_np.mean():.6f} std={out_nopre_np.std():.6f} min={out_nopre_np.min():.6f} max={out_nopre_np.max():.6f} rms={np.sqrt(np.mean(out_nopre_np**2)):.6f}")

# Save the full-pipeline reference
outpath = sys.argv[2] if len(sys.argv) > 2 else "reference_cartman.txt"
with open(outpath, 'w') as f:
    f.write(f"=== Full Pipeline (conv_pre → ups → MRF → conv_post) ===\n")
    f.write(f"n_samples={len(out_full_np)}\n")
    f.write(f"mean={out_full_np.mean():.6f}\n")
    f.write(f"std={out_full_np.std():.6f}\n")
    f.write(f"min={out_full_np.min():.6f}\n")
    f.write(f"max={out_full_np.max():.6f}\n")
    f.write(f"rms={np.sqrt(np.mean(out_full_np**2)):.6f}\n")
    for v in out_full_np:
        f.write(f"{v:.8e}\n")
print(f"\nReference saved to {outpath}")
