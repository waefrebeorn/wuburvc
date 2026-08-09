#!/usr/bin/env python3
"""Generate PyTorch reference output for Cartman v2 — exact forward pass.
Compares against our C11 kernel output to find discrepancies."""
import sys, os, numpy as np, torch
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.utils import weight_norm, remove_weight_norm


class ResBlock(torch.nn.Module):
    """HiFi-GAN ResBlock (Applio residuals.py exact copy)."""
    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = torch.nn.ModuleList()
        self.convs2 = torch.nn.ModuleList()
        for d in dilations:
            self.convs1.append(weight_norm(torch.nn.Conv1d(
                channels, channels, kernel_size, 1,
                padding=d * (kernel_size // 2), dilation=d
            )))
        for d in dilations:
            self.convs2.append(weight_norm(torch.nn.Conv1d(
                channels, channels, kernel_size, 1,
                padding=d * (kernel_size // 2), dilation=d
            )))

    def forward(self, x):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            x_residual = x
            x = F.leaky_relu(x, 0.1)
            x = conv1(x)
            x = F.leaky_relu(x, 0.1)
            x = conv2(x)
            x = x + x_residual
        return x

    def remove_weight_norm(self):
        for conv in self.convs1:
            remove_weight_norm(conv)
        for conv in self.convs2:
            remove_weight_norm(conv)


class HiFiGANGenerator(torch.nn.Module):
    """Exact copy of Applio/hifigan.py HiFiGANGenerator."""
    def __init__(self, initial_channel, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes,
                 gin_channels=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.ups = torch.nn.ModuleList()
        self.resblocks = torch.nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(torch.nn.ConvTranspose1d(
                upsample_initial_channel // (2**i),
                upsample_initial_channel // (2 ** (i + 1)),
                k, u, padding=(k - u) // 2
            ))
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (kk, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, kk, d))
        self.conv_post = torch.nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


def load_reference(pth_path):
    """Load model and return (gen, h, sd)."""
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

    # Strip 'dec.' prefix that RVC adds to generator keys
    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith('dec.'):
            new_sd[k[4:]] = v
    gen.load_state_dict(new_sd, strict=False)

    gen.eval()
    for m in gen.resblocks:
        m.remove_weight_norm()
    for i in range(len(gen.ups)):
        if hasattr(gen.ups[i], 'remove_weight_norm'):
            gen.ups[i].remove_weight_norm()

    return gen, h, sd


if __name__ == '__main__':
    pth_path = sys.argv[1]
    gen, h, sd = load_reference(pth_path)

    print(f"Config: inter={h[3]}, upsample_init={h[13]}, rates={h[12]}, kernels={h[14]}")
    print(f"resblock_kernels={h[10]} dilations={h[11]}")
    print(f"conv_pre: ({sd['dec.conv_pre.weight'].shape[1]}, {sd['dec.conv_pre.weight'].shape[0]}, 7)")

    # Generate mel (matching our C test: seed=42, shape (4, 80), scale 2.0)
    rng = np.random.RandomState(42)
    n_frames = 4
    mel_np = rng.randn(n_frames, 80).astype(np.float32) * 2.0

    # Save mel for C11 comparison test
    np.save("pytorch_ref_mel.npy", mel_np)

    # Build 192-channel input (inter_channels) from 80 mel
    inter_ch = h[3]  # 192
    gen_input = torch.zeros(1, inter_ch, n_frames)
    for c in range(min(80, inter_ch)):
        gen_input[0, c, :] = torch.from_numpy(mel_np[:, c])

    # Run full pipeline
    with torch.no_grad():
        out = gen(gen_input)

    out_np = out.squeeze().numpy()
    print(f"\nOutput: {len(out_np)} samples")
    print(f"Stats: mean={out_np.mean():.6f} std={out_np.std():.6f} min={out_np.min():.6f} max={out_np.max():.6f} rms={np.sqrt(np.mean(out_np**2)):.6f}")

    # Save for comparison
    np.save("pytorch_ref_output.npy", out_np)
    with open("pytorch_ref_stats.txt", "w") as f:
        f.write(f"n_samples={len(out_np)}\n")
        f.write(f"mean={out_np.mean():.6f}\n")
        f.write(f"std={out_np.std():.6f}\n")
        f.write(f"min={out_np.min():.6f}\n")
        f.write(f"max={out_np.max():.6f}\n")
        f.write(f"rms={np.sqrt(np.mean(out_np**2)):.6f}\n")
        for v in out_np:
            f.write(f"{v:.8e}\n")

    print(f"\nSaved PyTorch reference: {len(out_np)} samples")
