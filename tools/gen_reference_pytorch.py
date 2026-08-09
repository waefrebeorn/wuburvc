#!/usr/bin/env python3
"""Generate PyTorch reference output for Cartman v2 model.
Runs the full HiFi-GAN generator forward pass using the actual .pth weights."""
import sys
import torch
import torch.nn.functional as F
import numpy as np

class ResBlock(torch.nn.Module):
    """HiFi-GAN ResBlock with depthwise dilated convolutions."""
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.convs1 = torch.nn.ModuleList()
        self.convs2 = torch.nn.ModuleList()
        for d in dilation:
            conv1 = torch.nn.Conv1d(channels, channels, kernel_size,
                                     dilation=d, padding=d * (kernel_size // 2))
            self.convs1.append(conv1)
        for _ in dilation:
            conv2 = torch.nn.Conv1d(channels, channels, kernel_size,
                                     dilation=d, padding=d * (kernel_size // 2))
            self.convs2.append(conv2)
        # ... etc — we'll just load weights directly

def load_and_run(pth_path, n_frames=3, mel_channels=80, hidden=256, sample_rate=40000):
    """Load Cartman v2 model and run forward pass."""
    ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
    sd = ckpt['weight']
    config = ckpt['config']

    print(f"Config channels: {config[15]}")  # upsample_initial_channels
    print(f"Config hidden: hidden_channels={config[4]}, inter_channels={config[3]}")

    # Create random mel input (n_frames, mel_channels) → (mel_channels, n_frames) for Conv1d
    torch.manual_seed(42)
    mel = torch.randn(mel_channels, n_frames, dtype=torch.float32)
    print(f"\nInput mel: shape={tuple(mel.shape)}")

    # Conv1d expects (batch=1, channels, length)
    x = mel.unsqueeze(0)  # (1, mel_channels, n_frames)

    # conv_pre: Conv1d(192 → 512, k=7, s=1, p=3) — wait, input is 80 mel channels, not 192
    # Actually in RVC v2, the input goes through preprocesser first. The flow coupling
    # expands 80 → hidden (256). So the input to the generator is hidden=256 channels.
    # Let me simulate with 256 input channels (post-flow)

    # Actually, let's just verify the channel flow by checking what conv_pre expects:
    conv_pre_w = sd['dec.conv_pre.weight']
    print(f"\nconv_pre.weight: {tuple(conv_pre_w.shape)}")
    # If shape is (512, 192, 7), conv_pre expects 192 input channels.
    # But we feed 256 from the flow. This means there's a mismatch unless the flow
    # outputs 192 channels (inter_channels).

    # Let's check the flow coupling output channels:
    flow_keys = [k for k in sd if k.startswith('flow.') and 'weight' in k and 'weight_g' in k]
    print(f"\nFlow weight_norm layers: {len(flow_keys)}")
    for k in flow_keys:
        print(f"  {k}: weight_g={tuple(sd[k].shape)}")

    # Check dec.input_conv or similar
    dec_keys = [k for k in sd if k.startswith('dec.') and 'conv' in k and 'weight' in k and 'weight_g' not in k]
    print(f"\nDec conv weights (non-norm):")
    for k in sorted(dec_keys):
        print(f"  {k}: {tuple(sd[k].shape)}")

    # Simulate the forward pass
    # 1. Flow coupling: 80 → 256 (inter_channels in config)
    # 2. conv_pre: 256 → 512 (if in_channels=256)
    #    OR: 192 → 512 (if in_channels=192, which matches conv_pre.weight shape)

    # Let's check: in_channels of conv_pre = shape[1]
    in_ch = conv_pre_w.shape[1]
    print(f"\nconv_pre expects {in_ch} input channels")

    # 3. PixelShuffle: 512 → 256 (halves channels)
    # 4. ups loop

    # Let's trace the channel flow:
    print("\n=== Channel flow through generator ===")
    # After flow coupling + conv_pre + PixelShuffle: 256 channels
    ch = 256
    print(f"Input to ups loop: {ch}")
    for i in range(4):
        ups_ch = sd[f'dec.ups.{i}.weight_v']
        out_ch, in_ch, k = ups_ch.shape
        print(f"  ups.{i}: ConvTranspose1d(in={in_ch}, out={out_ch}) → {out_ch} ch")
        # ResBlock operates at what channel count?
        rb_start = i * 3
        for j in range(rb_start, rb_start + 3):
            rb_ch = sd[f'dec.resblocks.{j}.convs1.0.weight_g'].shape[0]
        print(f"  resblocks {rb_start}-{rb_start+2}: {rb_ch} ch")
        ch = out_ch  # after upsample

    print(f"\nconv_post: {tuple(sd['dec.conv_post.weight'].shape)}")

    # Now run the actual forward with proper channels
    print("\n=== Running forward pass ===")
    # The flow coupling produces hidden_channels features per frame.
    # Let's create synthetic flow output at the right channel count.
    flow_out_ch = sd['dec.conv_pre.weight'].shape[1]  # should be 256
    print(f"Flow output → conv_pre input channels: {flow_out_ch}")

    x = torch.randn(1, flow_out_ch, n_frames)

    # conv_pre
    w = sd['dec.conv_pre.weight'].float()
    b = sd['dec.conv_pre.bias'].float() if 'dec.conv_pre.bias' in sd else None
    x = F.conv_transpose1d(x, w, b, stride=1, padding=3)  # wait, conv_pre is Conv1d not ConvTranspose1d
    x = F.conv1d(x, w, b, stride=1, padding=3)
    print(f"After conv_pre: {tuple(x.shape)}")

    # PixelShuffle: 512 → 256
    x = F.pixel_shuffle(x.unsqueeze(-1), upscale_factor=2).squeeze(-1)
    print(f"After PixelShuffle: {tuple(x.shape)}")

    # Ups layer loop
    ups_rates = config[13] if len(config) > 13 else [10, 10, 2, 2]
    ups_kernels = config[14] if len(config) > 14 else [16, 16, 4, 4]
    ups_pads = [k // 2 - 1 if k > 2 else 0 for k in ups_kernels]

    for i in range(4):
        w = sd[f'dec.ups.{i}.weight_v'].float()
        g = sd[f'dec.ups.{i}.weight_g'].float()
        b = sd[f'dec.ups.{i}.bias'].float() if f'dec.ups.{i}.bias' in sd else None

        # De-normalize weight_norm: w = w_v * (w_g / ||w_v||)
        w_norm = torch.norm(w, p=2, dim=(1, 2), keepdim=True)
        w = w * (g / w_norm)

        # conv_transpose1d: PyTorch format is (in_ch, out_ch, k) for the weight
        # But w_v is stored as (out_ch, in_ch, k)
        # F.conv_transpose1d expects weight shape (in_ch, out_ch, k)
        w_pt = w.permute(1, 0, 2).contiguous()
        padding = k // 2 - 1 if k > 2 else 0
        # Actually the padding for RVC v2 upsampler: k//2 - 1?
        # Let's check: for k=16, s=10: PyTorch ConvTranspose1d padding affects output size
        # n_out = (n_in - 1)*stride - 2*padding + (kernel-1) + 1 + output_padding
        # RVC uses: padding = kernel_size // 2 - stride // 2 (for stride > 1)
        # For k=16, s=10: padding = 16//2 - 10//2 = 8 - 5 = 3... or k//2 - 1 = 7
        # Let's check both and see which matches the Cartman output length

        x = F.conv_transpose1d(x, w_pt, b, stride=ups_rates[i],
                                padding=ups_pads[i], output_padding=ups_pads[i])
        print(f"After ups.{i} (s={ups_rates[i]}, k={ups_kernels[i]}, p={ups_pads[i]}): {tuple(x.shape)}")

        # MRF: resblocks at stage i
        # We need to check the channel count
        rb_start = i * 3
        # The resblocks might need a channel projection if ch != rb_ch
        # In standard HiFi-GAN, the ResBlock doesn't change channels
        # But here ups.{i}.out_ch != resblock ch, so there must be a channel adjustment

        # Let's just skip MRF for now and check if conv_post works
        for j in range(rb_start, rb_start + 3):
            rb_w1 = sd[f'dec.resblocks.{j}.convs1.0.weight_v'].float()
            rb_ch = rb_w1.shape[0]
            rb_k = rb_w1.shape[2]
            rb_d = [1]  # all dilation=1 for now
            print(f"  resblock.{j}: ch={rb_ch}, k={rb_k}")

    print(f"\nFinal before conv_post: {tuple(x.shape)}")
    print(f"conv_post.weight: {tuple(sd['dec.conv_post.weight'].shape)}")
    print(f"conv_post expects {sd['dec.conv_post.weight'].shape[1]} input channels")
    print(f"But we have {x.shape[1]} channels")

    # The mismatch tells us about the architecture

if __name__ == '__main__':
    load_and_run(sys.argv[1] if len(sys.argv) > 1 else 'models/rvc/cartman/EricCartmanV1_e650_s10400.pth')
