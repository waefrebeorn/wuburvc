"""Debug: exact comparison of remove_weight_norm vs manual formula."""
import torch, torch.nn as nn, numpy as np, sys

ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
sd = ckpt['weight']

from torch.nn.utils import weight_norm, remove_weight_norm

# Method 1: Use weight_norm hook, then remove
conv = nn.ConvTranspose1d(512, 256, 16, 10, padding=3)
# The weight_norm will create weight_v and weight_g from the weight data
# So set the weight data first, then apply weight_norm, then override
conv.weight.data = torch.randn_like(conv.weight)  # dummy initial
conv = weight_norm(conv, name='weight', dim=0)
# Now override with checkpoint values
conv.weight_v.data.copy_(sd['dec.ups.0.weight_v'].float())
conv.weight_g.data.copy_(sd['dec.ups.0.weight_g'].float())
conv.bias.data.copy_(sd['dec.ups.0.bias'].float())

# Get the reparameterized weight (this is what forward() computes)
w_rep = conv.weight  # This is computed by the hook
print(f"Reparam weight shape: {w_rep.shape}")
print(f"Reparam first 5: {w_rep.flatten()[:5].tolist()}")

# Remove weight_norm and get final weight
remove_weight_norm(conv)
w_final = conv.weight.data.clone()
print(f"Final first 5: {w_final.flatten()[:5].tolist()}")

# Manual formula
wv = sd['dec.ups.0.weight_v'].float()
wg = sd['dec.ups.0.weight_g'].float()
# dim=0 for ConvTranspose1d (in_ch, out_ch, k) → norm per in_ch over (out_ch, k)
norm = torch.norm(wv, dim=(1, 2), keepdim=True)  # shape (in_ch, 1, 1)
print(f"norm shape: {norm.shape}")
w_manual = wg * wv / norm
print(f"Manual first 5: {w_manual.flatten()[:5].tolist()}")

diff = torch.abs(w_final - w_manual)
print(f"\nMax diff: {diff.max().item():.10f}")
print(f"Mean diff: {diff.mean().item():.10f}")

# Let's check if the formula uses 1e-8 epsilon
w_manual_eps = wg * wv / (norm + 1e-8)
diff_eps = torch.abs(w_final - w_manual_eps)
print(f"Max diff (with 1e-8 eps): {diff_eps.max().item():.10f}")
print(f"Mean diff (with 1e-8 eps): {diff_eps.mean().item():.10f}")

# Maybe the issue is that weight_norm normalizes with sqrt(norm_sq + eps)?
# Let's check PyTorch source for the formula
import inspect
print("\n--- weight_norm.py source (last 30 lines) ---")
src = inspect.getsource(sys.modules['torch.nn.utils.weight_norm'])
print('\n'.join(src.split('\n')[-30:]))
