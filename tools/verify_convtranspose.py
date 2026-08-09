"""Verify conv_transpose_1d against PyTorch ConvTranspose1d."""
import torch, numpy as np

# Simple test: ConvTranspose1d(3, 2, k=4, s=2, p=1)
torch.manual_seed(42)
conv = torch.nn.ConvTranspose1d(3, 2, 4, 2, padding=1)
con = torch.nn.Conv1d(2, 1, 3, 1, padding=1)

inp = torch.randn(3, 5)  # (channels, length)
out = conv(inp.unsqueeze(0)).squeeze(0)  # (2, 10)
out2 = con(out.unsqueeze(0)).squeeze(0)

print(f"conv_transpose output shape: {out.shape}")
print(f"conv output shape: {out2.shape}")
print(f"conv_transpose out[0, :5]: {out[0, :5]}")
print(f"conv out[:5]: {out2[:5]}")

# Now save weights and input for C verification
np.save("verify_mel.npy", inp.numpy().astype(np.float32))
np.save("verify_w.npy", conv.weight.detach().numpy())
np.save("verify_b.npy", conv.bias.detach().numpy())
np.save("verify_conv1d_w.npy", con.weight.detach().numpy())
np.save("verify_conv1d_b.npy", con.bias.detach().numpy())
np.save("verify_out.npy", out2.detach().numpy().astype(np.float32))
print("Saved verification files")
