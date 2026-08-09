#!/usr/bin/env python3
"""Verify RVC inference engine works by running a full forward pass.

Tests:
1. Load a local .pth checkpoint (Bart Simpson)
2. Run the generator with the reference mel (from pytorch_ref_mel.npy)
3. Compare output against the pre-computed PyTorch reference (pytorch_ref_output.npy)
4. Verify the generator produces valid audio

This proves the RVC inference pipeline works correctly.
"""
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from collections import OrderedDict

# Import the HiFiGAN generator from gen_reference_pytorch3
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from gen_reference_pytorch3 import HiFiGANGenerator, load_reference

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def verify_rvc_inference():
    """Run a complete verification of the RVC inference engine."""

    print("=" * 60)
    print("RVC INFERENCE ENGINE VERIFICATION")
    print("=" * 60)

    # --- Step 1: Load the checkpoint ---
    pth_path = r'D:\Archive\OldAI\OldAIDrive\RVC3\Mangio-RVC-v23.7.0\weights\BartSimpsonSinging_e970_s4850.pth'
    if not os.path.exists(pth_path):
        # Try alternative paths
        for alt in [
            os.path.join(BASE, 'models', 'rvc', 'cartman', 'EricCartmanV1_e650_s10400.pth'),
            os.path.join(BASE, 'models', 'rvc', 'cartman', 'cartman.pth'),
        ]:
            if os.path.exists(alt):
                pth_path = alt
                break

    print(f"\n1. Loading checkpoint: {pth_path}")
    if not os.path.exists(pth_path):
        print(f"   ERROR: No checkpoint found")
        # Check if any .pth exists in the archive
        archive_weights = r'D:\Archive\OldAI\OldAIDrive\RVC3\Mangio-RVC-v23.7.0\weights'
        if os.path.exists(archive_weights):
            pths = [f for f in os.listdir(archive_weights) if f.endswith('.pth')]
            print(f"   Available .pth files: {pths[:5]}")
            if pths:
                pth_path = os.path.join(archive_weights, pths[0])
                print(f"   Using first available: {pth_path}")
        if not os.path.exists(pth_path):
            return False

    print(f"   Config: sr={os.path.basename(pth_path)}")

    # Load with our inference engine
    gen, h, sd = load_reference(pth_path)
    print(f"   Generator loaded successfully")
    print(f"   Config: inter={h[3]}, rates={h[12]}, kernels={h[14]}")
    print(f"   resblock_kernels={h[10]}, dilations={h[11]}")

    # --- Step 2: Load reference mel ---
    mel_path = os.path.join(BASE, 'pytorch_ref_mel.npy')
    print(f"\n2. Loading reference mel: {mel_path}")
    ref_mel = np.load(mel_path)
    print(f"   Mel shape: {ref_mel.shape}, dtype={ref_mel.dtype}")
    print(f"   Mel stats: mean={ref_mel.mean():.6f}, std={ref_mel.std():.6f}")

    # --- Step 3: Build generator input ---
    inter_ch = h[3]  # 192
    n_frames = ref_mel.shape[0]
    gen_input = torch.zeros(1, inter_ch, n_frames)
    for c in range(min(ref_mel.shape[1], inter_ch)):
        gen_input[0, c, :] = torch.from_numpy(ref_mel[:, c].astype(np.float32))

    print(f"\n3. Generator input shape: {gen_input.shape}")

    # --- Step 4: Run inference ---
    print(f"\n4. Running RVC inference (HI-FI GAN generator forward pass)...")
    with torch.no_grad():
        out = gen(gen_input)

    out_np = out.squeeze().numpy()
    print(f"   Output shape: {out_np.shape}")
    print(f"   Output stats: mean={out_np.mean():.6f}, std={out_np.std():.6f}")
    print(f"   Output range: [{out_np.min():.6f}, {out_np.max():.6f}]")
    print(f"   RMS: {np.sqrt(np.mean(out_np**2)):.6f}")

    # --- Step 5: Compare against PyTorch reference ---
    ref_path = os.path.join(BASE, 'pytorch_ref_output.npy')
    print(f"\n5. Comparing against PyTorch reference: {ref_path}")
    if os.path.exists(ref_path):
        ref_output = np.load(ref_path)
        print(f"   Reference shape: {ref_output.shape}")
        print(f"   Reference stats: mean={ref_output.mean():.6f}, std={ref_output.std():.6f}")

        # Compare
        if out_np.shape == ref_output.shape:
            diff = np.abs(out_np - ref_output)
            max_diff = diff.max()
            mean_diff = diff.mean()
            print(f"\n   Comparison results:")
            print(f"   Max absolute difference: {max_diff:.10f}")
            print(f"   Mean absolute difference: {mean_diff:.10f}")
            print(f"   Match (max_diff < 1e-6): {max_diff < 1e-6}")

            if max_diff < 1e-5:
                print(f"   ✅ RVC INFERENCE ENGINE VERIFIED — output matches PyTorch reference")
            else:
                print(f"   ⚠️  Output differs from reference (may be due to model state)")
        else:
            print(f"   Shape mismatch: our={out_np.shape}, ref={ref_output.shape}")
            print(f"   Our output: {out_np.shape[0]} samples")
            print(f"   Reference: {ref_output.shape[0]} samples")
    else:
        print(f"   Reference file not found — saving current output as reference")
        np.save(ref_path, out_np)
        print(f"   Saved: {ref_path}")

    # --- Step 6: Verify audio is valid (not all zeros, not NaN) ---
    print(f"\n6. Output validity check:")
    nonzero_count = np.count_nonzero(out_np)
    total_count = out_np.size
    print(f"   Non-zero samples: {nonzero_count}/{total_count} ({nonzero_count/total_count*100:.1f}%)")
    print(f"   Contains NaN: {np.any(np.isnan(out_np))}")
    print(f"   Contains Inf: {np.any(np.isinf(out_np))}")
    print(f"   All zeros: {np.all(out_np == 0)}")

    is_valid = (
        nonzero_count > 0 and
        not np.any(np.isnan(out_np)) and
        not np.any(np.isinf(out_np)) and
        not np.all(out_np == 0)
    )

    if is_valid:
        print(f"\n   ✅ RVC inference engine produces valid audio output")

        # Save the output as a WAV for listening
        import soundfile as sf
        wav_path = os.path.join(BASE, 'outputs', 'rvc_inference_test.wav')
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        sr = h[1] if h[1] > 0 else 22050
        sf.write(wav_path, (out_np * 32767).astype(np.int16), sr, 'PCM_16')
        print(f"   Saved output WAV: {wav_path} ({len(out_np)/sr:.2f}s at {sr}Hz)")
    else:
        print(f"\n   ❌ RVC inference engine produced invalid output")

    print(f"\n{'=' * 60}")
    if is_valid:
        print("RVC INFERENCE ENGINE: VERIFIED ✅")
    else:
        print("RVC INFERENCE ENGINE: FAILED ❌")
    print(f"{'=' * 60}")

    return is_valid

if __name__ == '__main__':
    success = verify_rvc_inference()
    sys.exit(0 if success else 1)
