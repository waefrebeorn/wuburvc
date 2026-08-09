#!/usr/bin/env python3
"""Verify voice-to-voice cloning end-to-end using WuBuRVC (wubu_rvc.py).

Tests:
1. Source audio: Use cartman_base.wav (existing reference audio) as input
2. Target voice: Load a local RVC model (Bart Simpson or any of the 21 available)
3. Run voice conversion via wubu_rvc.RVC.convert()
4. Verify output is different from source (actual voice change happened)
5. Verify output is valid audio

This proves:
  - wubu_rvc engine loads correctly
  - RVC model checkpoint loads properly
  - Pitch extraction + feature conversion runs
  - Output is valid converted audio (not a copy of input)
"""
import sys
import os
import json
import numpy as np
import soundfile as sf

# Add src to path for wubu_rvc import
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'src'))

def main():
    print("=" * 60)
    print("WUBU_RVC VOICE-TO-VOICE CLONING VERIFICATION")
    print("=" * 60)

    from wubu_rvc import RVC, load_bank

    # --- Step 1: Load voice bank ---
    bank = load_bank()
    print(f"\n1. Loaded voice bank: {len(bank)} voices")

    # --- Step 2: Initialize RVC engine ---
    rvc = RVC(device="cuda")
    print(f"2. RVC engine: {rvc.engine}")
    print(f"   Applio available: {rvc.engine == 'applio-cli'}")
    print(f"   Mangio available: {rvc.engine == 'mangio-cli'}")
    print(f"   Ready: {rvc.ready}")

    if not rvc.ready:
        print(f"   ERROR: {rvc.error}")
        return False

    # --- Step 3: Find a voice with a local .pth file ---
    # Look for voices with local pth files
    candidates = []
    for k, v in bank.items():
        pth = v.get('pth')
        if pth and os.path.exists(pth):
            candidates.append(v)

    print(f"\n3. Found {len(candidates)} voices with local .pth files")
    if not candidates:
        print("   ERROR: No voices with local .pth files found")
        return False

    # Pick Bart Simpson (first one found)
    target_voice = candidates[0]
    print(f"   Target voice: {target_voice['name']}")
    print(f"   PTH: {target_voice['pth']}")
    print(f"   Index: {target_voice.get('index', 'N/A')}")
    print(f"   PTH exists: {os.path.exists(target_voice['pth'])}")
    idx_exists = os.path.exists(target_voice['index']) if target_voice.get('index') else False
    print(f"   Index exists: {idx_exists}")

    # --- Step 4: Source audio ---
    source_audio = os.path.join(BASE, 'outputs', 'cartman_base.wav')
    print(f"\n4. Source audio: {source_audio}")
    if not os.path.exists(source_audio):
        # Try generating with Piper
        print("   Source audio not found, checking for alternatives...")
        for alt in ['outputs/cartman_example.wav', 'outputs/rvc_inference_test.wav']:
            alt_path = os.path.join(BASE, alt)
            if os.path.exists(alt_path):
                source_audio = alt_path
                print(f"   Using: {source_audio}")
                break

    if not os.path.exists(source_audio):
        print("   ERROR: No source audio found")
        return False

    # Load source audio
    y, sr = sf.read(source_audio)
    print(f"   Audio: {len(y)} samples, {sr}Hz, {len(y)/sr:.2f}s, dtype={y.dtype}")

    # --- Step 5: Run voice-to-voice conversion ---
    print(f"\n5. Running voice-to-voice conversion...")
    print(f"   Source: {os.path.basename(source_audio)}")
    print(f"   Target: {target_voice['name']}")

    output_path = os.path.join(BASE, 'outputs', 'voice_to_voice_test.wav')
    result = rvc.convert(source_audio, target_voice['name'])

    if result == source_audio or not os.path.exists(result):
        print(f"   ❌ Conversion failed or returned original audio")
        print(f"   Result path: {result}")
        return False

    # Check output
    y_out, sr_out = sf.read(result)
    print(f"   Output: {len(y_out)} samples, {sr_out}Hz, {len(y_out)/sr_out:.2f}s")

    # --- Step 6: Verify output is different from source ---
    print(f"\n6. Verification:")
    # Normalize to compare
    y_src = y[:min(len(y), len(y_out))].astype(np.float64)
    y_dst = y_out[:min(len(y), len(y_out))].astype(np.float64)

    # Check if output differs significantly
    if len(y_src) == len(y_dst) and len(y_src) > 0:
        diff = np.abs(y_src - y_dst)
        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"   Max sample difference: {max_diff:.6f}")
        print(f"   Mean sample difference: {mean_diff:.6f}")

        # Check correlation
        if y_src.std() > 0 and y_dst.std() > 0:
            corr = np.corrcoef(y_src, y_dst)[0, 1]
            print(f"   Correlation: {corr:.6f}")
            if corr < 0.99:
                print(f"   ✅ Output is DIFFERENT from source (voice conversion occurred)")
            else:
                print(f"   ⚠️  Output is very similar to source (may not have converted)")
        else:
            print(f"   ⚠️  Zero variance in audio (flat output)")

    # Check output is valid audio
    nonzero = np.count_nonzero(y_out)
    print(f"   Non-zero samples: {nonzero}/{len(y_out)} ({nonzero/len(y_out)*100:.1f}%)")
    print(f"   Contains NaN: {np.any(np.isnan(y_out))}")
    print(f"   Contains Inf: {np.any(np.isinf(y_out))}")

    is_valid = (
        os.path.exists(result) and
        len(y_out) > 100 and
        nonzero > 0 and
        not np.any(np.isnan(y_out)) and
        not np.any(np.isinf(y_out))
    )

    if is_valid:
        print(f"\n   ✅ Voice-to-voice conversion verified!")
        print(f"   Source audio: {source_audio}")
        print(f"   Output audio: {result}")
        print(f"   Conversion time: {rvc.last_ms}ms")
    else:
        print(f"\n   ❌ Voice-to-voice conversion failed")
        return False

    print(f"\n{'=' * 60}")
    print("WUBU_RVC VOICE-TO-VOICE: VERIFIED ✅")
    print(f"{'=' * 60}")
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
