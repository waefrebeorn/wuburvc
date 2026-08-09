"""
Cartman RVC voice conversion: generate speech with Piper TTS, then
convert to Cartman's voice using the Cartman v1 RVC model checkpoint.

Full pipeline:
  1. Piper TTS generates base speech (English female voice)
  2. We extract content features (mel spectrogram) from the base audio
  3. We run the Cartman RVC generator (HiFi-GAN) to produce Cartman-style speech
  4. Save the result and create a video with Cartman's image
"""
import sys
import os
import numpy as np
import torch
import librosa
import soundfile as sf

# Import from our tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from gen_reference_pytorch3 import load_reference


def extract_mel(audio, sr, n_mel=32, n_fft=1025, hop=256, win=None):
    """Extract mel spectrogram from audio, matching RVC's config."""
    if win is None:
        win = n_fft
    spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop, win_length=win,
                        window='hann', center=True, pad_mode='reflect')
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mel,
                                     fmin=0, fmax=sr//2, htk=False)
    mel = mel_basis @ (np.abs(spec) ** 2)
    mel = mel.T  # (frames, mel_ch)
    mel = np.log(np.maximum(mel, 1e-5))
    # RVC normalization: (log_mel + 5.5) / 2.0  (standard RVC mel scaling)
    mel = (mel + 5.5) / 2.0
    return mel.astype(np.float32)


def rvc_convert(base_audio_path, output_path, pth_path, text=None):
    """Convert base audio to Cartman's voice using RVC HiFi-GAN generator."""
    print("Loading Cartman RVC model...")
    gen, h, sd = load_reference(pth_path)
    print(f"  Config: sr={h[1]}, mel_ch={h[2]}, inter={h[3]}, upsample_init={h[13]}")

    n_mel = int(h[1])  # RVC config: h[1] = n_mel (32 for v1, 80 for v2)
    inter_ch = int(h[2])  # 192
    n_fft = int(h[0])  # 1025
    sr_model = int(h[17]) if len(h) > 17 and isinstance(h[17], (int, float)) and h[17] > 1000 else 22050

    # Load base audio
    y, sr_orig = librosa.load(base_audio_path, sr=None, mono=True)
    # Resample to model's sample rate
    if sr_orig != sr_model:
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr_model)
        print(f"Resampled: {sr_orig} -> {sr_model} Hz")
    sr = sr_model
    y = y.astype(np.float32)
    print(f"Base audio: {len(y)} samples ({len(y)/sr:.2f}s) @ {sr}Hz")

    # Extract mel
    mel = extract_mel(y, sr, n_mel=n_mel, n_fft=n_fft)
    n_frames = mel.shape[0]
    print(f"Mel: {mel.shape}, mean={mel.mean():.4f}, std={mel.std():.4f}")

    # Build generator input (1, inter_ch, n_frames)
    gen_input = torch.zeros(1, inter_ch, n_frames)
    for c in range(min(n_mel, inter_ch)):
        gen_input[0, c, :] = torch.from_numpy(mel[:, c].astype(np.float32))

    print(f"gen_input: {gen_input.shape}")

    # Run generator
    with torch.no_grad():
        out = gen(gen_input)
    out_np = out.squeeze().numpy()

    print(f"Output: {len(out_np)} samples ({len(out_np)/sr:.2f}s)")
    print(f"  mean={out_np.mean():.6f}, std={out_np.std():.6f}, min={out_np.min():.6f}, max={out_np.max():.6f}")

    # Clip and save
    out_np = np.clip(out_np, -1.0, 1.0)
    sf.write(output_path, (out_np * 32767).astype(np.int16), sr, 'PCM_16')
    print(f"Saved to {output_path}")

    return out_np, sr


if __name__ == '__main__':
    pth_path = 'models/rvc/cartman/EricCartmanV1_e650_s10400.pth'
    base_audio = 'outputs/cartman_base.wav'
    out_audio = 'outputs/cartman_rvc.wav'

    if not os.path.exists(base_audio):
        print(f"ERROR: Base audio not found at {base_audio}")
        print("Generate it first with Piper TTS.")
        sys.exit(1)

    out, sr = rvc_convert(base_audio, out_audio, pth_path)
    print("\nDone! Cartman RVC audio generated.")
