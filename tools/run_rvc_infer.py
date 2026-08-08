"""
Run Cartman RVC model on base Piper TTS audio.
Generates Cartman-style speech from text.
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.utils import weight_norm, remove_weight_norm
import librosa
import soundfile as sf
from scipy import signal as scisignal

def load_rvc_model(pth_path):
    """Load RVC checkpoint and build the generator."""
    ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
    sd = ckpt['weight']
    config = ckpt['config']
    h = [x.item() if hasattr(x, 'item') else x for x in config]

    print(f"Config: inter={h[3]} (resblock_kernel={h[10]}), "
          f"upsample_init={h[13]}, rates={h[12]}, kernels={h[14]}")
    print(f"resblock_dilation={h[11]}")

    # Build the generator (HiFi-GAN based VITS/RVC architecture)
    from tools.gen_reference_pytorch3 import HiFiGANGenerator

    gen = HiFiGANGenerator(
        initial_channel=h[3],
        resblock_kernel_sizes=h[10],
        resblock_dilation_sizes=h[11],
        upsample_rates=h[12],
        upsample_initial_channel=h[13],
        upsample_kernel_sizes=h[14],
    )

    # Strip 'dec.' prefix from RVC checkpoint keys
    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith('dec.'):
            new_sd[k[4:]] = v
        else:
            new_sd[k] = v

    gen.load_state_dict(new_sd, strict=False)

    # Remove weight norm for inference
    gen.eval()
    for m in gen.resblocks:
        m.remove_weight_norm()
    for i in range(len(gen.ups)):
        if hasattr(gen.ups[i], 'remove_weight_norm'):
            gen.ups[i].remove_weight_norm()

    # Extract other necessary tensors
    # Post-rotation (flow) tensors
    post_rot = sd.get('dec.post.rot', None)
    post_conv = sd.get('dec.post.conv', None)
    emb = sd.get('dec.cond', None)  # conditional embedding
    # Text encoder / posterior encoder
    enc_p = sd.get('dpv2_enc_p.in_layers', None)  # depends on version

    return gen, h, sd


def rvc_infer(gen, h, audio_path, output_path):
    """Run RVC inference on audio file."""
    # Load audio at the model's sample rate (typically 22050 for RVC v1/v2)
    sr = 22050
    # Actually use the config's sample rate
    sr = h[1] if h[1] > 0 else 22050
    print(f"Sample rate: {sr}")

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    y = y.astype(np.float32)
    print(f"Input audio: {len(y)} samples ({len(y)/sr:.2f}s)")

    # Compute mel spectrogram
    # RVC uses a specific mel config
    n_fft = 1024
    n_mel = h[2]  # mel channels (usually 80 for v2, 192 for v1)
    hop = 256  # hop length
    win = 1024

    # Compute STFT -> mel
    spec = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win,
                        window='hann', center=True, pad_mode='reflect')
    # Mel filterbank
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mel,
                                     fmin=0, fmax=sr // 2, htk=False)
    mel = mel_basis @ np.abs(spec) ** 2  # power mel
    mel = mel.T  # (frames, mel)
    # Convert to log scale
    mel = np.log(np.maximum(mel, 1e-5))
    # Normalize using RVC's standard normalization
    mel = (mel - mel.mean()) / mel.std()  # rough normalization

    print(f"Mel shape: {mel.shape}, mean={mel.mean():.4f}, std={mel.std():.4f}")

    # Build tensor for the generator
    inter_ch = h[3]  # 192
    n_frames = mel.shape[0]

    # The generator expects (1, inter_ch, n_frames)
    # RVC: mel is (n_frames, mel_ch), generator pads to inter_ch channels
    gen_input = torch.zeros(1, inter_ch, n_frames)
    for c in range(min(n_mel, inter_ch)):
        gen_input[0, c, :] = torch.from_numpy(torch.tensor(mel[:, c], dtype=torch.float32))

    print(f"gen_input: {gen_input.shape}")

    # Run through the generator
    with torch.no_grad():
        out = gen(gen_input)
    out = out.squeeze().numpy()

    print(f"Output: {len(out)} samples ({len(out)/sr:.2f}s), "
          f"mean={out.mean():.6f}, std={out.std():.6f}")

    # Save as WAV
    sf.write(output_path, (out * 32767).astype(np.int16), sr, 'PCM_16')
    print(f"Saved to {output_path}")

    return out


if __name__ == '__main__':
    pth = 'models/rvc/cartman/EricCartmanV1_e650_s10400.pth'
    base_audio = 'outputs/cartman_base.wav'
    out_audio = 'outputs/cartman_rvc.wav'

    if len(sys.argv) > 1:
        # Use the provided text directly
        text = sys.argv[1]
    else:
        text = "Hello there! I am Eric Cartman and this is my text to speech system."
        print(f"Using default text: {text}")

    # If base audio doesn't exist, generate with Piper first
    if not os.path.exists(base_audio):
        print("Generating base TTS with Piper...")
        import subprocess
        result = subprocess.run([
            'python', '-m', 'piper',
            '--model', 'models/piper/en_US-amy-low.onnx',
            '--config', 'models/piper/en_US-amy-low.onnx.json',
            '--sentence-silence', '0.3',
            '--length-scale', '1.0',
            '-f', base_audio
        ], input=text, text=True, capture_output=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

    if not os.path.exists(base_audio):
        print("ERROR: Could not generate base audio")
        sys.exit(1)

    print("Loading Cartman RVC model...")
    gen, h, sd = load_rvc_model(pth)

    print("Running RVC inference...")
    out = rvc_infer(gen, h, base_audio, out_audio)
    print("Done!")
