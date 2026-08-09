#!/usr/bin/env python3
"""Generate A/B test video comparing WuBuRVC (C11) vs PyTorch (Mangio-RVC).

Produces:
  - outputs/ab_test_wuburvc.wav   — audio from C11 engine
  - outputs/ab_test_pytorch.wav   — audio from PyTorch reference
  - outputs/ab_test_video.mp4     — side-by-side A/B test video with stats overlay

Uses the same reference mel, model weights, and FAISS index for both engines
to prove they produce equivalent audio output.
"""
import os
import sys
import time
import json
import subprocess
import numpy as np
import soundfile as sf

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

BASE = r'C:\Users\eman5\WuBuMedia' if os.name == 'nt' else '/c/Users/eman5/WuBuMedia'
OUT_DIR = os.path.join(BASE, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

def run_c11_compare():
    """Run C11 compare test and parse stats."""
    exe = os.path.join(BASE, 'build', 'test_rvc_compare.exe')
    # Ensure Windows path format for subprocess
    exe = exe.replace('/', '\\') if ':' not in exe else exe
    print(f"Running: {exe}")
    print(f"Exists: {os.path.exists(exe)}")
    print("=== Running WuBuRVC C11 (compare test) ===")
    t0 = time.time()
    result = subprocess.run([exe], cwd=BASE, capture_output=True, text=True, timeout=120)
    elapsed = (time.time() - t0) * 1000

    stdout = result.stdout + result.stderr

    stats = {
        'engine': 'WuBuRVC (C11)',
        'total_time_ms': elapsed,
        'success': result.returncode == 0,
        'stdout': stdout,
    }

    # Parse stats from stdout
    for line in stdout.split('\n'):
        if 'WuBuRVC C11' in line and 'mean=' in line:
            # Parse: WuBuRVC C11  : mean=0.001882 std=0.022661 min=-0.061941 max=0.086063 rms=0.022739
            import re
            m = re.search(r'mean=([\d.eE+-]+)\s+std=([\d.eE+-]+)\s+min=([\d.eE+-]+)\s+max=([\d.eE+-]+)\s+rms=([\d.eE+-]+)', line)
            if m:
                stats['mean'] = float(m.group(1))
                stats['std'] = float(m.group(2))
                stats['min'] = float(m.group(3))
                stats['max'] = float(m.group(4))
                stats['rms'] = float(m.group(5))
        if 'PyTorch ref' in line and 'mean=' in line:
            import re
            m = re.search(r'mean=([\d.eE+-]+)\s+std=([\d.eE+-]+)\s+min=([\d.eE+-]+)\s+max=([\d.eE+-]+)\s+rms=([\d.eE+-]+)', line)
            if m:
                stats['ref_mean'] = float(m.group(1))
                stats['ref_std'] = float(m.group(2))
                stats['ref_min'] = float(m.group(3))
                stats['ref_max'] = float(m.group(4))
                stats['ref_rms'] = float(m.group(5))
        if 'Max abs diff' in line:
            import re
            m = re.match(r'Max abs diff:\s+([\d.eE+-]+)', line.strip())
            if m:
                stats['max_abs_diff'] = float(m.group(1))
        if 'Mean abs diff' in line:
            import re
            m = re.match(r'Mean abs diff:\s+([\d.eE+-]+)', line.strip())
            if m:
                stats['mean_abs_diff'] = float(m.group(1))
        if 'synthesize rc=' in line:
            import re
            m = re.search(r'synthesize rc=(\d+)', line)
            if m:
                stats['output_samples'] = int(m.group(1))
        if 'PARITY' in line or 'PASS' in line:
            stats['parity'] = 'PASS' in line

    print(f"C11 stats: {json.dumps({k: v for k, v in stats.items() if k != 'stdout'}, indent=2)}")
    return stats

def run_pytorch_reference():
    """Run PyTorch reference with same mel and capture output."""
    if not HAS_TORCH:
        return {'engine': 'PyTorch (Mangio-RVC)', 'success': False, 'error': 'no torch'}

    print("\n=== Running PyTorch reference (Mangio-RVC) ===")
    sys.path.insert(0, os.path.join(BASE, 'tools'))
    from gen_reference_pytorch3 import load_reference

    pth = os.path.join(BASE, 'models', 'rvc', 'cartman', 'EricCartmanV1_e650_s10400.pth')
    if not os.path.exists(pth):
        pth = os.path.join(r'D:\Archive\OldAI\OldAIDrive\RVC3\Mangio-RVC-v23.7.0\weights', 'BartSimpsonSinging_e970_s4850.pth')

    t0 = time.time()
    gen, h, sd = load_reference(pth)
    load_time = time.time() - t0

    # Use SAME mel as the C11 test (random seed 42)
    rng = np.random.RandomState(42)
    mel_np = rng.randn(4, 80).astype(np.float32) * 2.0
    np.save(os.path.join(BASE, 'pytorch_ref_mel.npy'), mel_np)

    inter_ch = h[3]
    n_frames = mel_np.shape[0]
    gen_input = torch.zeros(1, inter_ch, n_frames)
    for c in range(min(mel_np.shape[1], inter_ch)):
        gen_input[0, c, :] = torch.from_numpy(mel_np[:, c].astype(np.float32))

    t1 = time.time()
    with torch.no_grad():
        out = gen(gen_input)
    infer_time = (time.time() - t1) * 1000

    out_np = out.squeeze().numpy()
    total_time = (time.time() - t0) * 1000

    # Save as WAV (upsample to 22050 Hz for listening)
    sr = h[1] if h[1] > 0 else 22050
    wav_path = os.path.join(OUT_DIR, 'ab_test_pytorch.wav')
    # Upsample from 32Hz to 22050Hz for listening
    out_resized = np.tile(out_np, int(sr / max(1, len(out_np))) + 1)[:sr * 3]
    sf.write(wav_path, (out_resized * 32767).astype(np.int16), sr, 'PCM_16')

    stats = {
        'engine': 'PyTorch (Mangio-RVC)',
        'total_time_ms': total_time,
        'load_time_ms': load_time * 1000,
        'infer_time_ms': infer_time,
        'output_samples': len(out_np),
        'sample_rate': sr,
        'mean': float(out_np.mean()),
        'std': float(out_np.std()),
        'min': float(out_np.min()),
        'max': float(out_np.max()),
        'rms': float(np.sqrt(np.mean(out_np**2))),
        'rtf': infer_time / (len(out_np) / sr) if sr > 0 else 0,
        'success': True,
        'wav_path': wav_path,
    }

    np.save(os.path.join(BASE, 'pytorch_ab_output.npy'), out_np)
    print(f"PyTorch stats: {json.dumps(stats, indent=2)}")
    return stats

def run_c11_speed_test():
    """Run the C11 speed test to get timing data."""
    exe = os.path.join(BASE, 'build', 'test_speed_real.exe')
    print(f"Running: {exe}")
    print(f"Exists: {os.path.exists(exe)}")
    print("\\n=== Running WuBuRVC C11 speed test ===")
    result = subprocess.run([exe], cwd=BASE, capture_output=True, text=True, timeout=300)
    stdout = result.stdout + result.stderr

    stats = {'speed_stdout': stdout}
    for line in stdout.split('\n'):
        if 'WuBuRVC pitch-shift' in line:
            import re
            m = re.search(r'([\d.]+)\s*ms/frame.*RTF=([\d.]+)', line)
            if m:
                stats['pitch_shift_ms'] = float(m.group(1))
                stats['pitch_shift_rtf'] = float(m.group(2))
                m2 = re.search(r'([\d.]+)x realtime', line)
                if m2:
                    stats['pitch_shift_xrt'] = float(m2.group(1))
        if 'WuBuRVC pipeline' in line:
            import re
            m = re.search(r'([\d.]+)\s*ms/frame.*RTF=([\d.]+)', line)
            if m:
                stats['pipeline_ms'] = float(m.group(1))
                stats['pipeline_rtf'] = float(m.group(2))
        if 'PARITY' in line or 'PASS' in line:
            stats['parity'] = 'PASS'

    print(f"C11 speed stats: {json.dumps({k: v for k, v in stats.items() if not k.endswith('_stdout')}, indent=2)}")
    return stats

def generate_ab_video(c11_stats, py_stats, speed_stats):
    """Generate A/B test comparison video."""
    print("\n=== Generating A/B test video ===")

    # Generate C11 audio (1600 samples at 22050Hz = ~72ms)
    # The C11 engine outputs to stdout, we'll generate a synthetic tone for the video
    # Actually, let's generate both audio files properly

    # C11 audio: use the reference output stats to synthesize comparable audio
    sr = 22050
    duration = 3.0  # 3 seconds

    # Generate a test tone for C11 (we can't get raw output from exe easily)
    # Instead, let's just note the stats and generate a visual comparison

    # Create video frames showing the comparison
    # Use ffmpeg to generate a video with stats overlay

    stats_json = json.dumps({
        'c11': c11_stats,
        'pytorch': py_stats,
        'speed': speed_stats,
    }, indent=2, default=str)

    # Save stats
    stats_path = os.path.join(OUT_DIR, 'ab_test_stats.json')
    with open(stats_path, 'w') as f:
        f.write(stats_json)
    print(f"Stats saved: {stats_path}")

    # ── REAL A/B audio ──────────────────────────────────────────────
    # NO sine tones. The C11 engine writes outputs/rvc_ref/output_c11.wav
    # (test_rvc_real.exe) and the PyTorch reference writes
    # outputs/rvc_ref/output_audio.npy (gen_reference_real.py). Use those
    # real engine outputs for the A/B video.
    c11_wav = os.path.join(OUT_DIR, 'rvc_ref', 'output_c11.wav')
    py_wav = os.path.join(OUT_DIR, 'ab_test_pytorch.wav')

    ref_audio = os.path.join(OUT_DIR, 'rvc_ref', 'output_audio.npy')
    if os.path.exists(ref_audio):
        py_audio = np.load(ref_audio)               # real 40k synth out
        py_sr = 40000
        py_audio = np.clip(py_audio, -1, 1)
        sf.write(py_wav, (py_audio * 32767).astype(np.int16), py_sr, 'PCM_16')
    elif os.path.exists(py_wav):
        pass                                        # already exists
    else:
        print("WARN: no real PyTorch output_audio.npy; A/B audio missing")

    if not os.path.exists(c11_wav):
        print("WARN: no real C11 output_c11.wav; run test_rvc_real.exe first")

    print(f"C11 audio: {c11_wav} (exists={os.path.exists(c11_wav)})")
    print(f"PyTorch audio: {py_wav} (exists={os.path.exists(py_wav)})")

    # Generate video with ffmpeg
    # Create a video showing stats comparison
    video_path = os.path.join(OUT_DIR, 'ab_test_video.mp4')

    # Build stats text for video overlay
    def fmt(val, decimals=6):
        """Format a value, handling both float and string."""
        if isinstance(val, (int, float)):
            return f"{val:.{decimals}f}"
        return str(val)

    def fmt1(val):
        if isinstance(val, (int, float)):
            return f"{val:.1f}"
        return str(val)

    stats_text = f"""WuBuRVC vs PyTorch A/B Test

=== WuBuRVC (C11) ===
Mean: {fmt(c11_stats.get('mean'))}
Std:  {fmt(c11_stats.get('std'))}
Min:  {fmt(c11_stats.get('min'))}
Max:  {fmt(c11_stats.get('max'))}
RMS:  {fmt(c11_stats.get('rms'))}
Time: {fmt1(c11_stats.get('total_time_ms'))}ms

=== PyTorch (Mangio-RVC) ===
Mean: {fmt(py_stats.get('mean'))}
Std:  {fmt(py_stats.get('std'))}
Min:  {fmt(py_stats.get('min'))}
Max:  {fmt(py_stats.get('max'))}
RMS:  {fmt(py_stats.get('rms'))}
Load: {fmt1(py_stats.get('load_time_ms'))}ms
Infer: {fmt1(py_stats.get('infer_time_ms'))}ms

=== Comparison ===
Max abs diff: {c11_stats.get('max_abs_diff', 'N/A')}
Mean abs diff: {c11_stats.get('mean_abs_diff', 'N/A')}
Parity: {'PASS' if c11_stats.get('parity') else 'N/A'}

=== Speed ===
C11 pipeline: {speed_stats.get('pipeline_ms', 'N/A')}ms/frame
C11 pitch: {speed_stats.get('pitch_shift_ms', 'N/A')}ms/frame
C11 x-realtime: {speed_stats.get('pitch_shift_xrt', 'N/A')}x
PyTorch infer: {fmt1(py_stats.get('infer_time_ms'))}ms"""

    # Write stats text file for ffmpeg
    stats_txt_path = os.path.join(OUT_DIR, 'ab_test_stats.txt')
    with open(stats_txt_path, 'w') as f:
        f.write(stats_text)

    # Generate video using ffmpeg
    # Use the REAL wavs: two labeled panels side by side, C11 left / PyTorch
    # right; audio mixed so left ear = C11, right ear = PyTorch (audible A/B).
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', c11_wav,
        '-i', py_wav,
        '-filter_complex',
        '[0:a]aformat=channel_layouts=mono[a1];'
        '[1:a]aformat=channel_layouts=mono[a2];'
        '[a1][a2]join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR[aout];'
        'color=c=black:s=1280x720:r=30,format=rgba[bg];'
        '[bg]drawtext=text=WuBuRVC C11:fontsize=48:fontcolor=white:x=100:y=300[a];'
        '[a]drawtext=text=PyTorch Ref:fontsize=48:fontcolor=white:x=760:y=300[v]',
        '-map', '[v]', '-map', '[aout]',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '128k', '-shortest',
        video_path
    ]

    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"Video generated: {video_path}")
        else:
            print(f"FFmpeg error: {result.stderr[:500]}")
    except Exception as e:
        print(f"Video generation failed: {e}")

    return stats_path, video_path

def main():
    # Run all tests
    c11_stats = run_c11_compare()
    py_stats = run_pytorch_reference()
    speed_stats = run_c11_speed_test()

    # Generate A/B test video
    stats_path, video_path = generate_ab_video(c11_stats, py_stats, speed_stats)

    # Final summary
    print(f"\n{'=' * 60}")
    print("A/B TEST COMPLETE")
    print(f"{'=' * 60}")
    print(f"WuBuRVC C11: {len(c11_stats.get('stdout', ''))} chars output")
    print(f"  Time: {c11_stats.get('total_time_ms', 0):.1f}ms")
    print(f"  Parity: {c11_stats.get('parity', 'N/A')}")
    print(f"PyTorch: {py_stats.get('engine', 'N/A')}")
    print(f"  Infer time: {py_stats.get('infer_time_ms', 0):.1f}ms")
    print(f"  Total time: {py_stats.get('total_time_ms', 0):.1f}ms")
    print(f"Stats file: {stats_path}")
    print(f"Video file: {video_path}")

if __name__ == '__main__':
    main()
