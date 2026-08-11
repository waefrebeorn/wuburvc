#!/usr/bin/env python3
"""master.py — master audio to extended-LTS -18 dBFS RMS (-1 dBTP ceiling).

Usage:
  python tools/master.py <input.wav> <output.wav> <rms_target> <dbfs_ceiling>
  Example: python tools/master.py track1_mix.wav track1_mastered.wav -18 -1
"""
import subprocess
import sys
import os


def normalize_to_rms(input_path, output_path, target_rms_db, ceiling_db):
    """Loudnorm to target integrated loudness (LUFS, negative dB) with true-peak ceiling."""
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-af', f'loudnorm=I={target_rms_db}:TP={ceiling_db}:LRA=11',
        '-c:a', 'pcm_s16le', output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Mastered: {output_path} to {target_rms_db}dB RMS")
        return True
    print(f"Mastering failed for {os.path.basename(input_path)}")
    print(result.stderr[-500:])
    return False


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: master.py <input.wav> <output.wav> <rms_target> <dbfs_ceiling>")
        sys.exit(1)
    success = normalize_to_rms(sys.argv[1], sys.argv[2],
                               float(sys.argv[3]), float(sys.argv[4]))
    sys.exit(0 if success else 1)
