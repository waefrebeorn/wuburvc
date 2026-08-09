#!/usr/bin/env python3
"""setup_cuda_venv.py — build the CUDA inference venv used by wubu_gen.py.

The cohost's default .venv_win is CPU-only torch (good for kokoro/STT).
The generative nets (FLUX.2, Wan2.2, TripoSR) need a real CUDA torch.
This builds a dedicated venv at D:/venv_cuda with:
  - Python 3.11 (matching .venv_win so compiled ext match)
  - torch + torchvision cu124 (matches the installed CUDA 12.4 toolchain)
  - diffusers + transformers + accelerate + huggingface_hub

It uses `uv` if present (fast, deterministic), else falls back to python -m venv
+ pip. Designed to be re-runnable: it no-ops if the venv already has CUDA torch.

Triple-DA: verifies the venv python exists and torch cuda is importable at the end.

SPDX-License-Identifier: WaefreBeorn-UMV3
"""
import os
import shutil
import subprocess
import sys

CUDA_VENV = "D:/venv_cuda"
SRC_VENV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        ".venv_win", "Scripts", "python.exe")
CU124 = "https://download.pytorch.org/whl/cu124"


def _run(cmd, **kw):
    print("[setup]", " ".join(cmd))
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        print(f"[setup] FAILED ({r.returncode}): {' '.join(cmd)}")
        sys.exit(r.returncode)
    return r


def main():
    os.makedirs(os.path.dirname(CUDA_VENV) or ".", exist_ok=True)
    vpy = os.path.join(CUDA_VENV, "Scripts", "python.exe")

    # 1) create venv (uv preferred for speed/repro)
    if not os.path.exists(vpy):
        uv = shutil.which("uv")
        if uv:
            _run([uv, "venv", "--python", "3.11", CUDA_VENV])
        else:
            base = SRC_VENV if os.path.exists(SRC_VENV) else sys.executable
            _run([base, "-m", "venv", CUDA_VENV])
    else:
        print("[setup] venv exists, reusing", CUDA_VENV)

    # 2) install CUDA torch + friends
    uv = shutil.which("uv")
    if uv:
        _run([uv, "pip", "install", "--python", vpy,
              "torch", "torchvision", "--index-url", CU124])
        _run([uv, "pip", "install", "--python", vpy,
              "diffusers", "transformers", "accelerate", "huggingface_hub"])
    else:
        _run([vpy, "-m", "pip", "install", "--upgrade", "pip"])
        _run([vpy, "-m", "pip", "install", "torch", "torchvision",
              "--index-url", CU124])
        _run([vpy, "-m", "pip", "install", "diffusers", "transformers",
              "accelerate", "huggingface_hub"])

    # 3) Triple-DA verify
    clean_env = dict(os.environ); clean_env.pop("PYTHONPATH", None)
    out = subprocess.run([vpy, "-c",
        "import torch,diffusers; "
        "print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available()); "
        "print('diffusers', diffusers.__version__)"],
        capture_output=True, text=True, env=clean_env)
    print(out.stdout.strip())
    if out.returncode != 0:
        print("[setup] VERIFY FAILED:", out.stderr.strip())
        sys.exit(1)
    print("[setup] CUDA venv ready at", CUDA_VENV)


if __name__ == "__main__":
    main()
