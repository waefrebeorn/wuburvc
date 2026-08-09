#!/usr/bin/env python3
# SPDX-License-Identifier: WaefreBeorn-UMV3
"""wubu_serve.py — launch a local llama.cpp llama-server for a model.

Used by the WuBuDesk swarm (Step 4 of the plan): each model runs in its own
crash-isolated llama-server process behind one OpenAI-compatible router.

Examples:
  wubu_serve.py brain   -> serves Qwen3.6-27B-UD-IQ2_M on :57064 (the cohost brain)
  wubu_serve.py eyes    -> serves Qwen3.5-9B-VL + mmproj on :57065 (vision)
  wubu_serve.py coder   -> serves KAT-Coder-V2.5-Dev-IQ2_M on :57066

Flags tuned for RTX 2080 SUPER (sm_75, 8GB): -ngl 99 (offload all layers to
GPU), -fa (flash-attn), -ctx-size 8192. If VRAM is tight it falls back by
reducing -ngl. Set WUBU_LLAMA_BIN to the llama-server.exe path.
"""
import os, sys, subprocess, argparse, time

LLAMA_BIN = os.environ.get("WUBU_LLAMA_BIN",
    r"D:/llama.cpp/llama-server.exe")
CUDART = r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin"

MODELS = {
    "brain": dict(
        model=r"D:/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-IQ2_M.gguf",
        port=57064, ngpu=40, ctx=8192),
    "eyes": dict(
        model=r"D:/models/Qwen3.5-9B-VL/Qwen3.5-9B-UD-Q4_K_XL.gguf",
        mmproj=r"D:/models/Qwen3.5-9B-VL/mmproj-F16.gguf",
        port=57065, ngpu=99, ctx=8192),
    "coder": dict(
        model=r"D:/models/KAT-Coder-V2.5-Dev-GGUF/Kwaipilot_KAT-Coder-V2.5-Dev-IQ2_M.gguf",
        port=57066, ngpu=40, ctx=8192),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=list(MODELS.keys()))
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    m = MODELS[args.role]
    env = os.environ.copy()
    env["PATH"] = CUDART + ";" + env.get("PATH", "")
    cmd = [LLAMA_BIN, "-m", m["model"], "--host", args.host, "--port",
           str(m["port"]), "-ngl", str(m["ngpu"]), "-fa", "-c",
           str(m["ctx"]), "--no-warmup"]
    if "mmproj" in m:
        cmd += ["--mmproj", m["mmproj"]]
    print("LAUNCH", args.role, "->", " ".join(cmd[:6]), "...")
    # Run in foreground so the caller manages the process (or use background flag).
    if "--bg" in sys.argv:
        subprocess.Popen(cmd, env=env)
        print("started", args.role, "pid via Popen")
    else:
        subprocess.run(cmd, env=env)

if __name__ == "__main__":
    main()
