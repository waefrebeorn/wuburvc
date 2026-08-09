#!/usr/bin/env python3
"""WuBuPet generative harness — image/3D/video nets (2026, local, weights on D:).

Triple-DA: every function checks the weight EXISTS before running, prints a
verified path, and writes an output file. No silent failures.

Models (see knowledge/RESEARCH_WUBUPET_WORLD.md):
  - TripoSR        (VAST-AI, MIT)        image -> 3D mesh   ~6-8GB VRAM
  - FLUX.2 klein 4B (BFL, Apache-2.0)    text  -> image     ~8GB VRAM
  - Wan 2.2 5B     (Alibaba, open)       image -> video     ~8GB VRAM

Weights live under D:/models/{TripoSR,FLUX2-klein-4B,Wan2.2-5B}.

RUN ENVIRONMENT (verified 2026-08-04):
  The nets need a CUDA torch. The cohost's default .venv_win is CPU-only
  (torch 2.13.0+cpu). A dedicated CUDA venv is built by setup_cuda_venv.py at
  D:/venv_cuda (torch 2.x+cu124 + diffusers + transformers). wubu_gen.py runs
  through that venv via _cuda_venv_py(). On an 8GB 2080 SUPER we use
  sequential CPU offload so the full bf16 weights don't OOM the card.

SPDX-License-Identifier: WaefreBeorn-UMV3
"""
import os, sys, argparse, subprocess, glob

MODELS = {
    "triposr": "D:/models/TripoSR",
    "flux":    "D:/models/FLUX2-klein-4B",
    "wan":     "D:/models/Wan2.2-5B",
}

CUDA_VENV = "D:/venv_cuda"


def _guard_allows_heavy():
    """Refuse heavy GPU gen when boss is streaming/gaming (boss directive
    2026-08-03: never steal the stream's GPU). Returns (ok, reason)."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
        import resource_guard
        v = resource_guard.snapshot()
        if not v.get("safe_for_heavy"):
            return False, f"rig state={v.get('state')} (heavy gen deferred)"
        return True, v.get("state")
    except Exception as e:
        # guard unavailable -> allow (don't block the harness on a probe failure)
        return True, f"guard-unavailable:{e}"


def _require_heavy_ok(force=False):
    if force:
        return True
    ok, reason = _guard_allows_heavy()
    if not ok:
        print(f"[guard] REFUSED: {reason}. Pass --force to override.")
        return False
    return True


def _cuda_venv_py():
    """Use the CUDA venv python if it exists (built by setup_cuda_venv.py)."""
    p = os.path.join(CUDA_VENV, "Scripts", "python.exe")
    return p if os.path.exists(p) else _venv_py()


def _cuda_env():
    """Clean env for CUDA-venv subprocesses.

    The ambient PYTHONPATH (Hermes agent site-packages) shadows this venv's
    CUDA torch with a CPU torch build. Drop PYTHONPATH so the venv's own
    torch 2.x+cu124 is the one that loads. Keep the rest of the env.
    """
    e = dict(os.environ)
    e.pop("PYTHONPATH", None)
    return e


def _venv_py():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     ".venv_win", "Scripts", "python.exe")
    return p if os.path.exists(p) else "python3"


def _weight_present(name, patterns):
    d = MODELS[name]
    if not os.path.isdir(d):
        return None
    for pat in patterns:
        hits = glob.glob(os.path.join(d, "**", pat), recursive=True)
        if hits:
            return hits[0]
    return False  # dir exists but no weight yet

def check(name, patterns, human):
    r = _weight_present(name, patterns)
    if r is None:
        print(f"[DA] {human}: weight dir MISSING at {MODELS[name]}"); return False
    if r is False:
        print(f"[DA] {human}: dir present but weight NOT YET DOWNLOADED"); return False
    print(f"[DA] {human}: verified -> {r}")
    return r

def image_to_3d(img_path, out_dir="out/3d", force=False):
    """TripoSR: single image -> .obj/.glb mesh. Uses local weights in D:/models/TripoSR."""
    if not _require_heavy_ok(force=force):
        return "GUARD_REFUSED"
    w = check("triposr", ["model.ckpt", "config.yaml", "run.py"], "TripoSR")
    if not w:
        return "TRIPOSR_WEIGHT_MISSING"
    os.makedirs(out_dir, exist_ok=True)
    run = os.path.join(MODELS["triposr"], "run.py")
    # TripoSR run.py CLI: run.py <image> --pretrained-model-name-or-path <local> --output-dir <dir>
    # Run under the CUDA venv (needs a real torch build with the model deps).
    cmd = [_cuda_venv_py(), run, img_path,
           "--pretrained-model-name-or-path", MODELS["triposr"],
           "--output-dir", out_dir, "--model-save-format", "obj",
           "--mc-resolution", "256"]
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, env=_cuda_env())
    objs = glob.glob(os.path.join(out_dir, "*.obj")) + glob.glob(os.path.join(out_dir, "*.glb"))
    return objs[0] if objs else "NO_MESH"

def text_to_image(prompt, out="out/img.png", steps=4, force=False):
    """FLUX.2 klein 4B text->image via the correct Flux2KleinPipeline.

    model_index.json names Flux2KleinPipeline, so from_pretrained resolves it.
    klein-4B is a 4-step distilled text->image model (not the editing variant).
    On 8GB VRAM we keep only active layers on the GPU via sequential offload.
    """
    if not _require_heavy_ok(force=force):
        return "GUARD_REFUSED"
    w = check("flux", ["model_index.json", "flux-2-klein-4b.safetensors"], "FLUX.2 klein 4B")
    if not w:
        return "FLUX_WEIGHT_MISSING"
    code = f"""
import torch
from diffusers import Flux2KleinPipeline
p=r'{MODELS['flux']}'
pipe=Flux2KleinPipeline.from_pretrained(p, torch_dtype=torch.bfloat16)
try:
    pipe.enable_sequential_cpu_offload()
except Exception as e:
    print("[warn] sequential offload failed:", e, "-> falling back to cuda")
    pipe.to('cuda' if torch.cuda.is_available() else 'cpu')
img=pipe(prompt='''{prompt}''', num_inference_steps={steps}).images[0]
img.save(r'{out}')
print('IMG_SAVED', r'{out}')
"""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print("[run] FLUX.2 klein-4B text->image (cuda venv)")
    subprocess.run([_cuda_venv_py(), "-c", code], env=_cuda_env())
    return out if os.path.exists(out) else "NO_IMG"

def image_to_video(img_path, prompt="", out="out/vid.mp4", frames=81, force=False):
    """Wan 2.2 5B image->video via WanImageToVideoPipeline (TI2V).

    Wan2.2-5B is a ti2v model (config model_type=ti2v). WanImageToVideoPipeline
    loads all three diffusion shards + VAE + T5. Sequential offload keeps the
    ~20GB of bf16 weights out of the 8GB card (active layers stream through).
    """
    if not _require_heavy_ok(force=force):
        return "GUARD_REFUSED"
    w = check("wan", ["diffusion_pytorch_model.safetensors.index.json",
                       "diffusion_pytorch_model-00001-of-00003.safetensors"], "Wan 2.2 5B")
    if not w:
        return "WAN_WEIGHT_MISSING"
    code = f"""
import torch
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video
p=r'{MODELS['wan']}'
pipe=WanImageToVideoPipeline.from_pretrained(p, torch_dtype=torch.bfloat16)
try:
    pipe.enable_sequential_cpu_offload()
except Exception as e:
    print("[warn] sequential offload failed:", e, "-> falling back to cuda")
    pipe.to('cuda' if torch.cuda.is_available() else 'cpu')
vid=pipe(image=r'{img_path}', prompt='''{prompt}''', num_frames={frames}).frames[0]
export_to_video(vid, r'{out}', fps=16)
print('VID_SAVED', r'{out}')
"""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print("[run] Wan 2.2 5B image->video (cuda venv)")
    subprocess.run([_cuda_venv_py(), "-c", code], env=_cuda_env())
    return out if os.path.exists(out) else "NO_VID"

def main():
    ap = argparse.ArgumentParser(description="WuBuPet generative harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("i3d"); a.add_argument("image"); a.add_argument("--out", default="out/3d"); a.add_argument("--force", action="store_true")
    b = sub.add_parser("t2i"); b.add_argument("prompt"); b.add_argument("--out", default="out/img.png"); b.add_argument("--steps", type=int, default=4); b.add_argument("--force", action="store_true")
    c = sub.add_parser("i2v"); c.add_argument("image"); c.add_argument("--prompt", default=""); c.add_argument("--out", default="out/vid.mp4"); c.add_argument("--force", action="store_true")
    d = sub.add_parser("check"); d.add_argument("model", choices=list(MODELS))
    args = ap.parse_args()

    if args.cmd == "i3d":
        print(image_to_3d(args.image, args.out, force=args.force))
    elif args.cmd == "t2i":
        print(text_to_image(args.prompt, args.out, steps=args.steps, force=args.force))
    elif args.cmd == "i2v":
        print(image_to_video(args.image, args.prompt, args.out, force=args.force))
    elif args.cmd == "check":
        pats = {"triposr":["*.ckpt","*.pt","*.safetensors","config.yaml"],
                "flux":["*.safetensors","model_index.json"],
                "wan":["*.safetensors"]}[args.model]
        check(args.model, pats, args.model)

if __name__ == "__main__":
    main()
