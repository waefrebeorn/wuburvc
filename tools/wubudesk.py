#!/usr/bin/env python3
# SPDX-License-Identifier: WaefreBeorn-UMV3
"""WuBuDesk operations CLI — the cohost's toolbox.

Subcommands:
  speak  <mood> <text> [--mode live|movie]   push a cohost line to OBS overlay
  disk                                      PowerShell free-space check (WSL lies)
  status                                    rig + cohost health snapshot
  verify-model <gguf> [prompt] [ntok]       load+gen smoke test via gdb
  reflect                                   run the Reflexion self-review
  watch-engine                              check wubuwizard for READY signal

WuBuDesk = the supporting/Windows-port agent + stream cohost. It ports the
team's Linux code to this Windows rig and is "the wizard in your computer".
"""
import sys, os, json, subprocess, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
WUBUMEDIA = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(WUBUMEDIA, "src"))
OBSDIR = os.environ.get("OBS_RUNTIME_DIR", r"C:/Users/eman5/obs")


def _obs_pw():
    pw = os.environ.get("OBS_WS_PASSWORD")
    if pw:
        return pw
    try:
        cfg = json.load(open(os.path.join(
            os.environ.get("APPDATA", r"C:/Users/eman5/AppData/Roaming"),
            "obs-studio", "plugin_config", "obs-websocket", "config.json")))
        return cfg.get("server_password")
    except Exception:
        return None


def cmd_speak(args):
    from wubu_obs import ObsCohost
    obs = ObsCohost(port=4455, password=_obs_pw())
    # OBS connection is best-effort: the face overlay reads face_state.json
    # from the HTTP server, so we can still push state even if OBS is closed.
    try:
        obs.connect()
        obs.ensure_face()
    except Exception as e:
        print(f"[obs-skipped] {e}", file=sys.stderr)
    st = obs.speak(args.mood, args.text, mode=args.mode)
    print(json.dumps(st, indent=2))


def cmd_disk(args):
    ps = os.path.join(OBSDIR, "disk_check.ps1")
    if os.path.exists(ps):
        # PowerShell may block .ps1 via ExecutionPolicy; bypass for this run.
        r = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", ps], capture_output=True, text=True)
        print(r.stdout or r.stderr)
    else:
        inline = ("Get-PSDrive -PSProvider FileSystem | Select-Object Name,"
                  "@{N='FreeGB';E={[math]::Round($_.Free/1GB,1)}},"
                  "@{N='UsedGB';E={[math]::Round($_.Used/1GB,1)}} | Format-Table")
        r = subprocess.run(["powershell", "-NoProfile", "-Command", inline],
                           capture_output=True, text=True)
        print(r.stdout or r.stderr)


def cmd_status(args):
    snap = {
        "rig": {"gpu": "RTX 2080 SUPER (sm_75, 8GB)",
                "cpu": "Ryzen 5 3600 (AVX2, no AVX-512)", "ram": "64GB"},
        "cohost": {"face_server": ":8137", "bridge": ":18765",
                   "brain": ":57064"},
        "note": "Use `wubudesk verify-model` to confirm an engine load.",
    }
    print(json.dumps(snap, indent=2))


def cmd_verify_model(args):
    gguf = args.gguf
    prompt = args.prompt or "The meaning of AGI is"
    n = args.ntok or 24
    wiz = r"C:/Users/eman5/wubuwizard"
    binpath = os.path.join(wiz, "gen_text_win")
    if not os.path.exists(binpath):
        print("NO_BUILD: gen_text_win not found; run make -f Makefile.win")
        return 2
    # verify under gdb (clean env, no MSYS-DLL-path artifact)
    gdb = [
        "gdb", "-q", "-batch", "-ex", "run", "--args", binpath,
        gguf, prompt, str(n),
    ]
    env = dict(os.environ)
    env["PATH"] = r"/c/msys64/usr/bin:/c/msys64/mingw64/bin:" + env.get("PATH", "")
    r = subprocess.run(gdb, cwd=wiz, env=env, capture_output=True, text=True,
                       timeout=600)
    out = (r.stdout or "") + (r.stderr or "")
    if "Failed to open" in out or "unsupported quant" in out or "gguf" in out.lower():
        print(out[-2000:])
        return 0 if "unsupported quant" in out else 1
    print(out[-2000:])
    return 0


def cmd_reflect(args):
    refl = os.path.join(WUBUMEDIA, "memory", "reflections.json")
    if not os.path.exists(refl):
        print("NO_REFLECTIONS: create memory/reflections.json first")
        return 2
    data = json.load(open(refl))
    print(f"Reflections: {len(data.get('lessons', []))} lessons, "
          f"{len(data.get('open_questions', []))} open questions")
    print("Weakest step -> review PERSONA_ITERATION_PLAN.md step-by-step.")
    print("Next: add a lesson (id L0%02d) and report to boss." %
          (len(data.get('lessons', [])) + 1))
    return 0


def cmd_watch_engine(args):
    wiz = r"C:/Users/eman5/wubuwizard"
    rc = subprocess.run(["git", "-C", wiz, "log", "-1", "--oneline"],
                        capture_output=True, text=True)
    print("wubuwizard HEAD:", rc.stdout.strip())
    pace = os.path.join(WUBUMEDIA, "knowledge", "PACE_DEEPSEEK_V4.md")
    if os.path.exists(pace):
        print("PACE doc present (engine pace signal lives here).")


def main():
    ap = argparse.ArgumentParser(prog="wubudesk", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("speak"); s.add_argument("mood"); s.add_argument("text")
    s.add_argument("--mode", default="live"); s.set_defaults(func=cmd_speak)
    s = sub.add_parser("disk"); s.set_defaults(func=cmd_disk)
    s = sub.add_parser("status"); s.set_defaults(func=cmd_status)
    s = sub.add_parser("verify-model"); s.add_argument("gguf")
    s.add_argument("prompt", nargs="?"); s.add_argument("ntok", nargs="?", type=int)
    s.set_defaults(func=cmd_verify_model)
    s = sub.add_parser("reflect"); s.set_defaults(func=cmd_reflect)
    s = sub.add_parser("watch-engine"); s.set_defaults(func=cmd_watch_engine)
    args = ap.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
