#!/usr/bin/env python3
"""wubu_album_pipeline.py — full WuBuDesk album pipeline:
1. Slice clean stems from an Ardour interchange (lead vocal + instruments)
2. RVC-convert the lead vocal with our C11 engine (quality mode)
3. Mix + master with wubu_mixmaster (our C11 mastering chain)
4. Verify output (peak <= ceiling, no NaN) + deliver

Usage:
  python wubu_album_pipeline.py <project_name> <voice_model> [out_tag]
  project_name = folder under C:/Users/eman5/Documents/ (e.g. clevelandisgolden)
  voice_model  = model dir under WuBuMedia/models/rvc/ (e.g. cleveland, peter, seth)

Stems must follow Ardour interchange naming: "N StemName%L.wav" / "%R.wav".
Lead vocal = first stem containing 'vocals' (case-insensitive) that is NOT
'backing'. Everything else is mixed as instruments.
"""
import os, sys, glob, subprocess, shutil, wave, struct
import numpy as np

BASE = r'C:\Users\eman5\WuBuMedia'
TOOLS = os.path.join(BASE, 'tools')
sys.path.insert(0, TOOLS)
from slice_stems import read_float_wav, write_pcm16

RVC_EXE = os.path.join(BASE, 'build', 'wubu_rvc_fix.exe')
MIXMASTER = os.path.join(BASE, 'build', 'wubu_mixmaster.exe')
PY = os.path.join(BASE, '.venv_win', 'Scripts', 'python.exe')

def find_stems(proj_dir):
    """Return (lead_vocal_path_L, stems_list) where stems = (name, L_path, R_path)."""
    af = os.path.join(proj_dir, 'interchange', os.path.basename(proj_dir), 'audiofiles')
    if not os.path.isdir(af):
        af = os.path.join(proj_dir, 'interchange', os.path.basename(proj_dir))
    if not os.path.isdir(af):
        raise SystemExit(f'!! no interchange/audiofiles in {proj_dir}')
    files = sorted(glob.glob(os.path.join(af, '*.wav')))
    pairs = {}
    for f in files:
        base = os.path.basename(f)
        if base.endswith('%L.wav'):
            name = base[:-len('%L.wav')]
            pairs[name] = (f, os.path.join(af, name + '%R.wav'))
        elif base.endswith('%R.wav'):
            name = base[:-len('%R.wav')]
            if name not in pairs:
                pairs[name] = (os.path.join(af, name + '%L.wav'), f)
    # order: numeric prefix
    def key(name):
        m = name.split(' ')[0]
        return (not m.isdigit(), int(m) if m.isdigit() else 999)
    names = sorted(pairs.keys(), key=key)
    # lead = ALL stems that are vocals (not backing); multi-take projects like
    # sethslament have "0 Lead Vocals", "0 Lead Vocals-2", "0 Lead Vocals-3"
    # (separate takes mixed together). Everything else is instruments.
    lead_names = [n for n in names if 'vocal' in n.lower() and 'backing' not in n.lower()]
    if not lead_names:
        raise SystemExit(f'!! no lead vocal stem in {names}')
    lead = lead_names[0]
    stems = [(n, pairs[n][0], pairs[n][1]) for n in names if n not in lead_names]
    extra_lead = [(n, pairs[n][0], pairs[n][1]) for n in lead_names[1:]]
    return lead, pairs[lead][0], pairs[lead][1], stems, af, extra_lead

def rvc_convert(in_wav, model_dir, model_pth, out_wav, jobs=4):
    cmd = [RVC_EXE, in_wav, model_dir, out_wav, '--model', model_pth,
           '--noise', '0.33333', '--jobs', str(jobs), '--autokey', '8']
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if '[6]' in line: print('   ', line.strip())
    if not os.path.exists(out_wav):
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit(f'!! RVC failed for {in_wav}')
    return out_wav

def main():
    if len(sys.argv) < 3:
        print(__doc__); return 1
    proj = sys.argv[1]
    voice = sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else f'{proj}_{voice}'
    proj_dir = rf'C:\Users\eman5\Documents\{proj}'
    model_dir = os.path.join(BASE, 'models', 'rvc', voice)
    pth = glob.glob(os.path.join(model_dir, '*.pth'))
    if not pth:
        raise SystemExit(f'!! no model in {model_dir}')
    pth = pth[0]

    out = os.path.join(BASE, 'out', 'album', 'pipeline')
    os.makedirs(out, exist_ok=True)

    lead, leadL, leadR, stems, af, extra_lead = find_stems(proj_dir)
    print(f'[1] project={proj}  lead="{lead}" (+{len(extra_lead)} extra vocal takes)  instruments={len(stems)}')

    # --- 1. slice the WHOLE lead vocal (no time crop by default) ---
    print('[2] slicing lead vocal...')
    fmt, ch, sr, bits, leadL_data = read_float_wav(leadL)
    fmt, ch, sr, bits, leadR_data = read_float_wav(leadR)
    if ch == 2 and len(leadL_data.shape) > 1: leadL_data = leadL_data.mean(axis=1)
    if ch == 2 and len(leadR_data.shape) > 1: leadR_data = leadR_data.mean(axis=1)
    lead_wav = os.path.join(out, f'{tag}_lead_in.wav')
    write_pcm16(lead_wav, leadL_data, sr)
    print(f'   lead in: {len(leadL_data)/sr:.1f}s @{sr} -> {lead_wav}')

    # --- 2. RVC convert ---
    print(f'[3] RVC {voice} on clean lead vocal (quality mode)...')
    conv = os.path.join(out, f'{tag}_lead_{voice}.wav')
    rvc_convert(lead_wav, model_dir, pth, conv)
    # --- 3. slice instrument stems at the same rate ---
    print('[4] slicing instrument stems...')
    stem_args = []
    stem_wavs = {}
    for name, lp, rp in stems:
        fmt, ch, sr2, bits, ld = read_float_wav(lp)
        fmt, ch, sr2, bits, rd = read_float_wav(rp)
        if ch == 2 and len(ld.shape) > 1: ld = ld.mean(axis=1)
        if ch == 2 and len(rd.shape) > 1: rd = rd.mean(axis=1)
        n = min(len(ld), len(rd))
        # write L/R as separate mono files for mixmaster (it does the panning)
        tag2 = name.replace(' ', '_')[:30]
        lp2 = os.path.join(out, f'{tag2}_L.wav'); rp2 = os.path.join(out, f'{tag2}_R.wav')
        write_pcm16(lp2, ld[:n], sr2); write_pcm16(rp2, rd[:n], sr2)
        stem_wavs[name] = (lp2, rp2)
        print(f'   {name}: {n/sr2:.1f}s')

    # --- 4. mix + master (ARDOUR RECIPE — boss's chain) ---
    # The reference masters come from Ardour sessions with the ACE Expander
    # makeup +8.01 dB on the lead vocal (gain 2.512), ALL other stems at
    # UNITY (1.0), stereo pairs panned L/R, master bus clean → loudness is
    # achieved at export (-18 dBFS RMS extended-LTS, -1 dBTP). Drowning the
    # singer (lead 1.0, stems 0.8) is what made earlier pipeline outputs
    # muffled — never do that again.
    print('[5] mixing + mastering (Ardour recipe: vocal +8.01 dB, stems unity, -18 dBFS)...')
    VOCAL_GAIN = 2.512  # +8.01 dB — the session's ACE Expander makeup
    mix_args = [MIXMASTER, os.path.join(out, f'{tag}_mastered.wav'), str(sr)]
    mix_args += [f'{conv}:{VOCAL_GAIN}:0']
    # NOTE: extra lead takes (-2/-3) are ALTERNATE takes, not layers — the
    # Ardour recipe uses ONE take per project ("route 'seth' = the take").
    # Do NOT mix them raw (raw boss voice into an RVC track = wrong) and do
    # NOT re-convert them (the session's master uses a single take).
    for name, (lp2, rp2) in stem_wavs.items():
        # unity gain, pan stereo pairs L/R (matches the Ardour session faders)
        mix_args += [f'{lp2}:1.0:-1', f'{rp2}:1.0:1']
    r = subprocess.run(mix_args, capture_output=True, text=True)
    print(r.stdout[-1500:] if r.stdout else '')
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit('!! mixmaster failed')
    mastered = os.path.join(out, f'{tag}_mastered.wav')
    print(f'[6] MASTERED: {mastered}')

    # --- 5. verify ---
    print('[7] verify...')
    w = wave.open(mastered, 'rb')
    nch = w.getnchannels(); sr_v = w.getframerate(); nf = w.getnframes()
    d = np.frombuffer(w.readframes(nf), dtype=np.int16); w.close()
    a = d.astype(np.float32) / 32768
    if nch == 2:
        a = a.reshape(-1, 2).mean(axis=1)
    peak = np.abs(a).max()
    rms = np.sqrt(np.mean(a**2))
    ceiling = 10 ** (-1.0 / 20)  # -1 dBTP
    ok = peak <= ceiling + 0.001
    print(f'   peak={peak:.4f} (ceiling={ceiling:.4f}) rms={rms:.4f} len={len(a)/sr_v:.1f}s  {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())
