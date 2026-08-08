#!/usr/bin/env python3
"""Generate REAL RVC ground truth for WuBuRVC parity testing.

Pipeline (faithful to Mangio-RVC-Fork infer_batch_rvc.py):
  1. Load source audio -> 16kHz mono
  2. HuBERT content features (REAL hubert_base.pt via wubu_hubert_torch)
  3. F0 via RMVPE (real rmvpe.pt) or YIN fallback
  4. F0 -> coarse 1..255 (Mangio mel-quantization)
  5. SynthesizerTrnMs768NSFsid.infer() with the REAL checkpoint tensors
  6. Save: content.npy, f0_coarse.npy, nsff0.npy, output audio npy + wav

Usage:
  python gen_reference_real.py <model.pth> [wav_in] [out_dir]
"""
import os
import sys
import json
import math
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "tools"))
sys.path.insert(0, os.path.join(BASE, "src"))

import torch
import torch.nn.functional as F
import soundfile as sf

from wubu_hubert_torch import HuBERT

# Mangio source tree
MANGIO = r"D:\Archive\OldAI\OldAIDrive\RVC3\Mangio-RVC-v23.7.0"
sys.path.insert(0, MANGIO)
from lib.infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs768NSFsid
from lib.infer_pack.commons import sequence_mask


def load_audio_sf(path, sr=16000):
    y, orig_sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = torch.from_numpy(y.astype(np.float32))
    if orig_sr != sr:
        y = torch.from_numpy(
            np.interp(np.arange(0, len(y), orig_sr / sr), np.arange(len(y)), y.numpy()).astype(np.float32))
    return y.numpy()


def get_f0_rmvpe(pcm16k, model_path, sr=16000, hop=160, f0_min=50, f0_max=1100):
    """Real RMVPE f0 via the Mangio rmvpe module (torch, on CPU/GPU)."""
    sys.path.insert(0, MANGIO)
    from rmvpe import RMVPE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rmvpe = RMVPE(model_path, is_half=False, device=dev)
    f0 = rmvpe.infer_from_audio(pcm16k, thred=0.03)
    return f0


def get_f0_yin(pcm16k, sr=16000, hop=160, f0_min=50, f0_max=1100):
    """Fallback: call the C YIN extractor via a tiny bridge (or pure numpy)."""
    # Pure-numpy YIN (same algorithm as wubu_rvc_f0.c)
    window = 1024
    n_frames = (len(pcm16k) - window) // hop + 1
    if n_frames < 1:
        n_frames = 1
    tau_max = int(sr / f0_min)
    tau_min = int(sr / f0_max)
    if tau_max > window - 1:
        tau_max = window - 1
    if tau_min < 1:
        tau_min = 1
    f0 = np.zeros(n_frames, dtype=np.float32)
    for f in range(n_frames):
        x = pcm16k[f * hop:f * hop + window].astype(np.float64)
        diff = np.zeros(tau_max + 1)
        for tau in range(tau_max + 1):
            d = x[:window - tau] - x[tau:]
            diff[tau] = np.sum(d * d)
        cmnd = np.ones(tau_max + 1)
        running = 0.0
        for tau in range(1, tau_max + 1):
            running += diff[tau]
            cmnd[tau] = diff[tau] * tau / running if running > 0 else 1.0
        best = -1
        for tau in range(tau_min, tau_max + 1):
            if cmnd[tau] < 0.15:
                best = tau
                break
        if best < 0:
            mn = 1e30
            for tau in range(tau_min, tau_max + 1):
                if cmnd[tau] < mn:
                    mn = cmnd[tau]
                    best = tau
            if mn > 0.25:
                best = -1
        if best >= 0:
            a = cmnd[best - 1] if best > 0 else cmnd[best]
            b = cmnd[best]
            c = cmnd[best + 1] if best < tau_max else cmnd[best]
            den = a - 2 * b + c
            delta = 0.5 * (a - c) / den if den != 0 else 0.0
            delta = max(-1.0, min(1.0, delta))
            tau = best + delta
            f0[f] = sr / tau if tau > 0 else 0.0
        if f0[f] < f0_min or f0[f] > f0_max:
            f0[f] = 0.0
    return f0


def f0_to_coarse(f0, f0_min=50, f0_max=1100):
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    return np.rint(f0_mel).astype(np.int64), f0.copy()


def build_synth(pth):
    ckpt = torch.load(pth, map_location="cpu", weights_only=False)
    sd = ckpt["weight"]
    cfg = [x.item() if hasattr(x, "item") else x for x in ckpt["config"]]
    version = str(ckpt.get("version", "v2"))
    sr = cfg[17] if len(cfg) > 17 else 40000
    if version == "v1":
        net_g = SynthesizerTrnMs256NSFsid(
            cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], cfg[6], cfg[7],
            cfg[8], cfg[9], cfg[10], cfg[11], cfg[12], cfg[13], cfg[14],
            cfg[15], cfg[16], sr, is_half=False)
    else:
        net_g = SynthesizerTrnMs768NSFsid(
            cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], cfg[6], cfg[7],
            cfg[8], cfg[9], cfg[10], cfg[11], cfg[12], cfg[13], cfg[14],
            cfg[15], cfg[16], sr, is_half=False)
    net_g.load_state_dict(sd, strict=False)
    net_g.eval()
    net_g.remove_weight_norm()
    return net_g, sr, version, ckpt["config"]


def main():
    pth = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        BASE, "models", "rvc", "cartman", "EricCartmanV1_e650_s10400.pth")
    wav_in = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "outputs", "cartman_base.wav")
    out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(BASE, "outputs", "rvc_ref")
    os.makedirs(out_dir, exist_ok=True)

    # 1. audio
    pcm = load_audio_sf(wav_in, 16000)
    print(f"audio: {len(pcm)} samples @16k ({len(pcm)/16000:.2f}s)")

    # 2. HuBERT content
    hb = HuBERT(os.path.join(BASE, "models", "rvc", "hubert_base.pt"))
    content = hb.extract(pcm, version="v2")  # [T, 768]
    print(f"hubert content: {content.shape}")

    # 3. F0
    rmvpe_path = os.path.join(BASE, "models", "rvc", "rmvpe.pt")
    use_rmvpe = os.path.exists(rmvpe_path)
    if use_rmvpe:
        try:
            f0 = get_f0_rmvpe(pcm, rmvpe_path)
            print(f"f0 (RMVPE): {len(f0)} frames")
        except Exception as e:
            print(f"RMVPE failed ({e}); using YIN")
            f0 = get_f0_yin(pcm)
            use_rmvpe = False
    else:
        f0 = get_f0_yin(pcm)
    f0 = f0.astype(np.float32)

    # 4. coarse + align to content length (feats interpolated x2 in pipeline)
    p_len = len(pcm) // 160
    if len(f0) < p_len:
        f0 = np.pad(f0, (0, p_len - len(f0)))
    f0 = f0[:p_len]
    f0_coarse, f0bak = f0_to_coarse(f0)
    print(f"f0 coarse: {f0_coarse.shape} voiced={np.count_nonzero(f0bak)}")

    # content upsampled x2 (matches vc_infer_pipeline feats interpolation)
    content_t = torch.from_numpy(content).unsqueeze(0)  # [1, T, 768]
    content_up = F.interpolate(content_t.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
    n_frames = content_up.shape[1]
    if n_frames < p_len:
        p_len = n_frames
    content_up = content_up[:, :p_len, :]
    f0_coarse = torch.from_numpy(f0_coarse[:p_len]).unsqueeze(0).long()
    f0bak = torch.from_numpy(f0bak[:p_len]).unsqueeze(0).float()

    # 5. Synthesizer infer — replicate infer() manually to capture
    #    intermediates for C-side diffing (m_p, logs_p, z_p, z).
    device = "cpu"
    net_g, sr, version, cfg = build_synth(pth)
    sid = torch.tensor(0).unsqueeze(0).long()
    with torch.no_grad():
        g = net_g.emb_g(sid).unsqueeze(-1)          # [1, 256, 1]
        phone = content_up.to(device)
        pitch = f0_coarse.to(device)
        p_len = torch.tensor([phone.shape[1]], device=device)
        m_p, logs_p, x_mask = net_g.enc_p(phone, pitch, p_len)
        # Deterministic z_p (noise=0) — matches C test randn_scale=0 so the
        # pipeline math is compared honestly; random noise is a stochastic
        # overlay added at real inference time.
        z_p = m_p * x_mask
        z = net_g.flow(z_p, x_mask, g=g, reverse=True)
        nsff0 = f0bak.to(device)
        # Deterministic generator: replace SineGen.forward with a noise-free
        # version (C port is noise=0 everywhere). The default forward adds
        # random noise in unvoiced regions even at noise_std=0:
        #   noise = (1-uv)*sine_amp/3 * randn_like(sine_waves)
        # so we must strip it for the parity comparison.
        def _noisefree_sine(self, f0, upp):
            with torch.no_grad():
                f0 = f0[:, None].transpose(1, 2)
                f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim,
                                     device=f0.device)
                f0_buf[:, :, 0] = f0[:, :, 0]
                for idx in np.arange(self.harmonic_num):
                    f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)
                rad_values = (f0_buf / self.sampling_rate) % 1
                rand_ini = torch.rand(f0_buf.shape[0], f0_buf.shape[2],
                                      device=f0_buf.device)
                rand_ini[:, 0] = 0
                rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
                tmp_over_one = torch.cumsum(rad_values, 1)
                tmp_over_one *= upp
                tmp_over_one = F.interpolate(
                    tmp_over_one.transpose(2, 1), scale_factor=upp,
                    mode="linear", align_corners=True).transpose(2, 1)
                rad_values = F.interpolate(
                    rad_values.transpose(2, 1), scale_factor=upp,
                    mode="nearest").transpose(2, 1)
                tmp_over_one %= 1
                idx_wrap = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
                cumsum_shift = torch.zeros_like(rad_values)
                cumsum_shift[:, 1:, :] = idx_wrap * -1.0
                sine_waves = torch.sin(
                    torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)
                sine_waves = sine_waves * self.sine_amp
                uv = self._f02uv(f0)
                uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp,
                                   mode="nearest").transpose(2, 1)
                sine_waves = sine_waves * uv          # NO noise term
            return sine_waves, uv, None

        sine_gen = net_g.dec.m_source.l_sin_gen
        sine_gen.forward = _noisefree_sine.__get__(sine_gen, type(sine_gen))
        o = net_g.dec(z * x_mask, nsff0, g=g)
        audio = o[0, 0].cpu().float().numpy()

        # dump generator pre-conv + excitation for C-side diffing.
        # (SineGen.forward is already the noise-free patch above.)
        dec = net_g.dec
        with torch.no_grad():
            xd = dec.conv_pre(z * x_mask)
            xd = xd + dec.cond(g)
            np.save(os.path.join(out_dir, "inter_gen_pre.npy"),
                    np.ascontiguousarray(xd[0].cpu().numpy()))
            har, _, _ = dec.m_source(f0bak, dec.upp)
            np.save(os.path.join(out_dir, "inter_gen_sine.npy"),
                    np.ascontiguousarray(har[0, :, 0].cpu().numpy()))

        # dump enc_p encoder input/output for C-side diffing
        enc = net_g.enc_p
        with torch.no_grad():
            x0 = enc.emb_phone(phone) + enc.emb_pitch(pitch)
            x0 = x0 * math.sqrt(enc.hidden_channels)
            x0 = enc.lrelu(x0)
            x0 = torch.transpose(x0, 1, -1)       # [1, hidden, T]
            x0m = x0 * x_mask
            # hook each MHA to capture post-conv_o output per layer
            saved_mha = {}
            saved_attn = {}
            saved_q = {}
            saved_v = {}
            saved_pre = {}
            hooks = []
            for i, mha_mod in enumerate(enc.encoder.attn_layers):
                def mk(i):
                    def hk(mod, args, out):
                        saved_mha[i] = out.detach().cpu().numpy()
                        saved_attn[i] = mod.attn.detach().cpu().numpy()
                        # capture q: conv_q(args[0])
                        saved_q[i] = mod.conv_q(args[0]).detach().cpu().numpy()
                        saved_v[i] = mod.conv_v(args[1]).detach().cpu().numpy()
                        # recompute pre-conv_o: attention(q,k,v) then no conv_o
                        qq = mod.conv_q(args[0])
                        kk = mod.conv_k(args[1])
                        vv = mod.conv_v(args[1])
                        pre, _ = mod.attention(qq, kk, vv, mask=args[2])
                        saved_pre[i] = pre.detach().cpu().numpy()
                    return hk
                hooks.append(mha_mod.register_forward_hook(mk(i)))
            x1 = enc.encoder(x0m, x_mask)
            for hk in hooks:
                hk.remove()
            for i, v in saved_mha.items():
                np.save(os.path.join(out_dir, f"inter_enc_layer{i}_mha.npy"),
                        np.ascontiguousarray(v[0]))
            for i, v in saved_attn.items():
                np.save(os.path.join(out_dir, f"inter_enc_layer{i}_attn.npy"),
                        np.ascontiguousarray(v[0]))
            for i, v in saved_q.items():
                np.save(os.path.join(out_dir, f"inter_enc_layer{i}_q.npy"),
                        np.ascontiguousarray(v[0]))
            for i, v in saved_v.items():
                np.save(os.path.join(out_dir, f"inter_enc_layer{i}_v.npy"),
                        np.ascontiguousarray(v[0]))
            for i, v in saved_pre.items():
                np.save(os.path.join(out_dir, f"inter_enc_layer{i}_pre.npy"),
                        np.ascontiguousarray(v[0]))
            np.save(os.path.join(out_dir, "inter_enc_in.npy"),
                    np.ascontiguousarray(x0m[0].cpu().numpy()))
            np.save(os.path.join(out_dir, "inter_enc_out.npy"),
                    np.ascontiguousarray(x1[0].cpu().numpy()))

    np.save(os.path.join(out_dir, "inter_m_p.npy"), m_p[0].cpu().numpy())
    np.save(os.path.join(out_dir, "inter_logs_p.npy"), logs_p[0].cpu().numpy())
    np.save(os.path.join(out_dir, "inter_z_p.npy"), z_p[0].cpu().numpy())
    np.save(os.path.join(out_dir, "inter_z.npy"), z[0].cpu().numpy())
    np.save(os.path.join(out_dir, "inter_x_mask.npy"), x_mask[0].cpu().numpy())
    print("saved inter_*.npy intermediates")
    print(f"m_p {tuple(m_p.shape)} logs_p {tuple(logs_p.shape)} z {tuple(z.shape)}")
    print(f"synth out: {audio.shape} samples @{sr} ({len(audio)/sr:.2f}s)")

    # 6. save — ALWAYS ascontiguousarray: torch views save as F-order
    #    (fortran_order=True) which the C loader reads wrongly.
    np.save(os.path.join(out_dir, "content.npy"),
            np.ascontiguousarray(content))
    np.save(os.path.join(out_dir, "content_up.npy"),
            np.ascontiguousarray(content_up[0].numpy()))
    np.save(os.path.join(out_dir, "f0_coarse.npy"), f0_coarse[0].numpy())
    np.save(os.path.join(out_dir, "nsff0.npy"), f0bak[0].numpy())
    np.save(os.path.join(out_dir, "output_audio.npy"), audio)
    np.save(os.path.join(out_dir, "pcm16k.npy"), pcm)
    sf.write(os.path.join(out_dir, "output_audio.wav"), audio, sr)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({
            "model": pth, "wav_in": wav_in, "version": version,
            "sr": sr, "n_audio": int(len(audio)), "p_len": int(p_len),
            "f0_method": "rmvpe" if use_rmvpe else "yin",
            "config": [str(x) for x in cfg],
        }, f, indent=2)
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
