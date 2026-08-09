#!/usr/bin/env python3
"""Torch reference GeneratorNSF — ground truth for the Triton backend A/B."""
import numpy as np
import torch
import torch.nn.functional as F


def generator_nsf_torch(sd, z, nsff0, g, sr=32000.0, use_snake=False, inject_noise=False):
    """sd = state dict (float32 numpy), z (inter, T), nsff0 (T,), g (256,).
    Returns float32 numpy audio. Mirrors wubu_rvc_real.c + triton_rvc_gen."""
    dev = 'cpu'
    nF = z.shape[1]
    inter = z.shape[0]

    def T(name):
        return torch.as_tensor(sd[name], device=dev, dtype=torch.float32)

    def denorm(name):
        """weight_norm de-normalization: weight = weight_g * weight_v/||weight_v||
        per output channel (matches the C11 engine's load-time de-norm)."""
        wv = T(name)
        gname = name.replace('.weight', '.weight_g') if name.endswith('.weight') else name + '.weight_g'
        wg = T(gname) if gname in sd else None
        if wg is None:
            gname = name.replace('.weight_v', '.weight_g')
            wg = T(gname) if gname in sd else None
        if wg is not None:
            # wv shape (out, in, k) or (in, out, k) — norm over the last 2 dims
            dims = list(range(1, wv.dim()))
            norm = wv.norm(2, dim=dims, keepdim=True).clamp_min(1e-8)
            return wg * (wv / norm)
        return wv

    conv_pre_w = T('dec.conv_pre.weight')
    conv_pre_b = T('dec.conv_pre.bias')
    init_ch = conv_pre_w.shape[0]
    x = F.conv1d(torch.as_tensor(z, dtype=torch.float32).unsqueeze(0),
                 conv_pre_w, conv_pre_b, stride=1, padding=conv_pre_w.shape[2] // 2)[0]

    cond_w = sd.get('dec.cond.weight')
    if cond_w is not None:
        gin = cond_w.shape[1]
        cw = T('dec.cond.weight').reshape(init_ch, -1)
        cb = T('dec.cond.bias')
        off = cb.clone()
        gg = torch.as_tensor(g, dtype=torch.float32)
        for oc in range(init_ch):
            off[oc] = cb[oc] + (gg[:min(gin, 256)] * cw[oc, :min(gin, 256)]).sum()
        x = x + off[:, None]

    rates = [10, 8, 2, 2]
    n_ups = len(rates)
    ups_total = int(np.prod(rates))

    linw = float(T('dec.m_source.l_linear.weight').reshape(-1)[0]) if 'dec.m_source.l_linear.weight' in sd else 1.0
    linb = float(T('dec.m_source.l_linear.bias').reshape(-1)[0]) if 'dec.m_source.l_linear.bias' in sd else 0.0
    n_sine = nF * ups_total
    sine = np.zeros(n_sine, np.float32)
    rad = np.zeros(nF, np.float32)
    rad_acc = np.zeros(nF, np.float32)
    accum = 0.0
    for t in range(nF):
        f0 = nsff0[t] if nsff0[t] > 0 else 0.0
        rad[t] = f0 / sr
        c = rad[t] * ups_total
        r2 = np.fmod(c + 0.5, 1.0) - 0.5
        accum += r2
        rad_acc[t] = np.fmod(accum, 1.0)
    for j in range(n_sine):
        fi = j // ups_total
        if fi >= nF:
            fi = nF - 1
        u = j % ups_total
        phase = rad[fi] * (u + 1)
        if fi > 0:
            phase += rad_acc[fi - 1]
        s = np.sin(2.0 * np.pi * phase) * 0.1
        uv = 1.0 if nsff0[fi] > 0 else 0.0
        noise_amp = (1.0 - uv) * 0.1 / 3.0 if inject_noise else 0.0
        noise = noise_amp * (2.0 * np.random.random() - 1.0)
        sine[j] = np.tanh(linw * (s * uv + noise) + linb)
    sine_t = torch.as_tensor(sine, dtype=torch.float32).unsqueeze(0)

    def lrelu(t):
        return F.leaky_relu(t, 0.1)

    def snake(t):
        a = 0.2
        return t + (1.0 / a) * torch.sin(a * t) ** 2

    act = snake if use_snake else lrelu

    cur = x
    cur_ch, cur_n = init_ch, nF
    for L in range(n_ups):
        in_ch, in_n = cur_ch, cur_n
        out_ch = init_ch // (1 << (L + 1))
        st = rates[L]
        ut = sd[f'dec.ups.{L}.weight_v'] if f'dec.ups.{L}.weight_v' in sd else sd[f'dec.ups.{L}.weight']
        uw = denorm(f'dec.ups.{L}.weight_v') if f'dec.ups.{L}.weight_v' in sd else torch.as_tensor(ut, dtype=torch.float32)
        ub = T(f'dec.ups.{L}.bias')
        k = uw.shape[2]
        pad = (k - st) // 2
        out_n = (in_n - 1) * st - 2 * pad + k

        cur = act(cur)
        stage = F.conv_transpose1d(cur.unsqueeze(0), uw, ub, stride=st, padding=pad)[0]

        stride = 1
        for j in range(L + 1, n_ups):
            stride *= rates[j]
        nck = f'dec.noise_convs.{L}.weight'
        if nck in sd:
            ncw = torch.as_tensor(sd[nck], dtype=torch.float32)
            ncb = T(f'dec.noise_convs.{L}.bias')
            kk = ncw.shape[2] if ncw.ndim >= 3 else (1 if L == n_ups - 1 else stride * 2)
            npad = (kk - stride) // 2
            if npad < 0:
                npad = 0
            nc = F.conv1d(sine_t, ncw, ncb, stride=stride, padding=npad)
            stage = stage + nc

        ch = out_ch
        n_stacks = 3
        n_pairs = 3
        stage_pres = stage.clone()
        acc = torch.zeros((n_stacks * ch * out_n,), dtype=torch.float32)
        for s in range(n_stacks):
            rb = L * n_stacks + s
            kk = [3, 7, 11][s]
            xin = stage_pres.clone()
            for cp in range(n_pairs):
                r1k = f'dec.resblocks.{rb}.convs1.{cp}.weight_v'
                r2k = f'dec.resblocks.{rb}.convs2.{cp}.weight_v'
                r1v = denorm(r1k) if r1k in sd else torch.as_tensor(sd[f'dec.resblocks.{rb}.convs1.{cp}.weight'], dtype=torch.float32)
                r2v = denorm(r2k) if r2k in sd else torch.as_tensor(sd[f'dec.resblocks.{rb}.convs2.{cp}.weight'], dtype=torch.float32)
                r1b = T(f'dec.resblocks.{rb}.convs1.{cp}.bias')
                r2b = T(f'dec.resblocks.{rb}.convs2.{cp}.bias')
                dil = 1 + 2 * cp
                pad1 = dil * (kk - 1) // 2
                pad2 = kk // 2
                xt = F.conv1d(act(xin).unsqueeze(0), r1v, r1b, stride=1, padding=pad1, dilation=dil)[0]
                xt = F.conv1d(act(xt).unsqueeze(0), r2v, r2b, stride=1, padding=pad2)[0]
                xin = xt + xin
            acc[s * ch * out_n:(s + 1) * ch * out_n] = xin.reshape(-1)
        # stage = mean over stacks, keeping the (ch, out_n) layout
        stage = acc.reshape(n_stacks, ch, out_n).mean(0)
        cur = stage
        cur_ch, cur_n = ch, out_n

    post_w = torch.as_tensor(sd['dec.conv_post.weight'], dtype=torch.float32)
    post_in = post_w.shape[1] if post_w.ndim >= 2 else 32
    post_k = post_w.shape[2] if post_w.ndim >= 3 else 7
    post_pad = post_k // 2
    cur = act(cur)
    out = F.conv1d(cur.unsqueeze(0), post_w, None, stride=1, padding=post_pad)[0]
    out = torch.tanh(out)
    return out.reshape(-1).numpy(), cur_n


def main():
    import os
    import sys
    model = sys.argv[1]
    ckpt = torch.load(model, map_location='cpu', weights_only=False)
    sd = {}
    for k, v in ckpt['weight'].items():
        if isinstance(v, torch.Tensor):
            sd[k] = v.detach().float().numpy()
    z = np.fromfile('outputs/rvc_ref/c_gen_input.npy', dtype='<f4').reshape(192, -1)
    nsff0 = np.fromfile('outputs/rvc_ref/c_gen_nsff0.npy', dtype='<f4')
    g = np.fromfile('outputs/rvc_ref/c_gen_g.npy', dtype='<f4')
    out, out_n = generator_nsf_torch(sd, z, nsff0, g)
    np.save('outputs/rvc_ref/torch_gen_output.npy', out)
    print(f'[torch] wrote torch_gen_output.npy ({out_n} samples)')
    ref = np.fromfile('outputs/rvc_ref/c_gen_output.npy', dtype='<f4')
    m = min(out.size, ref.size)
    print(f'[torch] vs C11: corr={np.corrcoef(out[:m], ref[:m])[0,1]:.8f} maxdiff={np.max(np.abs(out[:m]-ref[:m])):.6f}')
    print(f'[torch] rms torch/cpu: {np.sqrt(np.mean(out**2)):.6f} / {np.sqrt(np.mean(ref**2)):.6f}')


if __name__ == '__main__':
    main()
