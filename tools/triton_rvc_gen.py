#!/usr/bin/env python3
"""Triton RVC generator — the first RVC-family Triton generator backend.

Implements the FULL RVC GeneratorNSF (conv_pre, NSF sine, upsample blocks
with noise convs + MRF resblocks, conv_post) as Triton kernels, running the
SAME math as the C11 CPU/CUDA/Vulkan backends in wuburvc.

Loads a real RVC v2 .pth (torch) for weights, takes the EXACT generator
inputs the C11 engine produced (dumped via WUBU_RVC_DUMP=1:
  outputs/rvc_ref/c_gen_input.npy   = z     [inter, T]
  outputs/rvc_ref/c_gen_nsff0.npy   = nsff0 [T]
  outputs/rvc_ref/c_gen_g.npy       = g     [256]
  outputs/rvc_ref/c_gen_output.npy  = CPU reference output (for A/B)
), runs the generator, and reports parity.

Usage:
  python tools/triton_rvc_gen.py MODEL.pth [--dump-dir outputs/rvc_ref] [--out OUT.wav]

Requires: triton (Windows build: pip install triton --index-url
https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-on-Windows/pypi/simple/),
torch with CUDA.

Design notes (verified math matches wubu_rvc_real.c):
  * conv1d: out[oc,j] = bias[oc] + sum_ic sum_tap in[ic, j*stride + tap*dil - pad] * w[oc,ic,tap]
  * convT1d: out[oc,j] = bias[oc] + sum_ic sum_i in[ic,i] * w[ic,oc, j + pad - i*stride]
  * MRF per stack s, pair cp: x = conv2(lrelu(conv1(lrelu(x)))) + x, carry per pair.
  * acc[s] += rb_in; stage = avg over stacks; conv_post -> tanh.
  * Sine stage matches CPU: rad = f0/sr, rad_acc cumulative fmod carry,
    phase = rad[fi]*(u+1) + rad_acc[fi-1], s = sin(2pi*phase)*0.1,
    uv = voiced mask, noise amp (1-uv)*0.1/3 when inject, sine = tanh(linw*(s*uv+noise)+linb).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import triton
import triton.language as tl


# ─────────────────────────── Triton kernels ───────────────────────────

@triton.jit
def k_conv1d(in_ptr, w_ptr, b_ptr, out_ptr,
             in_ch, n, out_ch, k, stride, pad, dil, n_out,
             BLOCK_J: tl.constexpr):
    """1D conv, 1 thread per (oc, j-tile). in is [in_ch, n], w [out_ch, in_ch, k]."""
    pid = tl.program_id(0)          # over oc * j-blocks
    BLOCK: tl.constexpr = BLOCK_J
    jb: tl.constexpr = tl.cdiv(n_out, BLOCK)
    j0 = (pid % jb) * BLOCK
    oc = pid // jb
    offs = j0 + tl.arange(0, BLOCK)
    m = offs < n_out
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    if b_ptr is not None:
        acc += tl.load(b_ptr + oc)
    for ic in range(0, in_ch):
        for tap in range(0, k):
            src = offs * stride + tap * dil - pad
            sm = m & (src >= 0) & (src < n)
            xv = tl.load(in_ptr + ic * n + src, mask=sm, other=0.0)
            wv = tl.load(w_ptr + oc * in_ch * k + ic * k + tap)
            acc += xv * wv
    tl.store(out_ptr + oc * n_out + offs, acc, mask=m)


@triton.jit
def k_convt1d(in_ptr, w_ptr, b_ptr, out_ptr,
              in_ch, n, out_ch, k, stride, pad, n_out,
              BLOCK_J: tl.constexpr):
    """ConvTranspose1d gather over taps (O(k) per output, not O(n)).
    w is [in_ch, out_ch, k] (PyTorch layout).
    out[oc,j] = bias[oc] + sum_ic sum_tap in[ic, (j+pad-tap)/stride] * w[ic,oc,tap]
    (only when (j+pad-tap) % stride == 0 and the source index is in range)."""
    pid = tl.program_id(0)
    BLOCK: tl.constexpr = BLOCK_J
    jb: tl.constexpr = tl.cdiv(n_out, BLOCK)
    j0 = (pid % jb) * BLOCK
    oc = pid // jb
    offs = j0 + tl.arange(0, BLOCK)
    m = offs < n_out
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    if b_ptr is not None:
        acc += tl.load(b_ptr + oc)
    for ic in range(0, in_ch):
        for tap in range(0, k):
            src = offs + pad - tap
            div_ok = (src % stride) == 0
            i = src // stride
            valid = div_ok & (i >= 0) & (i < n) & m
            xv = tl.load(in_ptr + ic * n + i, mask=valid, other=0.0)
            wv = tl.load(w_ptr + ic * out_ch * k + oc * k + tap)
            acc += xv * wv
    tl.store(out_ptr + oc * n_out + offs, acc, mask=m)


@triton.jit
def k_lrelu(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(x_ptr + offs, tl.where(x >= 0, x, x * 0.1), mask=m)


@triton.jit
def k_snake(x_ptr, n, BLOCK: tl.constexpr):
    """BigVGAN Snake: x + (1/a) sin^2(a x), a=0.2 (default in RVC)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    a = 0.2
    tl.store(x_ptr + offs, x + (1.0 / a) * tl.sin(a * x) * tl.sin(a * x), mask=m)


@triton.jit
def k_tanh(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(x_ptr + offs, 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0, mask=m)


@triton.jit
def k_add(a_ptr, b_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(a_ptr + offs, mask=m)
    b = tl.load(b_ptr + offs, mask=m)
    tl.store(a_ptr + offs, a + b, mask=m)


@triton.jit
def k_copy(dst_ptr, src_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=m), mask=m)


@triton.jit
def k_mul(x_ptr, n, s, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(x_ptr + offs, x * s, mask=m)


@triton.jit
def k_bias(x_ptr, off_ptr, total, frames, BLOCK: tl.constexpr):
    """x[oc, j] += off[oc] — per-channel bias add. Layout is channel-major
    [oc * frames + j], so oc = flat // frames."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    oc = offs // frames
    off = tl.load(off_ptr + oc, mask=m)
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(x_ptr + offs, x + off, mask=m)


def launch(fn, n, grid_extra=1, BLOCK=256):
    grid = (triton.cdiv(n, BLOCK) * grid_extra,)
    return grid


# ─────────────────────────── host helpers ───────────────────────────

def load_pth(model_path):
    """Load an RVC v2 .pth and return {name: float32 tensor}."""
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('weight', ckpt)
    out = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().float().numpy()
    return out, ckpt


def extract_config(ckpt):
    """Read upsample_rates + sample_rate from the .pth config.

    RVC v2 config is a LIST: [0]=version-ish, [1]=? [5]=mel_channels,
    [10]=resblock kernels, [12]=upsample_rates, [16]=hidden, [17]=sr.
    (Same mapping as tools/extract_rvc_weights.py.)"""
    rates = None
    sr = 32000.0
    cfg = ckpt.get('config')
    if isinstance(cfg, str):
        try:
            cfg = eval(cfg) if cfg.startswith('[') else []
        except Exception:
            cfg = []
    if isinstance(cfg, (list, tuple)):
        cfg = list(cfg)
        if len(cfg) >= 13 and isinstance(cfg[12], (list, tuple)):
            rates = [int(x) for x in cfg[12]]
        if len(cfg) >= 18:
            sr = float(cfg[17])
    elif isinstance(cfg, dict):
        ur = cfg.get('upsample_rates') or cfg.get('interp_rates') or cfg.get('upsample_rate')
        if ur is not None:
            rates = [int(x) for x in ur]
        if cfg.get('sampling_rate'):
            sr = float(cfg['sampling_rate'])
        elif cfg.get('sr'):
            sr = float(cfg['sr'])
    return rates, sr


def denorm_np(sd, name):
    """weight_norm de-normalization: W = weight_g * weight_v / (||weight_v|| + 1e-8)
    per channel (matches wubu_rvc_denormalize_weight). Returns float32 numpy."""
    wv = sd[name]
    gname = name.replace('.weight', '.weight_g') if name.endswith('.weight') else name + '.weight_g'
    if gname not in sd:
        gname = name.replace('.weight_v', '.weight_g')
    wg = sd.get(gname)
    if wg is None:
        return np.asarray(wv, np.float32)
    wv = np.asarray(wv, np.float32).reshape(-1)
    wg = np.asarray(wg, np.float32).reshape(-1)
    n_ch = wg.size
    per_ch = wv.size // n_ch
    out = np.empty_like(wv)
    for ch in range(n_ch):
        v = wv[ch * per_ch:(ch + 1) * per_ch]
        norm = np.sqrt(np.sum(v * v)) + 1e-8
        out[ch * per_ch:(ch + 1) * per_ch] = v * (wg[ch] / norm)
    return out.reshape(np.asarray(sd[name]).shape)


def act_gpu(x, use_snake):
    n = x.numel()
    if use_snake:
        k_snake[launch(k_snake, n)](x, n, BLOCK=256)
    else:
        k_lrelu[launch(k_lrelu, n)](x, n, BLOCK=256)


def conv1d_gpu(inp, w, b, out_ch, k, stride=1, pad=0, dil=1):
    """inp: (in_ch, n) torch tensor. Returns (out_ch, n_out)."""
    in_ch, n = inp.shape
    n_out = (n + 2 * pad - dil * (k - 1) - 1) // stride + 1
    out = torch.empty((out_ch, n_out), device=inp.device, dtype=torch.float32)
    wd = torch.as_tensor(w, device=inp.device, dtype=torch.float32).contiguous()
    bd = torch.as_tensor(b, device=inp.device, dtype=torch.float32).contiguous() if b is not None else None
    grid = (triton.cdiv(n_out, 64) * out_ch,)
    k_conv1d[grid](inp, wd, bd, out,
                   in_ch, n, out_ch, k, stride, pad, dil, n_out, BLOCK_J=64)
    return out


def convt1d_gpu(inp, w, b, out_ch, k, stride, pad):
    """inp: (in_ch, n). w: (in_ch, out_ch, k) PyTorch layout. Returns (out_ch, n_out)."""
    in_ch, n = inp.shape
    n_out = (n - 1) * stride - 2 * pad + k
    out = torch.empty((out_ch, n_out), device=inp.device, dtype=torch.float32)
    wd = torch.as_tensor(w, device=inp.device, dtype=torch.float32).contiguous()
    bd = torch.as_tensor(b, device=inp.device, dtype=torch.float32).contiguous() if b is not None else None
    grid = (triton.cdiv(n_out, 32) * out_ch,)
    k_convt1d[grid](inp, wd, bd, out,
                    in_ch, n, out_ch, k, stride, pad, n_out, BLOCK_J=32)
    return out


def add_gpu(a, b):
    n = a.numel()
    k_add[launch(k_add, n)](a, b, n, BLOCK=256)


def copy_gpu(dst, src):
    n = dst.numel()
    k_copy[launch(k_copy, n)](dst, src, n, BLOCK=256)


def mul_gpu(x, s):
    n = x.numel()
    k_mul[launch(k_mul, n)](x, n, s, BLOCK=256)


def bias_gpu(x, off, frames):
    total = x.numel()
    k_bias[launch(k_bias, total)](x, off, total, frames, BLOCK=256)


def tanh_gpu(x):
    n = x.numel()
    k_tanh[launch(k_tanh, n)](x, n, BLOCK=256)


# ─────────────────────────── the generator ───────────────────────────

def generator_nsf_triton(weights, z, nsff0, g, use_snake=False, inject_noise=False):
    """Mirror of wubu_generator_nsf (CPU) + wubu_generator_nsf_cuda.

    weights: {tensor-name: float32 numpy}. z: (inter, T) numpy.
    nsff0: (T,) Hz. g: (256,) speaker embedding.
    Returns (out_audio numpy float32, out_n).
    """
    dev = 'cuda'
    nF = z.shape[1]
    inter = z.shape[0]

    conv_pre_w = weights['dec.conv_pre.weight']
    conv_pre_b = weights.get('dec.conv_pre.bias', np.zeros(conv_pre_w.shape[0], np.float32))
    cond_w = weights.get('dec.cond.weight')
    cond_b = weights.get('dec.cond.bias', np.zeros(conv_pre_w.shape[0], np.float32))
    post_w = weights['dec.conv_post.weight']
    n_ups = 4
    # upsample rates come from the model config (dict inside the .pth), NOT
    # inferred from kernel sizes (Cleveland 32k = [10,8,2,2], Cartman 40k =
    # [10,10,2,2], Miku 48k = [12,10,2,2]). The caller passes them via the
    # global CONFIG when available; fall back to kernel-size heuristics.
    rates = list(getattr(sys.modules[__name__], 'UPSAMPLE_RATES', None) or [10, 10, 2, 2])
    init_ch = conv_pre_w.shape[0]

    # upsample params from weight shapes (kernel ALWAYS from dims[2]; stride
    # from the config/global above)
    ups_k, ups_pad = [], []
    ups_total = 1
    for L in range(n_ups):
        key = f'dec.ups.{L}.weight_v'
        if key not in weights:
            key = f'dec.ups.{L}.weight'
        ut = weights[key]
        k = ut.shape[2]
        st = rates[L] if L < len(rates) else 2
        ups_k.append(k)
        ups_pad.append((k - st) // 2)
        ups_total *= st

    # conv_pre: z -> x [init_ch, T]
    z_t = torch.as_tensor(z, device=dev, dtype=torch.float32).contiguous()
    x = conv1d_gpu(z_t, conv_pre_w, conv_pre_b, init_ch, conv_pre_w.shape[2],
                   stride=1, pad=conv_pre_w.shape[2] // 2, dil=1)
    if os.environ.get('TRITON_DUMP'):
        np.save(os.path.join(os.environ['TRITON_DUMP'], 't_pre.npy'),
                x.cpu().numpy())
    # cond(g): per-channel offset
    if cond_w is not None:
        gin = cond_w.shape[1]
        cw1 = np.asarray(cond_w).reshape(cond_w.shape[0], -1)
        cb1 = np.asarray(cond_b).reshape(-1) if cond_b is not None else np.zeros(init_ch, np.float32)
        off = cb1.copy()
        for oc in range(init_ch):
            acc = float(cb1[oc]) if oc < len(cb1) else 0.0
            for ic in range(min(gin, 256)):
                acc += float(g[ic]) * float(cw1[oc, ic])
            off[oc] = acc
        off_t = torch.as_tensor(off[:init_ch], device=dev, dtype=torch.float32)
        bias_gpu(x, off_t, nF)
    if os.environ.get('TRITON_DUMP'):
        np.save(os.path.join(os.environ['TRITON_DUMP'], 't_pre_cond.npy'),
                x.cpu().numpy())

    # sine excitation (host, exact CPU formula)
    sr = getattr(sys.modules[__name__], 'SAMPLE_RATE', 32000.0)
    # find sr from config if available
    linw = float(np.asarray(weights.get('dec.m_source.l_linear.weight', np.array([1.0], np.float32))).reshape(-1)[0])
    linb = float(np.asarray(weights.get('dec.m_source.l_linear.bias', np.array([0.0], np.float32))).reshape(-1)[0])
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
    sine_t = torch.as_tensor(sine, device=dev, dtype=torch.float32).contiguous()
    if os.environ.get('TRITON_DUMP'):
        np.save(os.path.join(os.environ['TRITON_DUMP'], 't_sine.npy'), sine)

    # upsample blocks
    cur = x.clone()
    cur_ch = init_ch
    cur_n = nF
    for L in range(n_ups):
        in_ch, in_n = cur_ch, cur_n
        out_ch = init_ch // (1 << (L + 1))
        out_n = (in_n - 1) * rates[L] - 2 * ups_pad[L] + ups_k[L]
        if out_n <= 0:
            return None, -1

        act_gpu(cur, use_snake)

        # convT
        ub = weights.get(f'dec.ups.{L}.bias', np.zeros(out_ch, np.float32))
        if f'dec.ups.{L}.weight_v' in weights:
            uw = denorm_np(weights, f'dec.ups.{L}.weight_v')
        else:
            uw = weights[f'dec.ups.{L}.weight']
        # PyTorch ConvTranspose1d weight: (in_ch, out_ch, k)
        stage = convt1d_gpu(cur, uw, ub, out_ch, ups_k[L], rates[L], ups_pad[L])

        # noise conv
        stride = 1
        for j in range(L + 1, n_ups):
            stride *= rates[j]
        nck = f'dec.noise_convs.{L}.weight'
        ncw = weights.get(nck)
        ncb = weights.get(f'dec.noise_convs.{L}.bias', np.zeros(out_ch, np.float32))
        if ncw is not None:
            kk = ncw.shape[2] if ncw.ndim >= 3 else (1 if L == n_ups - 1 else stride * 2)
            pad = (kk - stride) // 2
            if pad < 0:
                pad = 0
            nc = conv1d_gpu(sine_t.reshape(1, -1), ncw, ncb, out_ch, kk,
                            stride=stride, pad=pad, dil=1)
            add_gpu(stage, nc)

        # MRF
        ch = out_ch
        n_stacks = 3
        n_pairs = 3
        n2 = ch * out_n
        stage_pres = stage.clone()
        acc = torch.zeros((n_stacks * ch * out_n,), device=dev, dtype=torch.float32)
        tmp2 = stage_pres.clone()
        tmp = stage_pres.clone()
        for s in range(n_stacks):
            rb = L * n_stacks + s
            k = [3, 7, 11][s]
            copy_gpu(tmp2, stage_pres)
            copy_gpu(tmp, stage_pres)
            for cp in range(n_pairs):
                r1k = f'dec.resblocks.{rb}.convs1.{cp}.weight_v'
                r1b = weights.get(f'dec.resblocks.{rb}.convs1.{cp}.bias', np.zeros(ch, np.float32))
                r2k = f'dec.resblocks.{rb}.convs2.{cp}.weight_v'
                r2b = weights.get(f'dec.resblocks.{rb}.convs2.{cp}.bias', np.zeros(ch, np.float32))
                r1v = denorm_np(weights, r1k) if r1k in weights else weights[f'dec.resblocks.{rb}.convs1.{cp}.weight']
                r2v = denorm_np(weights, r2k) if r2k in weights else weights[f'dec.resblocks.{rb}.convs2.{cp}.weight']
                dil = 1 + 2 * cp
                pad1 = dil * (k - 1) // 2
                pad2 = k // 2
                act_gpu(tmp2, use_snake)
                cur = conv1d_gpu(tmp2, r1v, r1b, ch, k, stride=1, pad=pad1, dil=dil)
                act_gpu(cur, use_snake)
                tmp2 = conv1d_gpu(cur, r2v, r2b, ch, k, stride=1, pad=pad2, dil=1)
                add_gpu(tmp2, tmp)
                copy_gpu(tmp, tmp2)
            # acc[s] += tmp2
            acc_s = acc[s * n2:(s + 1) * n2]
            add_gpu(acc_s, tmp2)
        # stage = avg over stacks
        copy_gpu(tmp, acc[:n2])
        for s in range(1, n_stacks):
            add_gpu(tmp, acc[s * n2:(s + 1) * n2])
        mul_gpu(tmp, 1.0 / n_stacks)
        cur = tmp.clone()
        cur_ch = ch
        cur_n = out_n

    # conv_post + tanh
    post_in = post_w.shape[1] if post_w.ndim >= 2 else 32
    post_k = post_w.shape[2] if post_w.ndim >= 3 else 7
    post_pad = post_k // 2
    act_gpu(cur, use_snake)
    if os.environ.get('TRITON_DUMP'):
        np.save(os.path.join(os.environ['TRITON_DUMP'], 't_post_in.npy'),
                cur.cpu().numpy())
    out_n = cur_n
    out = conv1d_gpu(cur, post_w, None, 1, post_k, stride=1, pad=post_pad, dil=1)
    tanh_gpu(out)
    torch.cuda.synchronize()
    return out.reshape(-1).cpu().numpy(), out_n


# ─────────────────────────── CLI ───────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Triton RVC generator (parity harness)')
    ap.add_argument('model', help='path to RVC v2 .pth')
    ap.add_argument('--dump-dir', default='outputs/rvc_ref',
                    help='dir with c_gen_input/nsff0/g/output.npy (WUBU_RVC_DUMP=1)')
    ap.add_argument('--out', default='', help='write output wav here')
    ap.add_argument('--snake', action='store_true', help='use Snake activation')
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f'ERROR: model not found: {args.model}')
        sys.exit(2)

    print(f'[triton] loading model {args.model}')
    t0 = time.time()
    weights, ckpt = load_pth(args.model)
    print(f'[triton] loaded {len(weights)} tensors in {time.time()-t0:.2f}s')
    rates, sr = extract_config(ckpt)
    if rates:
        sys.modules[__name__].UPSAMPLE_RATES = rates
        print(f'[triton] config upsample_rates={rates} sr={sr}')
    else:
        print('[triton] no config rates — using kernel-size defaults')
    sys.modules[__name__].SAMPLE_RATE = sr

    dd = args.dump_dir
    z = np.fromfile(os.path.join(dd, 'c_gen_input.npy'), dtype='<f4')
    nsff0 = np.fromfile(os.path.join(dd, 'c_gen_nsff0.npy'), dtype='<f4')
    g = np.fromfile(os.path.join(dd, 'c_gen_g.npy'), dtype='<f4')
    ref = np.fromfile(os.path.join(dd, 'c_gen_output.npy'), dtype='<f4')
    if z.size == 0 or nsff0.size == 0:
        print('ERROR: missing dumps — run the C11 engine with WUBU_RVC_DUMP=1 first')
        sys.exit(2)
    nF = nsff0.size
    inter = z.size // nF
    z = z.reshape(inter, nF)
    print(f'[triton] z {z.shape}, nsff0 {nF}, g {g.size}, ref {ref.size}')

    print('[triton] running generator...')
    t0 = time.time()
    out, out_n = generator_nsf_triton(weights, z, nsff0, g,
                                      use_snake=args.snake, inject_noise=False)
    dt = time.time() - t0
    print(f'[triton] generator done in {dt:.3f}s ({dt/max(out_n,1)*32000:.2f}x realtime @32k)')

    m = min(out.size, ref.size)
    c = np.corrcoef(out[:m], ref[:m])[0, 1]
    md = np.max(np.abs(out[:m] - ref[:m]))
    print(f'[triton] PARITY vs C11: corr={c:.10f} maxdiff={md:.8f} n={m}')
    print(f'[triton] rms triton/cpu: {np.sqrt(np.mean(out**2)):.6f} / {np.sqrt(np.mean(ref**2)):.6f}')

    if args.out:
        import wave
        w = wave.open(args.out, 'wb')
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(32000)
        w.writeframes((np.clip(out, -1, 1) * 32767).astype('<i2').tobytes())
        w.close()
        print(f'[triton] wrote {args.out}')

    ok = c > 0.9999
    print('[triton] ' + ('PASS' if ok else 'FAIL') + ' (corr > 0.9999)')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
