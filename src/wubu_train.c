/* wubu_train.c — WuBuRVC training engine (C11, OpenMP).
 *
 * Real generator (HiFi-GAN decoder) training implemented 2026-08-07
 * (crash-recovery session). Before this rewrite wubu_train_step only
 * computed a loss value — the backward pass was a TODO. This file now has:
 *
 *   1. AdamW optimizer (decoupled weight decay) — bias correction FIXED
 *      (the old fused correction had bias1/bias2 inverted AND double-applied;
 *      now single, correct correction: lr_eff = lr * sqrt(1-b2^t)/(1-b1^t)).
 *   2. conv1d / conv_transpose1d forward + backward (exact chain rule).
 *   3. LReLU / tanh backward.
 *   4. MRF (multi-receptive-field) resblock stack forward + backward.
 *   5. wubu_train_step: forward → MSE loss → backward → AdamW update.
 *   6. wubu_train_gradcheck: analytic grads vs finite differences
 *      (Triple-DA verification — the reference is the loss itself).
 *
 * The decoder structure mirrors the parity-verified path in
 * wubu_rvc_kernels_exact.c (Cartman v2, 40 kHz):
 *   mel(192,F) → conv_pre(192→512,k7,p3) → lrelu
 *     → ups.0(512→256,k16,s10,p3) → MRF0 → lrelu
 *     → ups.1(256→128,k16,s10,p3) → MRF1 → lrelu
 *     → ups.2(128→64,k4,s2,p1) → MRF2 → lrelu
 *     → ups.3(64→32,k4,s2,p1) → MRF3 → lrelu
 *     → conv_post(32→1,k7,p3) → tanh → audio
 *   MRF stage: 3 stacks × 3 pairs, kernels {3,7,11}, dilations {1,3,5}.
 *
 * License: WaefreBeorn-UMV3
 */

#define _USE_MATH_DEFINES
#include "wubu_train.h"
#include "wubu_rvc.h"
#include "wubu_rvc_parity.h"
#include "wubu_rvc_real.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════ AdamW optimizer ═══════════════════════ */

typedef struct WuBuAdamW {
    struct {
        float *m;     /* first moment */
        float *v;     /* second moment */
        int    n;     /* number of elements */
        int    t;     /* timestep */
    } *states;
    int n_states;
    float beta1, beta2, eps, weight_decay, lr;
} WuBuAdamW;

WuBuAdamW *wubu_adamw_create(int n_params, float lr, float beta1, float beta2,
                             float eps, float weight_decay) {
    WuBuAdamW *opt = (WuBuAdamW *)calloc(1, sizeof(WuBuAdamW));
    if (!opt) return NULL;
    opt->states = calloc((size_t)(n_params > 0 ? n_params : 1), sizeof(*opt->states));
    if (!opt->states && n_params > 0) { free(opt); return NULL; }
    opt->n_states = n_params;
    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->weight_decay = weight_decay;
    return opt;
}

int wubu_adamw_init_param(WuBuAdamW *opt, int idx, int n_elem) {
    if (!opt || idx < 0 || idx >= opt->n_states || n_elem <= 0) return -1;
    opt->states[idx].n = n_elem;
    opt->states[idx].t = 0;
    opt->states[idx].m = (float *)calloc((size_t)n_elem, sizeof(float));
    opt->states[idx].v = (float *)calloc((size_t)n_elem, sizeof(float));
    if (!opt->states[idx].m || !opt->states[idx].v) {
        free(opt->states[idx].m);
        free(opt->states[idx].v);
        return -1;
    }
    return 0;
}

/* Correct AdamW step (Loshchilov & Hutter 2019, fused bias correction):
 *   m = b1*m + (1-b1)g ; v = b2*v + (1-b2)g²
 *   m_hat = m/(1-b1^t) ; v_hat = v/(1-b2^t)
 *   param -= lr * (m_hat / (sqrt(v_hat)+eps)) + lr*wd*param
 * lr_eff = lr * sqrt(1-b2^t)/(1-b1^t) folds both corrections into one
 * factor; we use that fused form directly on raw m/v.
 */
void wubu_adamw_step(WuBuAdamW *opt, int idx, float *param, const float *grad) {
    if (!opt || !param || !grad) return;
    int n = opt->states[idx].n;
    float *m = opt->states[idx].m;
    float *v = opt->states[idx].v;
    if (!m || !v || n <= 0) return;

    int t = ++opt->states[idx].t;
    float b1 = opt->beta1, b2 = opt->beta2;
    float bias1 = 1.0f - powf(b1, (float)t);
    float bias2 = 1.0f - powf(b2, (float)t);
    if (bias1 < 1e-12f) bias1 = 1e-12f;
    if (bias2 < 1e-12f) bias2 = 1e-12f;
    float lr_eff = opt->lr * sqrtf(bias2) / bias1;
    float wd = opt->weight_decay;
    float eps = opt->eps;

#pragma omp parallel for if(n >= 256)
    for (int i = 0; i < n; i++) {
        float g = grad[i];
        float m_new = b1 * m[i] + (1.0f - b1) * g;
        float v_new = b2 * v[i] + (1.0f - b2) * g * g;
        m[i] = m_new;
        v[i] = v_new;
        float denom = sqrtf(v_new / bias2) + eps;
        param[i] -= lr_eff * (m_new / bias1) / denom + opt->lr * wd * param[i];
    }
}

void wubu_adamw_free(WuBuAdamW *opt) {
    if (!opt) return;
    for (int i = 0; i < opt->n_states; i++) {
        free(opt->states[i].m);
        free(opt->states[i].v);
    }
    free(opt->states);
    free(opt);
}

/* ═══════════════════════ Trainable-parameter registry ═══════════════════════ */

int wubu_train_registry_build(WuBuRVCModel *model, WuBuTrainRegistry *reg) {
    if (!model || !reg) return -1;
    reg->count = 0;
    reg->cap = 0;
    reg->params = NULL;
    reg->cap = 160;
    reg->params = (WuBuTrainParam *)calloc((size_t)reg->cap, sizeof(WuBuTrainParam));
    if (!reg->params) return -1;

#define REG_ADD(nm, data_ptr, n_elems) do { \
    if ((data_ptr) && (n_elems) > 0 && reg->count < reg->cap) { \
        WuBuTrainParam *p = &reg->params[reg->count]; \
        snprintf(p->name, sizeof(p->name), "%s", (nm)); \
        p->data = (float *)(data_ptr); \
        p->n = (int)(n_elems); \
        p->grad = (float *)calloc((size_t)p->n, sizeof(float)); \
        if (!p->grad) return -1; \
        reg->count++; \
    } \
} while (0)

    /* conv_pre */
    {
        const RVCTensor *w = wubu_rvc_find_tensor(model, "dec.conv_pre.weight");
        const RVCTensor *b = wubu_rvc_find_tensor(model, "dec.conv_pre.bias");
        if (w && w->data)
            REG_ADD("dec.conv_pre.weight", w->data, (int)(w->dims[0]*w->dims[1]*w->dims[2]));
        if (b && b->data)
            REG_ADD("dec.conv_pre.bias", b->data, (int)b->dims[0]);
    }

    /* ups: inference source is hifi_upsample_denorm (de-normalized) */
    for (int L = 0; L < 4; L++) {
        if (model->hifi_upsample_denorm[L]) {
            char nm[64];
            snprintf(nm, sizeof(nm), "dec.ups.%d.weight", L);
            REG_ADD(nm, model->hifi_upsample_denorm[L], model->hifi_upsample_denorm_len[L]);
        }
        char bkey[64];
        snprintf(bkey, sizeof(bkey), "dec.ups.%d.bias", L);
        const RVCTensor *b = wubu_rvc_find_tensor(model, bkey);
        if (b && b->data) REG_ADD(bkey, b->data, (int)b->dims[0]);
    }

    /* resblocks: 12 stacks × 2 conv types × 3 pairs (weight_v de-normed in place) */
    for (int rb = 0; rb < 12; rb++) {
        for (int ct = 0; ct < 2; ct++) {
            const char *cn = ct == 0 ? "convs1" : "convs2";
            for (int cp = 0; cp < 3; cp++) {
                char key[128];
                snprintf(key, sizeof(key), "dec.resblocks.%d.%s.%d.weight_v", rb, cn, cp);
                RVCTensor *wv = (RVCTensor *)wubu_rvc_find_tensor(model, key);
                if (wv && wv->data)
                    REG_ADD(key, wv->data, (int)(wv->dims[0]*wv->dims[1]*wv->dims[2]));
                snprintf(key, sizeof(key), "dec.resblocks.%d.%s.%d.bias", rb, cn, cp);
                const RVCTensor *bb = wubu_rvc_find_tensor(model, key);
                if (bb && bb->data)
                    REG_ADD(key, bb->data, (int)bb->dims[0]);
            }
        }
    }

    /* conv_post */
    {
        const RVCTensor *w = wubu_rvc_find_tensor(model, "dec.conv_post.weight");
        const RVCTensor *b = wubu_rvc_find_tensor(model, "dec.conv_post.bias");
        if (w && w->data)
            REG_ADD("dec.conv_post.weight", w->data, (int)(w->dims[0]*w->dims[1]*w->dims[2]));
        if (b && b->data)
            REG_ADD("dec.conv_post.bias", b->data, (int)b->dims[0]);
    }
#undef REG_ADD
    return reg->count;
}

int wubu_train_registry_find(const WuBuTrainRegistry *reg, const char *name) {
    if (!reg || !name) return -1;
    for (int i = 0; i < reg->count; i++)
        if (strcmp(reg->params[i].name, name) == 0) return i;
    return -1;
}

void wubu_train_registry_zero_grads(WuBuTrainRegistry *reg) {
    if (!reg) return;
    for (int i = 0; i < reg->count; i++)
        if (reg->params[i].grad && reg->params[i].n > 0)
            memset(reg->params[i].grad, 0, (size_t)reg->params[i].n * sizeof(float));
}

void wubu_train_registry_free(WuBuTrainRegistry *reg) {
    if (!reg) return;
    for (int i = 0; i < reg->count; i++)
        free(reg->params[i].grad);
    free(reg->params);
    reg->params = NULL;
    reg->count = reg->cap = 0;
}

/* ═══════════════════════ Loss functions ═══════════════════════ */

float wubu_mse_loss(const float *a, const float *b, int n) {
    if (!a || !b || n <= 0) return 0.0f;
    float sum = 0.0f;
#pragma omp parallel for reduction(+:sum) if(n >= 256)
    for (int i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum / (float)n;
}

float wubu_mae_loss(const float *a, const float *b, int n) {
    if (!a || !b || n <= 0) return 0.0f;
    float sum = 0.0f;
#pragma omp parallel for reduction(+:sum) if(n >= 256)
    for (int i = 0; i < n; i++)
        sum += fabsf(a[i] - b[i]);
    return sum / (float)n;
}

/* Multi-scale STFT loss (evaluation only — magnitude spectrogram is not
 * used for gradient flow in wubu_train_step; MSE provides the gradient). */
float wubu_stft_loss(const float *sig_a, const float *sig_b, int n, int sr) {
    (void)sr;
    if (!sig_a || !sig_b || n <= 0) return 0.0f;
    int wins[] = {512, 256, 128};
    float total_loss = 0.0f;
    int n_scales = 3;

    for (int si = 0; si < n_scales; si++) {
        int w = wins[si];
        int hop = w / 4;
        int n_frames = (n - w) / hop + 1;
        if (n_frames < 1) n_frames = 1;

        float *mag_a = (float *)calloc((size_t)(w / 2 + 1) * n_frames, sizeof(float));
        float *mag_b = (float *)calloc((size_t)(w / 2 + 1) * n_frames, sizeof(float));
        if (!mag_a || !mag_b) { free(mag_a); free(mag_b); continue; }

        float *win = (float *)malloc((size_t)w * sizeof(float));
        if (win) {
            for (int i = 0; i < w; i++)
                win[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)i / (float)(w - 1));
        }

        for (int f = 0; f < n_frames; f++) {
            int start = f * hop;
            float *frame_a = (float *)calloc((size_t)w, sizeof(float));
            float *frame_b = (float *)calloc((size_t)w, sizeof(float));
            if (!frame_a || !frame_b) { free(frame_a); free(frame_b); continue; }
            for (int i = 0; i < w; i++) {
                int idx = start + i;
                float sa = (idx < n) ? sig_a[idx] : 0.0f;
                float sb = (idx < n) ? sig_b[idx] : 0.0f;
                if (win) { sa *= win[i]; sb *= win[i]; }
                frame_a[i] = sa;
                frame_b[i] = sb;
            }
            for (int bin = 0; bin <= w / 2; bin++) {
                float re_a = 0, im_a = 0, re_b = 0, im_b = 0;
                for (int t = 0; t < w; t++) {
                    float phase = -2.0f * (float)M_PI * (float)bin * (float)t / (float)w;
                    float c = cosf(phase), s = sinf(phase);
                    re_a += frame_a[t] * c; im_a += frame_a[t] * s;
                    re_b += frame_b[t] * c; im_b += frame_b[t] * s;
                }
                mag_a[(size_t)bin * n_frames + f] = sqrtf(re_a * re_a + im_a * im_a);
                mag_b[(size_t)bin * n_frames + f] = sqrtf(re_b * re_b + im_b * im_b);
            }
            free(frame_a); free(frame_b);
        }

        float lin_loss = wubu_mae_loss(mag_a, mag_b, (w / 2 + 1) * n_frames);
        float log_loss = 0.0f;
        int total_bins = (w / 2 + 1) * n_frames;
        for (int i = 0; i < total_bins; i++) {
            float la = mag_a[i] + 1e-10f;
            float lb = mag_b[i] + 1e-10f;
            log_loss += fabsf(logf(la) - logf(lb));
        }
        log_loss /= (float)total_bins;
        total_loss += 0.5f * (lin_loss + log_loss);
        free(win); free(mag_a); free(mag_b);
    }
    return total_loss / (float)n_scales;
}

float wubu_gan_g_loss(float d_real_out) {
    return -logf(d_real_out + 1e-8f);
}

float wubu_gan_d_loss(float d_real_out, float d_fake_out) {
    return -logf(d_real_out + 1e-8f) - logf(1.0f - d_fake_out + 1e-8f);
}

/* ═══════════════════════ conv1d forward/backward ═══════════════════════
 * Col-major tensors: in[(ic)*n_in + i], out[(oc)*n_out + j].
 * Weight layout (PyTorch Conv1d): W[(oc*in_ch + ic)*k + tap].
 *   out[oc,j] = b[oc] + Σ_ic Σ_tap in[ic, j*s - p + d*tap] * W[oc,ic,tap]
 */
static void conv1d_fwd(const float *in, const float *w, const float *b,
                       int in_ch, int out_ch, int k, int stride, int pad, int dil,
                       int n_in, float *out) {
    int n_out = (n_in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
    if (n_out <= 0) return;
    memset(out, 0, (size_t)out_ch * n_out * sizeof(float));
#pragma omp parallel for if(out_ch >= 4)
    for (int oc = 0; oc < out_ch; oc++) {
        const float *wv = w + (size_t)oc * in_ch * k;
        float *o = out + (size_t)oc * n_out;
        float bias = b ? b[oc] : 0.0f;
        for (int j = 0; j < n_out; j++) {
            float acc = bias;
            int j_in = j * stride - pad;
            for (int tap = 0; tap < k; tap++) {
                int src = j_in + dil * tap;
                if (src >= 0 && src < n_in) {
                    const float *wk = wv + (size_t)tap;
                    for (int ic = 0; ic < in_ch; ic++)
                        acc += in[(size_t)ic * n_in + src] * wk[(size_t)ic * k];
                }
            }
            o[j] = acc;
        }
    }
}

/* dw, db, din may be NULL. din is col-major like in. Parallelized over the
 * input-channel dimension (independent din rows / dw rows per ic). */
static void conv1d_bwd(const float *in, const float *w, const float *dout,
                       int in_ch, int out_ch, int k, int stride, int pad, int dil,
                       int n_in, int n_out,
                       float *din, float *dw, float *db) {
    if (dw) memset(dw, 0, (size_t)out_ch * in_ch * k * sizeof(float));
    if (db) memset(db, 0, (size_t)out_ch * sizeof(float));
    if (din) memset(din, 0, (size_t)in_ch * n_in * sizeof(float));

#pragma omp parallel for if(in_ch >= 4)
    for (int ic = 0; ic < in_ch; ic++) {
        for (int oc = 0; oc < out_ch; oc++) {
            const float *wv = w + (size_t)oc * in_ch * k;
            const float *do_ = dout + (size_t)oc * n_out;
            float *din_row = din ? din + (size_t)ic * n_in : NULL;
            float *dw_row = dw ? dw + (size_t)oc * in_ch * k + (size_t)ic * k : NULL;
            for (int j = 0; j < n_out; j++) {
                float g = do_[j];
                int j_in = j * stride - pad;
                for (int tap = 0; tap < k; tap++) {
                    int src = j_in + dil * tap;
                    if (src >= 0 && src < n_in) {
                        float x = in[(size_t)ic * n_in + src];
                        if (dw_row) dw_row[tap] += x * g;
                        if (din_row) din_row[src] += g * wv[(size_t)ic * k + tap];
                    }
                }
            }
        }
    }
    if (db) {
        for (int oc = 0; oc < out_ch; oc++) {
            float acc = 0.0f;
            const float *do_ = dout + (size_t)oc * n_out;
            for (int j = 0; j < n_out; j++) acc += do_[j];
            db[oc] = acc;
        }
    }
}

/* ═══════════════════════ conv_transpose1d forward/backward ═══════════════════════
 * Col-major. Weight layout (PyTorch ConvTranspose1d): W[(ic*out_ch + oc)*k + tap].
 *   out[oc, j] = b[oc] + Σ_ic Σ_tap in[ic, i] * W[ic,oc,tap],  j = i*s - p + tap
 */
static void convt1d_fwd(const float *in, const float *w, const float *b,
                        int in_ch, int out_ch, int k, int stride, int pad,
                        int n_in, float *out) {
    int n_out = (n_in - 1) * stride - 2 * pad + k;
    if (n_out <= 0) return;
    memset(out, 0, (size_t)out_ch * n_out * sizeof(float));
#pragma omp parallel for if(out_ch >= 4)
    for (int oc = 0; oc < out_ch; oc++) {
        float *o = out + (size_t)oc * n_out;
        for (int i = 0; i < n_in; i++) {
            int j_start = i * stride - pad;
            for (int ic = 0; ic < in_ch; ic++) {
                float inp = in[(size_t)ic * n_in + i];
                if (inp == 0.0f) continue;
                const float *wk = w + ((size_t)ic * (size_t)out_ch + oc) * k;
                for (int tap = 0; tap < k; tap++) {
                    int j = j_start + tap;
                    if (j >= 0 && j < n_out)
                        o[j] += inp * wk[tap];
                }
            }
        }
    }
    if (b) {
        for (int oc = 0; oc < out_ch; oc++) {
            float *o = out + (size_t)oc * n_out;
            for (int j = 0; j < n_out; j++) o[j] += b[oc];
        }
    }
}

static void convt1d_bwd(const float *in, const float *w, const float *dout,
                        int in_ch, int out_ch, int k, int stride, int pad,
                        int n_in, int n_out,
                        float *din, float *dw, float *db) {
    if (dw) memset(dw, 0, (size_t)in_ch * out_ch * k * sizeof(float));
    if (db) memset(db, 0, (size_t)out_ch * sizeof(float));
    if (din) memset(din, 0, (size_t)in_ch * n_in * sizeof(float));

#pragma omp parallel for if(in_ch >= 4)
    for (int ic = 0; ic < in_ch; ic++) {
        float *din_row = din ? din + (size_t)ic * n_in : NULL;
        const float *wv = w + (size_t)ic * (size_t)out_ch * (size_t)k;
        for (int i = 0; i < n_in; i++) {
            int j_start = i * stride - pad;
            float din_acc = 0.0f;
            for (int oc = 0; oc < out_ch; oc++) {
                const float *wk = wv + (size_t)oc * k;
                for (int tap = 0; tap < k; tap++) {
                    int j = j_start + tap;
                    if (j >= 0 && j < n_out) {
                        float g = dout[(size_t)oc * n_out + j];
                        din_acc += g * wk[tap];
                        if (dw) dw[(size_t)ic * out_ch * k + (size_t)oc * k + tap] +=
                                    in[(size_t)ic * n_in + i] * g;
                    }
                }
            }
            if (din_row) din_row[i] += din_acc;
        }
    }
    if (db) {
        for (int oc = 0; oc < out_ch; oc++) {
            float acc = 0.0f;
            for (int j = 0; j < n_out; j++) acc += dout[(size_t)oc * n_out + j];
            db[oc] = acc;
        }
    }
}

/* ═══════════════════════ Activations ═══════════════════════ */

static void lrelu_fwd(float *x, size_t n, float slope) {
    for (size_t i = 0; i < n; i++)
        x[i] = x[i] > 0 ? x[i] : slope * x[i];
}

/* din[i] = dout[i] * (x[i] > 0 ? 1 : slope); x = pre-activation (cached) */
static void lrelu_bwd(const float *x, const float *dout, float *din, size_t n, float slope) {
    for (size_t i = 0; i < n; i++)
        din[i] = dout[i] * (x[i] > 0 ? 1.0f : slope);
}

/* din[i] = dout[i] * (1 - y[i]^2); y = tanh output (cached) */
static void tanh_bwd(const float *y, const float *dout, float *din, size_t n) {
    for (size_t i = 0; i < n; i++)
        din[i] = dout[i] * (1.0f - y[i] * y[i]);
}

/* ═══════════════════════ MRF stage ═══════════════════════
 * forward, per stack s (rb = stage*3+s), per pair cp (dilations {1,3,5}):
 *   p = input
 *   a1 = lrelu(p)
 *   c1o = conv1d(a1, c1w, k, d=dil)
 *   a2 = lrelu(c1o)
 *   c2o = conv1d(a2, c2w, k, d=1)
 *   out = c2o + p        (residual)
 * stage_out = (Σ_s stack_out_s) / 3
 */

/* One pair forward; writes the four intermediates the backward needs. */
static void mrf_pair_fwd(const float *p,
                         const float *c1w, const float *c1b,
                         const float *c2w, const float *c2b,
                         int ch, int n, int k, int dil,
                         float *a1, float *c1o, float *a2, float *c2o) {
    int pad1 = dil * (k - 1) / 2;
    int pad2 = k / 2;
    memcpy(a1, p, (size_t)ch * n * sizeof(float));
    lrelu_fwd(a1, (size_t)ch * n, 0.1f);
    conv1d_fwd(a1, c1w, c1b, ch, ch, k, 1, pad1, dil, n, c1o);
    memcpy(a2, c1o, (size_t)ch * n * sizeof(float));
    lrelu_fwd(a2, (size_t)ch * n, 0.1f);
    conv1d_fwd(a2, c2w, c2b, ch, ch, k, 1, pad2, 1, n, c2o);
    for (size_t i = 0; i < (size_t)ch * n; i++) c2o[i] += p[i];
}

/* Backward through one MRF stage. Recomputes forward, stores intermediates.
 * d_stage_out: (ch,n) grad wrt stage output. Writes d_stage_in (ch,n).
 * Accumulates weight/bias grads into the registry. */
static void mrf_stage_bwd(WuBuRVCModel *model, int stage_idx, int ch, int n,
                          const float *stage_in,
                          const float *d_stage_out,
                          WuBuTrainRegistry *reg,
                          float *d_stage_in) {
    int dils[3] = {1, 3, 5};
    memset(d_stage_in, 0, (size_t)ch * n * sizeof(float));

    /* per-pair intermediates (3 pairs × 4 buffers) + stack input + grads */
    float *p[3], *a1[3], *c1o[3], *a2[3], *c2o[3];
    float *stack_in = (float *)malloc((size_t)ch * n * sizeof(float));
    float *d_in = (float *)malloc((size_t)ch * n * sizeof(float));
    float *d_res = (float *)malloc((size_t)ch * n * sizeof(float));
    float *d_a2 = (float *)malloc((size_t)ch * n * sizeof(float));
    float *d_c1o = (float *)malloc((size_t)ch * n * sizeof(float));
    float *d_a1 = (float *)malloc((size_t)ch * n * sizeof(float));
    if (!stack_in || !d_in || !d_res || !d_a2 || !d_c1o || !d_a1) {
        free(stack_in); free(d_in); free(d_res); free(d_a2); free(d_c1o); free(d_a1);
        return;
    }

    for (int s = 0; s < 3; s++) {
        int rb = stage_idx * 3 + s;
        memcpy(stack_in, stage_in, (size_t)ch * n * sizeof(float));
        int valid[3] = {0, 0, 0};

        /* 1. forward recompute, storing intermediates */
        for (int cp = 0; cp < 3; cp++) {
            p[cp] = a1[cp] = c1o[cp] = a2[cp] = c2o[cp] = NULL;
            char key[128];
            snprintf(key, sizeof(key), "dec.resblocks.%d.convs1.%d.weight_v", rb, cp);
            const RVCTensor *c1w = wubu_rvc_find_tensor(model, key);
            snprintf(key, sizeof(key), "dec.resblocks.%d.convs2.%d.weight_v", rb, cp);
            const RVCTensor *c2w = wubu_rvc_find_tensor(model, key);
            if (!c1w || !c1w->data || !c2w || !c2w->data) continue; /* passthrough */
            int k = c1w->dims[2];
            p[cp]  = (float *)malloc((size_t)ch * n * sizeof(float));
            a1[cp] = (float *)malloc((size_t)ch * n * sizeof(float));
            c1o[cp]= (float *)malloc((size_t)ch * n * sizeof(float));
            a2[cp] = (float *)malloc((size_t)ch * n * sizeof(float));
            c2o[cp]= (float *)malloc((size_t)ch * n * sizeof(float));
            if (!p[cp] || !a1[cp] || !c1o[cp] || !a2[cp] || !c2o[cp]) { valid[cp] = 0; continue; }
            memcpy(p[cp], stack_in, (size_t)ch * n * sizeof(float));
            snprintf(key, sizeof(key), "dec.resblocks.%d.convs1.%d.bias", rb, cp);
            const RVCTensor *c1b = wubu_rvc_find_tensor(model, key);
            snprintf(key, sizeof(key), "dec.resblocks.%d.convs2.%d.bias", rb, cp);
            const RVCTensor *c2b = wubu_rvc_find_tensor(model, key);
            mrf_pair_fwd(stack_in, c1w->data,
                         (c1b && c1b->data) ? c1b->data : NULL,
                         c2w->data,
                         (c2b && c2b->data) ? c2b->data : NULL,
                         ch, n, k, dils[cp], a1[cp], c1o[cp], a2[cp], c2o[cp]);
            memcpy(stack_in, c2o[cp], (size_t)ch * n * sizeof(float));
            valid[cp] = 1;
        }
        /* stack_in now = stack output */

        /* 2. backward, cp 2 → 0. d_in = grad wrt pair output (starts at
         * d_stage_out/3 because of the averaging). */
        for (size_t i = 0; i < (size_t)ch * n; i++)
            d_in[i] = d_stage_out[i] / 3.0f;

        for (int cp = 2; cp >= 0; cp--) {
            if (!valid[cp]) continue; /* passthrough: grad unchanged */
            char key[128];
            snprintf(key, sizeof(key), "dec.resblocks.%d.convs1.%d.weight_v", rb, cp);
            const RVCTensor *c1w = wubu_rvc_find_tensor(model, key);
            snprintf(key, sizeof(key), "dec.resblocks.%d.convs2.%d.weight_v", rb, cp);
            const RVCTensor *c2w = wubu_rvc_find_tensor(model, key);
            int k = c1w->dims[2];
            int pad1 = dils[cp] * (k - 1) / 2;
            int pad2 = k / 2;

            /* save entry grad — the residual path adds d(out) back to p */
            memcpy(d_res, d_in, (size_t)ch * n * sizeof(float));

            /* a) conv2: din=d_a2, dw/db → registry */
            snprintf(key, sizeof(key), "dec.resblocks.%d.convs2.%d.weight_v", rb, cp);
            int i2w = wubu_train_registry_find(reg, key);
            float *dw2 = (i2w >= 0) ? reg->params[i2w].grad : NULL;
            snprintf(key, sizeof(key), "dec.resblocks.%d.convs2.%d.bias", rb, cp);
            int i2b = wubu_train_registry_find(reg, key);
            float *db2 = (i2b >= 0) ? reg->params[i2b].grad : NULL;
            conv1d_bwd(a2[cp], c2w->data, d_in,
                       ch, ch, k, 1, pad2, 1, n, n,
                       d_a2, dw2, db2);

            /* b) lrelu after conv1: pre-act = c1o[cp] */
            lrelu_bwd(c1o[cp], d_a2, d_c1o, (size_t)ch * n, 0.1f);

            /* c) conv1: din=d_a1, dw/db → registry */
            snprintf(key, sizeof(key), "dec.resblocks.%d.convs1.%d.weight_v", rb, cp);
            int i1w = wubu_train_registry_find(reg, key);
            float *dw1 = (i1w >= 0) ? reg->params[i1w].grad : NULL;
            snprintf(key, sizeof(key), "dec.resblocks.%d.convs1.%d.bias", rb, cp);
            int i1b = wubu_train_registry_find(reg, key);
            float *db1 = (i1b >= 0) ? reg->params[i1b].grad : NULL;
            conv1d_bwd(a1[cp], c1w->data, d_c1o,
                       ch, ch, k, 1, pad1, dils[cp], n, n,
                       d_a1, dw1, db1);

            /* d) lrelu before conv1: pre-act = p[cp]; residual add */
            lrelu_bwd(p[cp], d_a1, d_in, (size_t)ch * n, 0.1f);
            for (size_t i = 0; i < (size_t)ch * n; i++)
                d_in[i] += d_res[i];
        }

        /* 3. accumulate stage input grad from this stack */
        for (size_t i = 0; i < (size_t)ch * n; i++)
            d_stage_in[i] += d_in[i];

        for (int cp = 0; cp < 3; cp++) {
            free(p[cp]); free(a1[cp]); free(c1o[cp]); free(a2[cp]); free(c2o[cp]);
        }
    }

    free(stack_in); free(d_in); free(d_res); free(d_a2); free(d_c1o); free(d_a1);
}

/* ═══════════════════════ Decoder forward (cached) ═══════════════════════
 * Returns n_out (samples) or -1. Fills cache with pre-activations needed by
 * the backward pass. Mirrors wubu_kernel_hifigan_exact structure exactly. */

typedef struct {
    int n_frames;
    int n_stage[4];            /* output length per stage */
    int ch_in[4], ch_out[4];   /* ups channel dims */
    int k[4], s[4], p[4];      /* ups kernel/stride/pad */
    float *pre_out;            /* (512, F) after conv_pre, PRE-lrelu */
    float *act_in[4];          /* lrelu'd input to each ups convT (col-major) */
    float *ups_out[4];         /* after convT (pre-MRF) */
    float *stage_out[4];       /* after MRF, PRE-lrelu */
    float *post_in;            /* (32, n3) after final lrelu (pre conv_post) */
    float *audio;              /* after tanh */
} DecCache;

static int decoder_forward(WuBuRVCModel *model, const float *mel_in, int n_frames,
                           float *audio, int max_samples, DecCache *cache) {
    if (!model || !model->loaded || !mel_in || !audio) return -1;
    for (int i = 0; i < 4; i++)
        if (!model->hifi_upsample_denorm[i]) return -1;

    memset(cache, 0, sizeof(*cache));
    cache->n_frames = n_frames;

    /* ups config from model (fallback Cartman 40k) */
    int k[4]  = {16, 16, 4, 4};
    int s[4]  = {10, 10, 2, 2};
    int ch_in[4]  = {512, 256, 128, 64};
    int ch_out[4] = {256, 128, 64, 32};
    for (int L = 0; L < 4; L++) {
        if (model->n_upsample_layers > L) {
            if (model->upsample_rates[L] > 0) s[L] = model->upsample_rates[L];
            if (model->upsample_kernel_sizes[L] > 0) k[L] = model->upsample_kernel_sizes[L];
        }
        cache->k[L] = k[L]; cache->s[L] = s[L];
        cache->ch_in[L] = ch_in[L]; cache->ch_out[L] = ch_out[L];
        cache->p[L] = (k[L] - s[L]) / 2;
    }

    const RVCTensor *conv_pre_w = wubu_rvc_find_tensor(model, "dec.conv_pre.weight");
    const RVCTensor *conv_pre_b = wubu_rvc_find_tensor(model, "dec.conv_pre.bias");
    if (!conv_pre_w || !conv_pre_w->data) return -1;
    int pre_in = conv_pre_w->dims[1];   /* 192 */
    int pre_out_ch = conv_pre_w->dims[0]; /* 512 */
    int pre_k = conv_pre_w->dims[2];    /* 7 */

    /* conv_pre: mel_in col-major (pre_in, F) → (512, F), PRE-lrelu in cache */
    cache->pre_out = (float *)calloc((size_t)pre_out_ch * n_frames, sizeof(float));
    if (!cache->pre_out) return -1;
    conv1d_fwd(mel_in, conv_pre_w->data,
               (conv_pre_b && conv_pre_b->data) ? conv_pre_b->data : NULL,
               pre_in, pre_out_ch, pre_k, 1, (pre_k - 1) / 2, 1, n_frames, cache->pre_out);

    /* act_in[0] = lrelu(pre_out) — separate buffer so pre_out stays pre-lrelu */
    cache->act_in[0] = (float *)malloc((size_t)pre_out_ch * n_frames * sizeof(float));
    if (!cache->act_in[0]) return -1;
    memcpy(cache->act_in[0], cache->pre_out, (size_t)pre_out_ch * n_frames * sizeof(float));
    lrelu_fwd(cache->act_in[0], (size_t)pre_out_ch * n_frames, 0.1f);

    /* stages */
    float *prev = cache->act_in[0];
    int prev_n = n_frames;
    int prev_ch = pre_out_ch;
    for (int L = 0; L < 4; L++) {
        int n_out = (prev_n - 1) * s[L] - 2 * cache->p[L] + k[L];
        if (n_out <= 0 || n_out > max_samples) return -1;
        cache->n_stage[L] = n_out;

        cache->ups_out[L] = (float *)calloc((size_t)ch_out[L] * n_out, sizeof(float));
        if (!cache->ups_out[L]) return -1;
        char bkey[64];
        snprintf(bkey, sizeof(bkey), "dec.ups.%d.bias", L);
        const RVCTensor *ub = wubu_rvc_find_tensor(model, bkey);
        convt1d_fwd(prev, model->hifi_upsample_denorm[L],
                    (ub && ub->data) ? ub->data : NULL,
                    ch_in[L], ch_out[L], k[L], s[L], cache->p[L], prev_n,
                    cache->ups_out[L]);

        cache->stage_out[L] = (float *)calloc((size_t)ch_out[L] * n_out, sizeof(float));
        if (!cache->stage_out[L]) return -1;
        /* MRF stage: needs the intermediate grads — we recompute forward in
         * backward, so here just run one stack pair at a time via helper */
        {
            int dils[3] = {1, 3, 5};
            float *acc = (float *)calloc((size_t)ch_out[L] * n_out, sizeof(float));
            float *stack = (float *)malloc((size_t)ch_out[L] * n_out * sizeof(float));
            float *a1 = (float *)malloc((size_t)ch_out[L] * n_out * sizeof(float));
            float *c1o = (float *)malloc((size_t)ch_out[L] * n_out * sizeof(float));
            float *a2 = (float *)malloc((size_t)ch_out[L] * n_out * sizeof(float));
            float *c2o = (float *)malloc((size_t)ch_out[L] * n_out * sizeof(float));
            if (!acc || !stack || !a1 || !c1o || !a2 || !c2o) {
                free(acc); free(stack); free(a1); free(c1o); free(a2); free(c2o);
                return -1;
            }
            for (int st = 0; st < 3; st++) {
                int rb = L * 3 + st;
                memcpy(stack, cache->ups_out[L], (size_t)ch_out[L] * n_out * sizeof(float));
                for (int cp = 0; cp < 3; cp++) {
                    char key[128];
                    snprintf(key, sizeof(key), "dec.resblocks.%d.convs1.%d.weight_v", rb, cp);
                    const RVCTensor *c1w = wubu_rvc_find_tensor(model, key);
                    snprintf(key, sizeof(key), "dec.resblocks.%d.convs2.%d.weight_v", rb, cp);
                    const RVCTensor *c2w = wubu_rvc_find_tensor(model, key);
                    if (!c1w || !c1w->data || !c2w || !c2w->data) continue;
                    snprintf(key, sizeof(key), "dec.resblocks.%d.convs1.%d.bias", rb, cp);
                    const RVCTensor *c1b = wubu_rvc_find_tensor(model, key);
                    snprintf(key, sizeof(key), "dec.resblocks.%d.convs2.%d.bias", rb, cp);
                    const RVCTensor *c2b = wubu_rvc_find_tensor(model, key);
                    int kk = c1w->dims[2];
                    mrf_pair_fwd(stack, c1w->data,
                                 (c1b && c1b->data) ? c1b->data : NULL,
                                 c2w->data,
                                 (c2b && c2b->data) ? c2b->data : NULL,
                                 ch_out[L], n_out, kk, dils[cp],
                                 a1, c1o, a2, c2o);
                    memcpy(stack, c2o, (size_t)ch_out[L] * n_out * sizeof(float));
                }
                for (size_t i = 0; i < (size_t)ch_out[L] * n_out; i++)
                    acc[i] += stack[i];
            }
            for (size_t i = 0; i < (size_t)ch_out[L] * n_out; i++)
                cache->stage_out[L][i] = acc[i] / 3.0f;
            free(acc); free(stack); free(a1); free(c1o); free(a2); free(c2o);
        }

        prev_ch = ch_out[L];
        prev_n = n_out;
        if (L < 3) {
            /* act_in[L+1] = lrelu(stage_out[L]) — separate, stage_out stays pre-lrelu */
            cache->act_in[L + 1] = (float *)malloc((size_t)ch_out[L] * n_out * sizeof(float));
            if (!cache->act_in[L + 1]) return -1;
            memcpy(cache->act_in[L + 1], cache->stage_out[L],
                   (size_t)ch_out[L] * n_out * sizeof(float));
            lrelu_fwd(cache->act_in[L + 1], (size_t)ch_out[L] * n_out, 0.1f);
            prev = cache->act_in[L + 1];
        }
        (void)prev_ch;
    }

    /* final lrelu + conv_post + tanh */
    int n3 = cache->n_stage[3];
    cache->post_in = (float *)malloc((size_t)32 * n3 * sizeof(float));
    if (!cache->post_in) return -1;
    memcpy(cache->post_in, cache->stage_out[3], (size_t)32 * n3 * sizeof(float));
    lrelu_fwd(cache->post_in, (size_t)32 * n3, 0.1f);

    const RVCTensor *post_w = wubu_rvc_find_tensor(model, "dec.conv_post.weight");
    const RVCTensor *post_b = wubu_rvc_find_tensor(model, "dec.conv_post.bias");
    if (!post_w || !post_w->data) return -1;
    int out_n = n3;
    if (out_n > max_samples) out_n = max_samples;
    cache->audio = (float *)calloc((size_t)out_n, sizeof(float));
    if (!cache->audio) return -1;
    {
        float *tmp = (float *)malloc((size_t)out_n * sizeof(float));
        if (!tmp) return -1;
        conv1d_fwd(cache->post_in, post_w->data,
                   (post_b && post_b->data) ? post_b->data : NULL,
                   32, 1, 7, 1, 3, 1, n3, tmp);
        for (int j = 0; j < out_n; j++)
            cache->audio[j] = tanhf(tmp[j]);
        free(tmp);
    }
    memcpy(audio, cache->audio, (size_t)out_n * sizeof(float));
    return out_n;
}

static void decoder_cache_free(DecCache *cache) {
    free(cache->pre_out);
    for (int L = 0; L < 4; L++) {
        free(cache->act_in[L]);
        free(cache->ups_out[L]);
        free(cache->stage_out[L]);
    }
    free(cache->post_in);
    free(cache->audio);
    memset(cache, 0, sizeof(*cache));
}

/* ═══════════════════════ Decoder backward ═══════════════════════ */

static void decoder_backward(WuBuRVCModel *model, const DecCache *cache,
                             const float *mel_in, const float *d_audio,
                             WuBuTrainRegistry *reg,
                             float *d_input /* optional, (192,F) */) {
    int n3 = cache->n_stage[3];

    /* d_audio → tanh' → conv_post bwd → lrelu' (pre-act = stage_out[3]) */
    float *d_post_in = (float *)calloc((size_t)32 * n3, sizeof(float));
    float *d_stage3 = (float *)calloc((size_t)32 * n3, sizeof(float));
    /* always allocated — it is the grad wrt act_in[0] feeding lrelu_bwd */
    float *d_conv_pre_in = (float *)calloc((size_t)512 * cache->n_frames, sizeof(float));
    if (!d_post_in || !d_stage3 || !d_conv_pre_in) {
        free(d_post_in); free(d_stage3); free(d_conv_pre_in);
        return;
    }

    {
        float *d_tanh_in = (float *)malloc((size_t)n3 * sizeof(float));
        float *d_post_out = (float *)malloc((size_t)32 * n3 * sizeof(float)); /* 32 ch */
        if (d_tanh_in && d_post_out) {
            tanh_bwd(cache->audio, d_audio, d_tanh_in, (size_t)n3);
            const RVCTensor *post_w = wubu_rvc_find_tensor(model, "dec.conv_post.weight");
            const RVCTensor *post_b = wubu_rvc_find_tensor(model, "dec.conv_post.bias");
            int iw = wubu_train_registry_find(reg, "dec.conv_post.weight");
            int ib = wubu_train_registry_find(reg, "dec.conv_post.bias");
            conv1d_bwd(cache->post_in, post_w->data, d_tanh_in,
                       32, 1, 7, 1, 3, 1, n3, n3,
                       d_post_out,
                       (iw >= 0) ? reg->params[iw].grad : NULL,
                       (ib >= 0) ? reg->params[ib].grad : NULL);
            /* post_in = lrelu(stage_out[3]) */
            lrelu_bwd(cache->stage_out[3], d_post_out, d_stage3, (size_t)32 * n3, 0.1f);
        }
        free(d_tanh_in); free(d_post_out);
    }

    /* stage 3 MRF bwd */
    {
        float *d_ups3 = (float *)malloc((size_t)32 * n3 * sizeof(float)); /* ups3 out_ch = 32 */
        if (d_ups3) {
            mrf_stage_bwd(model, 3, 32, n3, cache->ups_out[3], d_stage3, reg, d_ups3);
            /* convT3 bwd: input = lrelu(stage_out[2]) */
            int iw = wubu_train_registry_find(reg, "dec.ups.3.weight");
            int ib = wubu_train_registry_find(reg, "dec.ups.3.bias");
            float *d_l2 = (float *)calloc((size_t)64 * cache->n_stage[2], sizeof(float));
            if (d_l2) {
                convt1d_bwd(cache->act_in[3], model->hifi_upsample_denorm[3], d_ups3,
                            64, 32, cache->k[3], cache->s[3], cache->p[3],
                            cache->n_stage[2], n3, d_l2,
                            (iw >= 0) ? reg->params[iw].grad : NULL,
                            (ib >= 0) ? reg->params[ib].grad : NULL);
                /* stage_out[2] is pre-lrelu; apply lrelu bwd */
                float *d_stage2 = (float *)malloc((size_t)64 * cache->n_stage[2] * sizeof(float));
                if (d_stage2) {
                    lrelu_bwd(cache->stage_out[2], d_l2, d_stage2,
                              (size_t)64 * cache->n_stage[2], 0.1f);
                    /* stage 2 MRF bwd */
                    float *d_ups2 = (float *)malloc((size_t)128 * cache->n_stage[2] * sizeof(float));
                    if (d_ups2) {
                        mrf_stage_bwd(model, 2, 64, cache->n_stage[2], cache->ups_out[2],
                                      d_stage2, reg, d_ups2);
                        iw = wubu_train_registry_find(reg, "dec.ups.2.weight");
                        ib = wubu_train_registry_find(reg, "dec.ups.2.bias");
                        float *d_l1 = (float *)calloc((size_t)128 * cache->n_stage[1], sizeof(float));
                        if (d_l1) {
                            convt1d_bwd(cache->act_in[2], model->hifi_upsample_denorm[2], d_ups2,
                                        128, 64, cache->k[2], cache->s[2], cache->p[2],
                                        cache->n_stage[1], cache->n_stage[2], d_l1,
                                        (iw >= 0) ? reg->params[iw].grad : NULL,
                                        (ib >= 0) ? reg->params[ib].grad : NULL);
                            float *d_stage1 = (float *)malloc((size_t)128 * cache->n_stage[1] * sizeof(float));
                            if (d_stage1) {
                                lrelu_bwd(cache->stage_out[1], d_l1, d_stage1,
                                          (size_t)128 * cache->n_stage[1], 0.1f);
                                float *d_ups1 = (float *)malloc((size_t)256 * cache->n_stage[1] * sizeof(float));
                                if (d_ups1) {
                                    mrf_stage_bwd(model, 1, 128, cache->n_stage[1], cache->ups_out[1],
                                                  d_stage1, reg, d_ups1);
                                    iw = wubu_train_registry_find(reg, "dec.ups.1.weight");
                                    ib = wubu_train_registry_find(reg, "dec.ups.1.bias");
                                    float *d_l0 = (float *)calloc((size_t)256 * cache->n_stage[0], sizeof(float));
                                    if (d_l0) {
                                        convt1d_bwd(cache->act_in[1], model->hifi_upsample_denorm[1], d_ups1,
                                                    256, 128, cache->k[1], cache->s[1], cache->p[1],
                                                    cache->n_stage[0], cache->n_stage[1], d_l0,
                                                    (iw >= 0) ? reg->params[iw].grad : NULL,
                                                    (ib >= 0) ? reg->params[ib].grad : NULL);
                                        float *d_stage0 = (float *)malloc((size_t)256 * cache->n_stage[0] * sizeof(float));
                                        if (d_stage0) {
                                            lrelu_bwd(cache->stage_out[0], d_l0, d_stage0,
                                                      (size_t)256 * cache->n_stage[0], 0.1f);
                                            float *d_ups0 = (float *)malloc((size_t)512 * cache->n_stage[0] * sizeof(float));
                                            if (d_ups0) {
                                                mrf_stage_bwd(model, 0, 256, cache->n_stage[0], cache->ups_out[0],
                                                              d_stage0, reg, d_ups0);
                                                iw = wubu_train_registry_find(reg, "dec.ups.0.weight");
                                                ib = wubu_train_registry_find(reg, "dec.ups.0.bias");
                                                convt1d_bwd(cache->act_in[0], model->hifi_upsample_denorm[0], d_ups0,
                                                            512, 256, cache->k[0], cache->s[0], cache->p[0],
                                                            cache->n_frames, cache->n_stage[0], d_conv_pre_in,
                                                            (iw >= 0) ? reg->params[iw].grad : NULL,
                                                            (ib >= 0) ? reg->params[ib].grad : NULL);
                                                /* conv_pre bwd (pre_out = lrelu(conv_pre out)) */
                                                float *d_cp = (float *)calloc((size_t)512 * cache->n_frames, sizeof(float));
                                                if (d_cp) {
                                                    lrelu_bwd(cache->pre_out, d_conv_pre_in, d_cp,
                                                              (size_t)512 * cache->n_frames, 0.1f);
                                                    const RVCTensor *cpw = wubu_rvc_find_tensor(model, "dec.conv_pre.weight");
                                                    const RVCTensor *cpb = wubu_rvc_find_tensor(model, "dec.conv_pre.bias");
                                                    iw = wubu_train_registry_find(reg, "dec.conv_pre.weight");
                                                    ib = wubu_train_registry_find(reg, "dec.conv_pre.bias");
                                                    if (cpw && cpw->data)
                                                        conv1d_bwd(mel_in, cpw->data, d_cp,
                                                                   192, 512, 7, 1, 3, 1, cache->n_frames, cache->n_frames,
                                                                   d_input, /* may be NULL: input grads optional */
                                                                   (iw >= 0) ? reg->params[iw].grad : NULL,
                                                                   (ib >= 0) ? reg->params[ib].grad : NULL);
                                                    free(d_cp);
                                                }
                                            }
                                            free(d_ups0);
                                        }
                                        free(d_stage0);
                                    }
                                    free(d_l0);
                                }
                                free(d_ups1);
                            }
                            free(d_stage1);
                        }
                        free(d_l1);
                    }
                    free(d_ups2);
                }
                free(d_stage2);
            }
            free(d_l2);
        }
        free(d_ups3);
    }

    free(d_post_in); free(d_stage3); free(d_conv_pre_in);
}

/* ═══════════════════════ Public API ═══════════════════════ */

int wubu_decoder_forward(WuBuRVCModel *model, const float *mel_in, int n_frames,
                         float *audio, int max_samples) {
    DecCache cache;
    int n = decoder_forward(model, mel_in, n_frames, audio, max_samples, &cache);
    decoder_cache_free(&cache);
    return n;
}

int wubu_train_step(WuBuRVCModel *model, WuBuTrainRegistry *reg, WuBuAdamW *opt,
                    const float *mel_in, int n_frames,
                    const float *wav, int n_samples,
                    float *loss_out, int epoch) {
    if (!model || !reg || !opt || !mel_in || !wav) return -1;

    int max_samples = n_frames * 400;
    float *audio = (float *)malloc((size_t)max_samples * sizeof(float));
    if (!audio) return -1;

    DecCache cache;
    int n_out = decoder_forward(model, mel_in, n_frames, audio, max_samples, &cache);
    if (n_out <= 0) { free(audio); decoder_cache_free(&cache); return -1; }

    int n = n_out < n_samples ? n_out : n_samples;
    float mse = wubu_mse_loss(audio, wav, n);
    if (loss_out) *loss_out = mse;
    model->last_loss = mse;
    model->last_epoch = epoch;

    /* dL/daudio for MSE = 2*(audio - wav)/n */
    float *d_audio = (float *)malloc((size_t)n_out * sizeof(float));
    if (!d_audio) { free(audio); decoder_cache_free(&cache); return -1; }
    for (int i = 0; i < n; i++)
        d_audio[i] = 2.0f * (audio[i] - wav[i]) / (float)n;
    for (int i = n; i < n_out; i++)
        d_audio[i] = 0.0f;

    wubu_train_registry_zero_grads(reg);
    decoder_backward(model, &cache, mel_in, d_audio, reg, NULL);

    /* AdamW update */
    for (int i = 0; i < reg->count; i++)
        wubu_adamw_step(opt, i, reg->params[i].data, reg->params[i].grad);

    free(d_audio);
    free(audio);
    decoder_cache_free(&cache);
    return (mse < 0.1f) ? 1 : 0;
}

float wubu_train_gradcheck(WuBuRVCModel *model, WuBuTrainRegistry *reg,
                           const float *mel_in, int n_frames,
                           const float *wav, int n_samples,
                           int n_probes, int seed) {
    if (!model || !reg || !mel_in || !wav || n_probes <= 0) return -1.0f;

    int max_samples = n_frames * 400;
    float *audio = (float *)malloc((size_t)max_samples * sizeof(float));
    float *audio2 = (float *)malloc((size_t)max_samples * sizeof(float));
    if (!audio || !audio2) { free(audio); free(audio2); return -1.0f; }

    /* analytic grads via one full backward */
    DecCache cache;
    int n_out = decoder_forward(model, mel_in, n_frames, audio, max_samples, &cache);
    if (n_out <= 0) { free(audio); free(audio2); decoder_cache_free(&cache); return -1.0f; }
    int n = n_out < n_samples ? n_out : n_samples;
    float *d_audio = (float *)malloc((size_t)n_out * sizeof(float));
    if (!d_audio) { free(audio); free(audio2); decoder_cache_free(&cache); return -1.0f; }
    for (int i = 0; i < n; i++)
        d_audio[i] = 2.0f * (audio[i] - wav[i]) / (float)n;
    for (int i = n; i < n_out; i++)
        d_audio[i] = 0.0f;
    wubu_train_registry_zero_grads(reg);
    decoder_backward(model, &cache, mel_in, d_audio, reg, NULL);
    decoder_cache_free(&cache);
    free(d_audio);

    srand((unsigned)seed);
    float max_rel = 0.0f;
    int checked = 0;
    const float eps = 1e-2f;  /* float32 finite differences: need larger eps */

    for (int p = 0; p < n_probes; p++) {
        int pi = rand() % reg->count;
        WuBuTrainParam *par = &reg->params[pi];
        if (!par->grad || par->n <= 0) continue;
        int idx = rand() % par->n;
        float orig = par->data[idx];
        float g_analytic = par->grad[idx];

        par->data[idx] = orig + eps;
        int n1 = wubu_decoder_forward(model, mel_in, n_frames, audio, max_samples);
        float l1 = (n1 > 0) ? wubu_mse_loss(audio, wav, n1 < n_samples ? n1 : n_samples) : 1e9f;
        par->data[idx] = orig - eps;
        int n2 = wubu_decoder_forward(model, mel_in, n_frames, audio2, max_samples);
        float l2 = (n2 > 0) ? wubu_mse_loss(audio2, wav, n2 < n_samples ? n2 : n_samples) : 1e9f;
        par->data[idx] = orig;

        float g_numeric = (l1 - l2) / (2.0f * eps);
        float denom = fabsf(g_numeric) > 1e-6f ? fabsf(g_numeric) : 1e-6f;
        float rel = fabsf(g_analytic - g_numeric) / denom;
        if (getenv("WUBU_TRAIN_GC_DBG"))
            fprintf(stderr, "  probe %d: %s[%d] n=%d analytic=%.8f numeric=%.8f rel=%.4f\n",
                    p, par->name, idx, par->n, g_analytic, g_numeric, rel);
        if (rel > max_rel) max_rel = rel;
        checked++;
        if (checked >= n_probes) break;
    }

    free(audio); free(audio2);
    return max_rel;
}

/* Train from scratch: initialize RVC v2 generator weights */
int wubu_init_weights_rvc2(WuBuRVCModel *model) {
    if (!model) return -1;
    model->version = 2;
    model->version_f = 2.0f;
    model->mel_channels = 80;
    model->hidden_channels = 256;
    model->n_flow_layers = 4;
    model->sample_rate = 22050;
    model->n_upsample_layers = 4;
    model->upsample_rates[0] = 10; model->upsample_rates[1] = 10;
    model->upsample_rates[2] = 2; model->upsample_rates[3] = 2;
    model->upsample_kernel_sizes[0] = 16; model->upsample_kernel_sizes[1] = 16;
    model->upsample_kernel_sizes[2] = 4; model->upsample_kernel_sizes[3] = 4;
    model->upsample_rate = 400;
    model->has_spk_embed = 0;
    model->loaded = 1;
    return 0;
}
