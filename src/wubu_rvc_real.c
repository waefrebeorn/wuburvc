/* wubu_rvc_real.c — WuBuRVC REAL synthesis pipeline (v1 + v2).
 *
 * Faithful C11 port of Mangio-RVC-Fork lib/infer_pack/models.py
 * SynthesizerTrnMs768NSFsid.infer / SynthesizerTrnMs256NSFsid.infer:
 *   enc_p (TextEncoder) → z_p sampling → flow reverse (ResidualCouplingBlock)
 *   → GeneratorNSF (HiFi-GAN + f0 sine excitation).
 *
 * Layout convention: [channels, time] column-major (PyTorch (B,C,T), B=1),
 * matching the existing wubu_rvc_kernels_exact.c kernels.
 *
 * License: WaefreBeorn-UMV3
 */

#define _USE_MATH_DEFINES
#include "wubu_rvc_real.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Uniform random in [lo, hi) — used for NSF noise injection */
static inline float wubu_rand_uniform(float lo, float hi) {
    return lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}

/* ═══════════════════════ basic tensor helpers ═══════════════════════*/

static const RVCTensor *T(const WuBuRVCModel *m, const char *name) {
    return wubu_rvc_find_tensor(m, name);
}

/* weight_norm de-normalization: W = g * (v / ||v||) per channel. */
int wubu_denorm_tensor(const RVCTensor *g, const RVCTensor *v,
                       float *out, int n_elements) {
    if (!g || !v || !out || !g->data || !v->data || n_elements <= 0) return -1;
    int n_ch = g->dims[0];
    if (n_ch <= 0) return -1;
    int per_ch = n_elements / n_ch;
    for (int ch = 0; ch < n_ch; ch++) {
        float norm_sq = 0.0f;
        const float *vc = v->data + (size_t)ch * per_ch;
        for (int i = 0; i < per_ch; i++) norm_sq += vc[i] * vc[i];
        float scale = g->data[ch] / (sqrtf(norm_sq) + 1e-8f);
        float *oc = out + (size_t)ch * per_ch;
        for (int i = 0; i < per_ch; i++) oc[i] = vc[i] * scale;
    }
    return 0;
}

static float *denorm_cache(const WuBuRVCModel *m, const char *base,
                           float **cache, int *cache_len, int *ok) {
    /* Resolve weight_norm pair "<base>.weight_g" + "<base>.weight_v" into a
     * cached de-normalized array. Returns NULL if either tensor missing. */
    if (*cache) return *cache;
    char gn[256], vn[256];
    snprintf(gn, sizeof(gn), "%s.weight_g", base);
    snprintf(vn, sizeof(vn), "%s.weight_v", base);
    const RVCTensor *g = T(m, gn);
    const RVCTensor *v = T(m, vn);
    if (!g || !v || !g->data || !v->data) { *ok = 0; return NULL; }
    int n = 1;
    for (int d = 0; d < v->n_dims; d++) n *= v->dims[d];
    float *out = (float *)malloc((size_t)n * sizeof(float));
    if (!out) { *ok = 0; return NULL; }
    if (wubu_denorm_tensor(g, v, out, n) != 0) { free(out); *ok = 0; return NULL; }
    *cache = out;
    *cache_len = n;
    return out;
}

/* ═══════════════════════ conv / linear primitives ═══════════════════════
 * All operate on [C, T] column-major float buffers, single batch.
 */

/* Conv1d, PyTorch weight (out_ch, in_ch, k). out length: (n + 2p - d*(k-1) - 1)/s + 1
 * Loop order: oc → ic → tap → j (j innermost). This makes both `in` (stride s)
 * and `out` sequential per row → cache-friendly, auto-vectorizable. The old
 * oc → j → tap → ic order touched in_ch rows of 11KB+ each per output sample
 * (thousands of cache misses per position); for T=111k × 512ch that was the
 * dominant cost of the whole engine. */
static void conv1d_c(const float *in, int in_ch, int n,
                     const float *w, const float *b,
                     int out_ch, int k, int stride, int pad, int dil,
                     float *out) {
    int n_out = (n + 2 * pad - dil * (k - 1) - 1) / stride + 1;
    if (n_out <= 0) return;
    memset(out, 0, (size_t)out_ch * n_out * sizeof(float));
#pragma omp parallel for schedule(static) if(out_ch >= 32 && n_out >= 256)
    for (int oc = 0; oc < out_ch; oc++) {
        float *orow = out + (size_t)oc * n_out;
        float bias = b ? b[oc] : 0.0f;
        for (int j = 0; j < n_out; j++) orow[j] = bias;
        if (!w) continue;
        const float *wv = w + (size_t)oc * in_ch * k;
        for (int ic = 0; ic < in_ch; ic++) {
            const float *irow = in + (size_t)ic * n;
            for (int tap = 0; tap < k; tap++) {
                int off = tap * dil - pad;
                float wt = wv[(size_t)ic * k + tap];
                /* exact valid j range (src = j*stride + off in [0, n)):
                 * no branch inside the inner loop → vectorizes cleanly. */
                int j_lo = 0, j_hi = n_out;
                if (stride == 1) {
                    if (off < 0) j_lo = -off;
                    if (n - off < j_hi) j_hi = n - off;
                } else {
                    if (off < 0) j_lo = (-off + stride - 1) / stride;
                    if (n - off < 0) j_hi = 0;
                    else if ((n - off + stride - 1) / stride < j_hi)
                        j_hi = (n - off + stride - 1) / stride;
                }
                for (int j = j_lo; j < j_hi; j++)
                    orow[j] += irow[j * stride + off] * wt;
            }
        }
    }
}

/* ConvTranspose1d, PyTorch weight (in_ch, out_ch, k). out: (n-1)*s - 2p + k */
static void conv_transpose1d_c(const float *in, int in_ch, int n,
                               const float *w, const float *b,
                               int out_ch, int k, int stride, int pad,
                               float *out) {
    int n_out = (n - 1) * stride - 2 * pad + k;
    if (n_out <= 0) return;
    memset(out, 0, (size_t)out_ch * n_out * sizeof(float));
    /* oc → ic → i → tap: out row sequential, in row sequential per (oc,ic).
     * (Old i → ic → oc → tap scattered writes across out_ch rows.) */
#pragma omp parallel for schedule(static) if(out_ch >= 16 && n >= 512)
    for (int oc = 0; oc < out_ch; oc++) {
        float *orow = out + (size_t)oc * n_out;
        for (int ic = 0; ic < in_ch; ic++) {
            const float *irow = in + (size_t)ic * n;
            if (!w) continue;
            const float *wk = w + ((size_t)ic * out_ch + (size_t)oc) * k;
            for (int i = 0; i < n; i++) {
                float inp = irow[i];
                if (inp == 0.0f) continue;
                int j0 = i * stride - pad;
                for (int tap = 0; tap < k; tap++) {
                    int j = j0 + tap;
                    if (j >= 0 && j < n_out) orow[j] += inp * wk[tap];
                }
            }
        }
    }
    if (b) {
        for (int oc = 0; oc < out_ch; oc++) {
            float *orow = out + (size_t)oc * n_out;
            for (int j = 0; j < n_out; j++) orow[j] += b[oc];
        }
    }
}

/* Linear (out, in): y = W x + b, applied to [in, T] → [out, T] */
static void linear_c(const float *in, int in_d, int n,
                     const float *w, const float *b, int out_d, float *out) {
    for (int o = 0; o < out_d; o++) {
        const float *wv = w + (size_t)o * in_d;
        float bias = b ? b[o] : 0.0f;
        float *orow = out + (size_t)o * n;
        for (int j = 0; j < n; j++) orow[j] = bias;
        /* i outer, j inner: in[i*n+j] sequential per row → cache-friendly,
         * vectorizable. (Old j-outer/i-inner touched in_d rows per output.) */
        for (int i = 0; i < in_d; i++) {
            float wt = wv[i];
            const float *irow = in + (size_t)i * n;
            for (int j = 0; j < n; j++) orow[j] += irow[j] * wt;
        }
    }
}

/* Embedding (num, dim): gather rows by int index, out [dim, T] */
static void embedding_c(const float *emb, int num, int dim,
                        const int *idx, int n, float *out) {
    for (int j = 0; j < n; j++) {
        int id = idx[j];
        if (id < 0) id = 0;
        if (id >= num) id = num - 1;
        const float *row = emb + (size_t)id * dim;
        for (int d = 0; d < dim; d++) out[(size_t)d * n + j] = row[d];
    }
}

static void lrelu_c(float *x, size_t n, float slope) {
    for (size_t i = 0; i < n; i++) x[i] = x[i] > 0 ? x[i] : slope * x[i];
}

/* Snake activation (BigVGAN): f(x) = x + (1/a) * sin^2(a*x).
 * NOTE (2026-08-07, crash recovery): this is f(x) ALONE. Pretrained RVC
 * weights were trained with LeakyReLU(slope=0.1), whose negative compression
 * keeps residual-stack activations bounded. Snake (~identity pass-through)
 * removes that compression, so 12 MRF residual pairs compound and the final
 * tanh saturates to a pure ±1.0 square wave (verified: sat_frac=1.000).
 * The engine therefore detects saturation in wubu_rvc_synthesize_real and the
 * CLI falls back to LeakyReLU (parity-verified SNR 29.79 dB) with a warning.
 * Snake stays available for models actually TRAINED with it (BigVGAN). */
static inline float snake_c(float x, float a) {
    return x + (1.0f / a) * (sinf(a * x) * sinf(a * x));
}
static void snake_lrelu_c(float *x, size_t n, float slope) {
    (void)slope;  /* BigVGAN Snake has no LReLU term */
    for (size_t i = 0; i < n; i++) {
        x[i] = snake_c(x[i], 1.0f);
    }
}

/* GELU as used by Mangio-RVC lib/infer_pack/attentions.py FFN:
 * x * sigmoid(1.702 * x). This is a distinct tanh-family form — the
 * text encoder was trained with it, so match it EXACTLY. */
static float sigmoid_c(float x) { return 1.0f / (1.0f + expf(-x)); }
static float gelu_c(float x) { return x * sigmoid_c(1.702f * x); }

/* LayerNorm over channels (gamma/beta per channel), [C, T] */
static void layernorm_c(float *x, int C, int T,
                        const float *gamma, const float *beta, float eps) {
    for (int j = 0; j < T; j++) {
        float mean = 0, var = 0;
        for (int c = 0; c < C; c++) mean += x[(size_t)c * T + j];
        mean /= C;
        for (int c = 0; c < C; c++) { float d = x[(size_t)c * T + j] - mean; var += d * d; }
        var /= C;
        float inv = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < C; c++)
            x[(size_t)c * T + j] = (x[(size_t)c * T + j] - mean) * inv * (gamma ? gamma[c] : 1.0f) + (beta ? beta[c] : 0.0f);
    }
}

/* ═══════════════════════ MultiHeadAttention (RVC attentions.py) ═══════════════════════
 * window_size=10 relative position bias, 2 heads, k_channels = C/2 = 96.
 * x, c: [C, T]. attn_mask: [T, T] (1.0 or 0.0). out: [C, T].
 */
typedef struct {
    const float *wq, *bq, *wk, *bk, *wv, *bv, *wo, *bo; /* conv1d k=1 */
    const float *emb_rel_k, *emb_rel_v;                  /* [1, 2*win+1, kch] */
    int C, n_heads, window;
} MHA;

/* The mha stub above is replaced by mha_forward_t below; keep only the real one. */
static void mha_forward_t(const MHA *mha, const float *x, const float *c,
                          const float *attn_mask, int T,
                          float *q, float *k, float *v, float *scores,
                          float *p_attn, float *out) {
    int C = mha->C, H = mha->n_heads, kch = C / H, win = mha->window;
    /* q,k,v = conv1d k=1 over x (self-attn: c == x) */
    conv1d_c(x, C, T, mha->wq, mha->bq, C, 1, 1, 0, 1, q);
    conv1d_c(c, C, T, mha->wk, mha->bk, C, 1, 1, 0, 1, k);
    conv1d_c(c, C, T, mha->wv, mha->bv, C, 1, 1, 0, 1, v);

    /* scores[h][ti][tj] = sum_d q[h,d,ti]*k[h,d,tj] / sqrt(kch) */
#pragma omp parallel for schedule(static)
    for (int h = 0; h < H; h++) {
        float *srow = scores + (size_t)h * T * T;
        for (int ti = 0; ti < T; ti++) {
            for (int tj = 0; tj < T; tj++) {
                float acc = 0;
                for (int d = 0; d < kch; d++) {
                    acc += q[(size_t)(h * kch + d) * T + ti] *
                           k[(size_t)(h * kch + d) * T + tj];
                }
                srow[(size_t)ti * T + tj] = acc / sqrtf((float)kch);
            }
        }
    }

    /* relative position bias (emb_rel_k, window=10): scores_local[t][s].
     * PyTorch: rel_logits = matmul(q/sqrt(kch), emb_used), then
     * _relative_position_to_absolute_position maps rel_logits[i, l-1+j-i]
     * into scores_local[i,j]. emb_used is zero-padded outside the window
     * (see _get_relative_embeddings), so |j-i| > win contributes NOTHING —
     * do not clamp to the window edge (that was the m_p bug). */
    if (mha->emb_rel_k) {
#pragma omp parallel for schedule(static)
        for (int h = 0; h < H; h++) {
            float *srow = scores + (size_t)h * T * T;
            for (int ti = 0; ti < T; ti++) {
                for (int tj = 0; tj < T; tj++) {
                    int m = tj - ti + win;
                    if (m < 0 || m > 2 * win) continue;
                    float acc = 0;
                    for (int d = 0; d < kch; d++) {
                        acc += q[(size_t)(h * kch + d) * T + ti] *
                               mha->emb_rel_k[(size_t)m * kch + d];
                    }
                    srow[(size_t)ti * T + tj] += acc / sqrtf((float)kch);
                }
            }
        }
    }

    /* mask fill (-1e4) + softmax over tj */
#pragma omp parallel for schedule(static)
    for (int h = 0; h < H; h++) {
        float *srow = scores + (size_t)h * T * T;
        for (int ti = 0; ti < T; ti++) {
            float mx = -1e30f;
            for (int tj = 0; tj < T; tj++) {
                if (attn_mask && attn_mask[(size_t)ti * T + tj] == 0.0f)
                    srow[(size_t)ti * T + tj] = -1e4f;
                if (srow[(size_t)ti * T + tj] > mx) mx = srow[(size_t)ti * T + tj];
            }
            float sum = 0;
            for (int tj = 0; tj < T; tj++) {
                float e = expf(srow[(size_t)ti * T + tj] - mx);
                srow[(size_t)ti * T + tj] = e;
                sum += e;
            }
            for (int tj = 0; tj < T; tj++) srow[(size_t)ti * T + tj] /= (sum + 1e-12f);
        }
    }

    /* out[h,d,ti] = sum_tj p_attn[h,ti,tj] * v[h,d,tj]  (+ relative values) */
#pragma omp parallel for schedule(static)
    for (int h = 0; h < H; h++) {
        const float *srow = scores + (size_t)h * T * T;
        for (int ti = 0; ti < T; ti++) {
            for (int d = 0; d < kch; d++) {
                float acc = 0;
                for (int tj = 0; tj < T; tj++)
                    acc += srow[(size_t)ti * T + tj] * v[(size_t)(h * kch + d) * T + tj];
                out[(size_t)(h * kch + d) * T + ti] = acc;
            }
        }
    }

    /* relative values: output += sum_tj p_attn_rel * emb_rel_v (skip: RVC
     * forward uses it, but its contribution is a learned refinement; include
     * for full parity) */
    if (mha->emb_rel_v) {
        /* absolute -> relative attention weights, then matmul with emb_rel_v.
         * Same zero-outside-window rule as emb_rel_k: |j-i| > win → nothing. */
#pragma omp parallel for schedule(static)
        for (int h = 0; h < H; h++) {
            const float *srow = scores + (size_t)h * T * T;
            for (int d = 0; d < kch; d++) {
                for (int ti = 0; ti < T; ti++) {
                    float acc = 0;
                    for (int tj = 0; tj < T; tj++) {
                        int m = tj - ti + win;
                        if (m < 0 || m > 2 * win) continue;
                        acc += srow[(size_t)ti * T + tj] * mha->emb_rel_v[(size_t)m * kch + d];
                    }
                    out[(size_t)(h * kch + d) * T + ti] += acc;
                }
            }
        }
    }

    /* conv_o */
    float *attn_out = p_attn;
    memcpy(attn_out, out, (size_t)C * T * sizeof(float));
    conv1d_c(attn_out, C, T, mha->wo, mha->bo, C, 1, 1, 0, 1, out);
}

/* ═══════════════════════ FFN (RVC attentions.py FFN) ═══════════════════════
 * conv_1 (C->filter, k=3, same), ReLU (activation is None by default in the
 * Encoder — "gelu" would be x*sigmoid(1.702x), but the checkpoint was trained
 * with the default ReLU), conv_2 (filter->C, k=3, same) */
static void ffn_forward(const float *x, int C, int T, int filter_ch,
                        const float *w1, const float *b1,
                        const float *w2, const float *b2,
                        float *h, float *out) {
    conv1d_c(x, C, T, w1, b1, filter_ch, 3, 1, 1, 1, h);
    for (int i = 0; i < filter_ch * T; i++) h[i] = h[i] > 0 ? h[i] : 0.0f; /* ReLU */
    conv1d_c(h, filter_ch, T, w2, b2, C, 3, 1, 1, 1, out);
}

/* ═══════════════════════ TextEncoder (enc_p) ═══════════════════════
 * phone [content_dim, T], pitch [T] int, out m/logs [inter, T]. */
int wubu_enc_p_forward(WuBuRVCModel *model,
                       const float *phone, int n_frames, int content_dim,
                       const int *pitch,
                       float *m_out, float *logs_out, float *x_mask_out) {
    if (!model || !phone || !m_out || !logs_out || n_frames < 1) return -1;

    const RVCTensor *emb_w = T(model, "enc_p.emb_phone.weight");
    const RVCTensor *emb_b = T(model, "enc_p.emb_phone.bias");
    const RVCTensor *pitch_w = T(model, "enc_p.emb_pitch.weight");
    const RVCTensor *proj_w = T(model, "enc_p.proj.weight");
    const RVCTensor *proj_b = T(model, "enc_p.proj.bias");
    if (!emb_w || !emb_b || !pitch_w || !proj_w || !proj_b) return -1;

    int hidden = emb_w->dims[0];
    int inter = proj_w->dims[0] / 2;
    int nF = n_frames;

    /* x = emb_phone(phone) + emb_pitch(pitch); x *= sqrt(hidden); lrelu */
    float *x = (float *)malloc((size_t)hidden * nF * sizeof(float));
    float *xp = (float *)malloc((size_t)hidden * nF * sizeof(float));
    float *tmp = (float *)malloc((size_t)hidden * nF * sizeof(float));
    float *y = (float *)malloc((size_t)hidden * nF * sizeof(float));
    if (!x || !xp || !tmp || !y) { free(x); free(xp); free(tmp); free(y); return -1; }

    linear_c(phone, content_dim, nF, emb_w->data, emb_b->data, hidden, x);
    embedding_c(pitch_w->data, 256, hidden, pitch, nF, xp);
    for (int i = 0; i < hidden * nF; i++) {
        x[i] = (x[i] + xp[i]) * sqrtf((float)hidden);
    }
    lrelu_c(x, (size_t)hidden * nF, 0.1f);

    /* x_mask: all ones (full length) */
    if (x_mask_out)
        for (int j = 0; j < nF; j++) x_mask_out[j] = 1.0f;

    /* Encoder: n_layers (6) of MHA + LN1 + FFN + LN2 */
    char key[256];
    int n_layers = 0;
    for (int i = 0; i < 16; i++) {
        snprintf(key, sizeof(key), "enc_p.encoder.attn_layers.%d.conv_q.weight", i);
        if (T(model, key)) n_layers++; else break;
    }
    int n_heads = 2;
    int window = 10;
    int filter_ch = 768; /* from ffn conv_1 out channels */
    snprintf(key, sizeof(key), "enc_p.encoder.ffn_layers.0.conv_1.weight");
    const RVCTensor *ffn0 = T(model, key);
    if (ffn0) filter_ch = ffn0->dims[0];

    float *attn_mask = (float *)malloc((size_t)nF * nF * sizeof(float));
    float *q = (float *)malloc((size_t)hidden * nF * sizeof(float));
    float *k = (float *)malloc((size_t)hidden * nF * sizeof(float));
    float *v = (float *)malloc((size_t)hidden * nF * sizeof(float));
    float *scores = (float *)malloc((size_t)n_heads * nF * nF * sizeof(float));
    /* FFN needs a filter_channels-sized intermediate (768*nF), NOT hidden*nF */
    float *ffn_h = (float *)malloc((size_t)filter_ch * nF * sizeof(float));
    if (!attn_mask || !q || !k || !v || !scores || !ffn_h) {
        free(x); free(xp); free(tmp); free(y); free(attn_mask); free(q); free(k); free(v); free(scores); free(ffn_h);
        return -1;
    }
    for (int ti = 0; ti < nF; ti++)
        for (int tj = 0; tj < nF; tj++) attn_mask[(size_t)ti * nF + tj] = 1.0f;

    for (int L = 0; L < n_layers; L++) {
        MHA mha;
        memset(&mha, 0, sizeof(mha));
        snprintf(key, sizeof(key), "enc_p.encoder.attn_layers.%d.conv_q.weight", L);
        const RVCTensor *wq = T(model, key);
        snprintf(key, sizeof(key), "enc_p.encoder.attn_layers.%d.conv_q.bias", L);
        const RVCTensor *bq = T(model, key);
        snprintf(key, sizeof(key), "enc_p.encoder.attn_layers.%d.conv_k.weight", L);
        const RVCTensor *wk = T(model, key);
        snprintf(key, sizeof(key), "enc_p.encoder.attn_layers.%d.conv_k.bias", L);
        const RVCTensor *bk = T(model, key);
        snprintf(key, sizeof(key), "enc_p.encoder.attn_layers.%d.conv_v.weight", L);
        const RVCTensor *wv = T(model, key);
        snprintf(key, sizeof(key), "enc_p.encoder.attn_layers.%d.conv_v.bias", L);
        const RVCTensor *bv = T(model, key);
        snprintf(key, sizeof(key), "enc_p.encoder.attn_layers.%d.conv_o.weight", L);
        const RVCTensor *wo = T(model, key);
        snprintf(key, sizeof(key), "enc_p.encoder.attn_layers.%d.conv_o.bias", L);
        const RVCTensor *bo = T(model, key);
        snprintf(key, sizeof(key), "enc_p.encoder.attn_layers.%d.emb_rel_k", L);
        const RVCTensor *er_k = T(model, key);
        snprintf(key, sizeof(key), "enc_p.encoder.attn_layers.%d.emb_rel_v", L);
        const RVCTensor *er_v = T(model, key);
        if (!wq || !wk || !wv || !wo) { free(x); free(xp); free(tmp); free(y); free(attn_mask); free(q); free(k); free(v); free(scores); free(ffn_h); return -1; }
        mha.wq = wq->data; mha.bq = bq && bq->data ? bq->data : NULL;
        mha.wk = wk->data; mha.bk = bk && bk->data ? bk->data : NULL;
        mha.wv = wv->data; mha.bv = bv && bv->data ? bv->data : NULL;
        mha.wo = wo->data; mha.bo = bo && bo->data ? bo->data : NULL;
        mha.emb_rel_k = er_k && er_k->data ? er_k->data : NULL;
        mha.emb_rel_v = er_v && er_v->data ? er_v->data : NULL;
        mha.C = hidden; mha.n_heads = n_heads; mha.window = window;

        /* mha_forward_t(out) returns the conv_o result in its LAST arg.
         * Call with (p_attn=tmp, out=y) so the final MHA output lands in y
         * (the residual buffer) — passing (y, tmp) would add the PRE-conv_o
         * attention to x, which scrambles every channel. */
        mha_forward_t(&mha, x, x, attn_mask, nF, q, k, v, scores, tmp, y);

        if (getenv("WUBU_RVC_DUMP")) {
            char fn[128];
            snprintf(fn, sizeof(fn), "outputs/rvc_ref/c_enc_layer%d_mha.npy", L);
            FILE *df = fopen(fn, "wb");
            if (df) { fwrite(y, sizeof(float), (size_t)hidden * nF, df); fclose(df); }
            snprintf(fn, sizeof(fn), "outputs/rvc_ref/c_enc_layer%d_attn.npy", L);
            df = fopen(fn, "wb");
            if (df) { fwrite(scores, sizeof(float), (size_t)n_heads * nF * nF, df); fclose(df); }
            snprintf(fn, sizeof(fn), "outputs/rvc_ref/c_enc_layer%d_q.npy", L);
            df = fopen(fn, "wb");
            if (df) { fwrite(q, sizeof(float), (size_t)hidden * nF, df); fclose(df); }
            snprintf(fn, sizeof(fn), "outputs/rvc_ref/c_enc_layer%d_v.npy", L);
            df = fopen(fn, "wb");
            if (df) { fwrite(v, sizeof(float), (size_t)hidden * nF, df); fclose(df); }
            snprintf(fn, sizeof(fn), "outputs/rvc_ref/c_enc_layer%d_pre.npy", L);
            df = fopen(fn, "wb");
            if (df) { fwrite(tmp, sizeof(float), (size_t)hidden * nF, df); fclose(df); }
        }

        /* x = LayerNorm1(x + y) */
        for (int i = 0; i < hidden * nF; i++) x[i] += y[i];
        snprintf(key, sizeof(key), "enc_p.encoder.norm_layers_1.%d.gamma", L);
        const RVCTensor *n1g = T(model, key);
        snprintf(key, sizeof(key), "enc_p.encoder.norm_layers_1.%d.beta", L);
        const RVCTensor *n1b = T(model, key);
        layernorm_c(x, hidden, nF, n1g && n1g->data ? n1g->data : NULL,
                    n1b && n1b->data ? n1b->data : NULL, 1e-5f);

        /* y = FFN(x); x = LayerNorm2(x + y) */
        snprintf(key, sizeof(key), "enc_p.encoder.ffn_layers.%d.conv_1.weight", L);
        const RVCTensor *f1w = T(model, key);
        snprintf(key, sizeof(key), "enc_p.encoder.ffn_layers.%d.conv_1.bias", L);
        const RVCTensor *f1b = T(model, key);
        snprintf(key, sizeof(key), "enc_p.encoder.ffn_layers.%d.conv_2.weight", L);
        const RVCTensor *f2w = T(model, key);
        snprintf(key, sizeof(key), "enc_p.encoder.ffn_layers.%d.conv_2.bias", L);
        const RVCTensor *f2b = T(model, key);
        if (!f1w || !f2w) { free(x); free(xp); free(tmp); free(y); free(attn_mask); free(q); free(k); free(v); free(scores); free(ffn_h); return -1; }
        ffn_forward(x, hidden, nF, filter_ch, f1w->data, f1b ? f1b->data : NULL,
                    f2w->data, f2b ? f2b->data : NULL, ffn_h, y);
        for (int i = 0; i < hidden * nF; i++) x[i] += y[i];
        snprintf(key, sizeof(key), "enc_p.encoder.norm_layers_2.%d.gamma", L);
        const RVCTensor *n2g = T(model, key);
        snprintf(key, sizeof(key), "enc_p.encoder.norm_layers_2.%d.beta", L);
        const RVCTensor *n2b = T(model, key);
        layernorm_c(x, hidden, nF, n2g && n2g->data ? n2g->data : NULL,
                    n2b && n2b->data ? n2b->data : NULL, 1e-5f);
    }

    /* proj: Conv1d(hidden -> inter*2, k=1) -> m, logs */
    float *stats = (float *)malloc((size_t)(2 * inter) * nF * sizeof(float));
    if (!stats) { free(x); free(xp); free(tmp); free(y); free(attn_mask); free(q); free(k); free(v); free(scores); free(ffn_h); return -1; }

    if (getenv("WUBU_RVC_DUMP")) {
        FILE *df = fopen("outputs/rvc_ref/c_phone_in.npy", "wb");
        if (df) { fwrite(phone, sizeof(float), (size_t)content_dim * nF, df); fclose(df); }
        df = fopen("outputs/rvc_ref/c_enc_x.npy", "wb");
        if (df) { fwrite(x, sizeof(float), (size_t)hidden * nF, df); fclose(df); }
        df = fopen("outputs/rvc_ref/c_enc_x_input.npy", "wb");
        if (df) {
            /* redo emb + lrelu into a temp to dump the encoder INPUT */
            float *xi = (float *)malloc((size_t)hidden * nF * sizeof(float));
            if (xi) {
                linear_c(phone, content_dim, nF, emb_w->data, emb_b->data, hidden, xi);
                float *xpi = (float *)malloc((size_t)hidden * nF * sizeof(float));
                if (xpi) {
                    embedding_c(pitch_w->data, 256, hidden, pitch, nF, xpi);
                    for (int i = 0; i < hidden * nF; i++)
                        xi[i] = (xi[i] + xpi[i]) * sqrtf((float)hidden);
                    lrelu_c(xi, (size_t)hidden * nF, 0.1f);
                    fwrite(xi, sizeof(float), (size_t)hidden * nF, df);
                    free(xpi);
                }
                free(xi);
            }
        }
        fclose(df);
    }

    conv1d_c(x, hidden, nF, proj_w->data, proj_b->data, 2 * inter, 1, 1, 0, 1, stats);
    for (int j = 0; j < nF; j++) {
        for (int c = 0; c < inter; c++) {
            m_out[(size_t)c * nF + j] = stats[(size_t)c * nF + j];
            logs_out[(size_t)c * nF + j] = stats[(size_t)(inter + c) * nF + j];
        }
    }

    free(stats);
    free(x); free(xp); free(tmp); free(y); free(attn_mask); free(q); free(k); free(v); free(scores); free(ffn_h);
    return 0;
}

/* ═══════════════════════ WN (flow enc) ═══════════════════════
 * hidden=192, k=5, dilation=1, n_layers=3, gin=256 (cond). */
typedef struct {
    WuBuRVCModel *model;
    int hidden, n_layers, gin;
    const RVCTensor *cond_w, *cond_b;
    const RVCTensor *in_wg[4], *in_wv[4], *in_b[4];
    const RVCTensor *rs_wg[4], *rs_wv[4], *rs_b[4];
    float *in_denorm[4];
    float *rs_denorm[4];
    int in_denorm_len[4], rs_denorm_len[4];
    int flow_idx; /* 0,2,4,6 */
} WNBlock;

static int wn_init(WNBlock *wn, WuBuRVCModel *model, int flow_idx) {
    memset(wn, 0, sizeof(*wn));
    wn->model = model;
    wn->flow_idx = flow_idx;
    char key[256];
    snprintf(key, sizeof(key), "flow.flows.%d.enc.cond_layer", flow_idx);
    char gn[256], vn[256];
    snprintf(gn, sizeof(gn), "%s.weight_g", key);
    snprintf(vn, sizeof(vn), "%s.weight_v", key);
    wn->cond_w = wubu_rvc_find_tensor(model, gn);
    wn->cond_b = wubu_rvc_find_tensor(model, key);
    /* cond_b tensor name is "flow.flows.N.enc.cond_layer.bias" — find by base */
    if (!wn->cond_b) {
        char bn[256]; snprintf(bn, sizeof(bn), "%s.bias", key);
        wn->cond_b = wubu_rvc_find_tensor(model, bn);
    }
    if (!wn->cond_w) return -1;
    /* cond_w weight_g: (2*hidden*n_layers, 1, 1); weight_v: (2*hidden*n_layers, gin, 1) */
    char vn2[256];
    snprintf(vn2, sizeof(vn2), "flow.flows.%d.enc.cond_layer.weight_v", flow_idx);
    const RVCTensor *cond_v = wubu_rvc_find_tensor(model, vn2);
    if (!cond_v) return -1;
    wn->gin = cond_v->dims[1];
    wn->hidden = wn->cond_w->dims[0] / 6; /* 2*hidden*n_layers = 2*192*3 = 1152 */
    wn->n_layers = 3;
    for (int i = 0; i < 3; i++) {
        snprintf(key, sizeof(key), "flow.flows.%d.enc.in_layers.%d", flow_idx, i);
        snprintf(gn, sizeof(gn), "%s.weight_g", key);
        snprintf(vn, sizeof(vn), "%s.weight_v", key);
        wn->in_wg[i] = wubu_rvc_find_tensor(model, gn);
        wn->in_wv[i] = wubu_rvc_find_tensor(model, vn);
        char bn[256]; snprintf(bn, sizeof(bn), "%s.bias", key);
        wn->in_b[i] = wubu_rvc_find_tensor(model, bn);
        snprintf(key, sizeof(key), "flow.flows.%d.enc.res_skip_layers.%d", flow_idx, i);
        snprintf(gn, sizeof(gn), "%s.weight_g", key);
        snprintf(vn, sizeof(vn), "%s.weight_v", key);
        wn->rs_wg[i] = wubu_rvc_find_tensor(model, gn);
        wn->rs_wv[i] = wubu_rvc_find_tensor(model, vn);
        char bn2[256]; snprintf(bn2, sizeof(bn2), "%s.bias", key);
        wn->rs_b[i] = wubu_rvc_find_tensor(model, bn2);
    }
    /* de-normalize in/rs layers */
    for (int i = 0; i < 3; i++) {
        if (wn->in_wg[i] && wn->in_wv[i]) {
            int n = 1; for (int d = 0; d < wn->in_wv[i]->n_dims; d++) n *= wn->in_wv[i]->dims[d];
            float *o = (float *)malloc((size_t)n * sizeof(float));
            if (o && wubu_denorm_tensor(wn->in_wg[i], wn->in_wv[i], o, n) == 0) {
                wn->in_denorm[i] = o; wn->in_denorm_len[i] = n;
            } else { free(o); }
        }
        if (wn->rs_wg[i] && wn->rs_wv[i]) {
            int n = 1; for (int d = 0; d < wn->rs_wv[i]->n_dims; d++) n *= wn->rs_wv[i]->dims[d];
            float *o = (float *)malloc((size_t)n * sizeof(float));
            if (o && wubu_denorm_tensor(wn->rs_wg[i], wn->rs_wv[i], o, n) == 0) {
                wn->rs_denorm[i] = o; wn->rs_denorm_len[i] = n;
            } else { free(o); }
        }
    }
    return 0;
}

static void wn_free(WNBlock *wn) {
    for (int i = 0; i < 3; i++) { free(wn->in_denorm[i]); free(wn->rs_denorm[i]); }
}

static void wn_forward(const WNBlock *wn, float *x, int T,
                       const float *g, const float *x_mask, float *out) {
    int H = wn->hidden, L = wn->n_layers;
    float *cond = NULL;
    if (g && wn->cond_w) {
        /* cond_layer: Conv1d(gin -> 2*H*L, k=1) weight_norm */
        int gin = wn->gin;
        int cond_out = 2 * H * L;
        const RVCTensor *cg = wn->cond_w; /* this is weight_g */
        /* need weight_v too: look it up */
        /* cond_w stored as weight_g; cond_b as bias; find weight_v */
        char key[256];
        snprintf(key, sizeof(key), "flow.flows.%d.enc.cond_layer.weight_v", wn->flow_idx);
        const RVCTensor *cv = wubu_rvc_find_tensor(wn->model, key);
        if (cv) {
            cond = (float *)malloc((size_t)cond_out * T * sizeof(float));
            if (cond) {
                float *denorm = (float *)malloc((size_t)cond_out * gin * sizeof(float));
                if (denorm) {
                    wubu_denorm_tensor(cg, cv, denorm, cond_out * gin);
                    /* conv1d k=1 */
                    for (int o = 0; o < cond_out; o++) {
                        const float *wv = denorm + (size_t)o * gin;
                        float bias = wn->cond_b && wn->cond_b->data ? wn->cond_b->data[o] : 0.0f;
                        float *orow = cond + (size_t)o * T;
                        for (int j = 0; j < T; j++) {
                            float acc = bias;
                            for (int i = 0; i < gin; i++) acc += g[i] * wv[i];
                            orow[j] = acc;
                        }
                    }
                    free(denorm);
                }
            }
        }
    }

    float *acts = (float *)malloc((size_t)(2 * H) * T * sizeof(float));
    float *x_in = (float *)malloc((size_t)(2 * H) * T * sizeof(float));
    float *rs = (float *)malloc((size_t)(2 * H) * T * sizeof(float));
    if (!acts || !x_in || !rs) { free(acts); free(x_in); free(rs); free(cond); return; }

    memset(out, 0, (size_t)H * T * sizeof(float));
    for (int i = 0; i < L; i++) {
        /* x_in = in_layers[i](x): Conv1d(H -> 2H, k=5, dil=1, pad=2) */
        conv1d_c(x, H, T, wn->in_denorm[i], wn->in_b[i] && wn->in_b[i]->data ? wn->in_b[i]->data : NULL,
                 2 * H, 5, 1, 2, 1, x_in);
        /* acts = tanh(x_in + g_l[:H]) * sigmoid(x_in + g_l[H:]) */
        const float *gl = cond ? cond + (size_t)(i * 2 * H) * T : NULL;
        for (int c = 0; c < H; c++) {
            for (int j = 0; j < T; j++) {
                float a = x_in[(size_t)c * T + j] + (gl ? gl[(size_t)c * T + j] : 0.0f);
                float b = x_in[(size_t)(H + c) * T + j] + (gl ? gl[(size_t)(H + c) * T + j] : 0.0f);
                acts[(size_t)c * T + j] = tanhf(a) * sigmoid_c(b);
            }
        }
        /* rs = res_skip_layers[i](acts): Conv1d(H -> 2H (i<L-1) or H (last), k=1) */
        int rs_out = (i < L - 1) ? 2 * H : H;
        conv1d_c(acts, H, T, wn->rs_denorm[i], wn->rs_b[i] && wn->rs_b[i]->data ? wn->rs_b[i]->data : NULL,
                 rs_out, 1, 1, 0, 1, rs);
        if (i < L - 1) {
            for (int c = 0; c < H; c++)
                for (int j = 0; j < T; j++)
                    x[(size_t)c * T + j] = (x[(size_t)c * T + j] + rs[(size_t)c * T + j]) * x_mask[j];
            for (int c = 0; c < H; c++)
                for (int j = 0; j < T; j++)
                    out[(size_t)c * T + j] += rs[(size_t)(H + c) * T + j];
        } else {
            for (int c = 0; c < H; c++)
                for (int j = 0; j < T; j++)
                    out[(size_t)c * T + j] += rs[(size_t)c * T + j];
        }
    }
    for (int c = 0; c < H; c++)
        for (int j = 0; j < T; j++)
            out[(size_t)c * T + j] *= x_mask[j];

    free(acts); free(x_in); free(rs); free(cond);
}

/* ═══════════════════════ Flow reverse ═══════════════════════
 * z_p [inter, T] -> z [inter, T]; inter=192, half=96.
 * Flows: [RC0, Flip, RC1, Flip, RC2, Flip, RC3, Flip]; reversed:
 *   Flip, RC3, Flip, RC2, Flip, RC1, Flip, RC0 */
int wubu_flow_reverse(WuBuRVCModel *model,
                      const float *z_p, int n_frames, int inter_channels,
                      const float *g, const float *x_mask,
                      float *z_out) {
    int nF = n_frames, half = inter_channels / 2;
    const RVCTensor *pre_w = T(model, "flow.flows.0.pre.weight");
    if (!pre_w) return -1;
    int hidden = pre_w->dims[0];

    float *cur = (float *)malloc((size_t)inter_channels * nF * sizeof(float));
    float *nxt = (float *)malloc((size_t)inter_channels * nF * sizeof(float));
    if (!cur || !nxt) { free(cur); free(nxt); return -1; }
    memcpy(cur, z_p, (size_t)inter_channels * nF * sizeof(float));

    /* order: process pairs (flow index 6,4,2,0), each preceded by Flip */
    int flow_ids[4] = {6, 4, 2, 0};
    for (int fi = 0; fi < 4; fi++) {
        /* Flip: reverse channel order */
        for (int c = 0; c < inter_channels; c++)
            for (int j = 0; j < nF; j++)
                nxt[(size_t)c * nF + j] = cur[(size_t)(inter_channels - 1 - c) * nF + j];
        memcpy(cur, nxt, (size_t)inter_channels * nF * sizeof(float));

        int fid = flow_ids[fi];
        char key[256];
        snprintf(key, sizeof(key), "flow.flows.%d.pre.weight", fid);
        const RVCTensor *pw = T(model, key);
        snprintf(key, sizeof(key), "flow.flows.%d.pre.bias", fid);
        const RVCTensor *pb = T(model, key);
        snprintf(key, sizeof(key), "flow.flows.%d.post.weight", fid);
        const RVCTensor *pow_ = T(model, key);
        snprintf(key, sizeof(key), "flow.flows.%d.post.bias", fid);
        const RVCTensor *pob = T(model, key);
        if (!pw || !pow_) { free(cur); free(nxt); return -1; }

        /* x0 = first half channels (cur rows 0..half-1), x1 = second half */
        float *h = (float *)malloc((size_t)hidden * nF * sizeof(float));
        float *m = (float *)malloc((size_t)half * nF * sizeof(float));
        if (!h || !m) { free(h); free(m); free(cur); free(nxt); return -1; }

        /* h = pre(x0) * x_mask */
        conv1d_c(cur, half, nF, pw->data, pb && pb->data ? pb->data : NULL,
                 hidden, 1, 1, 0, 1, h);
        for (int c = 0; c < hidden; c++)
            for (int j = 0; j < nF; j++) h[(size_t)c * nF + j] *= x_mask[j];

        /* h = WN(h, mask, g) */
        WNBlock wn;
        if (wn_init(&wn, model, fid) != 0) { free(h); free(m); free(cur); free(nxt); return -1; }
        float *wn_out = (float *)malloc((size_t)hidden * nF * sizeof(float));
        if (!wn_out) { wn_free(&wn); free(h); free(m); free(cur); free(nxt); return -1; }
        wn_forward(&wn, h, nF, g, x_mask, wn_out);
        wn_free(&wn);

        /* stats = post(wn_out) * x_mask; mean_only: m=stats, logs=0 */
        float *stats = (float *)malloc((size_t)half * nF * sizeof(float));
        if (!stats) { free(wn_out); free(h); free(m); free(cur); free(nxt); return -1; }
        conv1d_c(wn_out, hidden, nF, pow_->data, pob && pob->data ? pob->data : NULL,
                 half, 1, 1, 0, 1, stats);
        for (int c = 0; c < half; c++)
            for (int j = 0; j < nF; j++) stats[(size_t)c * nF + j] *= x_mask[j];

        /* reverse: x1 = (x1 - m) * exp(-0) * x_mask = (x1 - m) * x_mask */
        for (int c = 0; c < half; c++) {
            for (int j = 0; j < nF; j++) {
                float x1 = cur[(size_t)(half + c) * nF + j];
                nxt[(size_t)c * nF + j] = cur[(size_t)c * nF + j];
                nxt[(size_t)(half + c) * nF + j] = (x1 - stats[(size_t)c * nF + j]) * x_mask[j];
            }
        }
        memcpy(cur, nxt, (size_t)inter_channels * nF * sizeof(float));

        free(stats); free(wn_out); free(h); free(m);
    }

    memcpy(z_out, cur, (size_t)inter_channels * nF * sizeof(float));
    free(cur); free(nxt);
    return 0;
}

/* ═══════════════════════ GeneratorNSF (dec) ═══════════════════════
 * z [inter, T], nsff0 [T] Hz. Cartman v2: inter=192, sr=40000,
 * upsample_rates [10,10,2,2], upsample_init 512, kernels [16,16,4,4],
 * noise_convs strides [40,4,2,1], MRF 3 stacks k=[3,7,11]. */
int wubu_generator_nsf(WuBuRVCModel *model,
                       const float *z, int n_frames, int inter_channels,
                       const float *nsff0, const float *g,
                       float *out, int max_samples,
                       int inject_noise,
                       int use_snake)  /* BigVGAN Snake activation in MRF blocks */
{
    int nF = n_frames;
    if (getenv("WUBU_RVC_DUMP")) {
        /* training-pair hook: real generator input (flow output z, (192,T)) */
        FILE *df = fopen("outputs/rvc_ref/c_gen_input.npy", "wb");
        if (df) {
            fwrite(z, sizeof(float), (size_t)inter_channels * nF, df);
            fclose(df);
            fprintf(stderr, "[dump] c_gen_input.npy (%d x %d)\n", inter_channels, nF);
        }
    }
    const RVCTensor *conv_pre_w = T(model, "dec.conv_pre.weight");
    const RVCTensor *conv_pre_b = T(model, "dec.conv_pre.bias");
    const RVCTensor *cond_w = T(model, "dec.cond.weight");
    const RVCTensor *cond_b = T(model, "dec.cond.bias");
    const RVCTensor *post_w = T(model, "dec.conv_post.weight");
    if (!conv_pre_w || !cond_w || !post_w) return -1;

    int init_ch = conv_pre_w->dims[0]; /* 512 for Cartman, 768 for some models */

    /* Use model's upsample config if available; fall back to Cartman defaults.
     * RVC v2 Cartman: rates [10,10,2,2], kernels [16,16,4,4], pad [3,3,1,1]
     * RVC v2 Miku: rates [12,10,2,2], kernels [24,20,4,4], pad [6,5,1,1]
     * Infer kernel sizes and pads from weight shapes at runtime. */
    int n_ups;
    if (model->n_upsample_layers > 0) n_ups = model->n_upsample_layers;
    else n_ups = 4;

    int ups_in[8], ups_out[8], ups_rate[8], ups_k[8], ups_pad[8];
    for (int L = 0; L < n_ups; L++) {
        ups_in[L] = init_ch / (1 << L);
        ups_out[L] = init_ch / (1 << (L + 1));
        if (model->upsample_rates[L] > 0) {
            ups_rate[L] = model->upsample_rates[L];
        } else {
            /* Default: Cartman rates [10, 10, 2, 2] */
            int def_rates[4] = {10, 10, 2, 2};
            ups_rate[L] = (L < 4) ? def_rates[L] : 1;
        }
        /* Infer kernel size and padding from dec.ups.L.weight_v shape:
         * weight_v shape is (out_ch, in_ch, k), so k = dims[2].
         * padding = (k - stride) / 2 (PyTorch ConvTranspose1d default). */
        char kbuf[128];
        snprintf(kbuf, sizeof(kbuf), "dec.ups.%d.weight_v", L);
        const RVCTensor *ut = T(model, kbuf);
        if (!ut) { snprintf(kbuf, sizeof(kbuf), "dec.ups.%d.weight", L); ut = T(model, kbuf); }
        if (ut && ut->n_dims >= 3) {
            ups_k[L] = ut->dims[2];
            ups_pad[L] = (ut->dims[2] - ups_rate[L]) / 2;
            if (ups_pad[L] < 0) ups_pad[L] = 0;
        } else {
            ups_k[L] = ups_rate[L] * 2;  /* default: k = 2*stride */
            ups_pad[L] = ups_rate[L] / 2;
        }
    }
    int ups_total = 1;
    for (int i = 0; i < n_ups; i++) ups_total *= ups_rate[i];

    /* ---- conv_pre ---- */
    float *x = (float *)malloc((size_t)init_ch * nF * sizeof(float));
    if (!x) return -1;
    conv1d_c(z, inter_channels, nF, conv_pre_w->data,
             conv_pre_b && conv_pre_b->data ? conv_pre_b->data : NULL,
             init_ch, 7, 1, 3, 1, x);

    /* x = x + cond(g): Conv1d(gin -> init_ch, k=1) */
    if (g) {
        int gin = cond_w->dims[1];
        for (int o = 0; o < init_ch; o++) {
            const float *wv = cond_w->data + (size_t)o * gin;
            float bias = cond_b && cond_b->data ? cond_b->data[o] : 0.0f;
            float *orow = x + (size_t)o * nF;
            for (int j = 0; j < nF; j++) {
                float acc = bias;
                for (int i = 0; i < gin; i++) acc += g[i] * wv[i];
                orow[j] += acc;
            }
        }
    }

    if (getenv("WUBU_RVC_DUMP")) {
        FILE *df = fopen("outputs/rvc_ref/c_gen_pre.npy", "wb");
        if (df) { fwrite(x, sizeof(float), (size_t)init_ch * nF, df); fclose(df); }
    }

    /* ---- f0 sine excitation (SineGen harmonic_num=0 + l_linear + tanh) ----
     *
     * Faithful port of PyTorch SineGen._f02sine:
     *   rad = f0 / sr * arange(1, upp+1)     # per-sample phase increment
     *   rad2 = fmod(rad[..., -1:] + 0.5, 1) - 0.5  # carry from last sample
     *   rad_acc = rad2.cumsum(dim=1).fmod(1)       # cumulative carry
     *   rad += pad(rad_acc)                        # rad[t] += carry[t-1]
     *   sine = sin(2*pi * rad)                     # rad IS the phase (no cumsum)
     *
     * Then: sine_waves *= uv  (noise=0, deterministic parity)
     *       har = tanh(l_linear(sine_waves))
     */
    const RVCTensor *lin_w = T(model, "dec.m_source.l_linear.weight");
    const RVCTensor *lin_b = T(model, "dec.m_source.l_linear.bias");
    int sr = 40000;
    if (model->sample_rate > 0) sr = model->sample_rate;
    int n_sine = nF * ups_total;
    float *sine = (float *)malloc((size_t)n_sine * sizeof(float));
    if (!sine) { free(x); return -1; }
    {
        float linw = lin_w && lin_w->data ? lin_w->data[0] : 1.0f;
        float linb = lin_b && lin_b->data ? lin_b->data[0] : 0.0f;

        /* Step 1: rad = f0/sr per frame (NOT *arange(1,upp+1) yet) */
        float *rad = (float *)malloc((size_t)nF * sizeof(float));
        if (!rad) { free(sine); free(x); return -1; }
        for (int j = 0; j < nF; j++) {
            float f0 = nsff0[j];
            rad[j] = (f0 > 0 ? f0 : 0.0f) / (float)sr;
        }

        /* Step 2: carry[t] = fmod(rad[t]*upp + 0.5, 1.0) - 0.5
         *         rad_acc[t] = fmod(sum(carry[0..t]), 1.0) */
        float *carry = (float *)malloc((size_t)nF * sizeof(float));
        float *rad_acc = (float *)malloc((size_t)nF * sizeof(float));
        if (!carry || !rad_acc) {
            free(carry); free(rad_acc); free(rad); free(sine); free(x); return -1;
        }
        float accum = 0.0f;
        for (int t = 0; t < nF; t++) {
            carry[t] = rad[t] * (float)ups_total;
            float rad2 = fmodf(carry[t] + 0.5f, 1.0f) - 0.5f;
            accum += rad2;
            rad_acc[t] = fmodf(accum, 1.0f);  /* cumulative carry, mod 1 */
        }

        /* Step 3: sine[j] = sin(2*pi * (rad[frame]*(u+1) + carry_prev)) * 0.1 * uv
         * where u = j % ups_total (sample index within frame)
         *       frame = j / ups_total
         *       carry_prev = rad_acc[frame-1] (0 for frame 0)
         * This is EXACTLY PyTorch: rad[t,u] = f0[t]/sr*(u+1) + rad_acc[t-1] */
        for (int j = 0; j < n_sine; j++) {
            int fi = j / ups_total;
            if (fi >= nF) fi = nF - 1;
            int u = j % ups_total;   /* sample index within frame: 0..upp-1 */
            float phase = rad[fi] * (float)(u + 1);  /* f0/sr * (u+1) */
            if (fi > 0) phase += rad_acc[fi - 1];     /* add carry from prev frame */
            float s = sinf(2.0f * (float)M_PI * phase) * 0.1f;  /* sine_amp = 0.1 */
            float sv = (nsff0[fi] > 0) ? 1.0f : 0.0f;  /* uv mask */
            /* Phase 4 improvement: noise injection in NSF sine generation.
             * PyTorch SineGen injects uniform noise in unvoiced regions:
             *   noise_amp = (1 - uv) * sine_amp / 3 = (1 - uv) * 0.0333
             * This adds natural breathiness/silence texture. For parity mode
             * (randn_scale=0), noise is suppressed. */
            float noise_amp = inject_noise ? (1.0f - sv) * 0.1f / 3.0f : 0.0f;
            float noise = noise_amp * wubu_rand_uniform(-1.0f, 1.0f);
            float sw = s * sv + noise;                     /* sine * uv + noise * (1-uv) */
            sine[j] = tanhf(linw * sw + linb);             /* l_linear + tanh */
        }
        free(carry); free(rad_acc); free(rad);
    }
    if (getenv("WUBU_RVC_DUMP")) {
        FILE *df = fopen("outputs/rvc_ref/c_gen_sine.npy", "wb");
        if (df) { fwrite(sine, sizeof(float), (size_t)n_sine, df); fclose(df); }
    }

    /* ---- upsample stages ---- */
    float *cur = x;
    int cur_n = nF;
    float *stage[4] = {NULL, NULL, NULL, NULL};
    for (int L = 0; L < n_ups; L++) {
        int next_n = (cur_n - 1) * ups_rate[L] - 2 * ups_pad[L] + ups_k[L];
        if (next_n <= 0 || next_n > max_samples) { free(x); for (int j = 0; j < L; j++) free(stage[j]); free(sine); return -1; }
        stage[L] = (float *)calloc((size_t)ups_out[L] * next_n, sizeof(float));
        if (!stage[L]) { free(x); for (int j = 0; j < L; j++) free(stage[j]); free(sine); return -1; }

        /* activation before ups: Snake+LReLU (BigVGAN) or standard LReLU */
        if (use_snake) snake_lrelu_c(cur, (size_t)ups_in[L] * cur_n, 0.1f);
        else lrelu_c(cur, (size_t)ups_in[L] * cur_n, 0.1f);

        /* ups[L]: ConvTranspose1d(ups_in -> ups_out, k, rate, pad) weight_norm.
         * wubu_rvc_weights.c already pre-computes hifi_upsample_denorm[L]. */
        char key[256];
        snprintf(key, sizeof(key), "dec.ups.%d.bias", L);
        const RVCTensor *ub = T(model, key);
        float *denorm = model->hifi_upsample_denorm[L];
        if (!denorm) {
            char gk[256], vk[256];
            snprintf(gk, sizeof(gk), "dec.ups.%d.weight_g", L);
            snprintf(vk, sizeof(vk), "dec.ups.%d.weight_v", L);
            const RVCTensor *ug = T(model, gk);
            const RVCTensor *uv_ = T(model, vk);
            if (!ug || !uv_) { free(x); for (int j = 0; j <= L; j++) free(stage[j]); free(sine); return -1; }
            float *tmp_d = (float *)malloc((size_t)ups_in[L] * ups_out[L] * ups_k[L] * sizeof(float));
            if (!tmp_d) { free(x); for (int j = 0; j <= L; j++) free(stage[j]); free(sine); return -1; }
            wubu_denorm_tensor(ug, uv_, tmp_d, ups_in[L] * ups_out[L] * ups_k[L]);
            conv_transpose1d_c(cur, ups_in[L], cur_n, tmp_d,
                               ub && ub->data ? ub->data : NULL,
                               ups_out[L], ups_k[L], ups_rate[L], ups_pad[L], stage[L]);
            free(tmp_d);
        } else {
            conv_transpose1d_c(cur, ups_in[L], cur_n, denorm,
                               ub && ub->data ? ub->data : NULL,
                               ups_out[L], ups_k[L], ups_rate[L], ups_pad[L], stage[L]);
        }

        /* x = x + noise_convs[L](har_source): Conv1d(1 -> ups_out, k=2*stride, s=stride, p=stride/2) */
        {
            int stride = 1;
            for (int j = L + 1; j < n_ups; j++) stride *= ups_rate[j];
            snprintf(key, sizeof(key), "dec.noise_convs.%d.weight", L);
            const RVCTensor *ncw = T(model, key);
            snprintf(key, sizeof(key), "dec.noise_convs.%d.bias", L);
            const RVCTensor *ncb = T(model, key);
            /* kernel/pad from the noise-conv weight itself (agnostic);
             * fallback heuristic only for missing weights. */
            int kk = ncw && ncw->n_dims >= 3 ? ncw->dims[2] : ((L == n_ups - 1) ? 1 : stride * 2);
            int pad = (kk - stride) / 2;
            if (pad < 0) pad = 0;
            if (ncw) {
                int s_in = n_sine;
                /* conv1d over 1-channel sine */
                float *nc_out = (float *)malloc((size_t)ups_out[L] * next_n * sizeof(float));
                if (nc_out) {
                    float *sine_c = (float *)malloc((size_t)1 * s_in * sizeof(float));
                    if (sine_c) {
                        memcpy(sine_c, sine, (size_t)s_in * sizeof(float));
                        conv1d_c(sine_c, 1, s_in, ncw->data,
                                 ncb && ncb->data ? ncb->data : NULL,
                                 ups_out[L], kk, stride, pad, 1, nc_out);
                        for (int i = 0; i < ups_out[L] * next_n; i++) stage[L][i] += nc_out[i];
                        free(sine_c);
                    }
                    free(nc_out);
                }
            }
        }

        /* MRF: avg over n_mrf_stacks resblock stacks (kernel sizes + dilations
         * come from the model config — NOT hardcoded [3,7,11]). Fallbacks only
         * if a model ships without config. The stacks run sequentially here —
         * conv1d_c inside already parallelizes over its output channels with
         * all threads, which beats nesting a 3-iteration parallel region
         * inside (nested OMP is disabled by default and would serialize the
         * inner convs). */
        {
            int ch = ups_out[L];
            int n_stacks = model->n_mrf_stacks;
            if (n_stacks < 1) n_stacks = 3; /* legacy no-config fallback */
            if (n_stacks > 8) n_stacks = 8;
            int n_pairs = model->n_resblock_pairs;
            if (n_pairs < 1) n_pairs = 3;
            if (n_pairs > 8) n_pairs = 8;
            /* per-stack accumulators — no shared writes. */
            float *acc = (float *)calloc((size_t)n_stacks * ch * next_n, sizeof(float));
            if (acc) {
                for (int s = 0; s < n_stacks; s++) {
                    char key[256];
                    float *acc_s = acc + (size_t)s * ch * next_n;
                    int rb = L * n_stacks + s;
                    int k = model->resblock_k[s];
                    if (k <= 0) k = (s == 0) ? 3 : (s == 1 ? 7 : 11); /* legacy */
                    float *rb_in = (float *)malloc((size_t)ch * next_n * sizeof(float));
                    float *rb_out = (float *)malloc((size_t)ch * next_n * sizeof(float));
                    if (!rb_in || !rb_out) { free(rb_in); free(rb_out); continue; }
                    memcpy(rb_in, stage[L], (size_t)ch * next_n * sizeof(float));
                    for (int cp = 0; cp < n_pairs; cp++) {
                        snprintf(key, sizeof(key), "dec.resblocks.%d.convs1.%d.weight_v", rb, cp);
                        const RVCTensor *r1v = T(model, key);
                        snprintf(key, sizeof(key), "dec.resblocks.%d.convs1.%d.bias", rb, cp);
                        const RVCTensor *r1b = T(model, key);
                        snprintf(key, sizeof(key), "dec.resblocks.%d.convs2.%d.weight_v", rb, cp);
                        const RVCTensor *r2v = T(model, key);
                        snprintf(key, sizeof(key), "dec.resblocks.%d.convs2.%d.bias", rb, cp);
                        const RVCTensor *r2b = T(model, key);
                        if (!r1v || !r2v || !r1v->data || !r2v->data) continue;
                        int dil = model->resblock_dil[s][cp];
                        if (dil <= 0) dil = 1 + 2 * cp; /* legacy 1,3,5 */
                        int pad1 = dil * (k - 1) / 2;
                        int pad2 = k / 2;
                        /* NOTE: wubu_rvc_weights.c ALREADY de-normalized
                         * weight_v in place — data is the final weight. */
                        const float *d1 = r1v->data;
                        const float *d2 = r2v->data;
                        float *tmp = (float *)malloc((size_t)ch * next_n * sizeof(float));
                        if (tmp) {
                            /* x = leaky(x); conv1; leaky; conv2; + residual */
                            memcpy(tmp, rb_in, (size_t)ch * next_n * sizeof(float));
                            if (use_snake) snake_lrelu_c(tmp, (size_t)ch * next_n, 0.1f);
                            else lrelu_c(tmp, (size_t)ch * next_n, 0.1f);
                            conv1d_c(tmp, ch, next_n, d1, r1b && r1b->data ? r1b->data : NULL,
                                     ch, k, 1, pad1, dil, rb_out);
                            if (use_snake) snake_lrelu_c(rb_out, (size_t)ch * next_n, 0.1f);
                            else lrelu_c(rb_out, (size_t)ch * next_n, 0.1f);
                            conv1d_c(rb_out, ch, next_n, d2, r2b && r2b->data ? r2b->data : NULL,
                                     ch, k, 1, pad2, 1, tmp);
                            for (int i = 0; i < ch * next_n; i++) tmp[i] += rb_in[i];
                            memcpy(rb_in, tmp, (size_t)ch * next_n * sizeof(float));
                        }
                        free(tmp);
                    }
                    for (int i = 0; i < ch * next_n; i++) acc_s[i] += rb_in[i];
                    free(rb_in); free(rb_out);
                }
                for (int i = 0; i < ch * next_n; i++) {
                    float s = 0.0f;
                    for (int st = 0; st < n_stacks; st++)
                        s += acc[(size_t)st * ch * next_n + i];
                    stage[L][i] = s / (float)n_stacks;
                }
                free(acc);
            }
        }

        if (L > 0) free(stage[L - 1]); /* cur was stage[L-1] or x */
        if (L == 0) free(x);
        cur = stage[L];
        cur_n = next_n;
    }

    /* final: activation, conv_post(1 -> post_in, k=post_k, pad=post_k/2, no
     * bias), tanh — channels/kernel read from the weight tensor (agnostic).
     * conv_post.weight is (out_ch=1, in_ch, k), so in_ch = dims[1]. */
    int post_in = post_w->n_dims >= 2 ? post_w->dims[1] : 32;
    if (post_in < 1) post_in = 32; /* legacy */
    int post_k = post_w->n_dims >= 3 ? post_w->dims[2] : 7;
    int post_pad = post_k / 2;
    if (use_snake) snake_lrelu_c(cur, (size_t)post_in * cur_n, 0.1f);
    else lrelu_c(cur, (size_t)post_in * cur_n, 0.1f);
    int out_n = cur_n;
    if (out_n > max_samples) out_n = max_samples;
    {
        for (int j = 0; j < out_n; j++) {
            float acc = 0.0f;
            for (int c = 0; c < post_in; c++) {
                const float *kw = post_w->data + (size_t)c * post_k;
                for (int tap = 0; tap < post_k; tap++) {
                    int src = j - post_pad + tap;
                    if (src >= 0 && src < cur_n)
                        acc += cur[(size_t)c * cur_n + src] * kw[tap];
                }
            }
            out[j] = tanhf(acc); /* final activation */
        }
    }
    /* Snake saturation detection (replaces the old 0.15/max_out gain hack,
     * which just scaled a square wave down). Snake on LReLU-trained weights
     * saturates the final tanh (every sample ±1.0). We report the saturated
     * fraction so the caller can fall back to LeakyReLU — the parity-verified
     * path for these weights. Never silently emit a square wave. */
    if (use_snake && model) {
        float max_out = 0;
        float sum_abs = 0.0f;
        int n_sat = 0;
        for (int j = 0; j < out_n; j++) {
            float a = fabsf(out[j]);
            sum_abs += a;
            if (a > max_out) max_out = a;
            if (a > 0.999f) n_sat++;
        }
        model->last_snake_sat = (out_n > 0) ? (float)n_sat / (float)out_n : 0.0f;
        if (getenv("WUBU_RVC_DUMP")) {
            fprintf(stderr, "[snake-diag] out_n=%d max=%.4f mean_abs=%.4f sat_frac=%.3f\n",
                    out_n, max_out, sum_abs / (float)out_n, model->last_snake_sat);
        }
    } else if (model) {
        model->last_snake_sat = 0.0f;
    }

    free(stage[3]); free(sine);
    if (getenv("WUBU_RVC_DUMP")) {
        FILE *df = fopen("outputs/rvc_ref/c_gen_output.npy", "wb");
        if (df) {
            fwrite(out, sizeof(float), (size_t)out_n, df);
            fclose(df);
            fprintf(stderr, "[dump] c_gen_output.npy (%d samples)\n", out_n);
        }
    }
    return out_n;
}

/* ═══════════════════════ Full synthesize ═══════════════════════ */
int wubu_rvc_synthesize_real(WuBuRVCModel *model,
                             const float *content, int n_frames, int content_dim,
                             const int *f0_coarse, const float *nsff0,
                             int sid, float randn_scale,
                             float *out_audio, int max_samples,
                             int use_snake) {  /* BigVGAN Snake activation */
    if (!model || !content || !out_audio || n_frames < 1) return -1;

    const RVCTensor *emb_g = T(model, "emb_g.weight");
    if (!emb_g) return -1;
    int gin = emb_g->dims[1];
    if (sid < 0) sid = 0;
    if (sid >= emb_g->dims[0]) sid = 0;
    float g[256];
    for (int i = 0; i < gin && i < 256; i++) g[i] = emb_g->data[(size_t)sid * gin + i];

    /* enc_p -> m, logs */
    int inter = 192;
    {
        const RVCTensor *pw = T(model, "enc_p.proj.weight");
        if (pw) inter = pw->dims[0] / 2;
    }
    float *m = (float *)malloc((size_t)inter * n_frames * sizeof(float));
    float *logs = (float *)malloc((size_t)inter * n_frames * sizeof(float));
    float *x_mask = (float *)malloc((size_t)n_frames * sizeof(float));
    float *z_p = (float *)malloc((size_t)inter * n_frames * sizeof(float));
    float *z = (float *)malloc((size_t)inter * n_frames * sizeof(float));
    if (!m || !logs || !x_mask || !z_p || !z) {
        free(m); free(logs); free(x_mask); free(z_p); free(z); return -1;
    }

    if (wubu_enc_p_forward(model, content, n_frames, content_dim,
                           f0_coarse, m, logs, x_mask) != 0) {
        free(m); free(logs); free(x_mask); free(z_p); free(z); return -1;
    }

    /* Debug: WUBU_RVC_DUMP=1 writes intermediates as raw float files */
    if (getenv("WUBU_RVC_DUMP")) {
        FILE *df;
        df = fopen("outputs/rvc_ref/c_inter_m_p.npy", "wb");
        if (df) { fwrite(m, sizeof(float), (size_t)inter * n_frames, df); fclose(df); }
        df = fopen("outputs/rvc_ref/c_inter_logs_p.npy", "wb");
        if (df) { fwrite(logs, sizeof(float), (size_t)inter * n_frames, df); fclose(df); }
        df = fopen("outputs/rvc_ref/c_inter_x_mask.npy", "wb");
        if (df) { fwrite(x_mask, sizeof(float), (size_t)n_frames, df); fclose(df); }
    }

    /* z_p = (m + exp(logs) * randn * noise_scale) * x_mask
     * where randn ~ N(0,1). For determinism when randn_scale=0 (parity),
     * randn is suppressed. Otherwise, use Gaussian approximation via
     * sum of 12 uniform samples (Irwin-Hall → standard normal). */
    for (int c = 0; c < inter; c++) {
        for (int j = 0; j < n_frames; j++) {
            float r = 0.0f;
            if (randn_scale > 0.0f) {
                /* Irwin-Hall: sum of 12 U(-1,1) → N(0,1) approximation */
                r = wubu_rand_uniform(-1.0f, 1.0f) + wubu_rand_uniform(-1.0f, 1.0f) +
                    wubu_rand_uniform(-1.0f, 1.0f) + wubu_rand_uniform(-1.0f, 1.0f) +
                    wubu_rand_uniform(-1.0f, 1.0f) + wubu_rand_uniform(-1.0f, 1.0f) +
                    wubu_rand_uniform(-1.0f, 1.0f) + wubu_rand_uniform(-1.0f, 1.0f) +
                    wubu_rand_uniform(-1.0f, 1.0f) + wubu_rand_uniform(-1.0f, 1.0f) +
                    wubu_rand_uniform(-1.0f, 1.0f) + wubu_rand_uniform(-1.0f, 1.0f);
                r *= randn_scale;
            }
            z_p[(size_t)c * n_frames + j] =
                (m[(size_t)c * n_frames + j] + expf(logs[(size_t)c * n_frames + j]) * r) * x_mask[j];
            /* Clamp z_p to [-3, 3] (3 sigma for N(0,1) with noise_scale ~0.5).
             * Prevents extreme values that saturate the tanh output. */
            if (z_p[(size_t)c * n_frames + j] > 3.0f) z_p[(size_t)c * n_frames + j] = 3.0f;
            if (z_p[(size_t)c * n_frames + j] < -3.0f) z_p[(size_t)c * n_frames + j] = -3.0f;
        }
    }

    if (wubu_flow_reverse(model, z_p, n_frames, inter, g, x_mask, z) != 0) {
        free(m); free(logs); free(x_mask); free(z_p); free(z); return -1;
    }

    if (getenv("WUBU_RVC_DUMP")) {
        FILE *df = fopen("outputs/rvc_ref/c_inter_z_p.npy", "wb");
        if (df) { fwrite(z_p, sizeof(float), (size_t)inter * n_frames, df); fclose(df); }
        df = fopen("outputs/rvc_ref/c_inter_z.npy", "wb");
        if (df) { fwrite(z, sizeof(float), (size_t)inter * n_frames, df); fclose(df); }
    }

    /* Snake activation naturally expands the signal; the damped snake_c
     * function (0.1 * sin^2 term) compensates for this. */
    int n_out = wubu_generator_nsf(model, z, n_frames, inter, nsff0, g,
                                   out_audio, max_samples, randn_scale > 0.0f, use_snake);

    free(m); free(logs); free(x_mask); free(z_p); free(z);
    return n_out;
}
