/* wubu_rmvpe.c — self-contained RMVPE F0 extractor (C11).
 *
 * Exact port of RVC's rmvpe.py (E2E model):
 *   mel(128, htk) -> DeepUnet(enc 5x4 blocks + inter 4 + dec 5, (2,2) pool)
 *                 -> cnn(16->3, 3x3) -> transpose+flatten(384)
 *                 -> BiGRU(384->256, 1 layer, bidir) -> Linear(512->360)
 *                 -> sigmoid -> local-average-cents decode -> F0 Hz
 *
 * Weight layout (PyTorch C-order):
 *   Conv2d:          (out, in, k, k)
 *   ConvTranspose2d: (in, out, k, k)
 *   GRU w_ih:        (3h, in) rows [W_ir; W_iz; W_in]
 *   Linear:          (out, in)
 * Activations: (C, H, W) flat [c*H*W + h*W + w].
 *
 * License: WaefreBeorn-UMV3
 */
#define _USE_MATH_DEFINES
#include "wubu_rmvpe.h"
#include "wubu_stft.h"
#include "wubu_gru.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ── flat-binary tensor map ── */
typedef struct {
    char  name[128];
    float *data;
    int   ndims;
    int   dims[4];
} RmvpeTensor;

struct WuBuRmvpe {
    RmvpeTensor *tensors;
    int   n_tensors;
    WuBuStft *stft;
    WuBuGru *gru;
    float cents[360];   /* 20*arange(360) + 1997.3794084376191 */
    float *mel_basis;   /* 128 x 513 */
};

static const RmvpeTensor *find_t(const WuBuRmvpe *r, const char *name) {
    for (int i = 0; i < r->n_tensors; i++)
        if (strcmp(r->tensors[i].name, name) == 0) return &r->tensors[i];
    return NULL;
}

static float *tdata(const WuBuRmvpe *r, const char *name) {
    const RmvpeTensor *t = find_t(r, name);
    return (t && t->data) ? t->data : NULL;
}

static int tnum(const RmvpeTensor *t) {
    if (!t) return 0;
    int n = 1;
    for (int i = 0; i < t->ndims; i++) n *= t->dims[i];
    return n;
}

/* ── flat-binary loader ── */
WuBuRmvpe *wubu_rmvpe_load(const char *bin_path) {
    FILE *f = fopen(bin_path, "rb");
    if (!f) return NULL;
    unsigned int count = 0;
    if (fread(&count, 4, 1, f) != 1) { fclose(f); return NULL; }
    if (count > 10000) { fclose(f); return NULL; }

    WuBuRmvpe *r = (WuBuRmvpe *)calloc(1, sizeof(WuBuRmvpe));
    if (!r) { fclose(f); return NULL; }
    r->n_tensors = (int)count;
    r->tensors = (RmvpeTensor *)calloc(count, sizeof(RmvpeTensor));
    if (!r->tensors) { free(r); fclose(f); return NULL; }

    for (unsigned int i = 0; i < count; i++) {
        unsigned int nlen = 0;
        if (fread(&nlen, 4, 1, f) != 1) goto fail;
        if (nlen >= sizeof(r->tensors[i].name)) goto fail;
        if (fread(r->tensors[i].name, 1, nlen, f) != nlen) goto fail;
        r->tensors[i].name[nlen] = 0;
        if (fread(&r->tensors[i].ndims, 4, 1, f) != 1) goto fail;
        if (r->tensors[i].ndims > 4) goto fail;
        int n = 1;
        for (int d = 0; d < r->tensors[i].ndims; d++) {
            unsigned int dim = 0;
            if (fread(&dim, 4, 1, f) != 1) goto fail;
            r->tensors[i].dims[d] = (int)dim;
            n *= (int)dim;
        }
        r->tensors[i].data = (float *)malloc((size_t)n * sizeof(float));
        if (!r->tensors[i].data) goto fail;
        if (fread(r->tensors[i].data, sizeof(float), (size_t)n, f) != (size_t)n) goto fail;
    }
    fclose(f);

    r->mel_basis = tdata(r, "mel_basis");
    if (!r->mel_basis) { wubu_rmvpe_free(r); return NULL; }
    r->stft = wubu_stft_create(1024, 160);
    if (!r->stft) { wubu_rmvpe_free(r); return NULL; }

    r->gru = wubu_gru_create(384, 256, 1);
    if (!r->gru) { wubu_rmvpe_free(r); return NULL; }
    if (wubu_gru_set(r->gru, 0, tdata(r, "fc.0.gru.weight_ih_l0"),
                     tdata(r, "fc.0.gru.weight_hh_l0"),
                     tdata(r, "fc.0.gru.bias_ih_l0"),
                     tdata(r, "fc.0.gru.bias_hh_l0")) != 0) { wubu_rmvpe_free(r); return NULL; }
    if (wubu_gru_set(r->gru, 1, tdata(r, "fc.0.gru.weight_ih_l0_reverse"),
                     tdata(r, "fc.0.gru.weight_hh_l0_reverse"),
                     tdata(r, "fc.0.gru.bias_ih_l0_reverse"),
                     tdata(r, "fc.0.gru.bias_hh_l0_reverse")) != 0) { wubu_rmvpe_free(r); return NULL; }

    for (int i = 0; i < 360; i++)
        r->cents[i] = 20.0f * (float)i + 1997.3794084376191f;
    return r;

fail:
    fclose(f);
    wubu_rmvpe_free(r);
    return NULL;
}

void wubu_rmvpe_free(WuBuRmvpe *r) {
    if (!r) return;
    for (int i = 0; i < r->n_tensors; i++)
        free(r->tensors[i].data);
    free(r->tensors);
    wubu_stft_free(r->stft);
    wubu_gru_free(r->gru);
    free(r);
}

/* ── 2D conv helpers (stride 1, optional pad) ── */
static void conv2d(const float *in, int cin, int H, int W,
                   const float *w, const float *b, int cout, int k, int pad,
                   float *out) {
    memset(out, 0, (size_t)cout * H * W * sizeof(float));
    /* parallel over output channels — the U-Net convs dominate RMVPE time
     * on long audio (serial = minutes on a 45s vocal) */
#pragma omp parallel for schedule(static) if(cout >= 4)
    for (int oc = 0; oc < cout; oc++) {
        const float *wrow = w + (size_t)oc * cin * k * k;
        float bias = b ? b[oc] : 0.0f;
        for (int h = 0; h < H; h++) {
            for (int wi = 0; wi < W; wi++) {
                float acc = bias;
                for (int ic = 0; ic < cin; ic++) {
                    const float *wch = wrow + (size_t)ic * k * k;
                    for (int dh = 0; dh < k; dh++) {
                        int hh = h + dh - pad;
                        if (hh < 0 || hh >= H) continue;
                        for (int dw = 0; dw < k; dw++) {
                            int ww = wi + dw - pad;
                            if (ww < 0 || ww >= W) continue;
                            acc += in[(size_t)ic * H * W + (size_t)hh * W + ww] *
                                   wch[(size_t)dh * k + dw];
                        }
                    }
                }
                out[(size_t)oc * H * W + (size_t)h * W + wi] = acc;
            }
        }
    }
}

/* ConvTranspose2d: k x k, stride s, pad p, output_pad op. H2 = (H-1)*s - 2p + k + op */
static void convt2d(const float *in, int cin, int H, int W,
                    const float *w, const float *b, int cout, int k, int s, int p, int op,
                    int H2, int W2, float *out) {
    memset(out, 0, (size_t)cout * H2 * W2 * sizeof(float));
    /* parallel over output channels; each oc accumulates over all ic */
#pragma omp parallel for schedule(static) if(cout >= 4)
    for (int oc = 0; oc < cout; oc++) {
        for (int ic = 0; ic < cin; ic++) {
            const float *wrow = w + ((size_t)ic * cout + oc) * k * k;
            for (int h = 0; h < H; h++) {
                int ho0 = h * s - p;
                for (int wi = 0; wi < W; wi++) {
                    float v = in[(size_t)ic * H * W + (size_t)h * W + wi];
                    if (v == 0.0f) continue;
                    int wo0 = wi * s - p;
                    for (int dh = 0; dh < k; dh++) {
                        int ho = ho0 + dh;
                        if (ho < 0 || ho >= H2) continue;
                        for (int dw = 0; dw < k; dw++) {
                            int wo = wo0 + dw;
                            if (wo < 0 || wo >= W2) continue;
                            out[(size_t)oc * H2 * W2 + (size_t)ho * W2 + wo] +=
                                v * wrow[(size_t)dh * k + dw];
                        }
                    }
                }
            }
        }
    }
    if (b) {
        for (int oc = 0; oc < cout; oc++) {
            float *o = out + (size_t)oc * H2 * W2;
            for (size_t i = 0; i < (size_t)H2 * W2; i++) o[i] += b[oc];
        }
    }
}

/* BatchNorm2d inference: y = (x - mean) * gamma / sqrt(var + eps) + beta */
static void bn2d(float *x, int C, int H, int W,
                 const float *g, const float *b, const float *mean, const float *var) {
    const float eps = 1e-5f;
    size_t hw = (size_t)H * W;
    for (int c = 0; c < C; c++) {
        float inv = g[c] / sqrtf(var[c] + eps);
        float bias = b[c] - mean[c] * inv;
        float *row = x + (size_t)c * hw;
        for (size_t i = 0; i < hw; i++)
            row[i] = row[i] * inv + bias;
    }
}

static void relu2d(float *x, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (x[i] < 0.0f) x[i] = 0.0f;
}

static void avgpool2d(const float *in, int C, int H, int W, float *out) {
    int H2 = H / 2, W2 = W / 2;
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H2; h++) {
            for (int w = 0; w < W2; w++) {
                float s = in[(size_t)c * H * W + (size_t)(2 * h) * W + 2 * w]
                        + in[(size_t)c * H * W + (size_t)(2 * h) * W + 2 * w + 1]
                        + in[(size_t)c * H * W + (size_t)(2 * h + 1) * W + 2 * w]
                        + in[(size_t)c * H * W + (size_t)(2 * h + 1) * W + 2 * w + 1];
                out[(size_t)c * H2 * W2 + (size_t)h * W2 + w] = s * 0.25f;
            }
        }
    }
}

/* One ConvBlockRes: conv(3x3)->BN->ReLU->conv(3x3)->BN->ReLU (+shortcut 1x1) */
static float *conv_block_res(const WuBuRmvpe *r, const char *prefix,
                             const float *in, int cin, int cout, int H, int W) {
    char key[160];
    float *out = (float *)malloc((size_t)cout * H * W * sizeof(float));
    float *out2 = (float *)malloc((size_t)cout * H * W * sizeof(float));
    if (!out || !out2) { free(out); free(out2); return NULL; }

    snprintf(key, sizeof(key), "%s.conv.0.weight", prefix);
    conv2d(in, cin, H, W, tdata(r, key), NULL, cout, 3, 1, out);
    snprintf(key, sizeof(key), "%s.conv.1.weight", prefix);
    const float *bg = tdata(r, key);
    snprintf(key, sizeof(key), "%s.conv.1.bias", prefix);
    const float *bb = tdata(r, key);
    snprintf(key, sizeof(key), "%s.conv.1.running_mean", prefix);
    const float *bm = tdata(r, key);
    snprintf(key, sizeof(key), "%s.conv.1.running_var", prefix);
    const float *bv = tdata(r, key);
    bn2d(out, cout, H, W, bg, bb, bm, bv);
    relu2d(out, (size_t)cout * H * W);

    snprintf(key, sizeof(key), "%s.conv.3.weight", prefix);
    conv2d(out, cout, H, W, tdata(r, key), NULL, cout, 3, 1, out2);
    snprintf(key, sizeof(key), "%s.conv.4.weight", prefix);
    bg = tdata(r, key);
    snprintf(key, sizeof(key), "%s.conv.4.bias", prefix);
    bb = tdata(r, key);
    snprintf(key, sizeof(key), "%s.conv.4.running_mean", prefix);
    bm = tdata(r, key);
    snprintf(key, sizeof(key), "%s.conv.4.running_var", prefix);
    bv = tdata(r, key);
    bn2d(out2, cout, H, W, bg, bb, bm, bv);
    relu2d(out2, (size_t)cout * H * W);

    /* shortcut: 1x1 conv with bias when channels change, else identity */
    if (cin != cout) {
        float *sc = (float *)malloc((size_t)cout * H * W * sizeof(float));
        if (sc) {
            snprintf(key, sizeof(key), "%s.shortcut.weight", prefix);
            const float *sw = tdata(r, key);
            snprintf(key, sizeof(key), "%s.shortcut.bias", prefix);
            const float *sb = tdata(r, key);
            conv2d(in, cin, H, W, sw, sb, cout, 1, 0, sc);
            for (size_t i = 0; i < (size_t)cout * H * W; i++) out2[i] += sc[i];
            free(sc);
        }
    } else {
        for (size_t i = 0; i < (size_t)cout * H * W; i++) out2[i] += in[i];
    }
    free(out);
    return out2;
}

/* ── cents decode (to_local_average_cents) ── */
static float decode_row(const float *s, const float *cents) {
    int center = 0;
    float maxv = s[0];
    for (int i = 1; i < 360; i++) {
        if (s[i] > maxv) { maxv = s[i]; center = i; }
    }
    if (maxv <= 0.03f) return 0.0f;
    float prod = 0.0f, wsum = 0.0f;
    for (int j = -4; j <= 4; j++) {
        int idx = center + j;
        float sv = (idx >= 0 && idx < 360) ? s[idx] : 0.0f;
        float cm = (idx >= 0 && idx < 360) ? cents[idx] : 0.0f;
        prod += sv * cm;
        wsum += sv;
    }
    if (wsum <= 0.0f) return 0.0f;
    float cents_val = prod / wsum;
    float f0 = 10.0f * powf(2.0f, cents_val / 1200.0f);
    return (f0 == 10.0f) ? 0.0f : f0;
}

/* ── full forward ── */
int wubu_rmvpe_f0(WuBuRmvpe *r, const float *pcm, int n_samples,
                  float *f0_out, int max_frames) {
    if (!r || !pcm || !f0_out) return -1;

    const int n_bins = 513;
    const int n_mels = 128;
    int T = wubu_stft_n_frames(r->stft, n_samples);
    if (T < 1) return -1;

    float *mag = (float *)malloc((size_t)n_bins * T * sizeof(float));
    float *mel = (float *)malloc((size_t)n_mels * T * sizeof(float));
    if (!mag || !mel) { free(mag); free(mel); return -1; }
    wubu_stft_magnitude(r->stft, pcm, n_samples, mag, T);
    wubu_mel_apply(r->mel_basis, n_mels, n_bins, mag, T, mel);
    free(mag);
    /* log(clamp(mel, 1e-5)) */
    for (int i = 0; i < n_mels * T; i++) {
        float v = mel[i] < 1e-5f ? 1e-5f : mel[i];
        mel[i] = logf(v);
    }
    if (getenv("WUBU_RMVPE_DUMP")) {
        float mf = 0, mf2 = 0, mn = 1e30f;
        for (int i = 0; i < n_mels * T; i++) {
            float v = fabsf(mel[i]);
            if (v > mf) mf = v;
            mf2 += v;
            if (mel[i] < mn) mn = mel[i];
        }
        fprintf(stderr, "[diag] mel: T=%d mean|.|=%.4f max=%.4f min=%.4f\n",
                T, mf2 / (float)(n_mels * T), mf, mn);
        FILE *df = fopen("outputs/rvc_ref/c_mel.bin", "wb");
        if (df) { fwrite(mel, sizeof(float), (size_t)n_mels * T, df); fclose(df); }
    }

    /* pad time to multiple of 32; x is (Tp, 128) ROW-major [t*128 + m]
     * while mel is (128, T) col-major [m*T + t] — transpose on copy. */
    int Tp = ((T + 31) / 32) * 32;
    float *x = (float *)calloc((size_t)Tp * n_mels, sizeof(float));
    if (!x) { free(mel); return -1; }
    for (int t = 0; t < T; t++)
        for (int m = 0; m < n_mels; m++)
            x[(size_t)t * n_mels + m] = mel[(size_t)m * T + t];
    free(mel);

    /* ── encoder: 5 layers, 4 conv blocks, avgpool (2,2) ── */
    int H = Tp, W = n_mels;
    int cin = 1;
    float *concat[5];
    int concat_c[5];
    char key[160];
    {
        /* encoder.bn on the single input channel */
        const float *bg = tdata(r, "unet.encoder.bn.weight");
        const float *bb = tdata(r, "unet.encoder.bn.bias");
        const float *bm = tdata(r, "unet.encoder.bn.running_mean");
        const float *bv = tdata(r, "unet.encoder.bn.running_var");
        bn2d(x, 1, H, W, bg, bb, bm, bv);
        if (getenv("WUBU_RMVPE_DUMP")) {
            FILE *df = fopen("outputs/rvc_ref/c_bn.bin", "wb");
            if (df) { fwrite(x, sizeof(float), (size_t)H * W, df); fclose(df); }
        }

        for (int L = 0; L < 5; L++) {
            int cout = 16 << L;
            float *cur = x;
            for (int B = 0; B < 4; B++) {
                snprintf(key, sizeof(key), "unet.encoder.layers.%d.conv.%d", L, B);
                float *nxt = conv_block_res(r, key, cur, (B == 0) ? cin : cout, cout, H, W);
                free(cur);
                cur = nxt;
                if (!cur) { free(x); return -1; }
            }
            concat[L] = cur;
            concat_c[L] = cout;
            if (getenv("WUBU_RMVPE_DUMP") && L == 0) {
                FILE *df = fopen("outputs/rvc_ref/c_l0.bin", "wb");
                if (df) { fwrite(cur, sizeof(float), (size_t)cout * H * W, df); fclose(df); }
            }
            x = (float *)malloc((size_t)cout * (H / 2) * (W / 2) * sizeof(float));
            if (!x) return -1;
            avgpool2d(concat[L], cout, H, W, x);
            /* NOTE: concat[L] stays alive — the decoder consumes it via
             * concat[4-L] (skip connections) and frees it there. */
            cin = cout;
            H /= 2;
            W /= 2;
        }
    }
    /* x: (256, H=9, W=4) */
    if (getenv("WUBU_RMVPE_DUMP")) {
        float mf = 0, mf2 = 0;
        for (int i = 0; i < 256 * H * W; i++) { float v = fabsf(x[i]); if (v > mf) mf = v; mf2 += v; }
        fprintf(stderr, "[diag] enc bottleneck: H=%d W=%d max|.|=%.4f mean|.|=%.4f\n",
                H, W, mf, mf2 / (float)(256 * H * W));
        FILE *df = fopen("outputs/rvc_ref/c_enc.bin", "wb");
        if (df) { fwrite(x, sizeof(float), (size_t)256 * H * W, df); fclose(df); }
    }

    /* ── intermediate: 4 layers, no pool (256 -> 512 -> 512...) ── */
    {
        int iin = 256;
        for (int L = 0; L < 4; L++) {
            int cout = 512;
            float *cur = x;
            for (int B = 0; B < 4; B++) {
                snprintf(key, sizeof(key), "unet.intermediate.layers.%d.conv.%d", L, B);
                float *nxt = conv_block_res(r, key, cur, (B == 0) ? iin : cout, cout, H, W);
                free(cur);
                cur = nxt;
                if (!cur) return -1;
            }
            x = cur;
            iin = cout;
        }
    }
    /* x: (512, 9, 4) */
    if (getenv("WUBU_RMVPE_DUMP")) {
        float mf = 0, mf2 = 0;
        for (int i = 0; i < 512 * H * W; i++) { float v = fabsf(x[i]); if (v > mf) mf = v; mf2 += v; }
        fprintf(stderr, "[diag] intermediate out: max|.|=%.4f mean|.|=%.4f\n", mf, mf2 / (float)(512 * H * W));
    }

    /* ── decoder: 5 layers, convT2d(2x) + concat skip + 4 conv blocks ── */
    {
        int din = 512;
        for (int L = 0; L < 5; L++) {
            int cout = 256 >> L;
            int H2 = 2 * H, W2 = 2 * W;
            float *d = (float *)malloc((size_t)cout * H2 * W2 * sizeof(float));
            if (!d) return -1;
            snprintf(key, sizeof(key), "unet.decoder.layers.%d.conv1.0.weight", L);
            convt2d(x, din, H, W, tdata(r, key), NULL, cout, 3, 2, 1, 1, H2, W2, d);
            snprintf(key, sizeof(key), "unet.decoder.layers.%d.conv1.1.weight", L);
            const float *bg = tdata(r, key);
            snprintf(key, sizeof(key), "unet.decoder.layers.%d.conv1.1.bias", L);
            const float *bb = tdata(r, key);
            snprintf(key, sizeof(key), "unet.decoder.layers.%d.conv1.1.running_mean", L);
            const float *bm = tdata(r, key);
            snprintf(key, sizeof(key), "unet.decoder.layers.%d.conv1.1.running_var", L);
            const float *bv = tdata(r, key);
            bn2d(d, cout, H2, W2, bg, bb, bm, bv);
            relu2d(d, (size_t)cout * H2 * W2);

            /* concat skip (encoder layer 4-L): cout channels, H2 x W2 */
            float *cat = (float *)malloc((size_t)(2 * cout) * H2 * W2 * sizeof(float));
            if (!cat) { free(d); return -1; }
            memcpy(cat, d, (size_t)cout * H2 * W2 * sizeof(float));
            const float *skip = concat[4 - L];
            memcpy(cat + (size_t)cout * H2 * W2, skip,
                   (size_t)cout * H2 * W2 * sizeof(float));
            free(d);
            free((float *)skip);

            float *cur = cat;
            for (int B = 0; B < 4; B++) {
                snprintf(key, sizeof(key), "unet.decoder.layers.%d.conv2.%d", L, B);
                float *nxt = conv_block_res(r, key, cur, (B == 0) ? 2 * cout : cout, cout, H2, W2);
                free(cur);
                cur = nxt;
                if (!cur) return -1;
            }
            free(x);
            x = cur;
            din = cout;
            H = H2;
            W = W2;
        }
    }
    /* x: (16, Tp, 128) */

    /* ── cnn: Conv2d(16->3, 3x3, pad 1) ── */
    {
        if (getenv("WUBU_RMVPE_DUMP")) {
            float mf = 0, mf2 = 0;
            for (int i = 0; i < 16 * H * W; i++) { float v = fabsf(x[i]); if (v > mf) mf = v; mf2 += v; }
            fprintf(stderr, "[diag] unet out: H=%d W=%d max|.|=%.4f mean|.|=%.4f\n",
                    H, W, mf, mf2 / (float)(16 * H * W));
        }
        float *c = (float *)malloc((size_t)3 * H * W * sizeof(float));
        if (!c) { free(x); return -1; }
        conv2d(x, 16, H, W, tdata(r, "cnn.weight"), tdata(r, "cnn.bias"), 3, 3, 1, c);
        free(x);

        /* transpose(1,2).flatten(-2): (H, 384) row-major */
        float *feat = (float *)malloc((size_t)H * 384 * sizeof(float));
        if (!feat) { free(c); return -1; }
        for (int t = 0; t < H; t++)
            for (int cc = 0; cc < 3; cc++)
                memcpy(feat + ((size_t)t * 3 + cc) * 128,
                       c + (size_t)cc * H * W + (size_t)t * W,
                       (size_t)W * sizeof(float));
        free(c);

        if (getenv("WUBU_RMVPE_DUMP")) {
            float mf = 0, mf2 = 0;
            for (int i = 0; i < H * 384; i++) { float v = fabsf(feat[i]); if (v > mf) mf = v; mf2 += v; }
            fprintf(stderr, "[diag] cnn->feat: H=%d max|.|=%.4f mean|.|=%.4f\n",
                    H, mf, mf2 / (float)(H * 384));
        }

        /* BiGRU(384->256) -> (H, 512) */
        float *gru_out = (float *)malloc((size_t)H * 512 * sizeof(float));
        if (!gru_out) { free(feat); return -1; }
        if (wubu_gru_forward(r->gru, feat, H, gru_out) != 0) { free(feat); free(gru_out); return -1; }
        free(feat);
        if (getenv("WUBU_RMVPE_DUMP")) {
            float mf = 0, mf2 = 0;
            for (int i = 0; i < H * 512; i++) { float v = fabsf(gru_out[i]); if (v > mf) mf = v; mf2 += v; }
            fprintf(stderr, "[diag] gru_out: max|.|=%.4f mean|.|=%.4f\n", mf, mf2 / (float)(H * 512));
            FILE *df = fopen("outputs/rvc_ref/c_gru.bin", "wb");
            if (df) { fwrite(gru_out, sizeof(float), (size_t)H * 512, df); fclose(df); }
        }

        /* Linear(512->360) + sigmoid -> (H, 360) */
        const float *fw = tdata(r, "fc.1.weight");
        const float *fb = tdata(r, "fc.1.bias");
        for (int t = 0; t < H; t++) {
            const float *gt = gru_out + (size_t)t * 512;
            for (int k = 0; k < 360; k++) {
                const float *wrow = fw + (size_t)k * 512;
                float acc = fb[k];
                for (int j = 0; j < 512; j++) acc += wrow[j] * gt[j];
                gru_out[(size_t)t * 360 + k] = 1.0f / (1.0f + expf(-acc));
            }
        }

        /* decode first T frames -> f0 */
        for (int t = 0; t < T && t < max_frames; t++)
            f0_out[t] = decode_row(gru_out + (size_t)t * 360, r->cents);
        if (getenv("WUBU_RMVPE_DUMP")) {
            for (int t = 0; t < T && t < 8; t++) {
                float mx = 0, sum = 0;
                const float *s = gru_out + (size_t)t * 360;
                for (int k = 0; k < 360; k++) { if (s[k] > mx) mx = s[k]; sum += s[k]; }
                fprintf(stderr, "[rmvpe-diag] frame %d: max=%.4f mean=%.4f f0=%.1f\n",
                        t, mx, sum / 360.0f, f0_out[t]);
            }
        }
        free(gru_out);
    }
    return T;
}
