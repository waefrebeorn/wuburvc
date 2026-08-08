/* wubu_rvc_hubert.c — Real HuBERT content encoder (C11).
 *
 * Faithful port of fairseq HubertModel for RVC's hubert_base.pt.
 * Zero dependencies at runtime; loads the WUBU flat binary produced by
 * tools/extract_hubert_weights.py.
 *
 * License: WaefreBeorn-UMV3
 */

#define _USE_MATH_DEFINES
#include "wubu_rvc_hubert.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* forward decls (primitives defined below, used by debug + extract) */
static void conv1d_nb(const float *in, int in_ch, int n,
                      const float *w, int out_ch, int k, int s, float *out);
static void layernorm_c(float *x, int C, int T,
                        const float *gamma, const float *beta, float eps);
static void groupnorm_c(float *x, int C, int T,
                        const float *gamma, const float *beta, float eps);
static void linear_c(const float *in, int in_d, int n,
                     const float *w, const float *b, int out_d, float *out);
static float gelu_c(float x);
static void pos_conv_apply(const WuBuHubert *h, const float *x, int T,
                           float *out, float *scratch);

/* ── WUBU binary loader (same format as extract_rvc_weights.py) ── */
typedef struct {
    char   name[256];
    float *data;
    int    dims[4];
    int    n_dims;
} WubTensor;

static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int parse_wubu(const char *path, WubTensor **out, int *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc((size_t)fsize);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)fsize, f) != (size_t)fsize) {
        free(buf); fclose(f); return -1;
    }
    fclose(f);

    if (memcmp(buf, "WUBU", 4) != 0) { free(buf); return -1; }
    uint32_t n = rd_u32(buf + 4);
    WubTensor *ts = (WubTensor *)calloc((size_t)n, sizeof(WubTensor));
    if (!ts) { free(buf); return -1; }

    size_t off = 8;
    int got = 0;
    for (uint32_t t = 0; t < n && off < (size_t)fsize; t++) {
        uint8_t nl = buf[off]; off += 1;
        if (off + nl > (size_t)fsize) break;
        memcpy(ts[got].name, buf + off, nl);
        ts[got].name[nl] = '\0';
        off += nl;

        uint32_t nd = rd_u32(buf + off); off += 4;
        uint32_t total = 1;
        for (uint32_t d = 0; d < nd && d < 4; d++) {
            ts[got].dims[d] = (int)rd_u32(buf + off);
            total *= (uint32_t)ts[got].dims[d];
            off += 4;
        }
        ts[got].n_dims = (int)nd;

        uint32_t dl = rd_u32(buf + off); off += 4;
        if (off + dl > (size_t)fsize) break;
        float *data = (float *)malloc((size_t)dl);
        if (!data) { off += dl; continue; }
        memcpy(data, buf + off, dl);
        off += dl;
        ts[got].data = data;
        got++;
    }
    free(buf);
    *out = ts;
    *n_out = got;
    return 0;
}

static void free_tensors(WubTensor *ts, int n) {
    if (!ts) return;
    for (int i = 0; i < n; i++) free(ts[i].data);
    free(ts);
}

static WubTensor *find_t(WubTensor *ts, int n, const char *name) {
    for (int i = 0; i < n; i++)
        if (strcmp(ts[i].name, name) == 0) return &ts[i];
    return NULL;
}

/* ── tiny tensor allocation helpers ── */
static float *dup_tensor(WubTensor *t) {
    if (!t || !t->data) return NULL;
    int total = 1;
    for (int d = 0; d < t->n_dims; d++) total *= t->dims[d];
    float *out = (float *)malloc((size_t)total * sizeof(float));
    if (out) memcpy(out, t->data, (size_t)total * sizeof(float));
    return out;
}

int wubu_hubert_load(WuBuHubert *h, const char *bin_path) {
    if (!h || !bin_path) return -1;
    memset(h, 0, sizeof(*h));

    WubTensor *ts = NULL;
    int n = 0;
    if (parse_wubu(bin_path, &ts, &n) != 0) return -1;

    char key[256];
    /* conv feature extractor blocks 0..6: conv + GroupNorm */
    int conv_in[7] = {1, 512, 512, 512, 512, 512, 512};
    int conv_k[7] = {10, 3, 3, 3, 3, 2, 2};
    int conv_s[7] = {5, 2, 2, 2, 2, 2, 2};
    for (int i = 0; i < 7; i++) {
        snprintf(key, sizeof(key), "feature_extractor.conv_layers.%d.0.weight", i);
        WubTensor *w = find_t(ts, n, key);
        h->conv_w[i] = dup_tensor(w);
        snprintf(key, sizeof(key), "feature_extractor.conv_layers.%d.2.weight", i);
        WubTensor *gnw = find_t(ts, n, key);
        h->conv_gn_w[i] = dup_tensor(gnw);
        snprintf(key, sizeof(key), "feature_extractor.conv_layers.%d.2.bias", i);
        WubTensor *gnb = find_t(ts, n, key);
        h->conv_gn_b[i] = dup_tensor(gnb);
        h->conv_k[i] = conv_k[i];
        h->conv_s[i] = conv_s[i];
        h->conv_in[i] = conv_in[i];
        if (!h->conv_w[i]) { free_tensors(ts, n); wubu_hubert_free(h); return -1; }
    }

    h->layer_norm_w = dup_tensor(find_t(ts, n, "layer_norm.weight"));
    h->layer_norm_b = dup_tensor(find_t(ts, n, "layer_norm.bias"));
    h->post_proj_w = dup_tensor(find_t(ts, n, "post_extract_proj.weight"));
    h->post_proj_b = dup_tensor(find_t(ts, n, "post_extract_proj.bias"));

    h->pos_w_g = dup_tensor(find_t(ts, n, "encoder.pos_conv.0.weight_g"));
    h->pos_w_v = dup_tensor(find_t(ts, n, "encoder.pos_conv.0.weight_v"));
    h->pos_b = dup_tensor(find_t(ts, n, "encoder.pos_conv.0.bias"));

    h->enc_ln_w = dup_tensor(find_t(ts, n, "encoder.layer_norm.weight"));
    h->enc_ln_b = dup_tensor(find_t(ts, n, "encoder.layer_norm.bias"));

    for (int i = 0; i < 12; i++) {
        snprintf(key, sizeof(key), "encoder.layers.%d.self_attn.q_proj.weight", i);
        h->attn_q_w[i] = dup_tensor(find_t(ts, n, key));
        snprintf(key, sizeof(key), "encoder.layers.%d.self_attn.q_proj.bias", i);
        h->attn_q_b[i] = dup_tensor(find_t(ts, n, key));
        snprintf(key, sizeof(key), "encoder.layers.%d.self_attn.k_proj.weight", i);
        h->attn_k_w[i] = dup_tensor(find_t(ts, n, key));
        snprintf(key, sizeof(key), "encoder.layers.%d.self_attn.k_proj.bias", i);
        h->attn_k_b[i] = dup_tensor(find_t(ts, n, key));
        snprintf(key, sizeof(key), "encoder.layers.%d.self_attn.v_proj.weight", i);
        h->attn_v_w[i] = dup_tensor(find_t(ts, n, key));
        snprintf(key, sizeof(key), "encoder.layers.%d.self_attn.v_proj.bias", i);
        h->attn_v_b[i] = dup_tensor(find_t(ts, n, key));
        snprintf(key, sizeof(key), "encoder.layers.%d.self_attn.out_proj.weight", i);
        h->attn_o_w[i] = dup_tensor(find_t(ts, n, key));
        snprintf(key, sizeof(key), "encoder.layers.%d.self_attn.out_proj.bias", i);
        h->attn_o_b[i] = dup_tensor(find_t(ts, n, key));
        snprintf(key, sizeof(key), "encoder.layers.%d.self_attn_layer_norm.weight", i);
        h->attn_ln_w[i] = dup_tensor(find_t(ts, n, key));
        snprintf(key, sizeof(key), "encoder.layers.%d.self_attn_layer_norm.bias", i);
        h->attn_ln_b[i] = dup_tensor(find_t(ts, n, key));
        snprintf(key, sizeof(key), "encoder.layers.%d.fc1.weight", i);
        h->fc1_w[i] = dup_tensor(find_t(ts, n, key));
        snprintf(key, sizeof(key), "encoder.layers.%d.fc1.bias", i);
        h->fc1_b[i] = dup_tensor(find_t(ts, n, key));
        snprintf(key, sizeof(key), "encoder.layers.%d.fc2.weight", i);
        h->fc2_w[i] = dup_tensor(find_t(ts, n, key));
        snprintf(key, sizeof(key), "encoder.layers.%d.fc2.bias", i);
        h->fc2_b[i] = dup_tensor(find_t(ts, n, key));
        snprintf(key, sizeof(key), "encoder.layers.%d.final_layer_norm.weight", i);
        h->fin_ln_w[i] = dup_tensor(find_t(ts, n, key));
        snprintf(key, sizeof(key), "encoder.layers.%d.final_layer_norm.bias", i);
        h->fin_ln_b[i] = dup_tensor(find_t(ts, n, key));
        if (!h->attn_q_w[i] || !h->attn_k_w[i] || !h->attn_v_w[i] ||
            !h->attn_o_w[i] || !h->fc1_w[i] || !h->fc2_w[i]) {
            free_tensors(ts, n);
            wubu_hubert_free(h);
            return -1;
        }
    }

    h->final_proj_w = dup_tensor(find_t(ts, n, "final_proj.weight"));
    h->final_proj_b = dup_tensor(find_t(ts, n, "final_proj.bias"));

    free_tensors(ts, n);
    h->loaded = 1;
    return 0;
}

void wubu_hubert_free(WuBuHubert *h) {
    if (!h) return;
    for (int i = 0; i < 7; i++) {
        free(h->conv_w[i]); h->conv_w[i] = NULL;
        free(h->conv_gn_w[i]); h->conv_gn_w[i] = NULL;
        free(h->conv_gn_b[i]); h->conv_gn_b[i] = NULL;
    }
    free(h->layer_norm_w); free(h->layer_norm_b);
    free(h->post_proj_w); free(h->post_proj_b);
    free(h->pos_w_g); free(h->pos_w_v); free(h->pos_b);
    free(h->enc_ln_w); free(h->enc_ln_b);
    for (int i = 0; i < 12; i++) {
        free(h->attn_q_w[i]); free(h->attn_q_b[i]);
        free(h->attn_k_w[i]); free(h->attn_k_b[i]);
        free(h->attn_v_w[i]); free(h->attn_v_b[i]);
        free(h->attn_o_w[i]); free(h->attn_o_b[i]);
        free(h->attn_ln_w[i]); free(h->attn_ln_b[i]);
        free(h->fc1_w[i]); free(h->fc1_b[i]);
        free(h->fc2_w[i]); free(h->fc2_b[i]);
        free(h->fin_ln_w[i]); free(h->fin_ln_b[i]);
    }
    free(h->final_proj_w); free(h->final_proj_b);
    h->loaded = 0;
}

int wubu_hubert_output_length(int n_samples) {
    int L = n_samples;
    int k[7] = {10, 3, 3, 3, 3, 2, 2};
    int s[7] = {5, 2, 2, 2, 2, 2, 2};
    for (int i = 0; i < 7; i++) {
        L = (L - k[i]) / s[i] + 1;
        if (L < 1) L = 1;
    }
    return L;
}

/* ── debug: dump intermediate stages as raw float files for C-vs-torch diff ── */
static void dbg_write(const char *name, const float *data, size_t n) {
    char path[512];
    snprintf(path, sizeof(path), "outputs/rvc_ref/%s", name);
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(data, sizeof(float), n, f); fclose(f); }
}

static void hubert_layer(const WuBuHubert *h, float *x768, int L, int T,
                         float *q, float *k, float *v, float *o,
                         float *scores, float *ffn, float *y);

int wubu_hubert_debug_dump(const WuBuHubert *h,
                           const float *pcm, int n_samples) {
    if (!h || !h->loaded || !pcm || n_samples < 100) return -1;
    int T0 = n_samples;
    int T = wubu_hubert_output_length(T0);
    if (T < 1) T = 1;
    const int C = 512;
    int max_inter = (T0 - 10) / 5 + 1;
    if (max_inter < T) max_inter = T;

    float *in = (float *)calloc((size_t)T0, sizeof(float));
    float *bufA = (float *)calloc((size_t)C * max_inter, sizeof(float));
    float *bufB = (float *)calloc((size_t)C * max_inter, sizeof(float));
    float *feats = (float *)calloc((size_t)768 * T, sizeof(float));
    float *feats2 = (float *)calloc((size_t)768 * T, sizeof(float));
    float *x768 = (float *)calloc((size_t)768 * T, sizeof(float));
    float *scratch = (float *)calloc((size_t)(768 * (T + 128) + 768 * 768 * 8), sizeof(float));
    float *q = (float *)calloc((size_t)768 * T, sizeof(float));
    float *k = (float *)calloc((size_t)768 * T, sizeof(float));
    float *v = (float *)calloc((size_t)768 * T, sizeof(float));
    float *o = (float *)calloc((size_t)768 * T, sizeof(float));
    float *scores = (float *)calloc((size_t)12 * T * T, sizeof(float));
    float *ffn = (float *)calloc((size_t)3072 * T, sizeof(float));
    float *y = (float *)calloc((size_t)768 * T, sizeof(float));
    if (!in || !bufA || !bufB || !feats || !feats2 || !x768 || !scratch ||
        !q || !k || !v || !o || !scores || !ffn || !y) return -1;
    memcpy(in, pcm, (size_t)n_samples * sizeof(float));

    /* conv front-end */
    float *src = in; int src_n = T0; int src_ch = 1;
    for (int blk = 0; blk < 7; blk++) {
        int out_ch = 512;
        int kk = h->conv_k[blk], ss = h->conv_s[blk];
        int n_out = (src_n - kk) / ss + 1;
        if (n_out < 1) n_out = 1;
        float *dst = (blk % 2 == 0) ? bufA : bufB;
        conv1d_nb(src, src_ch, src_n, h->conv_w[blk], out_ch, kk, ss, dst);
        if (blk == 0) {
            /* dump raw conv0 (before GroupNorm/GELU) for parity check */
            dbg_write("c_conv0_raw.npy", dst, (size_t)512 * n_out);
            groupnorm_c(dst, out_ch, n_out, h->conv_gn_w[blk], h->conv_gn_b[blk], 1e-5f);
            dbg_write("c_conv0_gn.npy", dst, (size_t)512 * n_out);
        }
        for (int i = 0; i < out_ch * n_out; i++) dst[i] = gelu_c(dst[i]);
        {
            char nm[64];
            snprintf(nm, sizeof(nm), "c_blk%d.npy", blk);
            dbg_write(nm, dst, (size_t)out_ch * n_out);
        }
        src = dst; src_n = n_out; src_ch = out_ch;
    }
    dbg_write("c_conv.npy", src, (size_t)512 * T);
    layernorm_c(src, 512, T, h->layer_norm_w, h->layer_norm_b, 1e-5f);
    dbg_write("c_conv_ln.npy", src, (size_t)512 * T);
    linear_c(src, 512, T, h->post_proj_w, h->post_proj_b, 768, feats);
    dbg_write("c_postproj.npy", feats, (size_t)768 * T);

    pos_conv_apply(h, feats, T, feats2, scratch);
    dbg_write("c_posconv.npy", feats2, (size_t)768 * T);
    for (int i = 0; i < 768 * T; i++) feats2[i] += feats[i];
    layernorm_c(feats2, 768, T, h->enc_ln_w, h->enc_ln_b, 1e-5f);
    memcpy(x768, feats2, (size_t)768 * T * sizeof(float));
    dbg_write("c_preln.npy", x768, (size_t)768 * T);

    /* pad to even is a masked no-op in fairseq (padded position gets
     * -inf attention), so run unpadded — bit-identical. */
    /* layer 1..12, dumping each */
    for (int L = 0; L < 12; L++) {
        hubert_layer(h, x768, L, T, q, k, v, o, scores, ffn, y);
        char nm[64];
        snprintf(nm, sizeof(nm), "c_layer%d.npy", L + 1);
        dbg_write(nm, x768, (size_t)768 * T);
    }
    dbg_write("c_final.npy", x768, (size_t)768 * T);

    free(in); free(bufA); free(bufB); free(feats); free(feats2); free(x768); free(scratch);
    free(q); free(k); free(v); free(o); free(scores); free(ffn); free(y);
    return T;
}

/* hubert_layer: run one transformer layer in place (post-LN, fairseq order) */
static void hubert_layer(const WuBuHubert *h, float *x768, int L, int T,
                         float *q, float *k, float *v, float *o,
                         float *scores, float *ffn, float *y) {
    const int H = 12, DH = 64;
    linear_c(x768, 768, T, h->attn_q_w[L], h->attn_q_b[L], 768, q);
    linear_c(x768, 768, T, h->attn_k_w[L], h->attn_k_b[L], 768, k);
    linear_c(x768, 768, T, h->attn_v_w[L], h->attn_v_b[L], 768, v);
#pragma omp parallel for schedule(static)
    for (int hh = 0; hh < H; hh++) {
        float *srow = scores + (size_t)hh * T * T;
        for (int ti = 0; ti < T; ti++) {
            for (int tj = 0; tj < T; tj++) {
                float acc = 0;
                for (int d = 0; d < DH; d++)
                    acc += q[(size_t)(hh * DH + d) * T + ti] * k[(size_t)(hh * DH + d) * T + tj];
                srow[(size_t)ti * T + tj] = acc / sqrtf((float)DH);
            }
        }
    }
#pragma omp parallel for schedule(static)
    for (int hh = 0; hh < H; hh++) {
        float *srow = scores + (size_t)hh * T * T;
        for (int ti = 0; ti < T; ti++) {
            float mx = -1e30f;
            for (int tj = 0; tj < T; tj++) if (srow[(size_t)ti * T + tj] > mx) mx = srow[(size_t)ti * T + tj];
            float sum = 0;
            for (int tj = 0; tj < T; tj++) { float e = expf(srow[(size_t)ti * T + tj] - mx); srow[(size_t)ti * T + tj] = e; sum += e; }
            for (int tj = 0; tj < T; tj++) srow[(size_t)ti * T + tj] /= (sum + 1e-12f);
        }
    }
#pragma omp parallel for schedule(static)
    for (int hh = 0; hh < H; hh++) {
        const float *srow = scores + (size_t)hh * T * T;
        for (int ti = 0; ti < T; ti++)
            for (int d = 0; d < DH; d++) {
                float acc = 0;
                for (int tj = 0; tj < T; tj++)
                    acc += srow[(size_t)ti * T + tj] * v[(size_t)(hh * DH + d) * T + tj];
                o[(size_t)(hh * DH + d) * T + ti] = acc;
            }
    }
    linear_c(o, 768, T, h->attn_o_w[L], h->attn_o_b[L], 768, y);
    for (int i = 0; i < 768 * T; i++) y[i] += x768[i];
    layernorm_c(y, 768, T, h->attn_ln_w[L], h->attn_ln_b[L], 1e-5f);
    linear_c(y, 768, T, h->fc1_w[L], h->fc1_b[L], 3072, ffn);
    for (int i = 0; i < 3072 * T; i++) ffn[i] = gelu_c(ffn[i]);
    linear_c(ffn, 3072, T, h->fc2_w[L], h->fc2_b[L], 768, x768);
    for (int i = 0; i < 768 * T; i++) x768[i] += y[i];
    layernorm_c(x768, 768, T, h->fin_ln_w[L], h->fin_ln_b[L], 1e-5f);
}


/* ── primitives (all [C, T] column-major, single batch) ── */

static void conv1d_nb(const float *in, int in_ch, int n,
                      const float *w, int out_ch, int k, int s,
                      float *out) {
    int n_out = (n - k) / s + 1;
    if (n_out < 1) n_out = 1;
    memset(out, 0, (size_t)out_ch * n_out * sizeof(float));
    /* oc → ic → tap → j (j innermost): sequential in/out rows, vectorizes.
     * Old oc → j → tap → ic order touched in_ch rows per sample — cache hell
     * on the 512ch front-end over 44k samples. */
#pragma omp parallel for schedule(static) if(out_ch >= 32 && n_out >= 256)
    for (int oc = 0; oc < out_ch; oc++) {
        float *orow = out + (size_t)oc * n_out;
        const float *wv = w + (size_t)oc * in_ch * k;
        for (int ic = 0; ic < in_ch; ic++) {
            const float *irow = in + (size_t)ic * n;
            for (int tap = 0; tap < k; tap++) {
                float wt = wv[(size_t)ic * k + tap];
                /* src = j*s + tap; exact valid range [0, n) — no inner
                 * branch, vectorizes. */
                int j_lo = 0, j_hi = n_out;
                if (tap >= n) { j_hi = 0; }
                else {
                    if ((n - tap + s - 1) / s < j_hi)
                        j_hi = (n - tap + s - 1) / s;
                }
                for (int j = j_lo; j < j_hi; j++)
                    orow[j] += irow[j * s + tap] * wt;
            }
        }
    }
}

/* LayerNorm over channels (transformer norms + post-conv layer_norm):
 * normalize across C at each time step, gamma/beta per channel. */
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

/* GroupNorm with num_groups == num_channels (fairseq block 0):
 * normalize EACH channel independently over the spatial (time) axis.
 * This is NOT LayerNorm-over-channels — the axis matters. */
static void groupnorm_c(float *x, int C, int T,
                        const float *gamma, const float *beta, float eps) {
    for (int c = 0; c < C; c++) {
        float *row = x + (size_t)c * T;
        float mean = 0, var = 0;
        for (int j = 0; j < T; j++) mean += row[j];
        mean /= T;
        for (int j = 0; j < T; j++) { float d = row[j] - mean; var += d * d; }
        var /= T;
        float inv = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < T; j++)
            row[j] = (row[j] - mean) * inv * (gamma ? gamma[c] : 1.0f) + (beta ? beta[c] : 0.0f);
    }
}

static void linear_c(const float *in, int in_d, int n,
                     const float *w, const float *b, int out_d, float *out) {
#pragma omp parallel for schedule(static) if(out_d >= 32 && n >= 64)
    for (int o = 0; o < out_d; o++) {
        const float *wv = w + (size_t)o * in_d;
        float bias = b ? b[o] : 0.0f;
        float *orow = out + (size_t)o * n;
        for (int j = 0; j < n; j++) orow[j] = bias;
        /* i outer, j inner: in[i*n+j] sequential per row → vectorizes. */
        for (int i = 0; i < in_d; i++) {
            float wt = wv[i];
            const float *irow = in + (size_t)i * n;
            for (int j = 0; j < n; j++) orow[j] += irow[j] * wt;
        }
    }
}

/* GELU as fairseq defines it (fairseq/modules/gelu.py) for hubert:
 * x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * NOTE: the cubic term is x^3 (fairseq uses pow(x,3)); the erf form of
 * torch.nn.GELU is NOT what hubert_base.pt was trained/run with. */
static float gelu_c(float x) { return x * (0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)))); }

/* ── pos_conv weight_norm (dim=2): W[oc,ic,k] = g[k] * v[oc,ic,k]/||v[oc,ic,:]|| ── */
static void pos_conv_apply(const WuBuHubert *h, const float *x, int T,
                           float *out, float *scratch) {
    /* x: [768, T] -> conv1d groups=16, k=128, pad=64 -> SamePad (remove 0
     * since k even) -> GELU. Effective weight precomputed per (oc,ic). */
    /* De-normalized weight: 768*48*128 = 4.7M floats — compute on the fly. */
    const int out_ch = 768, in_per_g = 48, k = 128;
    /* input is padded: T + 2*64 */
    int Tp = T + 2 * (k / 2);
    float *xpad = scratch;                       /* [768, Tp] */
    memset(xpad, 0, (size_t)768 * Tp * sizeof(float));
    for (int c = 0; c < 768; c++)
        for (int j = 0; j < T; j++)
            xpad[(size_t)c * Tp + (j + k / 2)] = x[(size_t)c * T + j];

    /* group conv: group g (48 in) -> 48 out per group; 16 groups.
     * out[oc][j] = sum_{ic} sum_{tap} xpad[icg][j+tap] * W[oc][ic][tap],
     * W[oc][ic][tap] = g[tap] * v[oc][ic][tap] / ||v[:,:,tap]||
     * (torch._weight_norm normalizes over dims 0,1 — per tap across ALL
     * (oc,ic) pairs — NOT per row; g is a per-tap gain). */
    memset(out, 0, (size_t)out_ch * T * sizeof(float));
    /* precompute per-tap norms across all (oc, ic) */
    float *tap_norm = (float *)malloc((size_t)k * sizeof(float));
    if (!tap_norm) return;
    for (int tap = 0; tap < k; tap++) {
        double acc = 0;
        for (int oc = 0; oc < out_ch; oc++) {
            const float *wv = h->pos_w_v + (size_t)oc * in_per_g * k;
            for (int ic = 0; ic < in_per_g; ic++) {
                float wvval = wv[(size_t)ic * k + tap];
                acc += (double)wvval * wvval;
            }
        }
        tap_norm[tap] = (float)(sqrt(acc) + 1e-8);
    }
#pragma omp parallel for schedule(static)
    for (int oc = 0; oc < out_ch; oc++) {
        int grp = oc / 48;
        const float *wv = h->pos_w_v + (size_t)oc * in_per_g * k;
        float *orow = out + (size_t)oc * T;
        float bias = h->pos_b ? h->pos_b[oc] : 0.0f;
        for (int j = 0; j < T; j++) orow[j] = bias;
        for (int ic = 0; ic < in_per_g; ic++) {
            int icg = grp * 48 + ic;
            const float *xin = xpad + (size_t)icg * Tp;
            const float *wvrow = wv + (size_t)ic * k;
            /* tap-outer, j-inner: xin is sequential, scaled weight is a
             * scalar per (ic,tap) — vectorizes instead of per-j division. */
            for (int tap = 0; tap < k; tap++) {
                float g = h->pos_w_g ? h->pos_w_g[tap] : 1.0f;
                float wt = g * wvrow[tap] / tap_norm[tap];
                for (int j = 0; j < T; j++)
                    orow[j] += xin[j + tap] * wt;
            }
        }
        for (int j = 0; j < T; j++) orow[j] = gelu_c(orow[j]);
    }
    free(tap_norm);
}

int wubu_hubert_extract_real(const WuBuHubert *h,
                             const float *pcm, int n_samples,
                             int version,
                             float *feats_out, int max_feats) {
    if (!h || !h->loaded || !pcm || !feats_out || n_samples < 100) return -1;
    int T0 = n_samples;
    int T = wubu_hubert_output_length(T0);
    if (T < 1) T = 1;
    int dim = (version == 1) ? 256 : 768;
    if (max_feats < T * dim) return -1;

    const int C = 512;
    float *feats = (float *)calloc((size_t)768 * T, sizeof(float)); /* post-proj */
    float *feats2 = (float *)calloc((size_t)768 * T, sizeof(float));
    float *x768 = (float *)calloc((size_t)768 * T, sizeof(float));
    float *scratch = (float *)calloc((size_t)(768 * (T + 128) + 768 * 768 * 8), sizeof(float));
    if (!feats || !feats2 || !x768 || !scratch) {
        free(feats); free(feats2); free(x768); free(scratch);
        return -1;
    }

    /* 1. conv feature extractor (7 blocks), ping-pong A/B.
     * NOTE: the first conv produces the LARGEST intermediate (T0/5 frames),
     * so size the ping-pong buffers for that, not the final T. */
    int max_inter = (T0 - 10) / 5 + 1;
    if (max_inter < T) max_inter = T;
    float *in = (float *)calloc((size_t)1 * T0, sizeof(float));
    float *bufA = (float *)calloc((size_t)C * max_inter, sizeof(float));
    float *bufB = (float *)calloc((size_t)C * max_inter, sizeof(float));
    if (!in || !bufA || !bufB) {
        free(in); free(bufA); free(bufB);
        free(feats); free(feats2); free(x768); free(scratch);
        return -1;
    }
    memcpy(in, pcm, (size_t)n_samples * sizeof(float));

    float *src = in;       /* [1, T0] initially */
    int src_n = T0;
    int src_ch = 1;
    for (int blk = 0; blk < 7; blk++) {
        int out_ch = 512;
        int k = h->conv_k[blk], s = h->conv_s[blk];
        int in_ch = src_ch;
        int n_out = (src_n - k) / s + 1;
        if (n_out < 1) n_out = 1;
        float *dst = (blk % 2 == 0) ? bufA : bufB;
        conv1d_nb(src, in_ch, src_n, h->conv_w[blk], out_ch, k, s, dst);
        /* fairseq mode="default": ONLY block 0 has GroupNorm(512,512);
         * blocks 1..6 are conv + GELU with NO norm. */
        if (blk == 0) {
            groupnorm_c(dst, out_ch, n_out, h->conv_gn_w[blk], h->conv_gn_b[blk], 1e-5f);
        }
        for (int i = 0; i < out_ch * n_out; i++) dst[i] = gelu_c(dst[i]);
        src = dst;
        src_n = n_out;
        src_ch = out_ch;
    }
    /* src now holds [512, T] */

    /* layer_norm over channels + post_extract_proj: [768, 512] x [512, T] -> [768, T] */
    layernorm_c(src, 512, T, h->layer_norm_w, h->layer_norm_b, 1e-5f);
    linear_c(src, 512, T, h->post_proj_w, h->post_proj_b, 768, feats);

    /* 2. pos_conv + pre-norm */
    pos_conv_apply(h, feats, T, feats2, scratch);
    for (int i = 0; i < 768 * T; i++) feats2[i] += feats[i];
    layernorm_c(feats2, 768, T, h->enc_ln_w, h->enc_ln_b, 1e-5f);
    memcpy(x768, feats2, (size_t)768 * T * sizeof(float));

    /* 3. 12 transformer layers (post-LN). fairseq pads to a multiple of
     * required_seq_len_multiple=2 but the padded position is MASKED in
     * attention (score=-inf -> softmax weight 0), so it has NO effect on
     * real positions. Running unpadded is therefore bit-identical. */
    const int H = 12, DH = 64;
    float *q = (float *)calloc((size_t)768 * T, sizeof(float));
    float *k = (float *)calloc((size_t)768 * T, sizeof(float));
    float *v = (float *)calloc((size_t)768 * T, sizeof(float));
    float *o = (float *)calloc((size_t)768 * T, sizeof(float));
    float *scores = (float *)calloc((size_t)H * T * T, sizeof(float));
    float *ffn = (float *)calloc((size_t)3072 * T, sizeof(float));
    float *y = (float *)calloc((size_t)768 * T, sizeof(float));
    if (!q || !k || !v || !o || !scores || !ffn || !y) {
        free(in); free(bufA); free(bufB); free(feats); free(feats2); free(x768); free(scratch);
        free(q); free(k); free(v); free(o); free(scores); free(ffn); free(y);
        return -1;
    }

    int n_layers = (version == 1) ? 9 : 12; /* v1: layer 9 (index 8? see below) */
    /* fairseq: layer index = output_layer (0-based); v1 uses layer 9 -> run
     * layers 0..8 then break at i==9? Actually extract_features breaks when
     * i == tgt_layer AFTER running that layer. v2: output_layer 12 -> never
     * reached (12 layers, indices 0..11) -> full 12. v1: output_layer 9 ->
     * runs layers 0..9 inclusive (10 layers), stops at index 9. */
    int stop_at = (version == 1) ? 9 : 12;
    for (int L = 0; L < 12; L++) {
        hubert_layer(h, x768, L, T, q, k, v, o, scores, ffn, y);
        if (L == stop_at - 1) break; /* reached target layer */
    }

    /* 4. output: v2 -> 768 features; v1 -> final_proj -> 256 */
    if (version == 1) {
        linear_c(x768, 768, T, h->final_proj_w, h->final_proj_b, 256, feats2);
        for (int j = 0; j < T; j++)
            for (int d = 0; d < 256; d++)
                feats_out[(size_t)j * 256 + d] = feats2[(size_t)d * T + j];
    } else {
        for (int j = 0; j < T; j++)
            for (int d = 0; d < 768; d++)
                feats_out[(size_t)j * 768 + d] = x768[(size_t)d * T + j];
    }

    free(in); free(bufA); free(bufB);
    free(feats); free(feats2); free(x768); free(scratch);
    free(q); free(k); free(v); free(o); free(scores); free(ffn); free(y);
    return T;
}
