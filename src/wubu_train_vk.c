/* wubu_train_vk.c — Vulkan-accelerated RVC training step.
 *
 * Mirrors the CPU training math in wubu_train.c EXACTLY (Triple-DA: the CPU
 * train_step is the reference, the CUDA backend in wubu_train_cuda.cu is the
 * cross-checked twin). Provides:
 *
 *   1. wubu_train_forward_vk  — cached decoder forward (conv_pre → lrelu →
 *      upsample convT+MRF ×4 → conv_post → tanh). NO cond/sine/noise (the
 *      training reference is decoder_forward, not the inference generator).
 *   2. wubu_train_backward_vk — conv1d_bwd / convT1d_bwd / lrelu_bwd /
 *      tanh_bwd via the Vulkan backward pipelines, writing grads into the
 *      training registry (identical layout/keys to the CPU+CUDA paths).
 *   3. wubu_train_step_vk     — forward → MSE → backward → AdamW.
 *
 * All VK ops are the synchronous host-buffer public API (wubu_vk_conv1d /
 * wubu_vk_convt1d / wubu_vk_act / wubu_vk_elt / wubu_vk_bwd_*). The cache is
 * host-side float arrays (the VK module uploads/downloads per op).
 *
 * License: WaefreBeorn-UMV3
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "wubu_rvc_parity.h"
#include "wubu_train.h"
#include "wubu_vk.h"

/* manual key builder (no snprintf — avoids MSVC CRT deps when linking the
 * nvcc object with MinGW). Builds e.g. "dec.ups.3.bias". The inference file
 * (wubu_rvc_cuda.cu) already defines kfmt; when compiled INTO that same
 * object (build/cuda_build.bat) the training copy must not collide. */
static void tfmt(char *o, const char *pre, int a,
                 const char *mid, int b, const char *suf) {
    char *p = o;
    const char *q;
    for (q = pre; *q; q++) *p++ = *q;
    if (a >= 0) {
        if (a == 0) *p++ = '0';
        else { char t[8]; int n = 0; while (a) { t[n++] = (char)('0' + a % 10); a /= 10; } while (n) *p++ = t[--n]; }
    }
    for (q = mid; *q; q++) *p++ = *q;
    if (b >= 0) {
        if (b == 0) *p++ = '0';
        else { char t[8]; int n = 0; while (b) { t[n++] = (char)('0' + b % 10); b /= 10; } while (n) *p++ = t[--n]; }
    }
    for (q = suf; *q; q++) *p++ = *q;
    *p = 0;
}

struct TrainCacheVk {
    int n_frames;
    int n_stage[4];
    int ch_in[4], ch_out[4];
    int k[4], s[4], p[4];
    int n_ups;
    float *pre_out;      /* (init_ch, F) after conv_pre, PRE-lrelu */
    float *act_in[5];    /* lrelu output per stage (act_in[0]=lrelu(pre_out)) */
    float *stage_out[4]; /* MRF average output per stage (pre-lrelu into act) */
    float *ups_out[4];   /* convT output per stage (pre-lrelu) — MRF input */
    float *mrf_a1[4][3][3], *mrf_c1[4][3][3], *mrf_c1o[4][3][3];
    float *mrf_a2[4][3][3], *mrf_c2o[4][3][3]; /* MRF intermediates for backward */
    float *post_in;      /* lrelu(stage_out[3]) — conv_post input */
    float *audio;        /* final tanh output (host) */
};

typedef struct TrainCacheVk TrainCacheVk;

/* ── forward kernels via the public VK ops ─────────────────────────── */

static int vk_conv1d_fwd(WuBuVk *vk,
                         const float *in, int in_ch, int n,
                         const float *w, const float *b,
                         int out_ch, int k, int stride, int pad, int dil,
                         float *out, int n_out) {
    return wubu_vk_conv1d(vk, in, in_ch, n, w, b, out_ch, k, stride, pad, dil,
                          out, n_out);
}

static int vk_convt1d_fwd(WuBuVk *vk,
                          const float *in, int in_ch, int n,
                          const float *w, const float *b,
                          int out_ch, int k, int stride, int pad,
                          float *out, int n_out) {
    return wubu_vk_convt1d(vk, in, in_ch, n, w, b, out_ch, k, stride, pad,
                           out, n_out);
}

static void vk_lrelu(WuBuVk *vk, float *x, int n_ch, int n) {
    wubu_vk_act(vk, x, NULL, 0, n_ch, n, 0.0f);
}

static void vk_tanh(WuBuVk *vk, float *x, int n_ch, int n) {
    wubu_vk_act(vk, x, NULL, 2, n_ch, n, 0.0f);
}

/* ── MRF forward helper (one stage): 3 stacks x 3 pairs, avg → stage_out ──
 * Pair math (EXACT match to wubu_train.c mrf_pair_fwd):
 *   a1 = lrelu(p); c1o = conv1(a1); a2 = lrelu(c1o); c2o = conv2(a2) + p
 * Intermediates cached for backward. */
static int mrf_stage_fwd_vk(WuBuVk *vk, WuBuRVCModel *model, int L,
                            const float *stack_in, float *stage_out,
                            TrainCacheVk *cache, int ch, int n) {
    const int dils[3] = { 1, 3, 5 };
    for (int s = 0; s < 3; s++) {
        int rb = L * 3 + s;
        const float *cur = stack_in;
        for (int cp = 0; cp < 3; cp++) {
            char key[128];
            tfmt(key, "dec.resblocks.", rb, ".convs1.", cp, ".weight_v");
            const RVCTensor *c1w = wubu_rvc_find_tensor(model, key);
            tfmt(key, "dec.resblocks.", rb, ".convs2.", cp, ".weight_v");
            const RVCTensor *c2w = wubu_rvc_find_tensor(model, key);
            if (!c1w || !c1w->data || !c2w || !c2w->data) return -1;
            int k = c1w->dims[2];
            int dil = dils[cp];
            int pad1 = dil * (k - 1) / 2;
            int pad2 = k / 2;
            tfmt(key, "dec.resblocks.", rb, ".convs1.", cp, ".bias");
            const RVCTensor *c1b = wubu_rvc_find_tensor(model, key);
            tfmt(key, "dec.resblocks.", rb, ".convs2.", cp, ".bias");
            const RVCTensor *c2b = wubu_rvc_find_tensor(model, key);
            /* allocate MRF intermediates (ch*n each) on first use */
            if (!cache->mrf_c1[L][s][cp]) {
                cache->mrf_c1[L][s][cp] = (float *)malloc((size_t)ch * n * sizeof(float));
                cache->mrf_a1[L][s][cp] = (float *)malloc((size_t)ch * n * sizeof(float));
                cache->mrf_c1o[L][s][cp] = (float *)malloc((size_t)ch * n * sizeof(float));
                cache->mrf_a2[L][s][cp] = (float *)malloc((size_t)ch * n * sizeof(float));
                cache->mrf_c2o[L][s][cp] = (float *)malloc((size_t)ch * n * sizeof(float));
                if (!cache->mrf_c1[L][s][cp] || !cache->mrf_a1[L][s][cp] ||
                    !cache->mrf_c1o[L][s][cp] || !cache->mrf_a2[L][s][cp] ||
                    !cache->mrf_c2o[L][s][cp]) return -1;
            }
            /* mrf_c1 = p (pair input, PRE-lrelu — for backward lrelu_bwd) */
            memcpy(cache->mrf_c1[L][s][cp], cur, (size_t)ch * n * sizeof(float));
            /* a1 = lrelu(p) */
            memcpy(cache->mrf_a1[L][s][cp], cur, (size_t)ch * n * sizeof(float));
            vk_lrelu(vk, cache->mrf_a1[L][s][cp], ch, n);
            /* c1o = conv1(a1) */
            vk_conv1d_fwd(vk, cache->mrf_a1[L][s][cp], ch, n, c1w->data,
                          (c1b && c1b->data) ? c1b->data : NULL,
                          ch, k, 1, pad1, dil, cache->mrf_c1o[L][s][cp], n);
            /* a2 = lrelu(c1o) */
            memcpy(cache->mrf_a2[L][s][cp], cache->mrf_c1o[L][s][cp], (size_t)ch * n * sizeof(float));
            vk_lrelu(vk, cache->mrf_a2[L][s][cp], ch, n);
            /* c2o = conv2(a2) + p (residual from ORIGINAL input) */
            vk_conv1d_fwd(vk, cache->mrf_a2[L][s][cp], ch, n, c2w->data,
                          (c2b && c2b->data) ? c2b->data : NULL,
                          ch, k, 1, pad2, 1, cache->mrf_c2o[L][s][cp], n);
            for (size_t i = 0; i < (size_t)ch * n; i++)
                cache->mrf_c2o[L][s][cp][i] += cur[i];
            cur = cache->mrf_c2o[L][s][cp];
        }
        /* accumulate this stack into stage_out (1/3 each) */
        if (s == 0)
            memcpy(stage_out, cur, (size_t)ch * n * sizeof(float));
        else
            for (size_t i = 0; i < (size_t)ch * n; i++)
                stage_out[i] += cur[i];
    }
    for (size_t i = 0; i < (size_t)ch * n; i++)
        stage_out[i] /= 3.0f;
    return 0;
}

/* ── cached forward ───────────────────────────────────────────────── */
int wubu_train_forward_vk(WuBuVk *vk, WuBuRVCModel *model, const float *mel_in,
                          int n_frames, float *audio, int max_samples,
                          TrainCacheVk *cache) {
    if (!vk || !model || !mel_in || !audio || !cache) return -1;
    memset(cache, 0, sizeof(*cache));
    cache->n_frames = n_frames;

    const RVCTensor *pre_w = wubu_rvc_find_tensor(model, "dec.conv_pre.weight");
    const RVCTensor *pre_b = wubu_rvc_find_tensor(model, "dec.conv_pre.bias");
    if (!pre_w || !pre_w->data) return -1;
    int pre_in = pre_w->dims[1], init_ch = pre_w->dims[0], pk = pre_w->dims[2];
    int n_ups = model->n_upsample_layers > 0 ? model->n_upsample_layers : 4;
    cache->n_ups = n_ups;
    int rate[4] = { 10, 10, 2, 2 }, kk[4] = { 16, 16, 4, 4 };
    for (int L = 0; L < n_ups && L < 4; L++) {
        if (model->upsample_rates[L] > 0) rate[L] = model->upsample_rates[L];
        if (model->upsample_kernel_sizes[L] > 0) kk[L] = model->upsample_kernel_sizes[L];
    }
    int ch_in[4] = { init_ch, init_ch / 2, init_ch / 4, init_ch / 8 };
    int ch_out[4] = { init_ch / 2, init_ch / 4, init_ch / 8, init_ch / 16 };

    /* conv_pre: mel_in col-major (pre_in, F) → (init_ch, F), PRE-lrelu in cache */
    cache->pre_out = (float *)malloc((size_t)init_ch * n_frames * sizeof(float));
    if (!cache->pre_out) return -1;
    if (vk_conv1d_fwd(vk, mel_in, pre_in, n_frames, pre_w->data,
                      (pre_b && pre_b->data) ? pre_b->data : NULL,
                      init_ch, pk, 1, pk / 2, 1, cache->pre_out, n_frames) != 0)
        return -1;
    cache->act_in[0] = (float *)malloc((size_t)init_ch * n_frames * sizeof(float));
    if (!cache->act_in[0]) return -1;
    memcpy(cache->act_in[0], cache->pre_out, (size_t)init_ch * n_frames * sizeof(float));
    vk_lrelu(vk, cache->act_in[0], init_ch, n_frames);

    /* upsample blocks */
    int cur_ch = init_ch, cur_n = n_frames;
    for (int L = 0; L < n_ups && L < 4; L++) {
        int out_ch = ch_out[L];
        int out_n = (cur_n - 1) * rate[L] - 2 * ((kk[L] - rate[L]) / 2) + kk[L];
        cache->ch_in[L] = cur_ch; cache->ch_out[L] = out_ch;
        cache->k[L] = kk[L]; cache->s[L] = rate[L]; cache->p[L] = (kk[L] - rate[L]) / 2;
        cache->n_stage[L] = out_n;
        cache->ups_out[L] = (float *)malloc((size_t)out_ch * out_n * sizeof(float));
        cache->stage_out[L] = (float *)malloc((size_t)out_ch * out_n * sizeof(float));
        if (!cache->ups_out[L] || !cache->stage_out[L]) return -1;
        char key[128];
        tfmt(key, "dec.ups.", L, "", -1, ".bias");
        const RVCTensor *ub = wubu_rvc_find_tensor(model, key);
        if (vk_convt1d_fwd(vk, cache->act_in[L], cur_ch, cur_n,
                           model->hifi_upsample_denorm[L],
                           (ub && ub->data) ? ub->data : NULL,
                           out_ch, kk[L], rate[L], (kk[L] - rate[L]) / 2,
                           cache->ups_out[L], out_n) != 0)
            return -1;
        if (mrf_stage_fwd_vk(vk, model, L, cache->ups_out[L], cache->stage_out[L],
                             cache, out_ch, out_n) != 0)
            return -1;
        if (L < n_ups - 1) {
            cache->act_in[L + 1] = (float *)malloc((size_t)out_ch * out_n * sizeof(float));
            if (!cache->act_in[L + 1]) return -1;
            memcpy(cache->act_in[L + 1], cache->stage_out[L], (size_t)out_ch * out_n * sizeof(float));
            vk_lrelu(vk, cache->act_in[L + 1], out_ch, out_n);
        }
        cur_ch = out_ch; cur_n = out_n;
    }

    /* final: lrelu, conv_post, tanh */
    int n3 = cache->n_stage[3];
    cache->post_in = (float *)malloc((size_t)32 * n3 * sizeof(float));
    if (!cache->post_in) return -1;
    memcpy(cache->post_in, cache->stage_out[3], (size_t)32 * n3 * sizeof(float));
    vk_lrelu(vk, cache->post_in, 32, n3);
    const RVCTensor *post_w = wubu_rvc_find_tensor(model, "dec.conv_post.weight");
    const RVCTensor *post_b = wubu_rvc_find_tensor(model, "dec.conv_post.bias");
    if (!post_w || !post_w->data) return -1;
    int post_in = post_w->dims[1], post_k = post_w->dims[2];
    int out_n = n3 > max_samples ? max_samples : n3;
    float *d_out = (float *)malloc((size_t)out_n * sizeof(float));
    if (!d_out) return -1;
    if (vk_conv1d_fwd(vk, cache->post_in, post_in, n3, post_w->data,
                      (post_b && post_b->data) ? post_b->data : NULL,
                      1, post_k, 1, post_k / 2, 1, d_out, out_n) != 0) {
        free(d_out); return -1;
    }
    vk_tanh(vk, d_out, 1, out_n);
    memcpy(audio, d_out, (size_t)out_n * sizeof(float));
    cache->audio = (float *)malloc((size_t)out_n * sizeof(float));
    if (cache->audio) memcpy(cache->audio, d_out, (size_t)out_n * sizeof(float));
    free(d_out);
    return out_n;
}

void wubu_train_cache_free_vk(TrainCacheVk *cache) {
    if (!cache) return;
    if (cache->pre_out) free(cache->pre_out);
    for (int L = 0; L < 5; L++) if (cache->act_in[L]) free(cache->act_in[L]);
    for (int L = 0; L < 4; L++) {
        if (cache->stage_out[L]) free(cache->stage_out[L]);
        if (cache->ups_out[L]) free(cache->ups_out[L]);
        for (int s = 0; s < 3; s++)
            for (int cp = 0; cp < 3; cp++) {
                if (cache->mrf_a1[L][s][cp]) free(cache->mrf_a1[L][s][cp]);
                if (cache->mrf_c1[L][s][cp]) free(cache->mrf_c1[L][s][cp]);
                if (cache->mrf_c1o[L][s][cp]) free(cache->mrf_c1o[L][s][cp]);
                if (cache->mrf_a2[L][s][cp]) free(cache->mrf_a2[L][s][cp]);
                if (cache->mrf_c2o[L][s][cp]) free(cache->mrf_c2o[L][s][cp]);
            }
    }
    if (cache->post_in) free(cache->post_in);
    if (cache->audio) free(cache->audio);
}

TrainCacheVk *wubu_train_cache_alloc_vk(void) {
    TrainCacheVk *c = (TrainCacheVk *)calloc(1, sizeof(TrainCacheVk));
    return c;
}

/* ── backward ─────────────────────────────────────────────────────── */

/* one MRF pair backward: mirror of CPU mrf_pair_bwd + CUDA path.
 * Pair forward: a1=lrelu(p); c1o=conv1(a1); a2=lrelu(c1o); c2o=conv2(a2)+p
 * Backward: d_c2o → conv2_bwd(a2) → lrelu_bwd(c1o) → conv1_bwd(a1) →
 *           lrelu_bwd(p) + d_res (residual adds d_out back to p input).
 * d_in = grad wrt pair output; on return d_in = grad wrt pair input. */
static void mrf_pair_bwd_vk(WuBuVk *vk, WuBuRVCModel *model,
                            TrainCacheVk *cache, int L, int s, int cp,
                            float *d_in, int ch, int n,
                            WuBuTrainRegistry *reg) {
    const int dils[3] = { 1, 3, 5 };
    int rb = L * 3 + s;
    int dil = dils[cp];
    if (getenv("WUBU_VK_TRAIN_DBG")) { fprintf(stderr, "[vkt] pair L=%d s=%d cp=%d\n", L, s, cp); fflush(stderr); }
    char key[128];
    tfmt(key, "dec.resblocks.", rb, ".convs1.", cp, ".weight_v");
    const RVCTensor *c1w = wubu_rvc_find_tensor(model, key);
    tfmt(key, "dec.resblocks.", rb, ".convs2.", cp, ".weight_v");
    const RVCTensor *c2w = wubu_rvc_find_tensor(model, key);
    if (!c1w || !c1w->data || !c2w || !c2w->data) return;
    int k = c1w->dims[2];
    int pad1 = dil * (k - 1) / 2;
    int pad2 = k / 2;

    float *d_res = (float *)malloc((size_t)ch * n * sizeof(float));
    float *d_a2 = (float *)malloc((size_t)ch * n * sizeof(float));
    float *d_c1o = (float *)malloc((size_t)ch * n * sizeof(float));
    float *d_a1 = (float *)malloc((size_t)ch * n * sizeof(float));
    float *d_p = (float *)malloc((size_t)ch * n * sizeof(float));
    if (!d_res || !d_a2 || !d_c1o || !d_a1 || !d_p) { free(d_res); free(d_a2); free(d_c1o); free(d_a1); free(d_p); return; }
    memcpy(d_res, d_in, (size_t)ch * n * sizeof(float));   /* residual path */
    if (getenv("WUBU_VK_TRAIN_DBG")) { fprintf(stderr, "[vkt] pair alloc ok\n"); fflush(stderr); }

    tfmt(key, "dec.resblocks.", rb, ".convs2.", cp, ".weight_v");
    int i2w = wubu_train_registry_find(reg, key);
    float *dw2 = (i2w >= 0) ? reg->params[i2w].grad : NULL;
    tfmt(key, "dec.resblocks.", rb, ".convs2.", cp, ".bias");
    int i2b = wubu_train_registry_find(reg, key);
    float *db2 = (i2b >= 0) ? reg->params[i2b].grad : NULL;
    /* conv2 bwd: in=a2, w=c2w, dout=d_in → d_a2, dw2, db2 */
    float *w2tmp = (float *)malloc((size_t)ch * ch * k * sizeof(float));
    if (!w2tmp) { free(d_res); free(d_a2); free(d_c1o); free(d_a1); free(d_p); return; }
    memcpy(w2tmp, c2w->data, (size_t)ch * ch * k * sizeof(float));
    memset(d_a2, 0, (size_t)ch * n * sizeof(float));
    if (getenv("WUBU_VK_TRAIN_DBG")) { fprintf(stderr, "[vkt] pair conv2 bwd ch=%d n=%d\n", ch, n); fflush(stderr); }
    wubu_vk_bwd_conv1d(vk, cache->mrf_a2[L][s][cp], ch, w2tmp, d_in,
                       ch, k, 1, pad2, 1, n, n, d_a2, dw2, db2);
    if (getenv("WUBU_VK_TRAIN_DBG")) { fprintf(stderr, "[vkt] pair conv2 bwd done\n"); fflush(stderr); }
    free(w2tmp);

    /* lrelu_bwd after conv1: pre-act = c1o */
    wubu_vk_bwd_act(vk, cache->mrf_c1o[L][s][cp], d_a2, d_c1o, 0, (size_t)ch * n);

    /* conv1 bwd: in=a1, w=c1w, dout=d_c1o → d_a1, dw1, db1 */
    tfmt(key, "dec.resblocks.", rb, ".convs1.", cp, ".weight_v");
    int i1w = wubu_train_registry_find(reg, key);
    float *dw1 = (i1w >= 0) ? reg->params[i1w].grad : NULL;
    tfmt(key, "dec.resblocks.", rb, ".convs1.", cp, ".bias");
    int i1b = wubu_train_registry_find(reg, key);
    float *db1 = (i1b >= 0) ? reg->params[i1b].grad : NULL;
    float *w1tmp = (float *)malloc((size_t)ch * ch * k * sizeof(float));
    if (!w1tmp) { free(d_res); free(d_a2); free(d_c1o); free(d_a1); free(d_p); return; }
    memcpy(w1tmp, c1w->data, (size_t)ch * ch * k * sizeof(float));
    memset(d_a1, 0, (size_t)ch * n * sizeof(float));
    wubu_vk_bwd_conv1d(vk, cache->mrf_a1[L][s][cp], ch, w1tmp, d_c1o,
                       ch, k, 1, pad1, dil, n, n, d_a1, dw1, db1);
    free(w1tmp);

    /* lrelu_bwd before conv1: pre-act = p (mrf_c1 slot); residual add */
    wubu_vk_bwd_act(vk, cache->mrf_c1[L][s][cp], d_a1, d_p, 0, (size_t)ch * n);
    for (size_t i = 0; i < (size_t)ch * n; i++)
        d_in[i] = d_p[i] + d_res[i];

    free(d_res); free(d_a2); free(d_c1o); free(d_a1); free(d_p);
}

/* one MRF stage backward: d_stage_out (grad wrt stage_out) → d_stage_in
 * (grad wrt ups_out = MRF input) */
static void mrf_stage_bwd_vk(WuBuVk *vk, WuBuRVCModel *model, int L,
                             TrainCacheVk *cache, const float *d_stage_out,
                             float *d_stage_in, int ch, int n,
                             WuBuTrainRegistry *reg) {
    for (size_t i = 0; i < (size_t)ch * n; i++) d_stage_in[i] = 0.0f;
    for (int s = 0; s < 3; s++) {
        float *d_in = (float *)malloc((size_t)ch * n * sizeof(float));
        if (!d_in) return;
        for (size_t i = 0; i < (size_t)ch * n; i++)
            d_in[i] = d_stage_out[i] / 3.0f;
        for (int cp = 2; cp >= 0; cp--)
            mrf_pair_bwd_vk(vk, model, cache, L, s, cp, d_in, ch, n, reg);
        for (size_t i = 0; i < (size_t)ch * n; i++)
            d_stage_in[i] += d_in[i];
        free(d_in);
    }
}

int wubu_train_backward_vk(WuBuVk *vk, WuBuRVCModel *model, TrainCacheVk *cache,
                           const float *mel_in, const float *d_audio,
                           WuBuTrainRegistry *reg) {
    if (!vk || !model || !cache || !d_audio || !reg) return -1;
    int n3 = cache->n_stage[3];

    /* d_audio → tanh' → conv_post bwd → lrelu' (pre-act = stage_out[3]) */
    float *d_tanh_in = (float *)malloc((size_t)n3 * sizeof(float));
    float *d_post_out = (float *)malloc((size_t)32 * n3 * sizeof(float));
    float *d_stage3 = (float *)malloc((size_t)32 * n3 * sizeof(float));
    if (!d_tanh_in || !d_post_out || !d_stage3) { free(d_tanh_in); free(d_post_out); free(d_stage3); return -1; }
    wubu_vk_bwd_act(vk, cache->audio, d_audio, d_tanh_in, 1, (size_t)n3);
    if (getenv("WUBU_VK_TRAIN_DBG")) fprintf(stderr, "[vkt] d_tanh_in[0]=%.6f last=%.6f d_audio[0]=%.6f\n", d_tanh_in[0], d_tanh_in[n3 - 1], d_audio[0]);
    const RVCTensor *post_w = wubu_rvc_find_tensor(model, "dec.conv_post.weight");
    if (!post_w || !post_w->data) { free(d_tanh_in); free(d_post_out); free(d_stage3); return -1; }
    int post_in = post_w->dims[1], post_k = post_w->dims[2];
    int ipw = wubu_train_registry_find(reg, "dec.conv_post.weight");
    float *dpw = (ipw >= 0) ? reg->params[ipw].grad : NULL;
    int ipb = wubu_train_registry_find(reg, "dec.conv_post.bias");
    float *dpb = (ipb >= 0) ? reg->params[ipb].grad : NULL;
    float *db_dummy = (float *)calloc((size_t)post_in, sizeof(float));
    if (!db_dummy) { free(d_tanh_in); free(d_post_out); free(d_stage3); return -1; }
    float *dpb_use = dpb ? dpb : db_dummy;
    if (getenv("WUBU_VK_TRAIN_DBG")) fprintf(stderr, "[vkt] conv_post ipw=%d ipb=%d dpw=%p dpb=%p\n", ipw, ipb, (void*)dpw, (void*)dpb);
    float *wpt = (float *)malloc((size_t)post_in * post_k * sizeof(float));
    if (!wpt) { free(db_dummy); free(d_tanh_in); free(d_post_out); free(d_stage3); return -1; }
    memcpy(wpt, post_w->data, (size_t)post_in * post_k * sizeof(float));
    memset(d_post_out, 0, (size_t)32 * n3 * sizeof(float));
    wubu_vk_bwd_conv1d(vk, cache->post_in, post_in, wpt, d_tanh_in,
                       1, post_k, 1, post_k / 2, 1, n3, n3,
                       d_post_out, dpw, dpb_use);
    free(wpt);
    free(db_dummy);
    wubu_vk_bwd_act(vk, cache->stage_out[3], d_post_out, d_stage3, 0, (size_t)32 * n3);
    if (getenv("WUBU_VK_TRAIN_DBG")) fprintf(stderr, "[vkt] d_stage3[0]=%.6f last=%.6f d_post_out[0]=%.6f\n", d_stage3[0], d_stage3[(size_t)32 * n3 - 1], d_post_out[0]);

    /* MRF backward, stage 3 → 0 */
    float *d_stage_in = NULL;
    float *d_ups = (float *)malloc((size_t)32 * n3 * sizeof(float));
    if (!d_ups) { free(d_tanh_in); free(d_post_out); free(d_stage3); return -1; }
    d_stage_in = d_ups;
    mrf_stage_bwd_vk(vk, model, 3, cache, d_stage3, d_stage_in, 32, n3, reg);
    int cur_ch = 32, cur_n = n3;
    float *d_act0 = NULL;   /* grad wrt act_in[0] — kept for conv_pre bwd */
    int ups_freed = 0;      /* d_ups was handed to the loop as d_stage_in */

    for (int L = 3; L >= 0; L--) {
        /* upsample backward: din wrt act_in[L] */
        int in_ch = cache->ch_in[L];
        int out_ch = cache->ch_out[L];
        int in_n = (L == 0) ? cache->n_frames : cache->n_stage[L - 1];
        int kk = cache->k[L], stride = cache->s[L], pad = cache->p[L];
        cur_n = cache->n_stage[L];   /* convT output length for stage L */
        char key[128];
        tfmt(key, "dec.ups.", L, "", -1, ".weight");
        int iuw = wubu_train_registry_find(reg, key);
        float *duw = (iuw >= 0) ? reg->params[iuw].grad : NULL;
        tfmt(key, "dec.ups.", L, "", -1, ".bias");
        int iub = wubu_train_registry_find(reg, key);
        float *dub = (iub >= 0) ? reg->params[iub].grad : NULL;
        float *d_act = (float *)calloc((size_t)in_ch * in_n, sizeof(float));
        if (!d_act) { if (!ups_freed) free(d_ups); free(d_tanh_in); free(d_post_out); free(d_stage3); return -1; }
        float *wt = (float *)malloc((size_t)in_ch * out_ch * kk * sizeof(float));
        if (!wt) { free(d_act); if (!ups_freed) free(d_ups); free(d_tanh_in); free(d_post_out); free(d_stage3); return -1; }
        memcpy(wt, model->hifi_upsample_denorm[L], (size_t)in_ch * out_ch * kk * sizeof(float));
        wubu_vk_bwd_convt1d(vk, cache->act_in[L], in_ch, wt, d_stage_in,
                            out_ch, kk, stride, pad, in_n, cur_n,
                            d_act, duw, dub);
        free(wt);

        if (L >= 1) {
            /* d_act = grad wrt act_in[L] = lrelu(stage_out[L-1]), so it has
             * in_ch × in_n dims (NOT the convT output dims). stage_out[L-1]
             * shares those dims: ch_out[L-1]==ch_in[L], n_stage[L-1]==in_n. */
            float *d_stage_prev = (float *)malloc((size_t)in_ch * in_n * sizeof(float));
            if (!d_stage_prev) { free(d_act); if (!ups_freed) free(d_ups); free(d_tanh_in); free(d_post_out); free(d_stage3); return -1; }
            wubu_vk_bwd_act(vk, cache->stage_out[L - 1], d_act, d_stage_prev, 0,
                            (size_t)in_ch * in_n);
            if (d_stage_in) { free(d_stage_in); if (d_stage_in == d_ups) ups_freed = 1; }
            d_stage_in = d_stage_prev;
            /* MRF bwd for stage L-1 — MRF runs on ch_out[L-1] channels
             * (NOT ch_in: the MRF input is ups_out[L-1] which has ch_out
             * channels; ch_in[L-1] is the *convT input* width). */
            int nch = cache->ch_out[L - 1];
            float *d_next = (float *)malloc((size_t)nch * cache->n_stage[L - 1] * sizeof(float));
            if (!d_next) { free(d_act); free(d_stage_in); if (!ups_freed) free(d_ups); free(d_tanh_in); free(d_post_out); free(d_stage3); return -1; }
            mrf_stage_bwd_vk(vk, model, L - 1, cache, d_stage_in, d_next,
                             nch, cache->n_stage[L - 1], reg);
            free(d_stage_in);
            d_stage_in = d_next;
        } else {
            /* L == 0: keep d_act (grad wrt act_in[0]) for conv_pre */
            d_act0 = d_act;
            d_act = NULL;
        }
        free(d_act);
    }
    /* d_stage_in = grad wrt ups_out[0] — consumed by convT0 above; free it */
    if (d_stage_in) { free(d_stage_in); if (d_stage_in == d_ups) ups_freed = 1; d_stage_in = NULL; }
    if (!ups_freed) free(d_ups);

    /* conv_pre backward: lrelu' (pre_out) then conv1d_bwd(mel) */
    {
        int init_ch = cache->ch_in[0];
        float *d_pre = (float *)malloc((size_t)init_ch * cache->n_frames * sizeof(float));
        if (!d_pre) { free(d_act0); free(d_ups); free(d_tanh_in); free(d_post_out); free(d_stage3); return -1; }
        const RVCTensor *pre_w = wubu_rvc_find_tensor(model, "dec.conv_pre.weight");
        int pre_in = pre_w->dims[1], pk = pre_w->dims[2];
        wubu_vk_bwd_act(vk, cache->pre_out, d_act0, d_pre, 0, (size_t)init_ch * cache->n_frames);
        free(d_act0); d_act0 = NULL;

        /* conv_pre weight grads */
        char key[128];
        tfmt(key, "dec.conv_pre.weight", -1, "", -1, "");
        int ipw2 = wubu_train_registry_find(reg, key);
        float *dpw2 = (ipw2 >= 0) ? reg->params[ipw2].grad : NULL;
        tfmt(key, "dec.conv_pre.bias", -1, "", -1, "");
        int ipb2 = wubu_train_registry_find(reg, key);
        float *dpb2 = (ipb2 >= 0) ? reg->params[ipb2].grad : NULL;
        float *wpt = (float *)malloc((size_t)init_ch * pre_in * pk * sizeof(float));
        if (wpt) {
            memcpy(wpt, pre_w->data, (size_t)init_ch * pre_in * pk * sizeof(float));
            float *d_mel = (float *)calloc((size_t)pre_in * cache->n_frames, sizeof(float));
            if (d_mel) {
                /* d_pre holds the lrelu grad (from the bwd_act above) — the
                 * bwd_conv1d zeroes its OWN din/dw/db slots; d_pre must NOT
                 * be wiped here (that was a bug: conv_pre grads came out 0). */
                wubu_vk_bwd_conv1d(vk, mel_in, pre_in, wpt, d_pre,
                                   init_ch, pk, 1, pk / 2, 1,
                                   cache->n_frames, cache->n_frames,
                                   d_mel, dpw2, dpb2);
                free(d_mel);
            }
            free(wpt);
        }
        free(d_pre);
    }

    if (!ups_freed) free(d_ups);
    free(d_tanh_in); free(d_post_out); free(d_stage3);
    return 0;
}

/* ── one full training step: forward → MSE → backward → AdamW ──────── */
int wubu_train_step_vk(WuBuVk *vk, WuBuRVCModel *model, WuBuTrainRegistry *reg,
                       WuBuAdamW *opt, const float *mel_in, int n_frames,
                       const float *wav, int n_samples,
                       float *loss_out, int epoch) {
    if (!vk || !model || !reg || !opt || !mel_in || !wav) return -1;
    int max_samples = n_frames * 400;
    float *audio = (float *)malloc((size_t)max_samples * sizeof(float));
    if (!audio) return -1;
    TrainCacheVk *cache = wubu_train_cache_alloc_vk();
    if (!cache) { free(audio); return -1; }
    int n_out = wubu_train_forward_vk(vk, model, mel_in, n_frames, audio, max_samples, cache);
    if (n_out <= 0) { free(audio); wubu_train_cache_free_vk(cache); free(cache); return -1; }
    int n = n_out < n_samples ? n_out : n_samples;
    float mse = wubu_mse_loss(audio, wav, n);
    if (loss_out) *loss_out = mse;
    model->last_loss = mse;
    model->last_epoch = epoch;
    float *d_audio = (float *)malloc((size_t)n_out * sizeof(float));
    if (!d_audio) { free(audio); wubu_train_cache_free_vk(cache); free(cache); return -1; }
    for (int i = 0; i < n; i++) d_audio[i] = 2.0f * (audio[i] - wav[i]) / (float)n;
    for (int i = n; i < n_out; i++) d_audio[i] = 0.0f;
    wubu_train_registry_zero_grads(reg);
    if (wubu_train_backward_vk(vk, model, cache, mel_in, d_audio, reg) != 0) {
        free(d_audio); free(audio); wubu_train_cache_free_vk(cache); free(cache); return -1;
    }
    for (int i = 0; i < reg->count; i++)
        wubu_adamw_step(opt, i, reg->params[i].data, reg->params[i].grad);
    free(d_audio); free(audio);
    wubu_train_cache_free_vk(cache); free(cache);
    return (mse < 0.1f) ? 1 : 0;
}
