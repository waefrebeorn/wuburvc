/* wubu_train_vk.c — Vulkan-accelerated RVC training step (RECORD MODE).
 *
 * Mirrors the CPU training math in wubu_train.c EXACTLY (Triple-DA: the CPU
 * train_step is the reference, the CUDA backend in wubu_train_cuda.cu is the
 * cross-checked twin). The VK backend records EVERY dispatch into one command
 * buffer per step: weights + mel + intermediates live in GPU slots (200+),
 * submitted once, grads downloaded once. Same math/shader kernels as the
 * per-op path, but ~100x less PCIe round-trips (per-op measured ~2 min/step
 * at F=96; record mode targets ~1-2 s/step).
 *
 * Slot map (wubu_train_vk.c owns these; WUBU_VK_TRAIN_BASE=200):
 *   W(i)          weights — registry param i (uploaded once per step)
 *   M             mel input (192 x F)
 *   A(n)          act_in[n]
 *   PRE           pre_out (conv_pre out, pre-lrelu)
 *   U(n)          ups_out[n]
 *   S(n)          stage_out[n]
 *   MRF(L,s,cp,c) MRF caches: c = 0 p, 1 a1, 2 c1o, 3 a2, 4 c2o
 *   POST          conv_post input
 *   AUD           audio out (final tanh)
 *   SCR(n)        backward scratch (d_audio, d_* working grads)
 *   G(i)          grad slots — registry param i (zeroed, accumulated, DL'd)
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
 * nvcc object with MinGW). */
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

#define TB WUBU_VK_TRAIN_BASE
#define SL_W(i)    (TB + (i))                    /* weights + grads, registry order */
#define SL_G(i)    (TB + 160 + (i))
#define SL_MEL     (TB + 330)
#define SL_ACT(n)  (TB + 340 + (n))
#define SL_PRE     (TB + 350)
#define SL_UPS(n)  (TB + 360 + (n))
#define SL_STG(n)  (TB + 370 + (n))
#define SL_MRF(L,s,cp,c) (TB + 390 + (((L)*3 + (s))*3 + (cp))*5 + (c))
#define SL_POST    (TB + 570)
#define SL_AUD     (TB + 580)
#define SL_SCR(n)  (TB + 590 + (n))              /* backward scratch */
#define SL_BIAS0   (TB + 320)                    /* shared zero bias slot (zeroed in begin) */
#define SL_STAGE   (TB + 590 + 49)              /* host-visible staging for grad downloads */
#define SL_DBG0    (TB + 590 + 46)              /* host-visible debug dump slot */
#define SL_DBG1    (TB + 590 + 47)              /* host-visible debug dump slot */

/* weight slot cache: registry idx -> slot (weights are registry order) */
static int w_slot(int reg_idx) { return SL_W(reg_idx); }
static int g_slot(int reg_idx) { return SL_G(reg_idx); }
/* bias slot: the tensor's own slot if present, else the shared zero slot */
static int b_slot(int reg_idx) { return reg_idx >= 0 ? SL_W(reg_idx) : SL_BIAS0; }

struct TrainCacheVk {
    int n_frames;
    int n_stage[4];
    int ch_in[4], ch_out[4];
    int k[4], s[4], p[4];
    int n_ups;
    int n_out;                 /* audio sample count */
    const WuBuTrainRegistry *reg;   /* registry for weight/grad slot mapping */
    /* per-tensor registry indices (found once per forward) */
    int ri_pre_w, ri_pre_b;
    int ri_post_w, ri_post_b;
    int ri_ups_w[4], ri_ups_b[4];
    int ri_rb_w[4][3][3][2], ri_rb_b[4][3][3][2];
};

typedef struct TrainCacheVk TrainCacheVk;

/* registry index helper — replaced by direct find_reg calls */

/* find registry idx by name in a provided registry; -1 if absent */
static int find_reg(const WuBuTrainRegistry *reg, const char *name) {
    return wubu_train_registry_find(reg, name);
}

/* upload a tensor's weights + zero its grad slot */
static int upload_w(WuBuVk *vk, const WuBuTrainRegistry *reg, int ri,
                    const float *data, size_t n) {
    (void)reg;
    if (ri < 0) return 0;         /* absent tensor (e.g. missing bias) */
    if (!data) {                  /* zero-fill the weight slot + grad slot */
        if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG uw zero ri=%d wslot=%d gslot=%d\n", ri, w_slot(ri), g_slot(ri));
        if (wubu_vk_train_zero(vk, w_slot(ri), n * 4)) return -1;
        if (wubu_vk_train_zero(vk, g_slot(ri), n * 4)) return -1;
        return 0;
    }
    /* weight slot holds the CURRENT weights (re-uploaded every step — the
     * model tensors are updated by AdamW on the host between steps). */
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG uw data ri=%d wslot=%d n=%zu\n", ri, w_slot(ri), n);
    {
        int rc_up = wubu_vk_train_upload(vk, w_slot(ri), data, n * 4);
        if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG uw upload rc=%d\n", rc_up);
        if (rc_up) return -1;
    }
    /* grad slot is zeroed at the START of the BACKWARD (backward zeroes every
     * grad slot it accumulates into before the first bwd op). NOT here — the
     * forward allocating 155 host-visible grad slots exhausts the PCIe BAR
     * before the backward even starts. */
    return 0;
}

/* ── forward (record mode) ────────────────────────────────────────── */

/* MRF stage forward recorded into GPU slots. cache arrays in slots. */
static int mrf_stage_fwd_rec(WuBuVk *vk, WuBuRVCModel *model,
                             const WuBuTrainRegistry *reg, int L,
                             int in_slot, int stage_slot, TrainCacheVk *cache,
                             int ch, int n) {
    const int dils[3] = { 1, 3, 5 };
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG mrf L=%d ch=%d n=%d\n", L, ch, n);
    for (int s = 0; s < 3; s++) {
        int rb = L * 3 + s;
        int cur_s = in_slot;   /* EVERY stack starts fresh from the stage input */
        if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG mrf L=%d s=%d\n", L, s);
        for (int cp = 0; cp < 3; cp++) {
            char key[128];
            tfmt(key, "dec.resblocks.", rb, ".convs1.", cp, ".weight_v");
            const RVCTensor *c1w = wubu_rvc_find_tensor(model, key);
            tfmt(key, "dec.resblocks.", rb, ".convs2.", cp, ".weight_v");
            const RVCTensor *c2w = wubu_rvc_find_tensor(model, key);
            if (!c1w || !c1w->data || !c2w || !c2w->data) {
                if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG mrf L=%d s=%d cp=%d key='%s' c1w=%p c2w=%p\n",
                        L, s, cp, key, (void*)c1w, (void*)c2w);
                return -1;
            }
            int k = c1w->dims[2];
            int dil = dils[cp];
            int pad1 = dil * (k - 1) / 2;
            int pad2 = k / 2;
            tfmt(key, "dec.resblocks.", rb, ".convs1.", cp, ".bias");
            const RVCTensor *c1b = wubu_rvc_find_tensor(model, key);
            tfmt(key, "dec.resblocks.", rb, ".convs2.", cp, ".bias");
            const RVCTensor *c2b = wubu_rvc_find_tensor(model, key);
            int ri1w = cache->ri_rb_w[L][s][cp][0];
            int ri1b = cache->ri_rb_b[L][s][cp][0];
            int ri2w = cache->ri_rb_w[L][s][cp][1];
            int ri2b = cache->ri_rb_b[L][s][cp][1];
            if (getenv("WUBU_VK_DBG"))
                fprintf(stderr, "VKDBG mrf L=%d s=%d cp=%d ri1w=%d ri1b=%d ri2w=%d ri2b=%d\n",
                        L, s, cp, ri1w, ri1b, ri2w, ri2b);
            if (upload_w(vk, reg, ri1w, c1w->data, (size_t)ch * ch * k) ||
                upload_w(vk, reg, ri1b, (c1b && c1b->data) ? c1b->data : NULL, (size_t)ch) ||
                upload_w(vk, reg, ri2w, c2w->data, (size_t)ch * ch * k) ||
                upload_w(vk, reg, ri2b, (c2b && c2b->data) ? c2b->data : NULL, (size_t)ch))
                return -1;
            if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG mrf L=%d s=%d cp=%d uploads ok\n", L, s, cp);
            int w1_s = w_slot(ri1w), b1_s = b_slot(ri1b);
            int w2_s = w_slot(ri2w), b2_s = b_slot(ri2b);
            int p_s = SL_MRF(L, s, cp, 0);   /* p = pair input (PRE-lrelu) */
            int a1_s = SL_MRF(L, s, cp, 1);
            int c1o_s = SL_MRF(L, s, cp, 2);
            int a2_s = SL_MRF(L, s, cp, 3);
            int c2o_s = SL_MRF(L, s, cp, 4);
            /* p = cur (copy) */
            wubu_vk_train_elt(vk, p_s, 0, cur_s, 0, 1, (size_t)ch * n);
            /* a1 = lrelu(p) */
            wubu_vk_train_elt(vk, a1_s, 0, p_s, 0, 1, (size_t)ch * n);
            wubu_vk_train_act(vk, a1_s, -1, 0, ch, n, 0.0f);
            /* c1o = conv1(a1) */
            wubu_vk_train_conv1d(vk, a1_s, w1_s, b1_s, c1o_s,
                                 ch, n, ch, k, 1, pad1, dil, n, 0, 0);
            /* a2 = lrelu(c1o); c2o = conv2(a2); then c2o += p */
            wubu_vk_train_elt(vk, a2_s, 0, c1o_s, 0, 1, (size_t)ch * n);
            wubu_vk_train_act(vk, a2_s, -1, 0, ch, n, 0.0f);
            wubu_vk_train_conv1d(vk, a2_s, w2_s, b2_s, c2o_s,
                                 ch, n, ch, k, 1, pad2, 1, n, 0, 0);
            wubu_vk_train_elt(vk, c2o_s, 0, p_s, 0, 0, (size_t)ch * n); /* += p */
            cur_s = c2o_s;
        }
        /* accumulate this stack: stage_out += cur */
        if (s == 0)
            wubu_vk_train_elt(vk, stage_slot, 0, cur_s, 0, 1, (size_t)ch * n);
        else
            wubu_vk_train_elt(vk, stage_slot, 0, cur_s, 0, 0, (size_t)ch * n);
    }
    /* scale by 1/3 */
    wubu_vk_train_act(vk, stage_slot, -1, 4, ch, n, 1.0f / 3.0f);
    return 0;
}

int wubu_train_forward_vk(WuBuVk *vk, WuBuRVCModel *model, const float *mel_in,
                          int n_frames, float *audio, int max_samples,
                          TrainCacheVk *cache) {
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG FWD-ENTER reg=%p\n", (void*)(cache ? cache->reg : NULL));
    if (!vk || !model || !mel_in || !audio || !cache) return -1;
    const WuBuTrainRegistry *reg = cache->reg;
    memset(cache, 0, sizeof(*cache));
    cache->reg = reg;
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
    /* NOTE: registry is passed via cache->reg (set by the step or test). */

    /* The forward needs the registry for weight slots — the step sets
     * cache->reg before calling forward. Tests must set it too. */
    if (!reg) { fprintf(stderr, "VKDBG fwd: no reg\n"); return -1; }

    /* resolve registry indices (once) */
    char key[128];
    tfmt(key, "dec.conv_pre.weight", -1, "", -1, "");
    cache->ri_pre_w = find_reg(reg, key);
    tfmt(key, "dec.conv_pre.bias", -1, "", -1, "");
    cache->ri_pre_b = find_reg(reg, key);
    for (int L = 0; L < 4; L++) {
        tfmt(key, "dec.ups.", L, "", -1, ".weight");
        cache->ri_ups_w[L] = find_reg(reg, key);
        tfmt(key, "dec.ups.", L, "", -1, ".bias");
        cache->ri_ups_b[L] = find_reg(reg, key);
        for (int s = 0; s < 3; s++)
            for (int cp = 0; cp < 3; cp++) {
                tfmt(key, "dec.resblocks.", L * 3 + s, ".convs1.", cp, ".weight_v");
                cache->ri_rb_w[L][s][cp][0] = find_reg(reg, key);
                tfmt(key, "dec.resblocks.", L * 3 + s, ".convs1.", cp, ".bias");
                cache->ri_rb_b[L][s][cp][0] = find_reg(reg, key);
                tfmt(key, "dec.resblocks.", L * 3 + s, ".convs2.", cp, ".weight_v");
                cache->ri_rb_w[L][s][cp][1] = find_reg(reg, key);
                tfmt(key, "dec.resblocks.", L * 3 + s, ".convs2.", cp, ".bias");
                cache->ri_rb_b[L][s][cp][1] = find_reg(reg, key);
            }
    }
    tfmt(key, "dec.conv_post.weight", -1, "", -1, "");
    cache->ri_post_w = find_reg(reg, key);
    tfmt(key, "dec.conv_post.bias", -1, "", -1, "");
    cache->ri_post_b = find_reg(reg, key);

    wubu_vk_train_begin(vk);
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG fwd: begin ok\n");
    /* shared zero-bias slot (2048 floats — enough for any bias + debug dumps) */
    if (wubu_vk_train_zero(vk, SL_BIAS0, 2048 * 4)) { wubu_vk_train_end(vk); return -1; }

    /* conv_pre: mel -> pre_out (PRE-lrelu), then act_in[0] = lrelu */
    if (cache->ri_pre_w < 0 || cache->ri_pre_b < 0) {
        fprintf(stderr, "VKDBG fwd: pre reg missing ri_pre_w=%d ri_pre_b=%d\n",
                cache->ri_pre_w, cache->ri_pre_b);
    }
    wubu_vk_train_upload(vk, SL_MEL, mel_in, (size_t)pre_in * n_frames * 4);
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG fwd: mel uploaded\n");
    if (upload_w(vk, reg, cache->ri_pre_w, pre_w->data, (size_t)init_ch * pre_in * pk) ||
        upload_w(vk, reg, cache->ri_pre_b, (pre_b && pre_b->data) ? pre_b->data : NULL, (size_t)init_ch))
        { wubu_vk_train_end(vk); return -1; }
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG fwd: pre weights ok\n");
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG fwd: pre_w[0]=%.6f mel[0]=%.6f bias0=%d\n",
            pre_w->data[0], mel_in[0], SL_BIAS0);
    wubu_vk_train_conv1d(vk, SL_MEL, w_slot(cache->ri_pre_w), b_slot(cache->ri_pre_b), SL_PRE,
                         pre_in, n_frames, init_ch, pk, 1, pk / 2, 1, n_frames, 0, 0);
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG fwd: conv_pre ok\n");
    wubu_vk_train_elt(vk, SL_ACT(0), 0, SL_PRE, 0, 1, (size_t)init_ch * n_frames);
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG fwd: act0 copy ok\n");
    wubu_vk_train_act(vk, SL_ACT(0), -1, 0, init_ch, n_frames, 0.0f);
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG fwd: act0 lrelu ok\n");

    /* upsample blocks */
    int cur_ch = init_ch, cur_n = n_frames;
    for (int L = 0; L < n_ups && L < 4; L++) {
        int out_ch = ch_out[L];
        int out_n = (cur_n - 1) * rate[L] - 2 * ((kk[L] - rate[L]) / 2) + kk[L];
        cache->ch_in[L] = cur_ch; cache->ch_out[L] = out_ch;
        cache->k[L] = kk[L]; cache->s[L] = rate[L]; cache->p[L] = (kk[L] - rate[L]) / 2;
        cache->n_stage[L] = out_n;
        tfmt(key, "dec.ups.", L, "", -1, ".bias");
        const RVCTensor *ub = wubu_rvc_find_tensor(model, key);
        if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG fwd: ups L=%d upload\n", L);
        if (upload_w(vk, reg, cache->ri_ups_w[L], model->hifi_upsample_denorm[L],
                     (size_t)cur_ch * out_ch * kk[L]) ||
            upload_w(vk, reg, cache->ri_ups_b[L], (ub && ub->data) ? ub->data : NULL,
                     (size_t)out_ch))
            { wubu_vk_train_end(vk); return -1; }
        if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG fwd: ups L=%d convt\n", L);
        wubu_vk_train_conv1d(vk, SL_ACT(L), w_slot(cache->ri_ups_w[L]), b_slot(cache->ri_ups_b[L]), SL_UPS(L),
                             cur_ch, cur_n, out_ch, kk[L], rate[L], (kk[L] - rate[L]) / 2,
                             1, out_n, 1, 0);
        if (mrf_stage_fwd_rec(vk, model, reg, L, SL_UPS(L), SL_STG(L), cache,
                              out_ch, out_n) != 0) {
            wubu_vk_train_end(vk);
            return -1;
        }
        if (L < n_ups - 1) {
            wubu_vk_train_elt(vk, SL_ACT(L + 1), 0, SL_STG(L), 0, 1, (size_t)out_ch * out_n);
            wubu_vk_train_act(vk, SL_ACT(L + 1), -1, 0, out_ch, out_n, 0.0f);
        }
        cur_ch = out_ch; cur_n = out_n;
    }

    /* final: lrelu, conv_post, tanh */
    int n3 = cache->n_stage[3];
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG fwd: final lrelu n3=%d\n", n3);
    wubu_vk_train_elt(vk, SL_POST, 0, SL_STG(3), 0, 1, (size_t)32 * n3);
    wubu_vk_train_act(vk, SL_POST, -1, 0, 32, n3, 0.0f);
    const RVCTensor *post_w = wubu_rvc_find_tensor(model, "dec.conv_post.weight");
    const RVCTensor *post_b = wubu_rvc_find_tensor(model, "dec.conv_post.bias");
    if (!post_w || !post_w->data) { wubu_vk_train_end(vk); return -1; }
    int post_in = post_w->dims[1], post_k = post_w->dims[2];
    int out_n = n3 > max_samples ? max_samples : n3;
    if (upload_w(vk, reg, cache->ri_post_w, post_w->data, (size_t)post_in * post_k) ||
        upload_w(vk, reg, cache->ri_post_b, (post_b && post_b->data) ? post_b->data : NULL, 1))
        { wubu_vk_train_end(vk); return -1; }
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG fwd: post weights ok ri_post_w=%d ri_post_b=%d\n", cache->ri_post_w, cache->ri_post_b);
    /* SL_AUD is downloaded at the end — force host-visible BEFORE the conv
     * would allocate it as DEVICE_LOCAL-only (pool_slot_dev keeps existing). */
    if (wubu_vk_train_zero(vk, SL_AUD, (size_t)out_n * 4)) { wubu_vk_train_end(vk); return -1; }
    wubu_vk_train_conv1d(vk, SL_POST, w_slot(cache->ri_post_w), b_slot(cache->ri_post_b), SL_AUD,
                         post_in, n3, 1, post_k, 1, post_k / 2, 1, out_n, 0, 2); /* tanh epi */
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG fwd: post conv ok\n");
    if (wubu_vk_train_end(vk) != 0) return -1;
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG fwd: end ok\n");
    wubu_vk_train_download(vk, SL_AUD, audio, (size_t)out_n * 4);
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG fwd: aud dl ok\n");
    cache->n_out = out_n;
    return out_n;
}

void wubu_train_cache_free_vk(TrainCacheVk *cache) {
    (void)cache;   /* slots are VK-owned; no host allocations */
}

void wubu_train_cache_set_reg_vk(TrainCacheVk *cache, const WuBuTrainRegistry *reg) {
    if (cache) cache->reg = reg;
}

TrainCacheVk *wubu_train_cache_alloc_vk(void) {
    return (TrainCacheVk *)calloc(1, sizeof(TrainCacheVk));
}

/* ── backward (record mode) ───────────────────────────────────────── */

/* GPU-side zero: record a vkCmdFillBuffer on the slot — works for
 * DEVICE_LOCAL-only slots that host memset cannot touch and doesn't depend
 * on a pre-zeroed source buffer. */
static void gzero(WuBuVk *vk, int dst, size_t n) {
    wubu_vk_train_fill_zero(vk, dst, n);
}

/* one MRF pair backward — recorded into slots. */
static void mrf_pair_bwd_rec(WuBuVk *vk, WuBuRVCModel *model, TrainCacheVk *cache,
                             int L, int s, int cp, int d_in_s, int ch, int n) {
    const int dils[3] = { 1, 3, 5 };
    int rb = L * 3 + s;
    int dil = dils[cp];
    char key[128];
    tfmt(key, "dec.resblocks.", rb, ".convs1.", cp, ".weight_v");
    const RVCTensor *c1w = wubu_rvc_find_tensor(model, key);
    tfmt(key, "dec.resblocks.", rb, ".convs2.", cp, ".weight_v");
    const RVCTensor *c2w = wubu_rvc_find_tensor(model, key);
    if (!c1w || !c1w->data || !c2w || !c2w->data) return;
    int k = c1w->dims[2];
    int pad1 = dil * (k - 1) / 2;
    int pad2 = k / 2;

    int p_s = SL_MRF(L, s, cp, 0);
    int a1_s = SL_MRF(L, s, cp, 1);
    int c1o_s = SL_MRF(L, s, cp, 2);
    int a2_s = SL_MRF(L, s, cp, 3);
    int c2o_s = SL_MRF(L, s, cp, 4);
    int w1_s = w_slot(cache->ri_rb_w[L][s][cp][0]);
    int b1_s = b_slot(cache->ri_rb_b[L][s][cp][0]);
    int w2_s = w_slot(cache->ri_rb_w[L][s][cp][1]);
    int b2_s = b_slot(cache->ri_rb_b[L][s][cp][1]);
    int dw1 = g_slot(cache->ri_rb_w[L][s][cp][0]);
    int db1 = g_slot(cache->ri_rb_b[L][s][cp][0]);
    int dw2 = g_slot(cache->ri_rb_w[L][s][cp][1]);
    int db2 = g_slot(cache->ri_rb_b[L][s][cp][1]);

    /* d_res = d_in (residual path) */
    int d_res = SL_SCR(10);
    wubu_vk_train_elt(vk, d_res, 0, d_in_s, 0, 1, (size_t)ch * n);
    /* conv2 bwd: in a2, w2, dout d_in -> d_a2, dw2, db2 */
    int d_a2 = SL_SCR(11);
    gzero(vk, d_a2, (size_t)ch * n);
    gzero(vk, dw2, (size_t)ch * ch * k);   /* grad slots accumulate — zero first */
    gzero(vk, db2, (size_t)ch);
    wubu_vk_train_bwd_conv1d(vk, a2_s, w2_s, d_in_s, d_a2, dw2, db2,
                             ch, ch, k, 1, pad2, 1, n, n);
    /* lrelu_bwd: pre-act c1o */
    int d_c1o = SL_SCR(12);
    wubu_vk_train_bwd_act(vk, c1o_s, d_a2, d_c1o, 0, (size_t)ch * n);
    /* conv1 bwd: in a1, w1, dout d_c1o -> d_a1, dw1, db1 */
    int d_a1 = SL_SCR(13);
    gzero(vk, d_a1, (size_t)ch * n);
    gzero(vk, dw1, (size_t)ch * ch * k);   /* grad slots accumulate — zero first */
    gzero(vk, db1, (size_t)ch);
    wubu_vk_train_bwd_conv1d(vk, a1_s, w1_s, d_c1o, d_a1, dw1, db1,
                             ch, ch, k, 1, pad1, dil, n, n);
    /* lrelu_bwd: pre-act p */
    int d_p = SL_SCR(14);
    wubu_vk_train_bwd_act(vk, p_s, d_a1, d_p, 0, (size_t)ch * n);
    /* d_in = d_p + d_res */
    wubu_vk_train_elt(vk, d_in_s, 0, d_p, 0, 1, (size_t)ch * n);
    wubu_vk_train_elt(vk, d_in_s, 0, d_res, 0, 0, (size_t)ch * n);
    (void)c2o_s;
}

static void mrf_stage_bwd_rec(WuBuVk *vk, WuBuRVCModel *model, TrainCacheVk *cache,
                              int L, int d_stage_in_s, int d_stage_out_s,
                              int ch, int n) {
    for (int s = 0; s < 3; s++) {
        /* distinct d_in per stack — the pipelined ring can have stack s-1's
         * final accumulate still in flight when stack s's first elt runs;
         * sharing one scratch slot would race (read vs overwrite). */
        int d_in = SL_SCR(20 + s);
        /* d_in = d_stage_out / 3 */
        wubu_vk_train_elt(vk, d_in, 0, d_stage_out_s, 0, 1, (size_t)ch * n);
        wubu_vk_train_act(vk, d_in, -1, 4, ch, n, 1.0f / 3.0f);
        for (int cp = 2; cp >= 0; cp--)
            mrf_pair_bwd_rec(vk, model, cache, L, s, cp, d_in, ch, n);
        /* d_stage_in += d_in */
        wubu_vk_train_elt(vk, d_stage_in_s, 0, d_in, 0, 0, (size_t)ch * n);
    }
}

int wubu_train_backward_vk(WuBuVk *vk, WuBuRVCModel *model, TrainCacheVk *cache,
                           const float *mel_in, const float *d_audio,
                           WuBuTrainRegistry *reg) {
    if (!vk || !model || !cache || !d_audio || !reg) return -1;
    int n3 = cache->n_stage[3];

    /* NOTE: grad slots are DEVICE_LOCAL-only (bwd record ops allocate them
     * that way to keep the PCIe BAR free for the weight uploads). All zeroing
     * below is GPU-side (gzero elt copies from the persistent GPU zero slot);
     * the final download stages each grad through the host-visible SL_STAGE. */
    wubu_vk_train_begin(vk);

    /* host-visible staging for grad downloads — SMALL (1MB) because the BAR
     * is already consumed by the weight uploads; each grad is staged in
     * 1MB chunks below. */
    {
        if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG bwd: stage alloc 1MB\n");
        if (wubu_vk_train_zero(vk, SL_STAGE, 1024 * 1024)) {
            if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG bwd: STAGE ALLOC FAIL\n");
            wubu_vk_train_end(vk); return -1;
        }
        if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG bwd: stage ok\n");
    }

    /* d_audio -> tanh_bwd -> conv_post bwd -> lrelu_bwd -> MRF3 */
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG bwd: begin ok\n");
    wubu_vk_train_upload(vk, SL_SCR(30), d_audio, (size_t)n3 * 4);
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG bwd: d_audio uploaded\n");
    int d_tanh = SL_SCR(31);
    wubu_vk_train_bwd_act(vk, SL_AUD, SL_SCR(30), d_tanh, 1, (size_t)n3);
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG bwd: tanh bwd done\n");
    const RVCTensor *post_w = wubu_rvc_find_tensor(model, "dec.conv_post.weight");
    if (!post_w || !post_w->data) { wubu_vk_train_end(vk); return -1; }
    int post_in = post_w->dims[1], post_k = post_w->dims[2];
    int d_post = SL_SCR(32);
    gzero(vk, d_post, (size_t)32 * n3);
    gzero(vk, g_slot(cache->ri_post_w), (size_t)post_in * post_k);
    if (cache->ri_post_b >= 0) gzero(vk, g_slot(cache->ri_post_b), 1);
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG bwd: d_post zeroed\n");
    {
        int dbp = (cache->ri_post_b >= 0) ? g_slot(cache->ri_post_b) : SL_SCR(48);
        wubu_vk_train_bwd_conv1d(vk, SL_POST, w_slot(cache->ri_post_w), d_tanh,
                                 d_post, g_slot(cache->ri_post_w), dbp,
                                 post_in, 1, post_k, 1, post_k / 2, 1, n3, n3);
    }
    if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG bwd: conv_post bwd done\n");
    /* conv_post uses post_in dims, not stage ch */
    int d_stage3 = SL_SCR(35);
    wubu_vk_train_bwd_act(vk, SL_STG(3), d_post, d_stage3, 0, (size_t)32 * n3);

    /* MRF backward stages 3 -> 0 */
    int d_stage_in = SL_SCR(36);
    gzero(vk, d_stage_in, (size_t)32 * n3);
    mrf_stage_bwd_rec(vk, model, cache, 3, d_stage_in, d_stage3, 32, n3);
    int d_act = SL_SCR(37);
    for (int L = 3; L >= 0; L--) {
        int in_ch = cache->ch_in[L];
        int out_ch = cache->ch_out[L];
        int in_n = (L == 0) ? cache->n_frames : cache->n_stage[L - 1];
        int kk = cache->k[L], stride = cache->s[L], pad = cache->p[L];
        int d_act_s = d_act;
        if (L >= 1) d_act_s = SL_SCR(37 + L);  /* distinct per stage */
        gzero(vk, d_act_s, (size_t)in_ch * in_n);
        gzero(vk, g_slot(cache->ri_ups_w[L]), (size_t)in_ch * out_ch * kk);
        if (cache->ri_ups_b[L] >= 0) gzero(vk, g_slot(cache->ri_ups_b[L]), (size_t)out_ch);
        {
            int dub = (cache->ri_ups_b[L] >= 0) ? g_slot(cache->ri_ups_b[L]) : SL_SCR(48);
            /* n_out for ups[L] convT bwd = n_stage[L] (CPU ref:
             * convt1d_bwd(..., n_in=n_stage[L-1], n_out=n_stage[L])). The old
             * cur_n bookkeeping drifted one stage high (L=2 used n_stage[3]…),
             * which made the shader read dout past its real length AND made
             * pool_slot_dev grow the d_stage_in slot mid-chain — silently
             * discarding the MRF output (first-call L1chain all-zero bug). */
            wubu_vk_train_bwd_convt1d(vk, SL_ACT(L), w_slot(cache->ri_ups_w[L]), d_stage_in,
                                      d_act_s, g_slot(cache->ri_ups_w[L]), dub,
                                      in_ch, out_ch, kk, stride, pad, in_n,
                                      cache->n_stage[L]);
        }
        if (L >= 1) {
            int d_stage_prev = SL_SCR(42 + L);
            wubu_vk_train_bwd_act(vk, SL_STG(L - 1), d_act_s, d_stage_prev, 0,
                                  (size_t)in_ch * in_n);
            int nch = cache->ch_out[L - 1];
            gzero(vk, d_stage_in, (size_t)nch * cache->n_stage[L - 1]);
            mrf_stage_bwd_rec(vk, model, cache, L - 1, d_stage_in, d_stage_prev,
                              nch, cache->n_stage[L - 1]);
        } else {
            /* L == 0: d_act = grad wrt act_in[0] — conv_pre */
            int init_ch = cache->ch_in[0];
            int d_pre = SL_SCR(44);
            if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG bwd: L=0 d_act_s=%d\n", d_act_s);
            wubu_vk_train_bwd_act(vk, SL_PRE, d_act_s, d_pre, 0,
                                  (size_t)init_ch * cache->n_frames);
            if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG bwd: L=0 pre bwd_act done\n");
            const RVCTensor *pre_w = wubu_rvc_find_tensor(model, "dec.conv_pre.weight");
            if (!pre_w || !pre_w->data) { wubu_vk_train_end(vk); return -1; }
            int pre_in = pre_w->dims[1], pk = pre_w->dims[2];
            gzero(vk, g_slot(cache->ri_pre_w), (size_t)init_ch * pre_in * pk);
            if (cache->ri_pre_b >= 0) gzero(vk, g_slot(cache->ri_pre_b), (size_t)init_ch);
            {
                int dbp = (cache->ri_pre_b >= 0) ? g_slot(cache->ri_pre_b) : SL_SCR(48);
                wubu_vk_train_bwd_conv1d(vk, SL_MEL, w_slot(cache->ri_pre_w), d_pre,
                                         SL_SCR(46), g_slot(cache->ri_pre_w), dbp,
                                         pre_in, init_ch, pk, 1, pk / 2, 1,
                                         cache->n_frames, cache->n_frames);
            }
        }
        if (L > 0) d_stage_in = SL_SCR(36); /* reuse */
    }

    if (wubu_vk_train_end(vk) != 0) return -1;

    if (getenv("WUBU_VK_DBG")) {
        /* registry index consistency check */
        for (int i = 0; i < reg->count && i < 160; i++) {
            if (strcmp(reg->params[i].name, "dec.resblocks.3.convs1.0.weight_v") == 0)
                fprintf(stderr, "VKDBG bwd: reg_vk idx of rb3c1w0 = %d\n", i);
        }
        /* dump the L=1 chain: d_stage_prev (833), d_act (828) + d_stage_in (826)
         * L=1 dims: d_act 256x160, d_stage_prev 256x160; d_stage_in 128x1280 */
        int slots[3] = { 833, 828, 826 };
        size_t sizes[3] = { (size_t)256 * 160, (size_t)256 * 160, (size_t)128 * 1280 };
        for (int si = 0; si < 3; si++) {
            size_t chunk = 256 * 1024;   /* fit SL_STAGE 1MB */
            float *tmp = malloc(sizes[si] * 4);
            for (size_t off = 0; off < sizes[si]; off += chunk) {
                size_t cnt = (sizes[si] - off < chunk) ? sizes[si] - off : chunk;
                if (wubu_vk_train_begin(vk) != 0) return -1;
                wubu_vk_train_elt(vk, SL_STAGE, 0, slots[si], off, 1, cnt);
                wubu_vk_train_end(vk);
                wubu_vk_train_download(vk, SL_STAGE, tmp + off, cnt * 4);
            }
            long z = 0, nz = 0; float vmin = 1e30f, vmax = -1e30f;
            for (size_t j = 0; j < sizes[si]; j++) {
                if (tmp[j] == 0.0f) z++; else nz++;
                if (tmp[j] < vmin) vmin = tmp[j];
                if (tmp[j] > vmax) vmax = tmp[j];
            }
            fprintf(stderr, "VKDBG bwd: L1chain slot%d zero=%ld nz=%ld vmin=%.6f vmax=%.6f\n",
                    slots[si], z, nz, vmin, vmax);
            free(tmp);
        }
    }

    /* staged download: grads are DEVICE_LOCAL; copy each into the small
     * host-visible staging slot in 256K-float chunks, then read it back. */
    {
        const int chunk = 256 * 1024;   /* 1MB floats*4 staging */
        for (int i = 0; i < reg->count; i++) {
            if (!reg->params[i].grad || reg->params[i].n <= 0) continue;
            int n = reg->params[i].n;
            for (int off = 0; off < n; off += chunk) {
                int cnt = (n - off < chunk) ? n - off : chunk;
                if (getenv("WUBU_VK_DBG")) fprintf(stderr, "VKDBG dl i=%d off=%d cnt=%d\n", i, off, cnt);
                if (wubu_vk_train_begin(vk) != 0) return -1;
                wubu_vk_train_elt(vk, SL_STAGE, 0, g_slot(i), off, 1, (size_t)cnt);
                if (wubu_vk_train_end(vk) != 0) return -1;
                wubu_vk_train_download(vk, SL_STAGE,
                                       reg->params[i].grad + off, (size_t)cnt * 4);
            }
        }
    }
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
    cache->reg = reg;
    int n_out = wubu_train_forward_vk(vk, model, mel_in, n_frames, audio, max_samples, cache);
    if (n_out <= 0) { free(audio); wubu_train_cache_free_vk(cache); free(cache); return -1; }
    int n = n_out < n_samples ? n_out : n_samples;
    if (epoch == 0 && getenv("WUBU_VK_DBG")) {
        float s = 0.0f;
        for (int i = 0; i < n && i < 256; i++) s += audio[i] * audio[i];
        fprintf(stderr, "VKDBG step0 audio rms0=%.6f (n=%d)\n", sqrtf(s / (n < 256 ? n : 256)), n);
    }
    float mse = wubu_mse_loss(audio, wav, n);

    /* Spectral supervision (Qwen3-TTS/HiFi-GAN recipe) — same as the CPU
     * trainer: add multi-scale STFT linear+log magnitude loss so harmonics
     * and formants stay sharp. WUBU_SPECTRAL_WEIGHT env tunes λ (default
     * 1.0, spectral grad RMS ≈ 0.5× MSE grad RMS); set 0 for pure MSE. */
    float spectral_weight = 1.0f;
    const char *sw = getenv("WUBU_SPECTRAL_WEIGHT");
    if (sw) spectral_weight = (float)atof(sw);
    float spec_loss = 0.0f;
    float *d_spec = NULL;
    if (spectral_weight > 0.0f && n > 128) {
        d_spec = (float *)malloc((size_t)n_out * sizeof(float));
        if (d_spec) {
            spec_loss = wubu_stft_loss_grad(audio, wav, n, 40000, d_spec, 3);
            float gnorm = 0.0f;
            for (int i = 0; i < n; i++) gnorm += d_spec[i] * d_spec[i];
            gnorm = sqrtf(gnorm / (float)n);
            if (gnorm > 1e-9f && mse > 1e-12f) {
                float scale = spectral_weight * 0.5f * (2.0f / (float)n) * sqrtf(mse) / gnorm;
                for (int i = 0; i < n; i++) d_spec[i] *= scale;
            } else {
                for (int i = 0; i < n; i++) d_spec[i] = 0.0f;
            }
        }
    }

    float total_loss = mse + spec_loss;
    if (loss_out) *loss_out = total_loss;
    model->last_loss = total_loss;
    model->last_epoch = epoch;
    float *d_audio = (float *)malloc((size_t)n_out * sizeof(float));
    if (!d_audio) { free(d_spec); free(audio); wubu_train_cache_free_vk(cache); free(cache); return -1; }
    for (int i = 0; i < n; i++) {
        d_audio[i] = 2.0f * (audio[i] - wav[i]) / (float)n;
        if (d_spec) d_audio[i] += d_spec[i];
    }
    for (int i = n; i < n_out; i++) d_audio[i] = 0.0f;
    wubu_train_registry_zero_grads(reg);
    if (wubu_train_backward_vk(vk, model, cache, mel_in, d_audio, reg) != 0) {
        free(d_spec); free(d_audio); free(audio); wubu_train_cache_free_vk(cache); free(cache); return -1;
    }
    for (int i = 0; i < reg->count; i++)
        wubu_adamw_step(opt, i, reg->params[i].data, reg->params[i].grad);
    free(d_spec); free(d_audio); free(audio);
    wubu_train_cache_free_vk(cache); free(cache);
    return (total_loss < 0.1f) ? 1 : 0;
}
