/* wubu_train_cuda.cu — GPU-accelerated RVC training step.
 *
 * Mirrors the CPU training math in wubu_train.c EXACTLY (Triple-DA:
 * the CPU train_step + gradcheck are the reference). Provides:
 *
 *   1. wubu_train_forward_cuda  — cached decoder forward (conv_pre, lrelu,
 *      upsample convT, MRF, conv_post, tanh). Caches the pre-activations the
 *      backward needs, like DecCache in wubu_train.c. NOTE: the CPU training
 *      decoder has NO cond, NO sine, NO noise convs — the training reference
 *      is decoder_forward, not the inference generator.
 *   2. wubu_train_backward_cuda — conv1d/convt1d/lrelu/tanh backward kernels
 *      writing grads straight into the training registry (host memory,
 *      pulled back per step).
 *   3. wubu_train_step_cuda     — forward → MSE loss → backward → AdamW.
 *
 * Kernel math mirrors the CPU loops; grads match to float32 tolerance.
 *
 * Weight layout (matches wubu_train.c / wubu_rvc_parity.c):
 *   conv1d:   W[(oc*in_ch + ic)*k + tap]
 *   convT1d:  W[(ic*out_ch + oc)*k + tap]  (PyTorch ConvTranspose1d layout)
 *   activations col-major [ch, n].
 *
 * BUILD NOTE: this file is #included at the bottom of wubu_rvc_cuda.cu and
 * compiled as ONE nvcc object (build/cuda_build.bat). Separate nvcc objects
 * each emit strong CRT math stubs (?tanh@@YAMM@Z etc.) that duplicate and
 * crash at load when linked together. No snprintf/fprintf in host code —
 * those pull MSVC CRT symbols that MinGW's link can't resolve.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" {
#include "wubu_rvc_parity.h"
#include "wubu_train.h"
}

/* manual key builder (no snprintf — avoids MSVC CRT deps when linking the
 * nvcc object with MinGW). Builds e.g. "dec.ups.3.bias". The inference file
 * (wubu_rvc_cuda.cu) already defines kfmt; when this file is #included into
 * that TU the static name would collide, so it's tfmt here. */
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

/* ── backward kernels ──────────────────────────────────────────────── */

__global__ void tk_conv1d_bwd(const float *in, const float *w, const float *dout,
                              float *din, float *dw, float *db,
                              int in_ch, int out_ch, int k, int stride, int pad, int dil,
                              int n_in, int n_out) {
    /* one block per (oc, ic) pair; grid = (out_ch, in_ch) */
    int oc = blockIdx.x;
    int ic = blockIdx.y;
    const float *wv = w + (size_t)oc * in_ch * k;
    const float *do_ = dout + (size_t)oc * n_out;
    float *din_row = din ? din + (size_t)ic * n_in : NULL;
    float *dw_row = dw ? dw + (size_t)oc * in_ch * k + (size_t)ic * k : NULL;
    __shared__ float dw_s[32];  /* k <= 32 in RVC (11 max) */
    for (int t = threadIdx.x; t < k; t += blockDim.x) dw_s[t] = 0.0f;
    __syncthreads();

    /* din accumulates via GLOBAL atomics — a shared din_s[1024] would
     * overflow when n_in > 1024 (stage 3 has n=5120). Global atomics are
     * correct across (oc,ic) blocks; dw is per-(oc,ic) so it stays shared. */
    for (int j = threadIdx.x; j < n_out; j += blockDim.x) {
        float g = do_[j];
        int j_in = j * stride - pad;
        for (int tap = 0; tap < k; tap++) {
            int src = j_in + dil * tap;
            if (src >= 0 && src < n_in) {
                float x = in[(size_t)ic * n_in + src];
                if (dw_row) atomicAdd(&dw_s[tap], x * g);
                if (din_row) atomicAdd(&din_row[src], g * wv[(size_t)ic * k + tap]);
            }
        }
    }
    __syncthreads();
    if (dw_row)
        for (int t = threadIdx.x; t < k; t += blockDim.x) atomicAdd(&dw_row[t], dw_s[t]);
    if (db && threadIdx.x == 0) {
        float acc = 0.0f;
        for (int j = 0; j < n_out; j++) acc += do_[j];
        atomicAdd(&db[oc], acc);
    }
}

__global__ void tk_convt1d_bwd(const float *in, const float *w, const float *dout,
                               float *din, float *dw, float *db,
                               int in_ch, int out_ch, int k, int stride, int pad,
                               int n_in, int n_out) {
    /* one block per (ic, i-tile); grid = (tiles, in_ch). Col-major in [ic,n]. */
    int ic = blockIdx.y;
    int tile = blockIdx.x;
    int TILE = 256;
    int i0 = tile * TILE;
    const float *wv = w + (size_t)ic * out_ch * k;
    float *din_row = din ? din + (size_t)ic * n_in : NULL;
    float *dw_row = dw ? dw + (size_t)ic * out_ch * k : NULL;
    for (int i = i0 + threadIdx.x; i < n_in && i < i0 + TILE; i += blockDim.x) {
        int j_start = i * stride - pad;
        float din_acc = 0.0f;
        for (int oc = 0; oc < out_ch; oc++) {
            const float *wk = wv + (size_t)oc * k;
            for (int tap = 0; tap < k; tap++) {
                int j = j_start + tap;
                if (j >= 0 && j < n_out) {
                    float g = dout[(size_t)oc * n_out + j];
                    din_acc += g * wk[tap];
                    if (dw_row) atomicAdd(&dw_row[(size_t)oc * k + tap], in[(size_t)ic * n_in + i] * g);
                }
            }
        }
        if (din_row) din_row[i] += din_acc;
    }
}

/* db for convT: db[oc] = sum_j dout[oc,j] — separate grid pass. */
__global__ void tk_convt1d_db(const float *dout, float *db, int out_ch, int n_out) {
    int oc = blockIdx.x;
    const float *do_ = dout + (size_t)oc * n_out;
    float acc = 0.0f;
    for (int j = threadIdx.x; j < n_out; j += blockDim.x) acc += do_[j];
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) db[oc] = red[0];
}

__global__ void tk_lrelu_bwd(const float *x, const float *dout, float *din, size_t n, float slope) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) din[i] = dout[i] * (x[i] > 0.0f ? 1.0f : slope);
}

__global__ void tk_tanh_bwd(const float *y, const float *dout, float *din, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) din[i] = dout[i] * (1.0f - y[i] * y[i]);
}

__global__ void tk_mse_grad(const float *audio, const float *wav, float *d_audio,
                            size_t n, size_t n_out, float inv_n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_audio[i] = 2.0f * (audio[i] - wav[i]) * inv_n;
    else if (i < n_out) d_audio[i] = 0.0f;
}

/* ── cached forward (like DecCache in wubu_train.c) ────────────────── */

typedef struct TrainCacheCuda {
    int n_frames;
    int n_stage[4];
    int ch_in[4], ch_out[4];
    int k[4], s[4], p[4];
    int n_ups;
    float *pre_out;      /* (init_ch, F) after conv_pre, PRE-lrelu */
    float *act_in[4];    /* lrelu'd input to each ups convT (col-major) */
    float *ups_out[4];   /* after convT (pre-MRF) */
    float *stage_out[4]; /* after MRF, PRE-lrelu */
    float *post_in;      /* (32, n3) after final lrelu (pre conv_post) */
    float *audio;        /* after tanh */
    /* MRF pair intermediates (for backward), per stage: [stack][pair] */
    float *mrf_p[4][3][3];   /* pair input (pre-lrelu) */
    float *mrf_a1[4][3][3];  /* lrelu(p) */
    float *mrf_c1[4][3][3];  /* conv1 out (pre-lrelu) */
    float *mrf_a2[4][3][3];  /* lrelu(c1) */
    float *mrf_c2[4][3][3];  /* conv2 out (pre-add) */
    float *d_stage_out[4];   /* upstream grad per stage (device) */
} TrainCacheCuda;

static void tcache_free(TrainCacheCuda *c) {
    if (!c) return;
    if (c->pre_out) cudaFree(c->pre_out);
    for (int L = 0; L < 4; L++) {
        if (c->act_in[L]) cudaFree(c->act_in[L]);
        if (c->ups_out[L]) cudaFree(c->ups_out[L]);
        if (c->stage_out[L]) cudaFree(c->stage_out[L]);
        if (c->d_stage_out[L]) cudaFree(c->d_stage_out[L]);
        for (int s = 0; s < 3; s++)
            for (int cp = 0; cp < 3; cp++) {
                if (c->mrf_p[L][s][cp]) cudaFree(c->mrf_p[L][s][cp]);
                if (c->mrf_a1[L][s][cp]) cudaFree(c->mrf_a1[L][s][cp]);
                if (c->mrf_c1[L][s][cp]) cudaFree(c->mrf_c1[L][s][cp]);
                if (c->mrf_a2[L][s][cp]) cudaFree(c->mrf_a2[L][s][cp]);
                if (c->mrf_c2[L][s][cp]) cudaFree(c->mrf_c2[L][s][cp]);
            }
    }
    memset(c, 0, sizeof(*c));
}

void wubu_train_cache_free_cuda(TrainCacheCuda *cache) {
    tcache_free(cache);
    free(cache);
}

TrainCacheCuda *wubu_train_cache_alloc_cuda(void) {
    TrainCacheCuda *c = (TrainCacheCuda *)calloc(1, sizeof(TrainCacheCuda));
    return c;
}

/* forward kernels (same math as wubu_rvc_cuda.cu but keeping intermediates) */
extern __global__ void k_conv1d(const float *in, int in_ch, int n,
                                const float *w, const float *b,
                                int out_ch, int k, int stride, int pad, int dil,
                                int n_out, float *out);
extern __global__ void k_convt1d(const float *in, int in_ch, int n,
                                 const float *w, const float *b,
                                 int out_ch, int k, int stride, int pad,
                                 int n_out, float *out);
extern __global__ void k_lrelu(float *x, size_t n, float slope);
extern __global__ void k_tanh(float *x, size_t n);
extern __global__ void k_add(float *a, const float *b, size_t n);
extern __global__ void k_copy(float *dst, const float *src, size_t n);
extern __global__ void k_mul(float *x, size_t n, float s);

static void t_conv1d(const float *in, int in_ch, int n, const float *w, const float *b,
                     int out_ch, int k, int stride, int pad, int dil, int n_out,
                     float *out, cudaStream_t st) {
    dim3 grid((unsigned)((n_out + 15) / 16), (unsigned)((out_ch + 3) / 4));
    dim3 block(16, 4);
    k_conv1d<<<grid, block, 0, st>>>(in, in_ch, n, w, b, out_ch, k, stride, pad, dil, n_out, out);
}

static void t_convt1d(const float *in, int in_ch, int n, const float *w, const float *b,
                      int out_ch, int k, int stride, int pad, int n_out,
                      float *out, cudaStream_t st) {
    dim3 grid((unsigned)((n_out + 31) / 32), (unsigned)out_ch);
    dim3 block(32, 1);
    k_convt1d<<<grid, block, 0, st>>>(in, in_ch, n, w, b, out_ch, k, stride, pad, n_out, out);
}

/* Get upsample config from model (same inference as decoder_forward). */
static void t_ups_config(const WuBuRVCModel *model, int *n_ups, int *rate, int *k, int *pad) {
    *n_ups = 4;
    for (int L = 0; L < 4; L++) {
        rate[L] = model->upsample_rates[L] > 0 ? model->upsample_rates[L] : 10;
        int fallback_k[4] = {16, 16, 4, 4};
        k[L] = model->upsample_kernel_sizes[L] > 0 ? model->upsample_kernel_sizes[L] : fallback_k[L];
        pad[L] = (k[L] - rate[L]) / 2;
    }
}

/* ── the cached forward ────────────────────────────────────────────── */

int wubu_train_forward_cuda(WuBuRVCModel *model, const float *mel_in, int n_frames,
                            float *audio, int max_samples, TrainCacheCuda *cache) {
    if (!model || !mel_in || !audio || !cache) return -1;
    memset(cache, 0, sizeof(*cache));
    cache->n_frames = n_frames;

    const RVCTensor *conv_pre_w = wubu_rvc_find_tensor(model, "dec.conv_pre.weight");
    const RVCTensor *conv_pre_b = wubu_rvc_find_tensor(model, "dec.conv_pre.bias");
    const RVCTensor *post_w = wubu_rvc_find_tensor(model, "dec.conv_post.weight");
    if (!conv_pre_w || !conv_pre_w->data || !post_w || !post_w->data) return -1;

    int init_ch = conv_pre_w->dims[0];
    int n_ups = 4;
    int rate[4], k[4], pad[4];
    t_ups_config(model, &n_ups, rate, k, pad);
    cache->n_ups = n_ups;

    /* conv_pre: mel_in col-major (pre_in, F) → (init_ch, F), PRE-lrelu in cache.
     * NOTE: matches CPU decoder_forward — NO cond, NO sine, NO noise convs. */
    cudaStream_t st = 0;
    float *d_z = NULL, *d_x = NULL, *d_cur = NULL;
    float *d_tmp = NULL, *d_tmp2 = NULL, *d_acc = NULL, *d_stage = NULL;
    float *d_out = NULL;
    cudaMalloc(&d_z, (size_t)conv_pre_w->dims[1] * n_frames * sizeof(float));
    cudaMemcpy(d_z, mel_in, (size_t)conv_pre_w->dims[1] * n_frames * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_x, (size_t)init_ch * n_frames * sizeof(float));
    cudaMalloc(&cache->pre_out, (size_t)init_ch * n_frames * sizeof(float));

    float *pre_w = (float *)malloc((size_t)conv_pre_w->dims[0] * conv_pre_w->dims[1] * conv_pre_w->dims[2] * sizeof(float));
    if (!pre_w) goto fail;
    memcpy(pre_w, conv_pre_w->data, (size_t)conv_pre_w->dims[0] * conv_pre_w->dims[1] * conv_pre_w->dims[2] * sizeof(float));
    float *d_prew = NULL, *d_preb = NULL;
    cudaMalloc(&d_prew, (size_t)conv_pre_w->dims[0] * conv_pre_w->dims[1] * conv_pre_w->dims[2] * sizeof(float));
    cudaMemcpy(d_prew, pre_w, (size_t)conv_pre_w->dims[0] * conv_pre_w->dims[1] * conv_pre_w->dims[2] * sizeof(float), cudaMemcpyHostToDevice);
    if (conv_pre_b && conv_pre_b->data) {
        cudaMalloc(&d_preb, (size_t)conv_pre_b->dims[0] * sizeof(float));
        cudaMemcpy(d_preb, conv_pre_b->data, (size_t)conv_pre_b->dims[0] * sizeof(float), cudaMemcpyHostToDevice);
    }
    int pk = conv_pre_w->dims[2];
    t_conv1d(d_z, conv_pre_w->dims[1], n_frames, d_prew, d_preb, init_ch, pk, 1, pk / 2, 1,
             n_frames, d_x, st);
    cudaMemcpy(cache->pre_out, d_x, (size_t)init_ch * n_frames * sizeof(float), cudaMemcpyDeviceToDevice);

    /* act_in[0] = lrelu(pre_out) */
    cudaMalloc(&cache->act_in[0], (size_t)init_ch * n_frames * sizeof(float));
    cudaMemcpy(cache->act_in[0], d_x, (size_t)init_ch * n_frames * sizeof(float), cudaMemcpyDeviceToDevice);
    k_lrelu<<<(unsigned)((init_ch * n_frames + 255) / 256), 256, 0, st>>>(cache->act_in[0], (size_t)init_ch * n_frames, 0.1f);

    /* upsample blocks */
    int cur_ch = init_ch;
    int cur_n = n_frames;
    cudaMalloc(&d_cur, (size_t)init_ch * n_frames * sizeof(float));
    cudaMemcpy(d_cur, cache->act_in[0], (size_t)init_ch * n_frames * sizeof(float), cudaMemcpyDeviceToDevice);

    for (int L = 0; L < n_ups; L++) {
        int in_ch = cur_ch;
        int in_n = cur_n;
        int out_ch = init_ch / (1 << (L + 1));
        int out_n = (in_n - 1) * rate[L] - 2 * pad[L] + k[L];
        if (out_n <= 0) goto fail;
        cache->ch_in[L] = in_ch; cache->ch_out[L] = out_ch;
        cache->n_stage[L] = out_n;
        cache->k[L] = k[L]; cache->s[L] = rate[L]; cache->p[L] = pad[L];

        /* ups convT — use the loader's de-normalized array (what the CPU
         * decoder_forward and the training registry both read) */
        char key[128];
        tfmt(key, "dec.ups.", L, "", -1, ".bias");
        const RVCTensor *ub = wubu_rvc_find_tensor(model, key);
        if (!model->hifi_upsample_denorm[L] || !ub || !ub->data) goto fail;
        int den_len = in_ch * out_ch * k[L];
        float *den = model->hifi_upsample_denorm[L];
        float *d_uw = NULL, *d_ub = NULL;
        cudaMalloc(&d_uw, (size_t)den_len * sizeof(float));
        cudaMemcpy(d_uw, den, (size_t)den_len * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_ub, (size_t)out_ch * sizeof(float));
        cudaMemcpy(d_ub, ub->data, (size_t)out_ch * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&cache->ups_out[L], (size_t)out_ch * out_n * sizeof(float));
        t_convt1d(d_cur, in_ch, in_n, d_uw, d_ub, out_ch, k[L], rate[L], pad[L],
                  out_n, cache->ups_out[L], st);
        cudaFree(d_uw); cudaFree(d_ub);

        /* MRF: 3 stacks x 3 pairs, avg -> stage_out[L] (keeps intermediates) */
        int ch = out_ch;
        int n2 = ch * out_n;
        cudaMalloc(&d_acc, (size_t)3 * ch * out_n * sizeof(float));
        cudaMalloc(&d_stage, (size_t)ch * out_n * sizeof(float));
        cudaMalloc(&cache->stage_out[L], (size_t)ch * out_n * sizeof(float));
        cudaMemset(d_acc, 0, (size_t)3 * ch * out_n * sizeof(float));
        cudaMalloc(&d_cur, (size_t)ch * out_n * sizeof(float));
        cudaMemcpy(d_cur, cache->ups_out[L], (size_t)ch * out_n * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMalloc(&d_tmp2, (size_t)ch * out_n * sizeof(float));
        cudaMalloc(&d_tmp, (size_t)ch * out_n * sizeof(float));

        for (int s = 0; s < 3; s++) {
            int rb = L * 3 + s;
            int kk = (s == 0) ? 3 : (s == 1 ? 7 : 11);
            cudaMemcpy(d_tmp, d_cur, (size_t)ch * out_n * sizeof(float), cudaMemcpyDeviceToDevice);
            for (int cp = 0; cp < 3; cp++) {
                tfmt(key, "dec.resblocks.", rb, ".convs1.", cp, ".weight_v");
                const RVCTensor *c1w = wubu_rvc_find_tensor(model, key);
                tfmt(key, "dec.resblocks.", rb, ".convs2.", cp, ".weight_v");
                const RVCTensor *c2w = wubu_rvc_find_tensor(model, key);
                if (!c1w || !c2w) goto fail;
                int dil = 1 + 2 * cp;
                int pad1 = dil * (kk - 1) / 2;
                int pad2 = kk / 2;
                int n1 = c1w->dims[0] * c1w->dims[1] * kk;
                int n2w = c2w->dims[0] * c2w->dims[1] * kk;

                float *d_c1 = NULL, *d_c2 = NULL, *d_c1b = NULL, *d_c2b = NULL;
                cudaMalloc(&d_c1, (size_t)n1 * sizeof(float));
                cudaMemcpy(d_c1, c1w->data, (size_t)n1 * sizeof(float), cudaMemcpyHostToDevice);
                cudaMalloc(&d_c2, (size_t)n2w * sizeof(float));
                cudaMemcpy(d_c2, c2w->data, (size_t)n2w * sizeof(float), cudaMemcpyHostToDevice);
                tfmt(key, "dec.resblocks.", rb, ".convs1.", cp, ".bias");
                const RVCTensor *c1b = wubu_rvc_find_tensor(model, key);
                tfmt(key, "dec.resblocks.", rb, ".convs2.", cp, ".bias");
                const RVCTensor *c2b = wubu_rvc_find_tensor(model, key);
                if (c1b && c1b->data) { cudaMalloc(&d_c1b, (size_t)c1b->dims[0] * sizeof(float)); cudaMemcpy(d_c1b, c1b->data, (size_t)c1b->dims[0] * sizeof(float), cudaMemcpyHostToDevice); }
                if (c2b && c2b->data) { cudaMalloc(&d_c2b, (size_t)c2b->dims[0] * sizeof(float)); cudaMemcpy(d_c2b, c2b->data, (size_t)c2b->dims[0] * sizeof(float), cudaMemcpyHostToDevice); }

                /* p = pair input (pre-lrelu) — save */
                cudaMalloc(&cache->mrf_p[L][s][cp], (size_t)ch * out_n * sizeof(float));
                cudaMemcpy(cache->mrf_p[L][s][cp], d_tmp, (size_t)ch * out_n * sizeof(float), cudaMemcpyDeviceToDevice);
                /* a1 = lrelu(p) — save */
                cudaMalloc(&cache->mrf_a1[L][s][cp], (size_t)ch * out_n * sizeof(float));
                cudaMemcpy(cache->mrf_a1[L][s][cp], d_tmp, (size_t)ch * out_n * sizeof(float), cudaMemcpyDeviceToDevice);
                k_lrelu<<<(unsigned)((ch * out_n + 255) / 256), 256, 0, st>>>(cache->mrf_a1[L][s][cp], (size_t)ch * out_n, 0.1f);
                /* c1o = conv1(a1) — save pre-act */
                cudaMalloc(&cache->mrf_c1[L][s][cp], (size_t)ch * out_n * sizeof(float));
                t_conv1d(cache->mrf_a1[L][s][cp], ch, out_n, d_c1, d_c1b, ch, kk, 1, pad1, dil,
                         out_n, cache->mrf_c1[L][s][cp], st);
                /* a2 = lrelu(c1o) — save */
                cudaMalloc(&cache->mrf_a2[L][s][cp], (size_t)ch * out_n * sizeof(float));
                cudaMemcpy(cache->mrf_a2[L][s][cp], cache->mrf_c1[L][s][cp], (size_t)ch * out_n * sizeof(float), cudaMemcpyDeviceToDevice);
                k_lrelu<<<(unsigned)((ch * out_n + 255) / 256), 256, 0, st>>>(cache->mrf_a2[L][s][cp], (size_t)ch * out_n, 0.1f);
                /* c2o = conv2(a2) — save pre-add */
                cudaMalloc(&cache->mrf_c2[L][s][cp], (size_t)ch * out_n * sizeof(float));
                t_conv1d(cache->mrf_a2[L][s][cp], ch, out_n, d_c2, d_c2b, ch, kk, 1, pad2, 1,
                         out_n, cache->mrf_c2[L][s][cp], st);
                /* out = c2o + p */
                k_add<<<(unsigned)((ch * out_n + 255) / 256), 256, 0, st>>>(cache->mrf_c2[L][s][cp], cache->mrf_p[L][s][cp], (size_t)ch * out_n);
                /* d_tmp = pair output (for next pair input) */
                cudaMemcpy(d_tmp, cache->mrf_c2[L][s][cp], (size_t)ch * out_n * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaFree(d_c1); cudaFree(d_c2); cudaFree(d_c1b); cudaFree(d_c2b);
            }
            /* acc[s] += stack_out */
            k_add<<<(unsigned)((ch * out_n + 255) / 256), 256, 0, st>>>(d_acc + (size_t)s * ch * out_n, d_tmp, (size_t)ch * out_n);
        }
        /* stage = avg(acc) */
        cudaMemcpy(d_stage, d_acc, (size_t)ch * out_n * sizeof(float), cudaMemcpyDeviceToDevice);
        for (int s = 1; s < 3; s++)
            k_add<<<(unsigned)((ch * out_n + 255) / 256), 256, 0, st>>>(d_stage, d_acc + (size_t)s * ch * out_n, (size_t)ch * out_n);
        k_mul<<<(unsigned)((ch * out_n + 255) / 256), 256, 0, st>>>(d_stage, (size_t)ch * out_n, 1.0f / 3.0f);
        cudaMemcpy(cache->stage_out[L], d_stage, (size_t)ch * out_n * sizeof(float), cudaMemcpyDeviceToDevice);

        /* save act_in for next upsample: lrelu(stage) — CPU gates this on
         * L < 3; for L=3 (final stage) the conv_post block applies the lrelu
         * itself, so creating act_in[4] would DOUBLE-lrelu the post input. */
        if (L < n_ups - 1) {
            cudaMalloc(&cache->act_in[L + 1], (size_t)out_ch * out_n * sizeof(float));
            cudaMemcpy(cache->act_in[L + 1], d_stage, (size_t)out_ch * out_n * sizeof(float), cudaMemcpyDeviceToDevice);
            k_lrelu<<<(unsigned)((out_ch * out_n + 255) / 256), 256, 0, st>>>(cache->act_in[L + 1], (size_t)out_ch * out_n, 0.1f);
            cudaMemcpy(d_cur, cache->act_in[L + 1], (size_t)out_ch * out_n * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpy(d_cur, d_stage, (size_t)out_ch * out_n * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        cur_ch = out_ch; cur_n = out_n;
        cudaFree(d_acc); cudaFree(d_stage); cudaFree(d_tmp2); cudaFree(d_tmp);
        d_acc = NULL; d_stage = NULL; d_tmp2 = NULL; d_tmp = NULL;
    }

    /* final: lrelu, conv_post, tanh */
    {
        int post_in = post_w->n_dims >= 2 ? post_w->dims[1] : 32;
        int post_k = post_w->n_dims >= 3 ? post_w->dims[2] : 7;
        int post_pad = post_k / 2;
        int out_n = cur_n;
        if (out_n > max_samples) out_n = max_samples;
        cudaMalloc(&cache->post_in, (size_t)post_in * cur_n * sizeof(float));
        cudaMemcpy(cache->post_in, d_cur, (size_t)post_in * cur_n * sizeof(float), cudaMemcpyDeviceToDevice);
        k_lrelu<<<(unsigned)((post_in * cur_n + 255) / 256), 256, 0, st>>>(cache->post_in, (size_t)post_in * cur_n, 0.1f);
        float *d_pw = NULL;
        cudaMalloc(&d_pw, (size_t)post_w->dims[0] * post_w->dims[1] * post_k * sizeof(float));
        cudaMemcpy(d_pw, post_w->data, (size_t)post_w->dims[0] * post_w->dims[1] * post_k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_out, (size_t)out_n * sizeof(float));
        t_conv1d(cache->post_in, post_in, cur_n, d_pw, NULL, 1, post_k, 1, post_pad, 1,
                 out_n, d_out, st);
        cudaFree(d_pw);
        k_tanh<<<(unsigned)((out_n + 255) / 256), 256, 0, st>>>(d_out, (size_t)out_n);
        cudaMemcpy(audio, d_out, (size_t)out_n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMalloc(&cache->audio, (size_t)out_n * sizeof(float));
        cudaMemcpy(cache->audio, d_out, (size_t)out_n * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(d_out); d_out = NULL;
        cudaFree(d_cur); d_cur = NULL;
        cudaFree(d_x); d_x = NULL;
        cudaFree(d_z); d_z = NULL;
        free(pre_w);
        return out_n;
    }

fail:
    cudaFree(d_z); cudaFree(d_x); cudaFree(d_cur);
    cudaFree(d_tmp); cudaFree(d_tmp2); cudaFree(d_acc); cudaFree(d_stage); cudaFree(d_out);
    free(pre_w);
    tcache_free(cache);
    return -1;
}

/* ═══════════════════════ backward (CUDA) ═══════════════════════
 * Mirrors decoder_backward in wubu_train.c exactly. */

int wubu_train_backward_cuda(WuBuRVCModel *model, TrainCacheCuda *cache,
                             const float *mel_in, const float *d_audio,
                             WuBuTrainRegistry *reg) {
    if (!model || !cache || !d_audio || !reg) return -1;
    cudaGetLastError(); /* clear any stale launch error */
    cudaStream_t st = 0;
    float *d_tanh_in = NULL;
    float *d_daudio = NULL;
    float *d_post_out = NULL, *d_stage3 = NULL;
    cudaMalloc(&d_tanh_in, (size_t)cache->n_stage[3] * sizeof(float));
    /* NOTE: d_audio is host memory — copy to device first, then tanh_bwd */
    cudaMalloc(&d_daudio, (size_t)cache->n_stage[3] * sizeof(float));
    cudaMemcpy(d_daudio, d_audio, (size_t)cache->n_stage[3] * sizeof(float), cudaMemcpyHostToDevice);
    tk_tanh_bwd<<<(unsigned)((cache->n_stage[3] + 255) / 256), 256, 0, st>>>(
        cache->audio, d_daudio, d_tanh_in, (size_t)cache->n_stage[3]);
    if (cudaGetLastError() != cudaSuccess) goto fail;

    /* conv_post backward: din wrt post_in = stage_out[3] (pre-lrelu) */
    const RVCTensor *post_w = wubu_rvc_find_tensor(model, "dec.conv_post.weight");
    if (!post_w || !post_w->data) goto fail;
    int post_in = post_w->dims[1], post_k = post_w->dims[2], post_pad = post_k / 2;
    int n3 = cache->n_stage[3];
    cudaMalloc(&d_post_out, (size_t)post_in * n3 * sizeof(float));
    cudaMemset(d_post_out, 0, (size_t)post_in * n3 * sizeof(float));
    cudaMalloc(&d_stage3, (size_t)32 * n3 * sizeof(float));
    cudaMemset(d_stage3, 0, (size_t)32 * n3 * sizeof(float));
    {
        float *d_pw = NULL;
        cudaMalloc(&d_pw, (size_t)post_w->dims[0] * post_in * post_k * sizeof(float));
        cudaMemcpy(d_pw, post_w->data, (size_t)post_w->dims[0] * post_in * post_k * sizeof(float), cudaMemcpyHostToDevice);
        float *d_pwg = NULL, *d_pbg = NULL;
        /* conv_post weight grads (registry name = dec.conv_post.weight) */
        int ipostw = wubu_train_registry_find(reg, "dec.conv_post.weight");
        float *pwg = (ipostw >= 0) ? reg->params[ipostw].grad : NULL;
        if (pwg) { cudaMalloc(&d_pwg, (size_t)post_w->dims[0] * post_in * post_k * sizeof(float)); cudaMemset(d_pwg, 0, (size_t)post_w->dims[0] * post_in * post_k * sizeof(float)); }
        /* conv_post bias grads (registry name = dec.conv_post.bias) */
        int ipostb = wubu_train_registry_find(reg, "dec.conv_post.bias");
        float *pbg = (ipostb >= 0) ? reg->params[ipostb].grad : NULL;
        if (pbg) { cudaMalloc(&d_pbg, (size_t)post_w->dims[0] * sizeof(float)); cudaMemset(d_pbg, 0, (size_t)post_w->dims[0] * sizeof(float)); }
        tk_conv1d_bwd<<<dim3(1, (unsigned)post_in), 256, 0, st>>>(
            cache->post_in, d_pw, d_tanh_in, d_post_out, d_pwg, d_pbg,
            post_in, 1, post_k, 1, post_pad, 1, n3, n3);
        if (d_pwg) { cudaMemcpy(pwg, d_pwg, (size_t)post_w->dims[0] * post_in * post_k * sizeof(float), cudaMemcpyDeviceToHost); cudaFree(d_pwg); }
        if (d_pbg) { cudaMemcpy(pbg, d_pbg, (size_t)post_w->dims[0] * sizeof(float), cudaMemcpyDeviceToHost); cudaFree(d_pbg); }
        cudaFree(d_pw);
    }
    /* lrelu_bwd(stage_out[3], d_post_out, d_stage3) */
    tk_lrelu_bwd<<<(unsigned)((32 * n3 + 255) / 256), 256, 0, st>>>(
        cache->stage_out[3], d_post_out, d_stage3, (size_t)32 * n3, 0.1f);

    /* ── MRF backward, stage 3 → 0 (mirror decoder_backward) ── */
    for (int L = 3; L >= 0; L--) {
        int ch = cache->ch_out[L];
        int out_n = cache->n_stage[L];
        size_t n2 = (size_t)ch * out_n;
        float *d_stage_in = NULL;
        cudaMalloc(&d_stage_in, n2 * sizeof(float));
        cudaMemset(d_stage_in, 0, n2 * sizeof(float));
        const float *d_stage_out = (L == 3) ? d_stage3 : cache->d_stage_out[L + 1];
        if (!d_stage_out) { cudaFree(d_stage_in); goto fail; }

        /* per-stack: d_in starts at d_stage_out/3 */
        for (int s = 0; s < 3; s++) {
            int rb = L * 3 + s;
            int kk = (s == 0) ? 3 : (s == 1 ? 7 : 11);
            float *d_in = NULL;
            cudaMalloc(&d_in, n2 * sizeof(float));
            k_mul<<<(unsigned)((n2 + 255) / 256), 256, 0, st>>>(d_in, n2, 0.0f); /* zero */
            /* d_in = d_stage_out / 3 */
            k_copy<<<(unsigned)((n2 + 255) / 256), 256, 0, st>>>(d_in, d_stage_out, n2);
            k_mul<<<(unsigned)((n2 + 255) / 256), 256, 0, st>>>(d_in, n2, 1.0f / 3.0f);

            for (int cp = 2; cp >= 0; cp--) {
                char key[128];
                tfmt(key, "dec.resblocks.", rb, ".convs1.", cp, ".weight_v");
                const RVCTensor *c1w = wubu_rvc_find_tensor(model, key);
                tfmt(key, "dec.resblocks.", rb, ".convs2.", cp, ".weight_v");
                const RVCTensor *c2w = wubu_rvc_find_tensor(model, key);
                if (!c1w || !c2w) { cudaFree(d_in); cudaFree(d_stage_in); goto fail; }
                int dil = 1 + 2 * cp;
                int pad1 = dil * (kk - 1) / 2;
                int pad2 = kk / 2;

                /* resblock weight_v tensors are de-normed in place — use data */
                int n1 = c1w->dims[0] * c1w->dims[1] * kk;
                int n2w = c2w->dims[0] * c2w->dims[1] * kk;
                float *d_c1 = NULL, *d_c2 = NULL;
                cudaMalloc(&d_c1, (size_t)n1 * sizeof(float));
                cudaMemcpy(d_c1, c1w->data, (size_t)n1 * sizeof(float), cudaMemcpyHostToDevice);
                cudaMalloc(&d_c2, (size_t)n2w * sizeof(float));
                cudaMemcpy(d_c2, c2w->data, (size_t)n2w * sizeof(float), cudaMemcpyHostToDevice);

                /* save entry grad = d_res */
                float *d_res = NULL;
                cudaMalloc(&d_res, n2 * sizeof(float));
                cudaMemcpy(d_res, d_in, n2 * sizeof(float), cudaMemcpyDeviceToDevice);

                /* a) conv2 bwd: din=d_a2, dw/db → registry */
                tfmt(key, "dec.resblocks.", rb, ".convs2.", cp, ".weight_v");
                int i2w = wubu_train_registry_find(reg, key);
                float *dw2 = (i2w >= 0) ? reg->params[i2w].grad : NULL;
                float *d_dw2 = NULL;
                if (dw2) { cudaMalloc(&d_dw2, (size_t)n2w * sizeof(float)); cudaMemset(d_dw2, 0, (size_t)n2w * sizeof(float)); }
                float *d_a2 = NULL;
                cudaMalloc(&d_a2, n2 * sizeof(float));
                cudaMemset(d_a2, 0, n2 * sizeof(float));
                tk_conv1d_bwd<<<dim3((unsigned)ch, (unsigned)ch), 256, 0, st>>>(
                    cache->mrf_a2[L][s][cp], d_c2, d_in, d_a2, d_dw2, NULL,
                    ch, ch, kk, 1, pad2, 1, out_n, out_n);
                if (d_dw2) { cudaMemcpy(dw2, d_dw2, (size_t)n2w * sizeof(float), cudaMemcpyDeviceToHost); cudaFree(d_dw2); }
                /* bias grad (host) — d_in is DEVICE memory: copy to host first */
                tfmt(key, "dec.resblocks.", rb, ".convs2.", cp, ".bias");
                int i2b = wubu_train_registry_find(reg, key);
                if (i2b >= 0 && reg->params[i2b].grad) {
                    float *h_in = (float *)malloc(n2 * sizeof(float));
                    if (h_in) {
                        cudaMemcpy(h_in, d_in, n2 * sizeof(float), cudaMemcpyDeviceToHost);
                        const float *do_ = h_in;
                        for (int oc = 0; oc < ch; oc++) {
                            float a = 0.0f;
                            for (int j = 0; j < out_n; j++) a += do_[(size_t)oc * out_n + j];
                            reg->params[i2b].grad[oc] = a;
                        }
                        free(h_in);
                    }
                }

                /* b) lrelu after conv1: pre-act = c1o */
                float *d_c1o = NULL;
                cudaMalloc(&d_c1o, n2 * sizeof(float));
                tk_lrelu_bwd<<<(unsigned)((n2 + 255) / 256), 256, 0, st>>>(
                    cache->mrf_c1[L][s][cp], d_a2, d_c1o, n2, 0.1f);

                /* c) conv1 bwd */
                tfmt(key, "dec.resblocks.", rb, ".convs1.", cp, ".weight_v");
                int i1w = wubu_train_registry_find(reg, key);
                float *dw1 = (i1w >= 0) ? reg->params[i1w].grad : NULL;
                float *d_dw1 = NULL;
                if (dw1) { cudaMalloc(&d_dw1, (size_t)n1 * sizeof(float)); cudaMemset(d_dw1, 0, (size_t)n1 * sizeof(float)); }
                float *d_a1 = NULL;
                cudaMalloc(&d_a1, n2 * sizeof(float));
                cudaMemset(d_a1, 0, n2 * sizeof(float));
                tk_conv1d_bwd<<<dim3((unsigned)ch, (unsigned)ch), 256, 0, st>>>(
                    cache->mrf_a1[L][s][cp], d_c1, d_c1o, d_a1, d_dw1, NULL,
                    ch, ch, kk, 1, pad1, dil, out_n, out_n);
                if (d_dw1) { cudaMemcpy(dw1, d_dw1, (size_t)n1 * sizeof(float), cudaMemcpyDeviceToHost); cudaFree(d_dw1); }
                tfmt(key, "dec.resblocks.", rb, ".convs1.", cp, ".bias");
                int i1b = wubu_train_registry_find(reg, key);
                if (i1b >= 0 && reg->params[i1b].grad) {
                    float *h_c1o = (float *)malloc(n2 * sizeof(float));
                    if (h_c1o) {
                        cudaMemcpy(h_c1o, d_c1o, n2 * sizeof(float), cudaMemcpyDeviceToHost);
                        const float *do_ = h_c1o;
                        for (int oc = 0; oc < ch; oc++) {
                            float a = 0.0f;
                            for (int j = 0; j < out_n; j++) a += do_[(size_t)oc * out_n + j];
                            reg->params[i1b].grad[oc] = a;
                        }
                        free(h_c1o);
                    }
                }

                /* d) lrelu before conv1: pre-act = p; residual add */
                float *d_p = NULL;
                cudaMalloc(&d_p, n2 * sizeof(float));
                tk_lrelu_bwd<<<(unsigned)((n2 + 255) / 256), 256, 0, st>>>(
                    cache->mrf_p[L][s][cp], d_a1, d_p, n2, 0.1f);
                k_add<<<(unsigned)((n2 + 255) / 256), 256, 0, st>>>(d_p, d_res, n2);
                cudaMemcpy(d_in, d_p, n2 * sizeof(float), cudaMemcpyDeviceToDevice);

                cudaFree(d_res); cudaFree(d_a2); cudaFree(d_c1o); cudaFree(d_a1); cudaFree(d_p);
                cudaFree(d_c1); cudaFree(d_c2);
            }
            /* accumulate stage input grad */
            k_add<<<(unsigned)((n2 + 255) / 256), 256, 0, st>>>(d_stage_in, d_in, n2);
            cudaFree(d_in);
        }
        /* save d_stage_out[L] for the next (upper) stage */
        cudaMalloc(&cache->d_stage_out[L], n2 * sizeof(float));
        cudaMemcpy(cache->d_stage_out[L], d_stage_in, n2 * sizeof(float), cudaMemcpyDeviceToDevice);

        /* upsample backward (convt1d) — din wrt act_in[L] */
        {
            char key[128];
            /* weight = hifi_upsample_denorm (de-normed, what the registry owns) */
            if (!model->hifi_upsample_denorm[L]) goto fail;
            int in_ch = cache->ch_in[L];
            int out_ch = cache->ch_out[L];
            int kk = cache->k[L];
            int in_n = (L == 0) ? cache->n_frames : cache->n_stage[L - 1];
            float *d_uw = NULL;
            int den_len = in_ch * out_ch * kk;
            cudaMalloc(&d_uw, (size_t)den_len * sizeof(float));
            cudaMemcpy(d_uw, model->hifi_upsample_denorm[L], (size_t)den_len * sizeof(float), cudaMemcpyHostToDevice);
            float *d_act = NULL;
            cudaMalloc(&d_act, (size_t)in_ch * in_n * sizeof(float));
            cudaMemset(d_act, 0, (size_t)in_ch * in_n * sizeof(float));
            /* grads for ups weight (registry name = dec.ups.%d.weight) */
            tfmt(key, "dec.ups.", L, "", -1, ".weight");
            int iuw = wubu_train_registry_find(reg, key);
            float *duw = (iuw >= 0) ? reg->params[iuw].grad : NULL;
            float *d_duw = NULL;
            if (duw) { cudaMalloc(&d_duw, (size_t)den_len * sizeof(float)); cudaMemset(d_duw, 0, (size_t)den_len * sizeof(float)); }
            tk_convt1d_bwd<<<dim3((unsigned)((in_n + 255) / 256), (unsigned)in_ch), 256, 0, st>>>(
                cache->act_in[L], d_uw, d_stage_in, d_act, d_duw, NULL,
                in_ch, out_ch, kk, cache->s[L], cache->p[L], in_n, out_n);
            if (d_duw) { cudaMemcpy(duw, d_duw, (size_t)den_len * sizeof(float), cudaMemcpyDeviceToHost); cudaFree(d_duw); }
            /* ups bias grad: db[oc] = sum_j d_stage_in[oc,j] — device sum then D2H */
            tfmt(key, "dec.ups.", L, "", -1, ".bias");
            int iub = wubu_train_registry_find(reg, key);
            if (iub >= 0 && reg->params[iub].grad) {
                float *d_db = NULL;
                cudaMalloc(&d_db, (size_t)out_ch * sizeof(float));
                cudaMemset(d_db, 0, (size_t)out_ch * sizeof(float));
                tk_convt1d_db<<<dim3((unsigned)out_ch, 1), 256, 0, st>>>(d_stage_in, d_db, out_ch, out_n);
                cudaMemcpy(reg->params[iub].grad, d_db, (size_t)out_ch * sizeof(float), cudaMemcpyDeviceToHost);
                cudaFree(d_db);
            }
            /* lrelu bwd: stage_out[L-1] pre-lrelu → act_in[L] */
            if (L >= 1) {
                float *d_stage_prev = NULL;
                cudaMalloc(&d_stage_prev, (size_t)out_ch * out_n * sizeof(float));
                tk_lrelu_bwd<<<(unsigned)((out_ch * out_n + 255) / 256), 256, 0, st>>>(
                    cache->stage_out[L - 1], d_act, d_stage_prev, (size_t)out_ch * out_n, 0.1f);
                cudaMalloc(&cache->d_stage_out[L], (size_t)out_ch * out_n * sizeof(float));
                cudaMemcpy(cache->d_stage_out[L], d_stage_prev, (size_t)out_ch * out_n * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaFree(d_stage_prev);
            } else {
                /* L==0: convT0 bwd → lrelu bwd (pre_out) → conv_pre bwd.
                 * (The convT0 backward ran above with the shared d_act —
                 * conv_pre consumes d_act0 = grad wrt act_in[0].) */
                float *d_act0 = d_act;
                (void)d_act0;

                /* conv_pre backward */
                const RVCTensor *conv_pre_w = wubu_rvc_find_tensor(model, "dec.conv_pre.weight");
                if (conv_pre_w && conv_pre_w->data) {
                    int init_ch = cache->ch_in[0];
                    float *d_pre = NULL;
                    cudaMalloc(&d_pre, (size_t)init_ch * cache->n_frames * sizeof(float));
                    float *d_zin = NULL;
                    cudaMalloc(&d_zin, (size_t)conv_pre_w->dims[1] * cache->n_frames * sizeof(float));
                    cudaMemset(d_zin, 0, (size_t)conv_pre_w->dims[1] * cache->n_frames * sizeof(float));
                    int pk = conv_pre_w->dims[2];
                    /* pre-lrelu: d(x_pre) = lrelu_bwd(pre_out, d_act0) */
                    tk_lrelu_bwd<<<(unsigned)((init_ch * cache->n_frames + 255) / 256), 256, 0, st>>>(
                        cache->pre_out, d_act0, d_pre, (size_t)init_ch * cache->n_frames, 0.1f);
                    /* conv_pre weight grads (registry name = dec.conv_pre.weight) */
                    tfmt(key, "dec.conv_pre.weight", -1, "", -1, "");
                    int ipw = wubu_train_registry_find(reg, key);
                    float *dpw = (ipw >= 0) ? reg->params[ipw].grad : NULL;
                    float *d_dpw = NULL;
                    int pre_n = conv_pre_w->dims[0] * conv_pre_w->dims[1] * pk;
                    if (dpw) { cudaMalloc(&d_dpw, (size_t)pre_n * sizeof(float)); cudaMemset(d_dpw, 0, (size_t)pre_n * sizeof(float)); }
                    float *d_mel = NULL;
                    cudaMalloc(&d_mel, (size_t)conv_pre_w->dims[1] * cache->n_frames * sizeof(float));
                    cudaMemcpy(d_mel, mel_in, (size_t)conv_pre_w->dims[1] * cache->n_frames * sizeof(float), cudaMemcpyHostToDevice);
                    float *d_pw2 = NULL;
                    cudaMalloc(&d_pw2, (size_t)pre_n * sizeof(float));
                    cudaMemcpy(d_pw2, conv_pre_w->data, (size_t)pre_n * sizeof(float), cudaMemcpyHostToDevice);
                    tk_conv1d_bwd<<<dim3((unsigned)init_ch, (unsigned)conv_pre_w->dims[1]), 256, 0, st>>>(
                        d_mel, d_pw2, d_pre, d_zin, d_dpw, NULL,
                        conv_pre_w->dims[1], init_ch, pk, 1, pk / 2, 1, cache->n_frames, cache->n_frames);
                    if (d_dpw) { cudaMemcpy(dpw, d_dpw, (size_t)pre_n * sizeof(float), cudaMemcpyDeviceToHost); cudaFree(d_dpw); }
                    /* DEBUG: verify conv_pre grads landed */
                    {
                        int ipwchk = wubu_train_registry_find(reg, "dec.conv_pre.weight");
                        if (ipwchk >= 0 && reg->params[ipwchk].grad) {
                            float s0 = 0.0f; int nz = 0;
                            for (int z = 0; z < reg->params[ipwchk].n; z++) {
                                s0 += fabsf(reg->params[ipwchk].grad[z]);
                                if (fabsf(reg->params[ipwchk].grad[z]) > 1e-9f) nz++;
                            }

                        }
                    }
                    tfmt(key, "dec.conv_pre.bias", -1, "", -1, "");
                    int ipb = wubu_train_registry_find(reg, key);
                    if (ipb >= 0 && reg->params[ipb].grad) {
                        float *h_pre = (float *)malloc((size_t)init_ch * cache->n_frames * sizeof(float));
                        if (h_pre) {
                            cudaMemcpy(h_pre, d_pre, (size_t)init_ch * cache->n_frames * sizeof(float), cudaMemcpyDeviceToHost);
                            const float *do_ = h_pre;
                            for (int oc = 0; oc < init_ch; oc++) {
                                float a = 0.0f;
                                for (int j = 0; j < cache->n_frames; j++) a += do_[(size_t)oc * cache->n_frames + j];
                                reg->params[ipb].grad[oc] = a;
                            }
                            free(h_pre);
                        }
                    }
                    cudaFree(d_pre); cudaFree(d_zin); cudaFree(d_mel); cudaFree(d_pw2);
                }
            }
            cudaFree(d_act);
            cudaFree(d_uw);
            cudaFree(d_stage_in);
        }
    }

    cudaFree(d_tanh_in);
    cudaFree(d_daudio);
    cudaFree(d_post_out);
    cudaFree(d_stage3);
    return 0;

fail:
    cudaFree(d_tanh_in);
    cudaFree(d_daudio);
    cudaFree(d_post_out);
    cudaFree(d_stage3);
    return -1;
}

/* ── full training step ─────────────────────────────────────────────── */

int wubu_train_step_cuda(WuBuRVCModel *model, WuBuTrainRegistry *reg, WuBuAdamW *opt,
                         const float *mel_in, int n_frames,
                         const float *wav, int n_samples,
                         float *loss_out, int epoch) {
    if (!model || !reg || !opt || !mel_in || !wav) return -1;

    int max_samples = n_frames * 400;
    float *audio = (float *)malloc((size_t)max_samples * sizeof(float));
    if (!audio) return -1;
    TrainCacheCuda *cache = wubu_train_cache_alloc_cuda();
    if (!cache) { free(audio); return -1; }
    int n_out = wubu_train_forward_cuda(model, mel_in, n_frames, audio, max_samples, cache);
    if (n_out <= 0) { free(audio); wubu_train_cache_free_cuda(cache); return -1; }

    int n = n_out < n_samples ? n_out : n_samples;
    /* MSE loss + grad (host) */
    float mse = 0.0f;
    for (int i = 0; i < n; i++) { float e = audio[i] - wav[i]; mse += e * e; }
    mse /= (float)n;

    /* Spectral supervision (Qwen3-TTS/HiFi-GAN recipe) — same as CPU/VK:
     * multi-scale STFT linear+log magnitude loss with gradient flow.
     * WUBU_SPECTRAL_WEIGHT tunes λ (default 1.0); 0 = pure MSE. */
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
    if (!d_audio) { free(d_spec); free(audio); wubu_train_cache_free_cuda(cache); return -1; }
    for (int i = 0; i < n; i++) {
        d_audio[i] = 2.0f * (audio[i] - wav[i]) / (float)n;
        if (d_spec) d_audio[i] += d_spec[i];
    }
    for (int i = n; i < n_out; i++) d_audio[i] = 0.0f;

    wubu_train_registry_zero_grads(reg);
    if (wubu_train_backward_cuda(model, cache, mel_in, d_audio, reg) != 0) {
        free(d_audio); free(audio); wubu_train_cache_free_cuda(cache); return -1;
    }

    /* AdamW update (host — params live on host) */
    for (int i = 0; i < reg->count; i++)
        wubu_adamw_step(opt, i, reg->params[i].data, reg->params[i].grad);

    free(d_spec);
    free(d_audio);
    free(audio);
    wubu_train_cache_free_cuda(cache);
    return (total_loss < 0.1f) ? 1 : 0;
}
