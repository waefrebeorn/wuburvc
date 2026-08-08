/* wubu_rvc_cuda.cu — CUDA GeneratorNSF inference kernels (sm_75+).
 *
 * The CPU synth is bounded by the generator convs over long audio
 * (~1.8x slower than realtime). The GPU does the same exact math with the
 * same [in_ch, n] col-major layout — parity matches the CPU float32 path to
 * float rounding.
 *
 * The flow (enc_p + posterior + prior + coupling) stays on the CPU (small
 * per chunk); the generator (conv_pre, NSF sine, upsample blocks with
 * noise convs + MRF resblocks, conv_post) runs here. The structure mirrors
 * wubu_generator_nsf in wubu_rvc_real.c EXACTLY (weight keys, pads,
 * strides, activations, averaging).
 *
 * License: WaefreBeorn-UMV3
 */
#ifdef __cplusplus
extern "C" {
#endif
#include "wubu_rvc_real.h"
#ifdef __cplusplus
}
#endif
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── kernels ─────────────────────────────────────────────────────────── */

__global__ void k_conv1d(const float *in, int in_ch, int n,
                         const float *w, const float *b,
                         int out_ch, int k, int stride, int pad, int dil,
                         int n_out, float *out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= n_out || oc >= out_ch) return;
    float acc = b ? b[oc] : 0.0f;
    const float *wv = w + (size_t)oc * in_ch * k;
    for (int ic = 0; ic < in_ch; ic++) {
        const float *irow = in + (size_t)ic * n;
        for (int tap = 0; tap < k; tap++) {
            int src = j * stride + tap * dil - pad;
            if (src >= 0 && src < n) acc += irow[src] * wv[(size_t)ic * k + tap];
        }
    }
    out[(size_t)oc * n_out + j] = acc;
}

__global__ void k_convt1d(const float *in, int in_ch, int n,
                          const float *w, const float *b,
                          int out_ch, int k, int stride, int pad,
                          int n_out, float *out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= n_out || oc >= out_ch) return;
    float acc = b ? b[oc] : 0.0f;
    /* accumulate in the SAME order as the CPU scatter (i increasing, one tap
     * per i): tap = j + pad - i*stride must be in [0, k). */
    int i_lo = (j + pad - (k - 1) + stride - 1) / stride;
    if (i_lo < 0) i_lo = 0;
    int i_hi = (j + pad) / stride;
    if (i_hi >= n) i_hi = n - 1;
    for (int ic = 0; ic < in_ch; ic++) {
        const float *irow = in + (size_t)ic * n;
        const float *wk = w + ((size_t)ic * out_ch + (size_t)oc) * k;
        for (int i = i_lo; i <= i_hi; i++) {
            int tap = j + pad - i * stride;
            if (tap >= 0 && tap < k) acc += irow[i] * wk[tap];
        }
    }
    out[(size_t)oc * n_out + j] = acc;
}

__global__ void k_lrelu(float *x, size_t n, float slope) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = v > 0.0f ? v : v * slope;
}

__global__ void k_snake(float *x, size_t n, float a) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = v + (1.0f / fmaxf(a * 2.0f, 1e-6f)) * sinf(a * v) * sinf(a * v);
}

__global__ void k_tanh(float *x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = tanhf(x[i]);
}

__global__ void k_add(float *a, const float *b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] += b[i];
}

__global__ void k_mul(float *x, size_t n, float s) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= s;
}

__global__ void k_copy(float *dst, const float *src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i];
}

/* add a per-channel offset: x[oc*n + j] += off[oc] */
__global__ void k_bias(float *x, const float *off, int n, int n_ch) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * n_ch) return;
    int oc = (int)(i / (size_t)n);
    x[i] += off[oc];
}

/* ── host-side launchers ─────────────────────────────────────────────── */

static void launch_act(float *x, size_t n, int snake, cudaStream_t st) {
    size_t th = 256;
    unsigned grid = (unsigned)((n + th - 1) / th);
    if (snake) {
        float a = 1.0f;
        void *args[] = {&x, &n, &a};
        cudaLaunchKernel((void *)k_snake, dim3(grid), dim3((unsigned)th), args, 0, st);
    } else {
        float slope = 0.1f;
        void *args[] = {&x, &n, &slope};
        cudaLaunchKernel((void *)k_lrelu, dim3(grid), dim3((unsigned)th), args, 0, st);
    }
}

static void launch_conv1d(const float *in, int in_ch, int n,
                          const float *w, const float *b,
                          int out_ch, int k, int stride, int pad, int dil,
                          float *out, int n_out, cudaStream_t st) {
    dim3 block(128, 4);
    dim3 grid((n_out + 127) / 128, (out_ch + 3) / 4);
    void *args[] = {(void *)&in, &in_ch, &n, &w, &b, &out_ch, &k, &stride,
                    &pad, &dil, &n_out, &out};
    cudaLaunchKernel((void *)k_conv1d, grid, block, args, 0, st);
    if (cudaGetLastError() != cudaSuccess) { /* tagged by caller */ }
}

static void launch_convt1d(const float *in, int in_ch, int n,
                           const float *w, const float *b,
                           int out_ch, int k, int stride, int pad,
                           float *out, int n_out, cudaStream_t st) {
    dim3 block(128, 4);
    dim3 grid((n_out + 127) / 128, (out_ch + 3) / 4);
    void *args[] = {(void *)&in, &in_ch, &n, &w, &b, &out_ch, &k, &stride,
                    &pad, &n_out, &out};
    cudaLaunchKernel((void *)k_convt1d, grid, block, args, 0, st);
}

static void launch_unary(float *x, size_t n, const void *fn, cudaStream_t st) {
    size_t th = 256;
    unsigned grid = (unsigned)((n + th - 1) / th);
    void *args[] = {&x, &n};
    cudaLaunchKernel(fn, dim3(grid), dim3((unsigned)th), args, 0, st);
}

static int gpu_ok(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        const char *err = cudaGetErrorString(e);
        char buf[512];
        size_t n = 0;
        const char *p = "[cuda] ";
        while (*p && n < sizeof(buf) - 1) buf[n++] = *p++;
        p = what; while (*p && n < sizeof(buf) - 1) buf[n++] = *p++;
        p = ": "; while (*p && n < sizeof(buf) - 1) buf[n++] = *p++;
        p = err; while (*p && n < sizeof(buf) - 1) buf[n++] = *p++;
        buf[n] = 0;
        DWORD w;
        WriteFile(GetStdHandle(STD_ERROR_HANDLE), buf, (DWORD)n, &w, NULL);
        return -1;
    }
    return 0;
}

/* check the async kernel error state (call after a kernel launch) */
static int gpu_ck(const char *tag) {
    return gpu_ok(cudaGetLastError(), tag);
}


/* BYTE count from the actual dims (biases are 1D - dims[1..] are garbage) */
static size_t tnum(const RVCTensor *t) {
    size_t n = 1;
    for (int i = 0; i < t->n_dims && i < 8; i++) n *= (size_t)t->dims[i];
    return n * sizeof(float);
}
/* manual key builder (no snprintf — avoids MSVC CRT deps when linking
 * the nvcc object with MinGW). Builds e.g. "dec.ups.3.bias" or
 * "dec.resblocks.9.convs1.2.weight_v". */
static void kfmt(char *o, const char *pre, int a,
                 const char *mid, int b, const char *suf) {
    char *p = o;
    const char *q;
    for (q = pre; *q; q++) *p++ = *q;
    if (a == 0) *p++ = '0';
    else { char t[8]; int n = 0; while (a) { t[n++] = (char)('0' + a % 10); a /= 10; } while (n) *p++ = t[--n]; }
    for (q = mid; *q; q++) *p++ = *q;
    if (b >= 0) {
        if (b == 0) *p++ = '0';
        else { char t[8]; int n = 0; while (b) { t[n++] = (char)('0' + b % 10); b /= 10; } while (n) *p++ = t[--n]; }
    }
    for (q = suf; *q; q++) *p++ = *q;
    *p = 0;
}

/* ── the CUDA GeneratorNSF driver ────────────────────────────────────── */
/* Mirrors wubu_generator_nsf (CPU) exactly; z is [inter, nF] col-major. */
int wubu_generator_nsf_cuda(WuBuRVCModel *model,
                            const float *z, int nF, int inter_channels,
                            const float *nsff0, const float *g,
                            float *out_audio, int max_samples,
                            int inject_noise, int use_snake) {
    (void)g;
    if (!model || !z || !out_audio) return -1;
    if (gpu_ok(cudaSetDevice(0), "set device")) return -1;
    cudaFree(0);
    cudaStream_t st;
    if (gpu_ok(cudaStreamCreate(&st), "stream")) return -1;

    int n_ups = model->n_upsample_layers > 0 ? model->n_upsample_layers : 4;
    const RVCTensor *pre_w0 = wubu_rvc_find_tensor(model, "dec.conv_pre.weight");
    int init_ch = pre_w0 && pre_w0->dims[0] > 0 ? pre_w0->dims[0] : 512;
    int ups_in[8], ups_out[8], ups_rate[8], ups_k[8], ups_pad[8];
    int ups_total = 1;
    for (int L = 0; L < n_ups; L++) {
        ups_in[L] = init_ch / (1 << L);
        ups_out[L] = init_ch / (1 << (L + 1));
        ups_rate[L] = model->upsample_rates[L] > 0 ? model->upsample_rates[L] : 2;
        /* kernel/pad come from the weight shape (like the CPU), NOT the
         * config field (which can be stale) */
        char kb[64];
        kfmt(kb, "dec.ups.", L, "", -1, ".weight_v");
        const RVCTensor *ut = wubu_rvc_find_tensor(model, kb);
        if (!ut) { kfmt(kb, "dec.ups.", L, "", -1, ".weight"); ut = wubu_rvc_find_tensor(model, kb); }
        ups_k[L] = ut && ut->n_dims >= 3 ? ut->dims[2] : 4;
        ups_pad[L] = (ups_k[L] - ups_rate[L]) / 2;
        if (ups_pad[L] < 0) ups_pad[L] = 0;
        ups_total *= ups_rate[L];
    }

    /* conv_pre: z [inter, T] -> x [ups_in[0], T] */
    const RVCTensor *pre_w = wubu_rvc_find_tensor(model, "dec.conv_pre.weight");
    const RVCTensor *pre_b = wubu_rvc_find_tensor(model, "dec.conv_pre.bias");
    if (!pre_w || !pre_w->data) return -1;
    int cur_ch = ups_in[0], cur_n = nF;

    float *d_z = NULL, *d_x = NULL, *d_sine = NULL, *d_cur = NULL;
    float *d_tmp = NULL, *d_tmp2 = NULL, *d_out = NULL;
    if (gpu_ok(cudaMalloc(&d_z, (size_t)inter_channels * nF * sizeof(float)), "z") ||
        gpu_ok(cudaMalloc(&d_x, (size_t)cur_ch * cur_n * sizeof(float)), "x") ||
        gpu_ok(cudaMalloc(&d_sine, (size_t)(nF * ups_total + 8) * sizeof(float)), "sine") ||
        gpu_ok(cudaMalloc(&d_cur, (size_t)cur_ch * cur_n * sizeof(float)), "cur") ||
        gpu_ok(cudaMalloc(&d_tmp, (size_t)cur_ch * cur_n * sizeof(float)), "tmp") ||
        gpu_ok(cudaMalloc(&d_tmp2, (size_t)cur_ch * cur_n * sizeof(float)), "tmp2") ||
        gpu_ok(cudaMalloc(&d_out, (size_t)(nF * ups_total + 8) * sizeof(float)), "out"))
        goto fail;
    if (gpu_ok(cudaMemcpy(d_z, z, (size_t)inter_channels * nF * sizeof(float),
                          cudaMemcpyHostToDevice), "z copy")) goto fail;

    {
        size_t pre_sz = (size_t)pre_w->dims[0] * pre_w->dims[1] * pre_w->dims[2];
        float *d_prew = NULL;
        cudaMalloc(&d_prew, pre_sz * sizeof(float));
        cudaMemcpy(d_prew, pre_w->data, pre_sz * sizeof(float), cudaMemcpyHostToDevice);
        float *d_preb = NULL;
        if (pre_b && pre_b->data) {
            cudaMalloc(&d_preb, (size_t)pre_b->dims[0] * sizeof(float));
            cudaMemcpy(d_preb, pre_b->data, (size_t)pre_b->dims[0] * sizeof(float), cudaMemcpyHostToDevice);
        }
        int pk = pre_w->dims[2];
        launch_conv1d(d_z, inter_channels, nF, d_prew, d_preb,
                      cur_ch, pk, 1, pk / 2, 1, d_x, cur_n, st);
        if (gpu_ck("conv_pre")) goto fail;
        cudaFree(d_prew); cudaFree(d_preb);
    }
    cudaFree(d_z); d_z = NULL;

    /* x = x + cond(g): Conv1d(gin -> init_ch, k=1) with bias - the speaker
     * conditioning the CPU applies right after conv_pre. */
    {
        const RVCTensor *cond_w = wubu_rvc_find_tensor(model, "dec.cond.weight");
        const RVCTensor *cond_b = wubu_rvc_find_tensor(model, "dec.cond.bias");
        if (cond_w && cond_w->data) {
            int gin = cond_w->n_dims >= 2 ? cond_w->dims[1] : 40;
            float *off = (float *)malloc((size_t)cur_ch * sizeof(float));
            float *d_off = NULL;
            for (int oc = 0; oc < cur_ch; oc++) {
                float acc = cond_b && cond_b->data ? cond_b->data[oc] : 0.0f;
                const float *wv = cond_w->data + (size_t)oc * gin;
                for (int ic = 0; ic < gin && ic < 256; ic++) acc += g[ic] * wv[ic];
                off[oc] = acc;
            }
            cudaMalloc(&d_off, (size_t)cur_ch * sizeof(float));
            cudaMemcpy(d_off, off, (size_t)cur_ch * sizeof(float), cudaMemcpyHostToDevice);
            size_t nb = (size_t)cur_ch * cur_n;
            size_t thb = 256;
            void *argsb[] = {&d_x, &d_off, &cur_n, &cur_ch};
            cudaLaunchKernel((void *)k_bias, dim3((unsigned)((nb + thb - 1) / thb)), dim3((unsigned)thb), argsb, 0, st);
            if (gpu_ck("cond")) goto fail;
            cudaFree(d_off);
            free(off);
        }
    }

    /* sine excitation - host (matches the CPU EXACTLY: rad=f0/sr per frame,
     * rad_acc cumulative carry, phase = rad[fi]*(u+1) + rad_acc[fi-1]) */
    {
        int n_sine = nF * ups_total;
        float sr = model->sample_rate > 0 ? (float)model->sample_rate : 40000.0f;
        float *sine = (float *)malloc((size_t)n_sine * sizeof(float));
        float *rad = (float *)malloc((size_t)nF * sizeof(float));
        float *rad_acc = (float *)malloc((size_t)nF * sizeof(float));
        const RVCTensor *linw = wubu_rvc_find_tensor(model, "dec.sine_linear.weight");
        const RVCTensor *linb = wubu_rvc_find_tensor(model, "dec.sine_linear.bias");
        float linw_v = (linw && linw->data) ? linw->data[0] : 1.0f;
        float linb_v = (linb && linb->data) ? linb->data[0] : 0.0f;
        if (sine && rad && rad_acc) {
            float accum = 0.0f;
            for (int t = 0; t < nF; t++) {
                float f0 = nsff0[t] > 0 ? nsff0[t] : 0.0f;
                rad[t] = f0 / sr;
                float c = rad[t] * (float)ups_total;
                float r2 = fmodf(c + 0.5f, 1.0f) - 0.5f;
                accum += r2;
                rad_acc[t] = fmodf(accum, 1.0f);
            }
            for (int j = 0; j < n_sine; j++) {
                int fi = j / ups_total;
                if (fi >= nF) fi = nF - 1;
                int u = j % ups_total;
                float phase = rad[fi] * (float)(u + 1);
                if (fi > 0) phase += rad_acc[fi - 1];
                float s = sinf(2.0f * (float)M_PI * phase) * 0.1f;
                float uv = (nsff0[fi] > 0) ? 1.0f : 0.0f;
                float noise_amp = inject_noise ? (1.0f - uv) * 0.1f / 3.0f : 0.0f;
                float noise = noise_amp * (2.0f * ((float)rand() / RAND_MAX) - 1.0f);
                float sw = s * uv + noise;
                sine[j] = tanhf(linw_v * sw + linb_v);
            }
        }
        cudaMemcpy(d_sine, sine, (size_t)n_sine * sizeof(float), cudaMemcpyHostToDevice);
        free(sine); free(rad); free(rad_acc);
    }

    /* upsample blocks */
    cudaMemcpy(d_cur, d_x, (size_t)cur_ch * cur_n * sizeof(float), cudaMemcpyDeviceToDevice);
    for (int L = 0; L < n_ups; L++) {
        int in_ch = cur_ch, in_n = cur_n;
        int out_ch = ups_out[L], out_n = (in_n - 1) * ups_rate[L] - 2 * ups_pad[L] + ups_k[L];
        if (out_n <= 0) goto fail;

        launch_act(d_cur, (size_t)in_ch * in_n, use_snake, st);

        /* convT1d: dec.ups.L (denorm precomputed) */
        float *den = model->hifi_upsample_denorm[L];
        int den_len = model->hifi_upsample_denorm_len[L];
        if (!den || den_len <= 0) goto fail;
        char key[256];
        kfmt(key, "dec.ups.", L, "", -1, ".bias");
        const RVCTensor *ub = wubu_rvc_find_tensor(model, key);
        float *d_uw = NULL, *d_ub = NULL;
        cudaMalloc(&d_uw, (size_t)den_len * sizeof(float));
        cudaMemcpy(d_uw, den, (size_t)den_len * sizeof(float), cudaMemcpyHostToDevice);
        if (ub && ub->data) {
            cudaMalloc(&d_ub, tnum(ub));
            cudaMemcpy(d_ub, ub->data, tnum(ub), cudaMemcpyHostToDevice);
        }
        if (gpu_ok(cudaMalloc(&d_tmp, (size_t)out_ch * out_n * sizeof(float)), "stage")) goto fail;
        launch_convt1d(d_cur, in_ch, in_n, d_uw, d_ub, out_ch, ups_k[L], ups_rate[L], ups_pad[L],
                       d_tmp, out_n, st);
        if (gpu_ck("ups")) goto fail;


        cudaFree(d_uw); cudaFree(d_ub);

        /* noise conv: conv1d(sine [1, n_sine] -> [out_ch, out_n], stride = product of remaining) */
        {
            int stride = 1;
            for (int j = L + 1; j < n_ups; j++) stride *= ups_rate[j];
            kfmt(key, "dec.noise_convs.", L, "", -1, ".weight");
            const RVCTensor *ncw = wubu_rvc_find_tensor(model, key);
            kfmt(key, "dec.noise_convs.", L, "", -1, ".bias");
            const RVCTensor *ncb = wubu_rvc_find_tensor(model, key);
            int kk = ncw && ncw->n_dims >= 3 ? ncw->dims[2] : ((L == n_ups - 1) ? 1 : stride * 2);
            int pad = (kk - stride) / 2;
            if (pad < 0) pad = 0;
            if (ncw && ncw->data) {
                int s_in = nF * ups_total;
                float *d_ncw = NULL, *d_ncb = NULL, *d_nc = NULL;
                cudaMalloc(&d_ncw, tnum(ncw));
                cudaMemcpy(d_ncw, ncw->data, tnum(ncw), cudaMemcpyHostToDevice);
                if (ncb && ncb->data) {
                    cudaMalloc(&d_ncb, tnum(ncb));
                    cudaMemcpy(d_ncb, ncb->data, tnum(ncb), cudaMemcpyHostToDevice);
                }
                cudaMalloc(&d_nc, (size_t)out_ch * out_n * sizeof(float));
                launch_conv1d(d_sine, 1, s_in, d_ncw, d_ncb, out_ch, kk, stride, pad, 1,
                              d_nc, out_n, st);
                if (gpu_ck("noise")) goto fail;
                /* stage += noise */
                size_t n2 = (size_t)out_ch * out_n;
                size_t th = 256;
                void *args[] = {&d_tmp, &d_nc, &n2};
                cudaLaunchKernel((void *)k_add, dim3((unsigned)((n2 + th - 1) / th)), dim3((unsigned)th), args, 0, st);
        cudaStreamSynchronize(st);
        if (gpu_ck("noise-add-sync")) goto fail;
                cudaFree(d_ncw); cudaFree(d_ncb); cudaFree(d_nc);
            }
        }

        /* MRF: n_stacks resblock stacks (n_pairs conv pairs each) -> avg -> stage */
        {
            int ch = out_ch;
            int n_stacks = model->n_mrf_stacks;
            if (n_stacks < 1) n_stacks = 3;
            if (n_stacks > 8) n_stacks = 8;
            int n_pairs = model->n_resblock_pairs;
            if (n_pairs < 1) n_pairs = 3;
            if (n_pairs > 8) n_pairs = 8;
            float *d_acc = NULL;
            float *d_stage = NULL;
            cudaMalloc(&d_acc, (size_t)n_stacks * ch * out_n * sizeof(float));
            cudaMemset(d_acc, 0, (size_t)n_stacks * ch * out_n * sizeof(float));
            cudaMalloc(&d_tmp2, (size_t)ch * out_n * sizeof(float));
            cudaMalloc(&d_cur, (size_t)ch * out_n * sizeof(float));
            /* preserve the stage: the carry copies below overwrite d_tmp */
            cudaMalloc(&d_stage, (size_t)ch * out_n * sizeof(float));
            {
                size_t n0 = (size_t)ch * out_n;
                size_t th0 = 256;
                void *args0[] = {&d_stage, &d_tmp, &n0};
                cudaLaunchKernel((void *)k_copy, dim3((unsigned)((n0 + th0 - 1) / th0)), dim3((unsigned)th0), args0, 0, st);
            }
            for (int s = 0; s < n_stacks; s++) {
                int rb = L * n_stacks + s;
                int k = model->resblock_k[s];
                if (k <= 0) k = (s == 0) ? 3 : (s == 1 ? 7 : 11);
                /* rb_in = stage (preserved in d_stage); ALSO reset the residual
                 * carry to the stage — the carry holds the PREVIOUS stack's
                 * output and must not leak into this stack's first pair. */
                size_t n2 = (size_t)ch * out_n;
                size_t th = 256;
                void *args_c[] = {&d_tmp2, &d_stage, &n2};
                cudaLaunchKernel((void *)k_copy, dim3((unsigned)((n2 + th - 1) / th)), dim3((unsigned)th), args_c, 0, st);
                void *args_cr[] = {&d_tmp, &d_stage, &n2};
                cudaLaunchKernel((void *)k_copy, dim3((unsigned)((n2 + th - 1) / th)), dim3((unsigned)th), args_cr, 0, st);
        if (gpu_ck("copy")) goto fail;
                for (int cp = 0; cp < n_pairs; cp++) {
                    char key[256];
                    kfmt(key, "dec.resblocks.", rb, ".convs1.", cp, ".weight_v");
                    const RVCTensor *r1v = wubu_rvc_find_tensor(model, key);
                    kfmt(key, "dec.resblocks.", rb, ".convs1.", cp, ".bias");
                    const RVCTensor *r1b = wubu_rvc_find_tensor(model, key);
                    kfmt(key, "dec.resblocks.", rb, ".convs2.", cp, ".weight_v");
                    const RVCTensor *r2v = wubu_rvc_find_tensor(model, key);
                    kfmt(key, "dec.resblocks.", rb, ".convs2.", cp, ".bias");
                    const RVCTensor *r2b = wubu_rvc_find_tensor(model, key);
                    if (!r1v || !r2v || !r1v->data || !r2v->data) continue;
                    int dil = model->resblock_dil[s][cp];
                    if (dil <= 0) dil = 1 + 2 * cp;
                    int pad1 = dil * (k - 1) / 2;
                    int pad2 = k / 2;
                    float *d_c1 = NULL, *d_c2 = NULL, *d_c1b = NULL, *d_c2b = NULL;
                    cudaMalloc(&d_c1, tnum(r1v));
                    cudaMalloc(&d_c2, tnum(r2v));
                    cudaMemcpy(d_c1, r1v->data, tnum(r1v), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_c2, r2v->data, tnum(r2v), cudaMemcpyHostToDevice);
                    if (r1b && r1b->data) {
                        cudaMalloc(&d_c1b, tnum(r1b));
                        cudaMemcpy(d_c1b, r1b->data, tnum(r1b), cudaMemcpyHostToDevice);
                    }
                    if (r2b && r2b->data) {
                        cudaMalloc(&d_c2b, tnum(r2b));
                        cudaMemcpy(d_c2b, r2b->data, tnum(r2b), cudaMemcpyHostToDevice);
                    }
                    /* x = act(rb_in); conv1; act; conv2; + rb_in */
                    launch_act(d_tmp2, n2, use_snake, st);
                    launch_conv1d(d_tmp2, ch, out_n, d_c1, d_c1b, ch, k, 1, pad1, dil,
                                  d_cur, out_n, st);
                    if (gpu_ck("mrf1")) goto fail;
                    launch_act(d_cur, n2, use_snake, st);
                    launch_conv1d(d_cur, ch, out_n, d_c2, d_c2b, ch, k, 1, pad2, 1,
                                  d_tmp2, out_n, st);
                    if (gpu_ck("mrf2")) goto fail;
                    void *args_a[] = {&d_tmp2, &d_tmp, &n2};
                    cudaLaunchKernel((void *)k_add, dim3((unsigned)((n2 + th - 1) / th)), dim3((unsigned)th), args_a, 0, st);
                    cudaStreamSynchronize(st);
                    if (gpu_ck("mrf-skip-add")) goto fail;
                    /* carry: d_tmp = the pair output = the NEXT pair's residual base
                     * (the CPU keeps rb_in updated; the residual adds the PREVIOUS
                     * pair's output, NOT the stage - except pair 0, which uses the
                     * stage that d_tmp still holds). */
                    void *args_c2[] = {&d_tmp, &d_tmp2, &n2};
                    cudaLaunchKernel((void *)k_copy, dim3((unsigned)((n2 + th - 1) / th)), dim3((unsigned)th), args_c2, 0, st);
                    if (gpu_ck("mrf-carry")) goto fail;
        if (gpu_ck("add")) goto fail;
                    cudaFree(d_c1); cudaFree(d_c2); cudaFree(d_c1b); cudaFree(d_c2b);
                }
                /* acc[s] += rb_in (d_tmp2) */
                float *d_acc_s = d_acc + (size_t)s * ch * out_n;
                size_t n3 = (size_t)ch * out_n;
                size_t th3 = 256;
                void *args_s[] = {&d_acc_s, &d_tmp2, &n3};
                cudaLaunchKernel((void *)k_add, dim3((unsigned)((n3 + th3 - 1) / th3)), dim3((unsigned)th3), args_s, 0, st);
                cudaStreamSynchronize(st);
                if (gpu_ck("mrf-acc-add")) goto fail;
        if (gpu_ck("add")) goto fail;
            }
            /* stage = avg over stacks */
            size_t n2 = (size_t)ch * out_n;
            size_t th = 256;
            /* start from acc[0] */
            void *args_c0[] = {&d_tmp, &d_acc, &n2};
            cudaLaunchKernel((void *)k_copy, dim3((unsigned)((n2 + th - 1) / th)), dim3((unsigned)th), args_c0, 0, st);
        if (gpu_ck("add")) goto fail;
        if (gpu_ck("copy")) goto fail;
            for (int s = 1; s < n_stacks; s++) {
                const float *d_acc_s = d_acc + (size_t)s * ch * out_n;
                void *args_a[] = {&d_tmp, &d_acc_s, &n2};
                cudaLaunchKernel((void *)k_add, dim3((unsigned)((n2 + th - 1) / th)), dim3((unsigned)th), args_a, 0, st);
        if (gpu_ck("mul")) goto fail;
            }
            float inv = 1.0f / (float)n_stacks;
            void *args_m[] = {&d_tmp, &n2, &inv};
            cudaLaunchKernel((void *)k_mul, dim3((unsigned)((n2 + th - 1) / th)), dim3((unsigned)th), args_m, 0, st);
        if (gpu_ck("copy")) goto fail;
            cudaFree(d_acc); cudaFree(d_stage); cudaFree(d_tmp2); cudaFree(d_cur);
            /* d_cur = stage (d_tmp) */
            cudaMalloc(&d_cur, (size_t)ch * out_n * sizeof(float));
            void *args_c[] = {&d_cur, &d_tmp, &n2};
            cudaLaunchKernel((void *)k_copy, dim3((unsigned)((n2 + th - 1) / th)), dim3((unsigned)th), args_c, 0, st);
            cudaFree(d_tmp);
            d_tmp = NULL;
            cur_ch = ch; cur_n = out_n;
        }
    }

    /* final: act, conv_post(1 -> post_in, k, pad=k/2, no bias), tanh */
    {
        const RVCTensor *post_w = wubu_rvc_find_tensor(model, "dec.conv_post.weight");
        if (!post_w || !post_w->data) goto fail;
        int post_in = post_w->n_dims >= 2 ? post_w->dims[1] : 32;
        if (post_in < 1) post_in = 32;
        int post_k = post_w->n_dims >= 3 ? post_w->dims[2] : 7;
        int post_pad = post_k / 2;
        int out_n = cur_n;
        if (out_n > max_samples) out_n = max_samples;
        launch_act(d_cur, (size_t)post_in * cur_n, use_snake, st);
        float *d_pw = NULL;
        cudaMalloc(&d_pw, tnum(post_w));
        cudaMemcpy(d_pw, post_w->data, tnum(post_w), cudaMemcpyHostToDevice);
        launch_conv1d(d_cur, post_in, cur_n, d_pw, NULL, 1, post_k, 1, post_pad, 1,
                      d_out, out_n, st);
        if (gpu_ck("post")) goto fail;
        cudaFree(d_pw);
        launch_unary(d_out, (size_t)out_n, (void *)k_tanh, st);
        cudaStreamSynchronize(st);
        if (gpu_ck("tanh")) goto fail;
        cudaMemcpy(out_audio, d_out, (size_t)out_n * sizeof(float), cudaMemcpyDeviceToHost);
        if (gpu_ck("memcpy-out")) goto fail;
        cudaFree(d_cur); cudaFree(d_tmp); cudaFree(d_tmp2); cudaFree(d_x);
        cudaFree(d_sine); cudaFree(d_out);
        cudaStreamDestroy(st);
        return out_n;
    }

fail:
    cudaFree(d_z); cudaFree(d_x); cudaFree(d_sine); cudaFree(d_cur);
    cudaFree(d_tmp); cudaFree(d_tmp2); cudaFree(d_out);
    cudaStreamDestroy(st);
    return -1;
}
