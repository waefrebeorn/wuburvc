/*
 * wubu_rvc_kernels_exact.c — Exact HiFi-GAN transposed convolution kernel
 *
 * Matches PyTorch ConvTranspose1d + MRF residual blocks + conv_post.
 * Reads de-normalized weight_norm tensors from WuBuRVCModel by name lookup.
 *
 * Cartman v2 architecture (verified via PyTorch — Applio/hifigan.py):
 *   Input: 256-channel mel (post-flow coupling), n_frames time steps
 *   conv_pre: Conv1d(192→512, k=7, s=1, p=3)
 *   ups.0: ConvTranspose1d(512→256, k=16, s=10, p=7) → 256 ch
 *   MRF stage 0: resblocks 0-2, 256 ch, k=3
 *   ups.1: ConvTranspose1d(256→128, k=16, s=10, p=7) → 128 ch
 *   MRF stage 1: resblocks 3-5, 128 ch, k=7
 *   ups.2: ConvTranspose1d(128→64, k=4, s=2, p=1) → 64 ch
 *   MRF stage 2: resblocks 6-8, 64 ch, k=11
 *   ups.3: ConvTranspose1d(64→32, k=4, s=2, p=1) → 32 ch
 *   MRF stage 3: resblocks 9-11, 32 ch, k=3
 *   conv_post: Conv1d(32→1, k=7, s=1, p=3) → 1 ch, then tanh
 *
 * Weight layouts (WUBU binary = PyTorch C-order flatten):
 *   Conv1d: (out_ch, in_ch, k) → flat[oc * in_ch * k + ic * k + tap]
 *   ConvTranspose1d: (in_ch, out_ch, k) → flat[ic * out_ch * k + oc * k + tap]
 *   weight_norm: weight_g (ch,1,1), weight_v (ch,ch,k) → de-normalized in-place
 *
 * License: WaefreBeorn-UMV3
 */

#include "wubu_rvc_parity.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── ConvTranspose1d (PyTorch-compatible) ── */
static void conv_transpose_1d(const float *input,
                               const float *weight,
                               const float *bias,
                               int in_ch, int out_ch, int k, int s, int p,
                               int n_in, float *output)
{
    int n_out = (n_in - 1) * s - 2 * p + k;
    if (!weight) { memset(output, 0, (size_t)out_ch * n_out * sizeof(float)); return; }
    if (n_out <= 0) return;
    memset(output, 0, (size_t)out_ch * n_out * sizeof(float));

    for (int i = 0; i < n_in; i++) {
        int j_start = i * s - p;
        for (int ic = 0; ic < in_ch; ic++) {
            float inp = input[(size_t)ic * n_in + i];
            if (inp == 0.0f) continue;
            const float *wv = weight + (size_t)ic * (size_t)out_ch * (size_t)k;
            for (int oc = 0; oc < out_ch; oc++) {
                float *outrow = output + (size_t)oc * n_out;
                const float *wk = wv + (size_t)oc * k;
                for (int tap = 0; tap < k; tap++) {
                    int j = j_start + tap;
                    if (j >= 0 && j < n_out)
                        outrow[j] += inp * wk[tap];
                }
            }
        }
    }
    if (bias) {
        for (int oc = 0; oc < out_ch; oc++) {
            float b = bias[oc];
            float *outrow = output + (size_t)oc * n_out;
            for (int j = 0; j < n_out; j++) outrow[j] += b;
        }
    }
}

/* ── Conv1d (groups=1), PyTorch Conv1d weight: (out_ch, in_ch, k) ──
 * Input: (in_ch, n) col-major. Output: (out_ch, n_out) col-major.
 */
static void conv1d(const float *input, int in_ch,
                   const float *weight, const float *bias,
                   int out_ch, int k, int stride, int pad, int dilation,
                   int n_in, float *output)
{
    int n_out = (n_in + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
    if (!weight) { memset(output, 0, (size_t)out_ch * n_out * sizeof(float)); return; }
    if (n_out <= 0) return;
    memset(output, 0, (size_t)out_ch * n_out * sizeof(float));

    for (int oc = 0; oc < out_ch; oc++) {
        const float *wv = weight + (size_t)oc * in_ch * k;
        float *out = output + (size_t)oc * n_out;
        float b = bias ? bias[oc] : 0.0f;
        for (int j = 0; j < n_out; j++) {
            float acc = b;
            int j_in = j * stride - pad;
            for (int tap = 0; tap < k; tap++) {
                int src = j_in + dilation * tap;
                if (src >= 0 && src < n_in) {
                    const float *wk = wv + (size_t)tap;  /* weight[oc*in_ch*k + ic*k + tap] */
                    for (int ic = 0; ic < in_ch; ic++) {
                        acc += input[(size_t)ic * n_in + src] * wk[(size_t)ic * k];
                    }
                }
            }
            out[j] = acc;
        }
    }
}

/* ── LeakyReLU in-place (slope=0.1 per Applio LRELU_SLOPE) ── */
static void lrelu(float *data, size_t n) {
    for (size_t i = 0; i < n; i++)
        data[i] = data[i] > 0 ? data[i] : 0.1f * data[i];
}

/* ── Snake activation (BigVGAN): f(x) = x + (1/α)sin²(αx) ──
 * Periodic inductive bias for audio signals. α=1 is the standard choice.
 * NOTE (2026-08-07, crash recovery): this is f(x) ALONE — adding LeakyReLU
 * on top (f = snake + lrelu ≈ 2x for x>0) doubles the signal each MRF
 * stage, blows up activations through 4 stages, saturates the final tanh
 * and produces a square wave (pitch CV 0.0014). BigVGAN's Snake module is
 * exactly x + (1/α)sin²(αx); there is no LReLU term. */
static inline float snake(float x, float a) {
    return x + (1.0f / a) * sinf(a * x) * sinf(a * x);
}

/* ── Snake activation applied in place (BigVGAN, no LReLU term) ── */
static void snake_lrelu(float *data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = snake(data[i], 1.0f);
    }
}

/* ── MRF ResBlock pair: convs1 → LeakyReLU → convs2 + residual ──
 * Applio ResBlock.forward:
 *   for conv1, conv2 in zip(convs1, convs2):
 *     x_residual = x
 *     x = leaky_relu(x)
 *     x = conv1(x)
 *     x = leaky_relu(x)
 *     x = conv2(x)
 *     x = x + x_residual
 * convs1: Conv1d(ch, ch, k, dilation=d)  convs2: Conv1d(ch, ch, k, dilation=1)
 * Weights are de-normalized in-place (stored as (ch, ch, k) groups=1).
 */
static void mrf_resblock_pair(const float *input, int ch, int n, int k_size,
                               int dil1,
                               const float *conv1_w, const float *conv1_b,
                               const float *conv2_w, const float *conv2_b,
                               float *output, int use_snake)
{
    /* 1. Activation on input */
    float *tmp = (float *)malloc((size_t)ch * n * sizeof(float));
    if (!tmp) { memcpy(output, input, (size_t)ch * n * sizeof(float)); return; }
    memcpy(tmp, input, (size_t)ch * n * sizeof(float));
    if (use_snake) snake_lrelu(tmp, (size_t)ch * n);
    else lrelu(tmp, (size_t)ch * n);

    /* 2. conv1 (dilated) */
    int pad1 = dil1 * (k_size - 1) / 2;
    conv1d(tmp, ch, conv1_w, conv1_b, ch, k_size, 1, pad1, dil1, n, output);
    free(tmp);

    /* 3. Activation after conv1 */
    if (use_snake) snake_lrelu(output, (size_t)ch * n);
    else lrelu(output, (size_t)ch * n);

    /* 4. conv2 (dilation=1) → + residual */
    float *tmp2 = (float *)malloc((size_t)ch * n * sizeof(float));
    if (!tmp2) return;
    int pad2 = k_size / 2;
    conv1d(output, ch, conv2_w, conv2_b, ch, k_size, 1, pad2, 1, n, tmp2);
    for (size_t i = 0; (size_t)ch * n > i; i++)
        tmp2[i] += input[i];
    memcpy(output, tmp2, (size_t)ch * n * sizeof(float));
    free(tmp2);
}

/* ── Apply MRF to a stage: average over 3 resblock stacks ── */
static void apply_mrf_stage(const float *input, int ch, int n,
                             int stage_idx, const WuBuRVCModel *model,
                             float *output, int use_snake)
{
    int dilations[3] = {1, 3, 5};
    int num_stacks = 3;

    float *acc = (float *)calloc((size_t)ch * n, sizeof(float));
    if (!acc) {
        memcpy(output, input, (size_t)ch * n * sizeof(float));
        return;
    }

    char key[256];
    float *stack_out = (float *)malloc((size_t)ch * n * sizeof(float));
    float *tmp_in = (float *)malloc((size_t)ch * n * sizeof(float));

    for (int s = 0; s < num_stacks; s++) {
        int rb = stage_idx * 3 + s;
        if (!tmp_in || !stack_out) continue;
        memcpy(tmp_in, input, (size_t)ch * n * sizeof(float));

        for (int cp = 0; cp < 3; cp++) {
            snprintf(key, sizeof(key), "dec.resblocks.%d.convs1.%d.weight_v", rb, cp);
            const RVCTensor *c1w_t = wubu_rvc_find_tensor(model, key);
            snprintf(key, sizeof(key), "dec.resblocks.%d.convs1.%d.bias", rb, cp);
            const RVCTensor *c1b_t = wubu_rvc_find_tensor(model, key);
            snprintf(key, sizeof(key), "dec.resblocks.%d.convs2.%d.weight_v", rb, cp);
            const RVCTensor *c2w_t = wubu_rvc_find_tensor(model, key);
            snprintf(key, sizeof(key), "dec.resblocks.%d.convs2.%d.bias", rb, cp);
            const RVCTensor *c2b_t = wubu_rvc_find_tensor(model, key);

            if (!c1w_t || !c1w_t->data || !c2w_t || !c2w_t->data) {
                memcpy(stack_out, tmp_in, (size_t)ch * n * sizeof(float));
            } else {
                if (c1w_t->dims[0] != ch || c1w_t->dims[1] != ch) {
                    memcpy(stack_out, tmp_in, (size_t)ch * n * sizeof(float));
                    memcpy(tmp_in, stack_out, (size_t)ch * n * sizeof(float));
                    continue;
                }
                int k = c1w_t->dims[2];  /* kernel size from weight shape */
                mrf_resblock_pair(tmp_in, ch, n, k, dilations[cp],
                                   c1w_t->data,
                                   (c1b_t && c1b_t->data) ? c1b_t->data : NULL,
                                   c2w_t->data,
                                   (c2b_t && c2b_t->data) ? c2b_t->data : NULL,
                                   stack_out, use_snake);
                memcpy(tmp_in, stack_out, (size_t)ch * n * sizeof(float));
            }
        }
        for (size_t i = 0; (size_t)ch * n > i; i++)
            acc[i] += stack_out[i];
    }

    for (size_t i = 0; (size_t)ch * n > i; i++)
        output[i] = acc[i] / num_stacks;
    free(acc);
    free(stack_out);
    free(tmp_in);
}

/* ── Exact HiFi-GAN forward pass ──
 * mel_input: (n_frames, 256) row-major — 256 channels from flow coupling.
 * Output: audio samples (n_frames * 400).
 */
int wubu_kernel_hifigan_exact(const WuBuRVCModel *model,
                               const float *mel_input,
                               int n_frames,
                               float *output,
                               int max_output,
                               int use_snake)
{
    if (!model || !model->loaded || !mel_input || !output)
        return -1;

    for (int i = 0; i < 4; i++) {
        if (!model->hifi_upsample_denorm[i]) return -1;
    }

    /* Cartman v2: conv_pre(192→512), then each ups stage halves channels */
    int ups_in_ch[4]  = {512, 256, 128, 64};
    int ups_out_ch[4] = {256, 128, 64, 32};
    int ups_rates[4]  = {10, 10, 2, 2};
    int ups_kernels[4] = {16, 16, 4, 4};
    int ups_pads[4]   = {3, 3, 1, 1};

    int n_cur = n_frames;
    int n0 = (n_cur - 1) * ups_rates[0] - 2 * ups_pads[0] + ups_kernels[0];
    if (n0 <= 0 || n0 > max_output) return -1;

    float *buf0 = (float *)calloc((size_t)ups_out_ch[0] * n0, sizeof(float));
    if (!buf0) return -1;

    /* Step 1: Transpose mel_input row-major (n_frames, 256) → col-major (192, n_frames) */
    float *mel_col = (float *)calloc((size_t)192 * n_cur, sizeof(float));
    if (!mel_col) { free(buf0); return -1; }
    for (int f = 0; f < n_cur; f++)
        for (int c = 0; c < 192; c++)
            mel_col[(size_t)c * n_cur + f] = mel_input[(size_t)f * 256 + c];

    /* Step 2: conv_pre: Conv1d(192→512, k=7, s=1, p=3) → [NO LeakyReLU after conv_pre] */
    const RVCTensor *conv_pre_w = wubu_rvc_find_tensor(model, "dec.conv_pre.weight");
    const RVCTensor *conv_pre_b = wubu_rvc_find_tensor(model, "dec.conv_pre.bias");
    float *pre_out = (float *)calloc((size_t)512 * n_cur, sizeof(float));
    if (pre_out && conv_pre_w && conv_pre_w->data) {
        conv1d(mel_col, 192, conv_pre_w->data,
               (conv_pre_b && conv_pre_b->data) ? conv_pre_b->data : NULL,
               512, 7, 1, 3, 1, n_cur, pre_out);
    } else if (pre_out) {
        /* Fallback: zero-pad 192 → 512 */
        for (int f = 0; f < n_cur; f++)
            for (int c = 0; c < 192; c++)
                pre_out[(size_t)c * n_cur + f] = mel_col[(size_t)c * n_cur + f];
    }
    free(mel_col);
    if (!pre_out) { free(buf0); return -1; }

    char bias_name[64];

    /* Step 3: Stage 0 — leaky_relu → ups.0 → MRF (no lrelu between ups and MRF) */
    lrelu(pre_out, (size_t)ups_in_ch[0] * n_cur);  /* LeakyReLU BEFORE ups.0 (matches PyTorch) */
    snprintf(bias_name, sizeof(bias_name), "dec.ups.0.bias");
    const RVCTensor *ups_b_t = wubu_rvc_find_tensor(model, bias_name);
    conv_transpose_1d(pre_out, model->hifi_upsample_denorm[0],
                      ups_b_t && ups_b_t->data ? ups_b_t->data : NULL,
                      ups_in_ch[0], ups_out_ch[0], ups_kernels[0],
                      ups_rates[0], ups_pads[0], n_cur, buf0);
    free(pre_out);
    /* NO lrelu here — PyTorch does: x = ups[i](x) → MRF(x), lrelu is inside ResBlock */
    apply_mrf_stage(buf0, ups_out_ch[0], n0, 0, model, buf0, use_snake);

    /* Steps 4-6: Stages 1-3 */
    float *prev = buf0;
    int prev_n = n0;
    float *layer_bufs[3] = {NULL, NULL, NULL};

    for (int L = 1; L < 4; L++) {
        int n_next = (prev_n - 1) * ups_rates[L] - 2 * ups_pads[L] + ups_kernels[L];
        if (n_next <= 0 || n_next > max_output) {
            free(buf0);
            for (int j = 0; j < L - 1; j++) free(layer_bufs[j]);
            return -1;
        }
        layer_bufs[L - 1] = (float *)calloc((size_t)ups_out_ch[L] * n_next, sizeof(float));
        if (!layer_bufs[L - 1]) {
            free(buf0);
            for (int j = 0; j < L - 1; j++) free(layer_bufs[j]);
            return -1;
        }
        snprintf(bias_name, sizeof(bias_name), "dec.ups.%d.bias", L);
        ups_b_t = wubu_rvc_find_tensor(model, bias_name);
        /* LeakyReLU BEFORE ups.L (matches PyTorch: lrelu → ups) */
        {
            int prev_ch = L == 1 ? ups_out_ch[0] : ups_out_ch[L - 1];
            lrelu(prev, (size_t)prev_ch * prev_n);
        }
        conv_transpose_1d(prev, model->hifi_upsample_denorm[L],
                          ups_b_t && ups_b_t->data ? ups_b_t->data : NULL,
                          ups_in_ch[L], ups_out_ch[L], ups_kernels[L],
                          ups_rates[L], ups_pads[L], prev_n, layer_bufs[L - 1]);
        /* NO lrelu here — PyTorch does: x = ups[i](x) → MRF(x), lrelu is inside ResBlock */
        apply_mrf_stage(layer_bufs[L - 1], ups_out_ch[L], n_next, L, model, layer_bufs[L - 1], use_snake);
        prev = layer_bufs[L - 1];
        prev_n = n_next;
    }

    /* Final LeakyReLU + conv_post: Conv1d(32→1, k=7, s=1, p=3) → tanh */
    lrelu(prev, (size_t)32 * prev_n);
    const RVCTensor *t_post = wubu_rvc_find_tensor(model, "dec.conv_post.weight");
    const float *post_w = (t_post && t_post->data) ? t_post->data : NULL;

    int out_n = prev_n;
    if (post_w) {
        for (int j = 0; j < prev_n; j++) {
            float acc = 0.0f;
            for (int c = 0; c < 32; c++) {
                const float *kw = post_w + (size_t)c * 7;
                for (int tap = 0; tap < 7; tap++) {
                    int src = j - 3 + tap;
                    if (src >= 0 && src < prev_n)
                        acc += prev[(size_t)c * prev_n + src] * kw[tap];
                }
            }
            output[j] = tanhf(acc);
        }
    } else {
        for (int j = 0; j < prev_n; j++) {
            float acc = 0.0f;
            for (int c = 0; c < 32; c++)
                acc += prev[(size_t)c * prev_n + j];
            output[j] = tanhf(acc / 32.0f);
        }
    }

    free(buf0);
    for (int L = 0; L < 3; L++) free(layer_bufs[L]);
    return out_n;
}
