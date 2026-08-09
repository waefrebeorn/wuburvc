/* tt_vkconvt0.c — isolate convT0 backward at REAL L=0 dims in RECORD mode.
 * Build: gcc ... src/tt_vkconvt0.c src/wubu_vk.c src/wubu_rvc.c ... build/wubu_cuda.o
 * Runs: forward convT (512->256, k16, s10, p3) on random input, then bwd, A/B vs CPU.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_vk.h"
#include "wubu_train.h"

#define IN_CH 512
#define OUT_CH 256
#define K 16
#define STRIDE 10
#define PAD 3
#define N_IN 16
#define N_OUT 160
#define TB WUBU_VK_TRAIN_BASE
#define SL_IN (TB + 10)
#define SL_W  (TB + 11)
#define SL_DOUT (TB + 12)
#define SL_DIN  (TB + 13)
#define SL_DW   (TB + 14)
#define SL_DB   (TB + 15)
#define SL_STAGE (TB + 16)

static void convt1d_fwd_cpu(const float *in, const float *w, const float *b,
                            int in_ch, int out_ch, int k, int stride, int pad,
                            int n_in, int n_out, float *out) {
    memset(out, 0, (size_t)out_ch * n_out * sizeof(float));
    for (int oc = 0; oc < out_ch; oc++) {
        for (int i = 0; i < n_in; i++) {
            int j_start = i * stride - pad;
            for (int tap = 0; tap < k; tap++) {
                int j = j_start + tap;
                if (j < 0 || j >= n_out) continue;
                float acc = 0;
                for (int ic = 0; ic < in_ch; ic++)
                    acc += in[(size_t)ic * n_in + i] * w[((size_t)ic * out_ch + oc) * k + tap];
                out[(size_t)oc * n_out + j] += acc;
            }
        }
    }
    for (int oc = 0; oc < out_ch; oc++)
        for (int j = 0; j < n_out; j++)
            out[(size_t)oc * n_out + j] += b[oc];
}

static void convt1d_bwd_cpu(const float *in, const float *w, const float *dout,
                            int in_ch, int out_ch, int k, int stride, int pad,
                            int n_in, int n_out, float *din, float *dw, float *db) {
    memset(dw, 0, (size_t)in_ch * out_ch * k * sizeof(float));
    memset(db, 0, (size_t)out_ch * sizeof(float));
    memset(din, 0, (size_t)in_ch * n_in * sizeof(float));
    for (int ic = 0; ic < in_ch; ic++) {
        for (int i = 0; i < n_in; i++) {
            int j_start = i * stride - pad;
            float din_acc = 0;
            for (int oc = 0; oc < out_ch; oc++) {
                for (int tap = 0; tap < k; tap++) {
                    int j = j_start + tap;
                    if (j < 0 || j >= n_out) continue;
                    float g = dout[(size_t)oc * n_out + j];
                    din_acc += g * w[((size_t)ic * out_ch + oc) * k + tap];
                    dw[((size_t)ic * out_ch + oc) * k + tap] += in[(size_t)ic * n_in + i] * g;
                }
            }
            din[(size_t)ic * n_in + i] += din_acc;
        }
    }
    for (int oc = 0; oc < out_ch; oc++) {
        float acc = 0;
        for (int j = 0; j < n_out; j++) acc += dout[(size_t)oc * n_out + j];
        db[oc] = acc;
    }
}

int main(void) {
    WuBuVk *vk = wubu_vk_create();
    if (!vk) { fprintf(stderr, "vk create FAIL\n"); return 1; }
    float *in = malloc((size_t)IN_CH * N_IN * 4);
    float *w  = malloc((size_t)IN_CH * OUT_CH * K * 4);
    float *b  = malloc((size_t)OUT_CH * 4);
    float *dout = malloc((size_t)OUT_CH * N_OUT * 4);
    float *din_cpu = malloc((size_t)IN_CH * N_IN * 4);
    float *dw_cpu = malloc((size_t)IN_CH * OUT_CH * K * 4);
    float *db_cpu = malloc((size_t)OUT_CH * 4);
    float *din_vk = malloc((size_t)IN_CH * N_IN * 4);
    float *dw_vk = malloc((size_t)IN_CH * OUT_CH * K * 4);
    float *db_vk = malloc((size_t)OUT_CH * 4);
    srand(42);
    for (size_t i = 0; i < (size_t)IN_CH * N_IN; i++) in[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (size_t i = 0; i < (size_t)IN_CH * OUT_CH * K; i++) w[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (size_t i = 0; i < (size_t)OUT_CH; i++) b[i] = 0.01f * (float)(rand() % 100) - 0.05f;
    for (size_t i = 0; i < (size_t)OUT_CH * N_OUT; i++) dout[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

    convt1d_bwd_cpu(in, w, dout, IN_CH, OUT_CH, K, STRIDE, PAD, N_IN, N_OUT,
                    din_cpu, dw_cpu, db_cpu);

    /* forward to make sure the record path executes the bwd with real data */
    if (wubu_vk_train_begin(vk)) { fprintf(stderr, "begin FAIL\n"); return 1; }
    /* host-visible staging for downloads (max tensor = dw 8MB) */
    if (wubu_vk_train_zero(vk, SL_STAGE, (size_t)IN_CH * OUT_CH * K * 4)) { fprintf(stderr, "stage alloc FAIL\n"); return 1; }
    if (wubu_vk_train_upload(vk, SL_IN, in, (size_t)IN_CH * N_IN * 4)) { fprintf(stderr, "upload in FAIL\n"); return 1; }
    if (wubu_vk_train_upload(vk, SL_W, w, (size_t)IN_CH * OUT_CH * K * 4)) { fprintf(stderr, "upload w FAIL\n"); return 1; }
    if (wubu_vk_train_upload(vk, SL_DOUT, dout, (size_t)OUT_CH * N_OUT * 4)) { fprintf(stderr, "upload dout FAIL\n"); return 1; }
    /* din_s/dw/db zeroed via GPU fill */
    wubu_vk_train_fill_zero(vk, SL_DIN, (size_t)IN_CH * N_IN);
    wubu_vk_train_fill_zero(vk, SL_DW, (size_t)IN_CH * OUT_CH * K);
    wubu_vk_train_fill_zero(vk, SL_DB, (size_t)OUT_CH);
    wubu_vk_train_bwd_convt1d(vk, SL_IN, SL_W, SL_DOUT, SL_DIN, SL_DW, SL_DB,
                              IN_CH, OUT_CH, K, STRIDE, PAD, N_IN, N_OUT);
    if (wubu_vk_train_end(vk)) { fprintf(stderr, "end FAIL\n"); return 1; }

    /* staged download via elt */
    wubu_vk_train_begin(vk);
    wubu_vk_train_elt(vk, SL_STAGE, 0, SL_DIN, 0, 1, (size_t)IN_CH * N_IN);
    wubu_vk_train_end(vk);
    wubu_vk_train_download(vk, SL_STAGE, din_vk, (size_t)IN_CH * N_IN * 4);
    wubu_vk_train_begin(vk);
    wubu_vk_train_elt(vk, SL_STAGE, 0, SL_DW, 0, 1, (size_t)IN_CH * OUT_CH * K);
    wubu_vk_train_end(vk);
    wubu_vk_train_download(vk, SL_STAGE, dw_vk, (size_t)IN_CH * OUT_CH * K * 4);
    wubu_vk_train_begin(vk);
    wubu_vk_train_elt(vk, SL_STAGE, 0, SL_DB, 0, 1, (size_t)OUT_CH);
    wubu_vk_train_end(vk);
    wubu_vk_train_download(vk, SL_STAGE, db_vk, (size_t)OUT_CH * 4);

    double max_rel = 0, max_abs = 0; long bad = 0;
    for (size_t i = 0; i < (size_t)IN_CH * OUT_CH * K; i++) {
        double denom = fabsf(dw_cpu[i]) > 1e-6f ? fabsf(dw_cpu[i]) : 1e-6f;
        double rel = fabsf(dw_cpu[i] - dw_vk[i]) / denom;
        float ad = fabsf(dw_cpu[i] - dw_vk[i]);
        if (ad > max_abs) max_abs = ad;
        if (rel > max_rel) max_rel = rel;
        if (rel > 0.15 && fabsf(dw_cpu[i]) > 1e-4f) { bad++; if (bad < 4) fprintf(stderr, "  DW mismatch[%zu] cpu=%.6f vk=%.6f\n", i, dw_cpu[i], dw_vk[i]); }
    }
    fprintf(stderr, "[tt] DW A/B checked=%zu bad=%ld max_rel=%.6f max_abs=%.6f %s\n",
            (size_t)IN_CH * OUT_CH * K, bad, max_rel, max_abs, bad == 0 ? "PASS" : "FAIL");
    long badz = 0;
    for (size_t i = 0; i < (size_t)IN_CH * OUT_CH * K; i++)
        if (dw_vk[i] == 0.0f && fabsf(dw_cpu[i]) > 1e-4f) badz++;
    fprintf(stderr, "[tt] DW zero-but-cpu-nonzero=%ld\n", badz);
    /* din */
    double max_rel_d = 0; long badd = 0;
    for (size_t i = 0; i < (size_t)IN_CH * N_IN; i++) {
        double denom = fabsf(din_cpu[i]) > 1e-6f ? fabsf(din_cpu[i]) : 1e-6f;
        double rel = fabsf(din_cpu[i] - din_vk[i]) / denom;
        if (rel > max_rel_d) max_rel_d = rel;
        if (rel > 0.15 && fabsf(din_cpu[i]) > 1e-4f) badd++;
    }
    fprintf(stderr, "[tt] DIN A/B checked=%zu bad=%ld max_rel=%.6f\n", (size_t)IN_CH * N_IN, badd, max_rel_d);

    wubu_vk_destroy(vk);
    free(in); free(w); free(b); free(dout);
    free(din_cpu); free(dw_cpu); free(db_cpu);
    free(din_vk); free(dw_vk); free(db_vk);
    return bad == 0 && badd == 0 ? 0 : 1;
}
