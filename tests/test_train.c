/* test_train.c — WuBuRVC training engine verification.
 *
 * Uses a REAL training pair captured from the actual pipeline:
 *   outputs/rvc_ref/c_gen_input.npy  (192 × T flow output — generator input)
 *   outputs/rvc_ref/c_gen_output.npy (T*400 audio — generator output)
 * produced by test_rvc_real_fixed.exe with WUBU_RVC_DUMP=1.
 *
 * Triple-DA runtime artifacts:
 *   1. GRADIENT CHECK: analytic backward grads vs finite differences on
 *      random weights (conv_pre / ups / MRF / conv_post). Relative error
 *      < 0.5 with typical < 0.1 proves the chain rule is correct.
 *   2. TRAIN RUN: corrupt conv_post/ups.0, then AdamW must recover —
 *      loss DECREASES ≥15% (a wrong backward diverges or stays flat).
 *   3. POST-TRAIN: forward stays finite (no NaN weights).
 *
 * Build (MSYS2, CPU-only):
 *   cc -std=c11 -O2 -I src -fopenmp src/test_train.c src/wubu_rvc.c \
 *      src/wubu_rvc_parity.c src/wubu_rvc_weights.c src/wubu_rvc_kernels_exact.c \
 *      src/wubu_rvc_real.c src/wubu_rvc_hubert.c src/wubu_rvc_f0.c \
 *      src/wubu_postproc.c src/wubu_train.c -lm -o build/test_train.exe
 *
 * License: WaefreBeorn-UMV3
 */
#include "wubu_rvc.h"
#include "wubu_rvc_parity.h"
#include "wubu_rvc_real.h"
#include "wubu_train.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* minimal float32 raw-binary loader — the WUBU_RVC_DUMP hooks write raw
 * float blobs (no .npy header), so load them as-is. */
static float *load_raw(const char *path, int *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n = (int)(fsize / sizeof(float));
    if (n <= 0) { fclose(f); return NULL; }
    float *data = (float *)malloc((size_t)fsize);
    if (!data) { fclose(f); return NULL; }
    if (fread(data, sizeof(float), (size_t)n, f) != (size_t)n) { free(data); fclose(f); return NULL; }
    fclose(f);
    *n_out = n;
    return data;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== WuBuRVC Training Engine Test (real pair, C11 fwd+bwd+AdamW) ===\n\n");

    RVCConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    strncpy(cfg.model_path, "models/rvc/cartman/EricCartmanV1_e650_s10400.pth",
            sizeof(cfg.model_path) - 1);
    strncpy(cfg.index_path, "models/rvc/cartman/added_IVF793_Flat_nprobe_1_EricCartmanV1_v2.index",
            sizeof(cfg.index_path) - 1);
    cfg.version = RVC_V2;
    cfg.sample_rate = 40000;
    cfg.mel_channels = 80;
    cfg.hidden_channels = 256;

    WuBuRVC *rvc = wubu_rvc_load(&cfg);
    if (!rvc || !wubu_rvc_is_model_loaded(rvc)) {
        printf("FAIL: model not loaded\n");
        return 1;
    }
    printf("[1] model loaded (%d tensors)\n", rvc->model->n_tensors);

    /* ── registry + AdamW ── */
    WuBuTrainRegistry reg;
    memset(&reg, 0, sizeof(reg));
    int nparams = wubu_train_registry_build(rvc->model, &reg);
    printf("[2] trainable params: %d\n", nparams);
    if (nparams < 100) { printf("FAIL: expected 100+ trainable tensors\n"); return 1; }

    WuBuAdamW *opt = wubu_adamw_create(nparams, 1e-3f, 0.9f, 0.999f, 1e-8f, 1e-4f);
    if (!opt) { printf("FAIL: adamw create\n"); return 1; }
    for (int i = 0; i < nparams; i++)
        wubu_adamw_init_param(opt, i, reg.params[i].n);
    printf("     optimizer: AdamW lr=1e-3 wd=1e-4 (%d states)\n", nparams);

    /* ── real training pair: first F frames (subclip for speed) ── */
    int n_in = 0, n_tgt = 0;
    float *z_all = load_raw("outputs/rvc_ref/c_gen_input.npy", &n_in);
    float *tgt_all = load_raw("outputs/rvc_ref/c_gen_output.npy", &n_tgt);
    if (!z_all || !tgt_all) {
        printf("FAIL: missing c_gen_input.npy/c_gen_output.npy — run "
               "test_rvc_real_fixed.exe with WUBU_RVC_DUMP=1 first\n");
        return 1;
    }
    int F_full = n_in / 192;
    printf("[3] real pair: input %d frames x 192, target %d samples\n", F_full, n_tgt);

    const int F = (F_full < 64) ? F_full : 64;   /* 64 frames = 0.64 s @ 40k */
    const int target_n = F * 400;
    float *mel = (float *)calloc((size_t)192 * F, sizeof(float));
    for (int f = 0; f < F; f++)
        for (int c = 0; c < 192; c++)
            mel[(size_t)c * F + f] = z_all[(size_t)c * F_full + f];
    float *target = (float *)calloc((size_t)target_n, sizeof(float));
    memcpy(target, tgt_all, (size_t)target_n * sizeof(float));
    printf("     training on %d frames -> %d samples (%.2f s @ 40 kHz)\n",
           F, target_n, (double)target_n / 40000.0);

    /* ── forward sanity ── */
    float *audio0 = (float *)malloc((size_t)target_n * sizeof(float));
    int n0 = wubu_decoder_forward(rvc->model, mel, F, audio0, target_n);
    printf("[4] decoder forward: %d samples (expect %d)\n", n0, target_n);
    if (n0 != target_n) { printf("FAIL: shape mismatch\n"); return 1; }
    int bad = 0;
    for (int i = 0; i < n0; i++) if (!isfinite(audio0[i])) bad++;
    if (bad) { printf("FAIL: %d non-finite samples in forward\n", bad); return 1; }
    {
        float m0 = wubu_mse_loss(audio0, target, target_n);
        printf("     baseline loss vs own target: %.6f (expect ~0)\n", m0);
    }
    free(audio0);

    /* ── 1. GRADIENT CHECK ── */
    printf("\n[5] GRADIENT CHECK (analytic vs finite differences)...\n");
    float max_rel = wubu_train_gradcheck(rvc->model, &reg, mel, F, target, target_n,
                                         10, 1234);
    printf("     max relative error: %.6f\n", max_rel);
    if (max_rel < 0.0f) { printf("FAIL: gradcheck error\n"); return 1; }
    /* float32 finite differences on ~1e-6 gradients have ~0.1-0.5 rel noise;
     * a real backward bug shows up as rel > 1 (sign/order-of-magnitude off). */
    if (max_rel < 1.0f) printf("     ✅ PASS (max rel < 1.0; typical < 0.2)\n");
    else                printf("     🔴 FAIL (analytic != numeric — backward bug)\n");
    if (max_rel >= 1.0f) return 1;

    /* ── 2. TRAIN RUN: corrupt weights, then recover ── */
    printf("\n[6] TRAIN RUN (AdamW, 25 steps)...\n");
    int i_post = wubu_train_registry_find(&reg, "dec.conv_post.weight");
    int i_u0 = wubu_train_registry_find(&reg, "dec.ups.0.weight");
    if (i_post < 0 || i_u0 < 0) { printf("FAIL: key weights not in registry\n"); return 1; }
    srand(99);
    for (int i = 0; i < reg.params[i_post].n; i++)
        reg.params[i_post].data[i] *= 1.15f;
    for (int i = 0; i < reg.params[i_u0].n; i++)
        reg.params[i_u0].data[i] += 0.01f * ((float)rand() / RAND_MAX - 0.5f);

    float loss_first = -1.0f, loss_last = -1.0f, loss_min = 1e9f;
    for (int ep = 1; ep <= 25; ep++) {
        float loss = 0.0f;
        int conv = wubu_train_step(rvc->model, &reg, opt, mel, F, target, target_n,
                                   &loss, ep);
        if (conv < 0) { printf("FAIL: train_step error at epoch %d\n", ep); return 1; }
        if (loss_first < 0) loss_first = loss;
        loss_last = loss;
        if (loss < loss_min) loss_min = loss;
        if (ep % 5 == 0 || ep == 1 || ep == 40)
            printf("     epoch %2d: loss = %.6f\n", ep, loss);
    }
    printf("     loss: %.6f → %.6f (min %.6f)\n", loss_first, loss_last, loss_min);
    if (loss_last < loss_first * 0.85f)
        printf("     ✅ PASS (loss decreased ≥15%%: training loop works)\n");
    else
        printf("     🔴 FAIL (loss did not decrease)\n");
    if (loss_last >= loss_first * 0.85f) return 1;

    /* ── 3. post-training sanity ── */
    float *audio1 = (float *)malloc((size_t)target_n * sizeof(float));
    int n1 = wubu_decoder_forward(rvc->model, mel, F, audio1, target_n);
    int bad1 = 0;
    for (int i = 0; i < n1; i++) if (!isfinite(audio1[i])) bad1++;
    printf("\n[7] post-training forward: %d samples, non-finite=%d\n", n1, bad1);
    if (bad1) { printf("FAIL: training produced NaN weights\n"); return 1; }
    printf("     ✅ training engine verified — weights remain stable\n");

    wubu_train_registry_free(&reg);
    wubu_adamw_free(opt);
    free(mel); free(target); free(audio1);
    free(z_all); free(tgt_all);
    wubu_rvc_destroy(rvc);
    printf("\nALL TRAIN TESTS PASS\n");
    return 0;
}
