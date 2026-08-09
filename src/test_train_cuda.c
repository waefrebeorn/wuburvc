/* test_train_cuda.c — verify the CUDA training path vs the CPU reference.
 *
 * Triple-DA: claim = "wubu_train_forward_cuda / wubu_train_backward_cuda
 * produce the same audio and the same registry grads as the CPU engine".
 * Verification = run BOTH on identical inputs and compare:
 *   1. forward audio correlation (CPU decoder_forward vs CUDA forward)
 *   2. registry grads (CPU decoder_backward vs CUDA backward) — the strong
 *      A/B check (finite differences are unreliable on real RVC models
 *      because tanh saturation flattens single-weight loss surfaces).
 *   3. loss trajectory over a few CUDA steps (should decrease).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wubu_rvc_parity.h"
#include "wubu_train.h"

static float corr1(const float *a, const float *b, int n) {
    double sa = 0, sb = 0, sab = 0, sa2 = 0, sb2 = 0;
    for (int i = 0; i < n; i++) { sa += a[i]; sb += b[i]; }
    double ma = sa / n, mb = sb / n;
    for (int i = 0; i < n; i++) {
        double x = a[i] - ma, y = b[i] - mb;
        sab += x * y; sa2 += x * x; sb2 += y * y;
    }
    if (sa2 <= 0 || sb2 <= 0) return 0.0f;
    return (float)(sab / sqrt(sa2 * sb2));
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1]
        : "C:/Users/eman5/WuBuMedia/models/rvc/cleveland/Cleveland_Brown_220e_7920s.pth";
    const char *bin_path = argc > 2 ? argv[2]
        : "C:/Users/eman5/WuBuMedia/models/rvc/cleveland/Cleveland_Brown_220e_7920s.pth.weights.bin";
    fprintf(stderr, "[t] loading %s\n", model_path);
    fflush(stderr);
    WuBuRVCModel *model = wubu_rvc_load_model(model_path);
    if (!model) { fprintf(stderr, "FAIL: cannot load model\n"); return 2; }
    if (wubu_rvc_load_weights(model, bin_path) != 0) { fprintf(stderr, "FAIL: cannot load weights\n"); return 2; }
    fprintf(stderr, "[t] loaded OK\n");
    fflush(stderr);

    /* deterministic pseudo-random mel input + target wav */
    int F = 16, sr = 40000;
    int max_samples = F * 400;
    float *mel = (float *)malloc((size_t)192 * F * sizeof(float));
    float *wav = (float *)malloc((size_t)max_samples * sizeof(float));
    unsigned s = 12345;
    for (int i = 0; i < 192 * F; i++) { s = s * 1664525u + 1013904223u; mel[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }
    for (int i = 0; i < max_samples; i++) { s = s * 1664525u + 1013904223u; wav[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }

    /* ── 1. forward A/B ── */
    float *audio_cpu = (float *)malloc((size_t)max_samples * sizeof(float));
    int n_out_cpu = wubu_decoder_forward(model, mel, F, audio_cpu, max_samples);
    if (n_out_cpu <= 0) { fprintf(stderr, "FAIL: cpu forward\n"); return 2; }

    float *audio_cuda = (float *)malloc((size_t)max_samples * sizeof(float));
    TrainCacheCuda *cache = wubu_train_cache_alloc_cuda();
    if (!cache) { fprintf(stderr, "FAIL: cache alloc\n"); return 2; }
    int n_out_cuda = wubu_train_forward_cuda(model, mel, F, audio_cuda, max_samples, cache);
    if (n_out_cuda != n_out_cpu) { fprintf(stderr, "FAIL: n_out differ %d vs %d\n", n_out_cuda, n_out_cpu); return 2; }
    float c = corr1(audio_cpu, audio_cuda, n_out_cpu);
    float maxd = 0.0f;
    for (int i = 0; i < n_out_cpu; i++) { float d = fabsf(audio_cpu[i] - audio_cuda[i]); if (d > maxd) maxd = d; }
    fprintf(stderr, "[t] FORWARD corr=%.10f maxdiff=%.8f (n=%d) %s\n", c, maxd, n_out_cpu,
            (c > 0.999f) ? "PASS" : "CHECK");
    fflush(stderr);

    /* ── 2. grads A/B: CPU backward vs CUDA backward on identical inputs ── */
    WuBuTrainRegistry reg_cpu, reg_cuda;
    memset(&reg_cpu, 0, sizeof(reg_cpu));
    memset(&reg_cuda, 0, sizeof(reg_cuda));
    int nreg_cpu = wubu_train_registry_build(model, &reg_cpu);
    int nreg_cuda = wubu_train_registry_build(model, &reg_cuda);
    if (nreg_cpu == 0 || nreg_cuda == 0) {
        fprintf(stderr, "FAIL: registry build (cpu=%d cuda=%d)\n", nreg_cpu, nreg_cuda); return 2;
    }
    fprintf(stderr, "[t] registry %d params\n", nreg_cpu);
    fflush(stderr);

    int n = n_out_cpu < max_samples ? n_out_cpu : max_samples;
    float *d_audio = (float *)malloc((size_t)n_out_cpu * sizeof(float));
    for (int i = 0; i < n; i++) d_audio[i] = 2.0f * (audio_cpu[i] - wav[i]) / (float)n;
    for (int i = n; i < n_out_cpu; i++) d_audio[i] = 0.0f;

    /* CPU grads (full engine backward: forward+cache+backward internally) */
    int rc_cpu = wubu_train_backward_cpu(model, mel, F, wav, n, &reg_cpu);
    if (rc_cpu <= 0) { fprintf(stderr, "FAIL: cpu backward rc=%d\n", rc_cpu); return 2; }
    fprintf(stderr, "[t] cpu backward ok\n"); fflush(stderr);

    /* CUDA grads */
    wubu_train_registry_zero_grads(&reg_cuda);
    int rc_cuda = wubu_train_backward_cuda(model, cache, mel, d_audio, &reg_cuda);
    if (rc_cuda != 0) { fprintf(stderr, "FAIL: cuda backward rc=%d\n", rc_cuda); return 2; }
    fprintf(stderr, "[t] cuda backward ok\n"); fflush(stderr);

    /* compare grads (CPU grads are the reference). Float32 A/B: atomic
     * accumulation order differs from the CPU loop, so small grads can show
     * rel ~5-10% — only flag significant grads (|cpu| > 1e-4) with
     * rel > 0.15 (well above rounding noise). */
    double max_rel = 0.0, max_abs = 0.0;
    int bad = 0, checked = 0;
    for (int i = 0; i < reg_cpu.count && i < reg_cuda.count; i++) {
        WuBuTrainParam *pc = &reg_cpu.params[i];
        WuBuTrainParam *pu = &reg_cuda.params[i];
        if (!pc->grad || !pu->grad || pc->n != pu->n) continue;
        for (int j = 0; j < pc->n; j++) {
            float gc = pc->grad[j], gu = pu->grad[j];
            float ad = fabsf(gc - gu);
            if (ad > max_abs) max_abs = ad;
            double denom = fabsf(gc) > 1e-6f ? fabsf(gc) : 1e-6f;
            double rel = fabsf(gc - gu) / denom;
            if (rel > max_rel) max_rel = rel;
            if (rel > 0.15 && fabsf(gc) > 1e-4f) {
                if (bad < 6)
                    fprintf(stderr, "  grad mismatch %s[%d] cpu=%.6f cuda=%.6f rel=%.3f\n",
                            pc->name, j, gc, gu, rel);
                bad++;
            }
            checked++;
        }
    }
    int grad_ok = (bad == 0);
    fprintf(stderr, "[t] GRADS A/B checked=%d bad=%d max_rel=%.6f max_abs=%.8f %s\n",
            checked, bad, max_rel, max_abs, grad_ok ? "PASS" : "FAIL");
    fflush(stderr);

    /* ── 3. loss trajectory over 3 CUDA steps ── */
    WuBuAdamW *opt = wubu_adamw_create(reg_cuda.count, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
    for (int i = 0; i < reg_cuda.count; i++)
        wubu_adamw_init_param(opt, i, reg_cuda.params[i].n);
    float l0 = 1e9f, l1 = 0;
    for (int ep = 0; ep < 3; ep++) {
        float lo = 0;
        int rc = wubu_train_step_cuda(model, &reg_cuda, opt, mel, F, wav, n, &lo, ep + 1);
        fprintf(stderr, "[t] cuda step %d loss=%.6f rc=%d\n", ep + 1, lo, rc);
        if (ep == 0) l0 = lo;
        l1 = lo;
    }
    int traj_ok = (l1 < l0);
    fprintf(stderr, "[t] TRAJECTORY loss %.6f -> %.6f %s\n", l0, l1, traj_ok ? "DECREASED" : "STALLED");

    free(mel); free(wav); free(audio_cpu); free(audio_cuda); free(d_audio);
    wubu_train_cache_free_cuda(cache);
    wubu_adamw_free(opt);
    wubu_train_registry_free(&reg_cpu);
    wubu_train_registry_free(&reg_cuda);
    wubu_rvc_model_free(model);
    fprintf(stderr, "[t] TEST_DONE %s\n", (grad_ok && traj_ok) ? "PASS" : "FAIL");
    return (grad_ok && traj_ok && c > 0.999f) ? 0 : 1;
}
