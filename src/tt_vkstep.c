#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_rvc_parity.h"
#include "wubu_train.h"
#include "wubu_vk.h"

int main(void) {
    const char *model_path = "C:/Users/eman5/WuBuMedia/models/rvc/cleveland/Cleveland_Brown_220e_7920s.pth";
    const char *bin_path = "C:/Users/eman5/WuBuMedia/models/rvc/cleveland/Cleveland_Brown_220e_7920s.pth.weights.bin";
    WuBuRVCModel *model = wubu_rvc_load_model(model_path);
    if (!model) { fprintf(stderr, "load FAIL\n"); return 1; }
    wubu_rvc_load_weights(model, bin_path);
    int F = 16, max_samples = F * 400;
    const char *fs = getenv("WUBU_TRAIN_F");
    if (fs) F = atoi(fs);
    max_samples = F * 400;
    fprintf(stderr, "ENVF: F=%d max=%d\n", F, max_samples); fflush(stderr);
    float *mel = (float *)malloc((size_t)192 * F * sizeof(float));
    float *wav = (float *)malloc((size_t)max_samples * sizeof(float));
    float *audio = (float *)malloc((size_t)max_samples * sizeof(float));
    unsigned s = 12345;
    for (int i = 0; i < 192 * F; i++) { s = s * 1664525u + 1013904223u; mel[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }
    for (int i = 0; i < max_samples; i++) { s = s * 1664525u + 1013904223u; wav[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }
    fprintf(stderr, "STEP1: model loaded\n"); fflush(stderr);

    WuBuVk *vk = wubu_vk_create();
    fprintf(stderr, "STEP2: vk created %s\n", vk ? "OK" : "FAIL"); fflush(stderr);
    if (!vk) return 1;
    TrainCacheVk *cache = wubu_train_cache_alloc_vk();
    int n_out = wubu_train_forward_vk(vk, model, mel, F, audio, max_samples, cache);
    fprintf(stderr, "STEP3: vk forward n=%d\n", n_out); fflush(stderr);
    if (n_out <= 0) return 1;

    WuBuTrainRegistry reg;
    memset(&reg, 0, sizeof(reg));
    wubu_train_registry_build(model, &reg);
    int n = n_out;
    float *d_audio = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) d_audio[i] = 2.0f * (audio[i] - wav[i]) / (float)n;

    fprintf(stderr, "STEP4: calling backward...\n"); fflush(stderr);
    int rc = wubu_train_backward_vk(vk, model, cache, mel, d_audio, &reg);
    fprintf(stderr, "STEP5: backward rc=%d\n", rc); fflush(stderr);

    wubu_train_registry_free(&reg);
    wubu_train_cache_free_vk(cache); free(cache);
    wubu_vk_destroy(vk);
    wubu_rvc_model_free(model);
    fprintf(stderr, "DONE\n");
    return 0;
}
