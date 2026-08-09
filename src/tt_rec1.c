/* tt_rec1.c — isolate record-mode FORWARD vs per-op path. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_rvc_parity.h"
#include "wubu_train.h"
#include "wubu_vk.h"

static float frand(void) { static unsigned s = 12345; s = s * 1103515245 + 12345; return ((s >> 16) & 0x7fff) / 32768.0f - 0.5f; }

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "C:/Users/eman5/WuBuMedia/models/rvc/cleveland/Cleveland_Brown_220e_7920s.pth";
    WuBuRVCModel *model = wubu_rvc_load_model(model_path);
    if (!model) { fprintf(stderr, "FAIL load\n"); return 1; }
    wubu_rvc_load_weights(model, "C:/Users/eman5/WuBuMedia/models/rvc/cleveland/Cleveland_Brown_220e_7920s.pth.weights.bin");
    int F = argc > 2 ? atoi(argv[2]) : 16;
    int max_samples = F * 400;
    float *mel = (float *)malloc(192 * F * 4);
    float *audio = (float *)malloc(max_samples * 4);
    float *audio2 = (float *)malloc(max_samples * 4);
    for (int i = 0; i < 80 * F; i++) mel[i] = frand() * 0.5f;
    for (int i = 80 * F; i < 192 * F; i++) mel[i] = 0.0f;

    /* CPU reference */
    TrainCacheVk *cache = wubu_train_cache_alloc_vk();
    WuBuTrainRegistry reg;
    memset(&reg, 0, sizeof(reg));
    wubu_train_registry_build(model, &reg);
    wubu_train_cache_set_reg_vk(cache, &reg);
    int n_out = wubu_decoder_forward(model, mel, F, audio, max_samples);
    printf("cpu n=%d audio[0..3]=%.6f %.6f %.6f %.6f\n", n_out, audio[0], audio[1], audio[2], audio[3]);

    /* VK record mode */
    WuBuVk *vk = wubu_vk_create();
    if (!vk) { fprintf(stderr, "FAIL vk\n"); return 1; }
    int n2 = wubu_train_forward_vk(vk, model, mel, F, audio2, max_samples, cache);
    printf("vk  n=%d audio[0..3]=%.6f %.6f %.6f %.6f\n", n2, audio2[0], audio2[1], audio2[2], audio2[3]);
    /* VK PER-OP: use the generator's wubu_vk_generator_nsf to cross-check the
     * same model+mel on the proven path. Build a full generator call (with
     * zero noise/cond so it reduces to the decoder). */
    {
        int ups_total = 1;
        for (int L = 0; L < 4; L++) ups_total *= (model->upsample_rates[L] > 0 ? model->upsample_rates[L] : 2);
        int inter = model->hidden_channels > 0 ? model->hidden_channels : 192;
        float *z = (float *)calloc((size_t)inter * F, sizeof(float));
        float *nsff0 = (float *)calloc((size_t)F, sizeof(float));
        float *g = (float *)calloc((size_t)256, sizeof(float));
        int n3 = wubu_vk_generator_nsf(vk, model, z, F, inter, nsff0, g,
                                       audio2, max_samples, 0, 1);
        printf("perop n=%d audio[0..3]=%.6f %.6f %.6f %.6f\n", n3, audio2[0], audio2[1], audio2[2], audio2[3]);
        free(z); free(nsff0); free(g);
    }
    double ma = 0, mb = 0, mc = 0;
    int nn = n2 < n_out ? n2 : n_out;
    for (int i = 0; i < nn; i++) { ma += audio[i] * audio2[i]; mb += audio[i] * audio[i]; mc += audio2[i] * audio2[i]; }
    printf("corr=%.6f (n=%d) maxdiff=%.6f\n", ma / sqrt(mb * mc + 1e-30), nn,
           fabsf(audio[0] - audio2[0]));
    wubu_vk_destroy(vk);
    return 0;
}
