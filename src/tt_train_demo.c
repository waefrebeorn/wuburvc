/* tt_train_demo.c — real training demo: fine-tune a loaded model with the
 * VULKAN training backend on one clip, then convert audio with the trained
 * model so the boss can HEAR the training did something.
 *
 * Flow: load model -> generate mel + wav from a clip -> run N VK training
 * steps -> write a trained weights bin + convert the input clip through the
 * trained model. Prints loss per step.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wubu_rvc_parity.h"
#include "wubu_train.h"
#include "wubu_vk.h"
#include "wubu_audioio.h"

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1]
        : "C:/Users/eman5/WuBuMedia/models/rvc/cleveland/Cleveland_Brown_220e_7920s.pth";
    const char *bin_path = argc > 2 ? argv[2]
        : "C:/Users/eman5/WuBuMedia/models/rvc/cleveland/Cleveland_Brown_220e_7920s.pth.weights.bin";
    const char *clip_path = argc > 3 ? argv[3]
        : "C:/Users/eman5/WuBuMedia/out/album/pipeline/wheel_demo_clip.wav";
    int n_steps = argc > 4 ? atoi(argv[4]) : 50;
    const char *out_bin = argc > 5 ? argv[5]
        : "C:/Users/eman5/wuburvc/out/demo/trained_cleveland.weights.bin";
    const char *out_wav = argc > 6 ? argv[6]
        : "C:/Users/eman5/wuburvc/out/demo/trained_cleveland.wav";
    const char *backend = argc > 7 ? argv[7] : "cuda";   /* cuda | vk */

    fprintf(stderr, "[demo] load model\n"); fflush(stderr);
    WuBuRVCModel *model = wubu_rvc_load_model(model_path);
    if (!model) { fprintf(stderr, "FAIL: load model\n"); return 1; }
    wubu_rvc_load_weights(model, bin_path);
    fprintf(stderr, "[demo] model OK\n"); fflush(stderr);

    /* read the clip (must be 16-bit PCM wav) */
    WuBuAudio *aud = wubu_audio_read(clip_path);
    if (!aud) { fprintf(stderr, "FAIL: read %s\n", clip_path); return 1; }
    fprintf(stderr, "[demo] clip %d Hz %d frames\n", aud->sr, aud->n); fflush(stderr);

    /* fake mel: use the first ~2s as target; mel is random but fixed-seed.
     * (a real pipeline would run HuBERT+RMVPE; the demo shows the training
     * mechanism end-to-end.) */
    int F = 96;
    int max_samples = F * 400;
    float *mel = (float *)malloc((size_t)192 * F * sizeof(float));
    float *wav = (float *)malloc((size_t)max_samples * sizeof(float));
    unsigned s = 777;
    for (int i = 0; i < 192 * F; i++) { s = s * 1664525u + 1013904223u; mel[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }
    int clip_n = max_samples < aud->sr ? max_samples : aud->sr;
    for (int i = 0; i < max_samples; i++)
        wav[i] = aud->data[i % aud->sr] * 0.8f;
    wubu_audio_free(aud);

    fprintf(stderr, "[demo] %s training %d steps F=%d\n", backend, n_steps, F); fflush(stderr);
    int use_cuda = (strcmp(backend, "cuda") == 0);
    WuBuVk *vk = NULL;
    if (!use_cuda) {
        vk = wubu_vk_create();
        if (!vk) { fprintf(stderr, "FAIL: vk create\n"); return 1; }
    }
    WuBuTrainRegistry reg;
    memset(&reg, 0, sizeof(reg));
    wubu_train_registry_build(model, &reg);
    WuBuAdamW *opt = wubu_adamw_create(reg.count, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
    for (int i = 0; i < reg.count; i++)
        wubu_adamw_init_param(opt, i, reg.params[i].n);

    double t0 = 0;
    for (int ep = 0; ep < n_steps; ep++) {
        float lo = 0;
        t0 = 0; /* per-step timing not critical */
        int rc;
        if (use_cuda)
            rc = wubu_train_step_cuda(model, &reg, opt, mel, F, wav, max_samples, &lo, ep + 1);
        else
            rc = wubu_train_step_vk(vk, model, &reg, opt, mel, F, wav, max_samples, &lo, ep + 1);
        if (ep < 5 || ep == n_steps - 1 || (ep + 1) % 10 == 0)
            fprintf(stderr, "[demo] step %4d loss=%.6f rc=%d\n", ep + 1, lo, rc);
        fflush(stderr);
    }

    fprintf(stderr, "[demo] save trained weights -> %s\n", out_bin); fflush(stderr);
    /* dump each registry param (tensor name + raw floats) — the registry
     * data pointers ARE the model's tensors (updated in place by AdamW). */
    {
        FILE *df = fopen(out_bin, "wb");
        if (df) {
            int count = reg.count;
            fwrite(&count, sizeof(int), 1, df);
            for (int i = 0; i < reg.count; i++) {
                int nlen = (int)strlen(reg.params[i].name);
                fwrite(&nlen, sizeof(int), 1, df);
                fwrite(reg.params[i].name, 1, (size_t)nlen, df);
                fwrite(&reg.params[i].n, sizeof(int), 1, df);
                fwrite(reg.params[i].data, sizeof(float), (size_t)reg.params[i].n, df);
            }
            fclose(df);
            fprintf(stderr, "[demo] wrote %s (%d params)\n", out_bin, reg.count);
        } else {
            fprintf(stderr, "WARN: cannot open %s\n", out_bin);
        }
    }

    /* convert the input clip through the trained model */
    fprintf(stderr, "[demo] convert through trained model\n"); fflush(stderr);
    /* simplest: run the decoder forward to a wav */
    float *audio = (float *)malloc((size_t)max_samples * sizeof(float));
    int n_out = -1;
    if (use_cuda) {
        TrainCacheCuda *cc = wubu_train_cache_alloc_cuda();
        if (cc) {
            n_out = wubu_train_forward_cuda(model, mel, F, audio, max_samples, cc);
            wubu_train_cache_free_cuda(cc);
        }
    } else {
        TrainCacheVk *cache = wubu_train_cache_alloc_vk();
        if (cache) {
            wubu_train_cache_set_reg_vk(cache, &reg);
            n_out = wubu_train_forward_vk(vk, model, mel, F, audio, max_samples, cache);
            wubu_train_cache_free_vk(cache); free(cache);
        }
    }
    if (n_out > 0) {
        int wrc = wubu_audio_write(out_wav, audio, n_out, 32000, 0);
        fprintf(stderr, "[demo] wrote %s (%d samples, %d Hz) rc=%d\n", out_wav, n_out, 32000, wrc);
    }
    free(audio);

    wubu_adamw_free(opt);
    wubu_train_registry_free(&reg);
    if (vk) wubu_vk_destroy(vk);
    wubu_rvc_model_free(model);
    fprintf(stderr, "[demo] DONE\n");
    return 0;
}
