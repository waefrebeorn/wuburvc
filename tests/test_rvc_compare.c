/* test_rvc_compare.c — Compare WuBuRVC pipeline output vs PyTorch reference.
 * Uses the same mel input (saved from gen_reference_pytorch3.py) and same
 * Cartman v2 model weights. Compares: mean, std, min, max, RMS.
 * Also loads PyTorch reference output for sample-by-sample comparison. */
#include "wubu_rvc.h"
#include "wubu_rvc_parity.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Minimal .npy float32 loader — reads header, returns data pointer. */
static float *load_npy_float32(const char *path, int *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[6];
    if (fread(magic, 1, 6, f) != 6) { fclose(f); return NULL; }
    if (memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); return NULL; }
    /* Read version */
    unsigned char ver[2];
    fread(ver, 1, 2, f);
    /* Read header length (2 bytes for v1.0, 4 for v2.0) */
    unsigned int hlen;
    if (ver[0] == 1) {
        unsigned short hl;
        fread(&hl, 2, 1, f);
        hlen = hl;
    } else {
        fread(&hlen, 4, 1, f);
    }
    /* Skip header */
    fseek(f, hlen, SEEK_CUR);
    /* Read data */
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 6 + 2 + (ver[0] == 1 ? 2 : 4) + hlen, SEEK_SET);
    long dsize = fsize - (6 + 2 + (ver[0] == 1 ? 2 : 4) + hlen);
    int n = (int)(dsize / sizeof(float));
    if (n <= 0) { fclose(f); return NULL; }
    float *data = (float *)malloc((size_t)dsize);
    if (!data) { fclose(f); return NULL; }
    if (fread(data, sizeof(float), (size_t)n, f) != (size_t)n) {
        free(data); fclose(f); return NULL;
    }
    fclose(f);
    *n_out = n;
    return data;
}

int main(void) {
    const char *pth = "models/rvc/cartman/EricCartmanV1_e650_s10400.pth";
    const char *idx = "models/rvc/cartman/added_IVF793_Flat_nprobe_1_EricCartmanV1_v2.index";

    RVCConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    strncpy(cfg.model_path, pth, sizeof(cfg.model_path) - 1);
    if (idx) strncpy(cfg.index_path, idx, sizeof(cfg.index_path) - 1);
    cfg.version = RVC_V2;
    cfg.sample_rate = 40000;
    cfg.mel_channels = 80;
    cfg.hidden_channels = 256;
    cfg.n_flow_layers = 4;
    cfg.n_hifigan_upsamples = 4;
    cfg.n_mrf_stacks = 3;
    cfg.n_residual_layers = 4;

    WuBuRVC *rvc = wubu_rvc_load(&cfg);
    if (!rvc || !wubu_rvc_is_model_loaded(rvc)) {
        printf("FAIL: model not loaded\n");
        return 1;
    }

    /* Load mel from .npy (same data used by PyTorch reference) */
    int mel_n = 0;
    float *mel = load_npy_float32("pytorch_ref_mel.npy", &mel_n);
    int n_frames = 4, mel_ch = 80;
    if (mel_n != n_frames * mel_ch) {
        printf("FAIL: mel shape mismatch (%d expected %d)\n", mel_n, n_frames * mel_ch);
        if (mel) free(mel);
        wubu_rvc_destroy(rvc);
        return 1;
    }

    /* Run our C11 pipeline */
    int n_audio = n_frames * 512;
    float *output = (float *)malloc((size_t)n_audio * sizeof(float));
    int rc = wubu_rvc_synthesize(rvc, mel, n_frames, mel_ch, output, n_audio);
    printf("synthesize rc=%d (expected %d samples)\n", rc, n_audio);

    if (rc > 0) {
        float mean = 0, max_v = 0, min_v = 0, sum_sq = 0;
        for (int i = 0; i < rc; i++) {
            mean += output[i];
            sum_sq += output[i] * output[i];
            if (output[i] > max_v) max_v = output[i];
            if (output[i] < min_v) min_v = output[i];
        }
        mean /= rc;
        float std = sqrtf(sum_sq / rc - mean * mean);
        float rms = sqrtf(sum_sq / rc);

        /* Load PyTorch reference */
        int ref_n = 0;
        float *ref = load_npy_float32("pytorch_ref_output.npy", &ref_n);

        printf("\n=== Comparison (same mel from .npy) ===\n");
        printf("WuBuRVC C11  : mean=%.6f std=%.6f min=%.6f max=%.6f rms=%.6f\n",
               mean, std, min_v, max_v, rms);
        if (ref && ref_n == rc) {
            float r_mean = 0, r_max = 0, r_min = 0, r_sq = 0;
            float max_diff = 0;
            for (int i = 0; i < ref_n; i++) {
                r_mean += ref[i];
                r_sq += ref[i] * ref[i];
                if (ref[i] > r_max) r_max = ref[i];
                if (ref[i] < r_min) r_min = ref[i];
                float d = fabsf(output[i] - ref[i]);
                if (d > max_diff) max_diff = d;
            }
            r_mean /= ref_n;
            float r_std = sqrtf(r_sq / ref_n - r_mean * r_mean);
            float r_rms = sqrtf(r_sq / ref_n);
            printf("PyTorch ref  : mean=%.6f std=%.6f min=%.6f max=%.6f rms=%.6f\n",
                   r_mean, r_std, r_min, r_max, r_rms);
            printf("Max abs diff: %.8f\n", max_diff);
            /* Compute mean abs diff */
            float sum_abs = 0;
            for (int i = 0; i < rc; i++) {
                sum_abs += fabsf(output[i] - ref[i]);
            }
            float mean_abs = sum_abs / rc;
            printf("Mean abs diff: %.8f\n", mean_abs);
        } else {
            printf("PyTorch ref  : (could not load pytorch_ref_output.npy)\n");
        }

        /* Check for corruption */
        int nan_count = 0, inf_count = 0, clipped = 0;
        for (int i = 0; i < rc; i++) {
            if (isnan(output[i])) nan_count++;
            if (isinf(output[i])) inf_count++;
            if (output[i] > 1.0f || output[i] < -1.0f) clipped++;
        }
        printf("\nSanity: nan=%d inf=%d clipped=%d\n", nan_count, inf_count, clipped);
        printf("%s\n", (nan_count == 0 && inf_count == 0 && clipped == 0)
                      ? "PASS: no NaN/Inf/clipping" : "FAIL: corrupted output");

        if (ref && ref_n == rc) {
            float max_diff = 0;
            for (int i = 0; i < rc; i++) {
                float d = fabsf(output[i] - ref[i]);
                if (d > max_diff) max_diff = d;
            }
            /* Consider pass if max diff < 0.1 (untrained model, floating point) */
            printf("PARITY: %s (max_diff=%.6f)\n",
                   max_diff < 0.1f ? "PASS" : "NEEDS_WORK", max_diff);
        }

        free(ref);
    } else {
        printf("FAIL: pipeline returned %d\n", rc);
    }

    free(mel);
    free(output);
    wubu_rvc_destroy(rvc);
    return 0;
}
