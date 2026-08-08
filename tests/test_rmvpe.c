/* test_rmvpe.c — WuBuRmvpe C11 port verification vs Python reference.
 *
 * Loads models/rvc/rmvpe_weights.bin, extracts F0 from outputs/rvc_ref/pcm16k.npy
 * and compares against the reference outputs/rvc_ref/nsff0.npy (RMVPE via
 * Mangio torch). A correct port shows corr > 0.99 and small Hz diff.
 *
 * Build (MSYS2):
 *   cc -std=c11 -O2 -I src -fopenmp src/test_rmvpe.c src/wubu_rmvpe.c \
 *      src/wubu_stft.c src/wubu_gru.c -lm -o build/test_rmvpe.exe
 *
 * License: WaefreBeorn-UMV3
 */
#include "wubu_rmvpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float *load_npy(const char *path, int *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[6];
    if (fread(magic, 1, 6, f) != 6) { fclose(f); return NULL; }
    if (memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); return NULL; }
    unsigned char ver[2]; fread(ver, 1, 2, f);
    unsigned int hlen;
    if (ver[0] == 1) { unsigned short hl; fread(&hl, 2, 1, f); hlen = hl; }
    else { fread(&hlen, 4, 1, f); }
    fseek(f, hlen, SEEK_CUR);
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 6 + 2 + (ver[0] == 1 ? 2 : 4) + hlen, SEEK_SET);
    long dsize = fsize - (6 + 2 + (ver[0] == 1 ? 2 : 4) + hlen);
    int n = (int)(dsize / sizeof(float));
    if (n <= 0) { fclose(f); return NULL; }
    float *data = (float *)malloc((size_t)dsize);
    if (!data) { fclose(f); return NULL; }
    if (fread(data, sizeof(float), (size_t)n, f) != (size_t)n) { free(data); fclose(f); return NULL; }
    fclose(f);
    *n_out = n;
    return data;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== WuBuRmvpe C11 port vs Python reference ===\n\n");

    WuBuRmvpe *rm = wubu_rmvpe_load("models/rvc/rmvpe_weights.bin");
    if (!rm) { printf("FAIL: cannot load rmvpe_weights.bin\n"); return 1; }
    printf("[1] rmvpe weights loaded\n");

    int npcm = 0, nref = 0;
    float *pcm = load_npy("outputs/rvc_ref/pcm16k.npy", &npcm);
    float *ref = load_npy("outputs/rvc_ref/nsff0.npy", &nref);
    if (!pcm || !ref) { printf("FAIL: pcm16k.npy / nsff0.npy missing\n"); return 1; }
    printf("[2] pcm %d samples, reference f0 %d frames\n", npcm, nref);

    float *f0 = (float *)calloc((size_t)(npcm / 160 + 8), sizeof(float));
    int T = wubu_rmvpe_f0(rm, pcm, npcm, f0, npcm / 160 + 8);
    printf("[3] C11 f0 frames: %d (reference %d)\n", T, nref);
    if (T <= 0) { printf("FAIL: extraction error\n"); return 1; }

    int n = T < nref ? T : nref;
    int voiced = 0;
    double sxy = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sum_abs = 0;
    for (int i = 0; i < n; i++) {
        if (f0[i] > 0 && ref[i] > 0) {
            voiced++;
            double a = f0[i], b = ref[i];
            sxy += a * b; sx += a; sy += b; sxx += a * a; syy += b * b;
            sum_abs += fabs(a - b);
        }
    }
    printf("     voiced frames (both): %d / %d\n", voiced, n);
    if (voiced > 10) {
        double corr = (voiced * sxy - sx * sy) /
                      sqrt((voiced * sxx - sx * sx) * (voiced * syy - sy * sy) + 1e-12);
        double mean_diff = sum_abs / voiced;
        printf("     corr (voiced): %.6f\n", corr);
        printf("     mean |diff| (Hz): %.3f\n", mean_diff);
        printf("     first 10 C11: ");
        int shown = 0;
        for (int i = 0; i < n && shown < 10; i++) {
            if (f0[i] > 0) { printf("%.1f ", f0[i]); shown++; }
        }
        printf("\n     first 10 ref: ");
        shown = 0;
        for (int i = 0; i < n && shown < 10; i++) {
            if (ref[i] > 0) { printf("%.1f ", ref[i]); shown++; }
        }
        printf("\n");
        if (corr > 0.99 && mean_diff < 15.0)
            printf("     ✅ PASS (port matches Python RMVPE)\n");
        else
            printf("     🔴 FAIL (port diverges from reference)\n");
        if (corr <= 0.99 || mean_diff >= 15.0) { free(pcm); free(ref); free(f0); return 1; }
    } else {
        printf("     🔴 FAIL: not enough voiced frames\n");
        free(pcm); free(ref); free(f0);
        return 1;
    }

    wubu_rmvpe_free(rm);
    free(pcm); free(ref); free(f0);
    printf("\nRMVPE PORT VERIFIED\n");
    return 0;
}
