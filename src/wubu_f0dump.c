/* wubu_f0dump.c — RMVPE f0 dump for pitch-drift analysis.
 * Usage: wubu_f0dump in.wav out.f0 (float32, 0 = unvoiced, 100 fps)
 * Links wubu_rmvpe/wubu_stft/wubu_gru/wubu_audioio. */
#include "wubu_rmvpe.h"
#include "wubu_audioio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: wubu_f0dump in.wav out.f0\n"); return 1; }
    WuBuAudio *a = wubu_audio_read(argv[1]);
    if (!a) { fprintf(stderr, "read failed\n"); return 1; }
    int n16 = 0;
    float *pcm = (float *)malloc((size_t)(a->n * 2 + 1024) * sizeof(float));
    if (!pcm) return 1;
    n16 = wubu_audio_resample_sinc(a->data, a->n, a->sr, 16000, pcm);
    wubu_audio_free(a);
    WuBuRmvpe *rm = wubu_rmvpe_load("models/rvc/rmvpe_weights.bin");
    if (!rm) { fprintf(stderr, "rmvpe load failed\n"); return 1; }
    int maxf = n16 / 160 + 8;
    float *f0 = (float *)calloc((size_t)maxf, sizeof(float));
    int n_f0 = wubu_rmvpe_f0(rm, pcm, n16, f0, maxf);
    wubu_rmvpe_free(rm);
    free(pcm);
    FILE *f = fopen(argv[2], "wb");
    if (!f) { free(f0); return 1; }
    fwrite(f0, sizeof(float), (size_t)n_f0, f);
    fclose(f);
    fprintf(stderr, "wrote %d f0 frames\n", n_f0);
    free(f0);
    return 0;
}
