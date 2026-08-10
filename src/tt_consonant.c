/* tt_consonant.c — probe: validate wubu_consonant_uv.
 * Synthetic: 1s voiced (220Hz sawtooth-ish with harmonics), 1s fricative
 * (white noise shaped), 1s voiced again. Expect uv=1 for voiced, 0 for
 * noise. Also run on a real wordy vocal clip. */
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_consonant.h"

static float *load_pcm16(const char *path, int *n_out, int *sr_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    unsigned char hdr[44];
    if (fread(hdr, 1, 44, f) != 44) { fclose(f); return NULL; }
    int sr = hdr[24] | (hdr[25] << 8) | (hdr[26] << 16) | (hdr[27] << 24);
    int ch = hdr[22] | (hdr[23] << 8);
    int n = hdr[40] | (hdr[41] << 8) | (hdr[42] << 16) | (hdr[43] << 24);
    n /= 2;
    short *pcm = (short *)malloc((size_t)n * 2);
    if (fread(pcm, 2, (size_t)n, f) != (size_t)n) { fclose(f); free(pcm); return NULL; }
    fclose(f);
    float *x = (float *)malloc((size_t)n * 4);
    for (int i = 0; i < n; i++) x[i] = pcm[i] / 32768.0f;
    free(pcm);
    *n_out = n / ch; *sr_out = sr;
    return x;
}

/* tiny xorshift for noise */
static unsigned s = 12345;
static float urand(void) { s = s * 1103515245 + 12345; return (float)((s >> 16) & 0x7fff) / 32767.0f; }

int main(int argc, char **argv) {
    float *x = NULL; int n = 0, sr = 16000;
    if (argc > 1) {
        x = load_pcm16(argv[1], &n, &sr);
        if (!x) { fprintf(stderr, "load fail %s\n", argv[1]); return 1; }
        printf("loaded %s: %d samples @%d\n", argv[1], n, sr);
    } else {
        sr = 16000; n = sr * 3;
        x = (float *)malloc((size_t)n * 4);
        for (int i = 0; i < sr; i++) {         /* voiced 1s: 220Hz + 3 harmonics */
            float t = (float)i / sr;
            x[i] = 0.5f * sinf(2 * (float)M_PI * 220 * t)
                 + 0.2f * sinf(2 * (float)M_PI * 440 * t)
                 + 0.1f * sinf(2 * (float)M_PI * 660 * t)
                 + 0.05f * sinf(2 * (float)M_PI * 880 * t);
        }
        for (int i = sr; i < 2 * sr; i++) {    /* fricative 1s: shaped noise */
            x[i] = 0.3f * (urand() * 2 - 1);
        }
        for (int i = 2 * sr; i < 3 * sr; i++) { /* voiced 1s */
            float t = (float)(i - 2 * sr) / sr;
            x[i] = 0.5f * sinf(2 * (float)M_PI * 330 * t)
                 + 0.2f * sinf(2 * (float)M_PI * 660 * t);
        }
        printf("synthetic: 1s voiced(220) + 1s fricative(noise) + 1s voiced(330)\n");
    }

    int n_frames = n / 160;
    float *f0 = (float *)calloc((size_t)n_frames, sizeof(float));
    float *uv = (float *)calloc((size_t)n_frames, sizeof(float));
    float *fl = (float *)calloc((size_t)n_frames, sizeof(float));
    /* synthetic f0: 220 in sec1, 0 in sec2, 330 in sec3 */
    if (argc <= 1) {
        for (int j = 0; j < n_frames; j++) {
            float t = j / 100.0f;
            if (t < 1.0f) f0[j] = 220;
            else if (t < 2.0f) f0[j] = 0;
            else f0[j] = 330;
        }
    }
    int got = wubu_consonant_uv(x, n, sr, f0, n_frames, uv, fl, 1024, 160);
    printf("frames=%d got=%d\n", n_frames, got);

    /* summary per second */
    for (int sec = 0; sec < n_frames / 100; sec++) {
        int vc = 0, uc = 0; float fs = 0;
        for (int j = sec * 100; j < (sec + 1) * 100 && j < n_frames; j++) {
            if (uv[j] > 0.5f) vc++; else uc++;
            fs += fl[j];
        }
        fs /= 100;
        printf("sec %d: voiced=%3d unvoiced=%3d flatness=%.2f  %s\n",
               sec, vc, uc, fs, vc > uc ? "[VOICED]" : "[UNVOICED]");
    }
    free(x); free(f0); free(uv); free(fl);
    return 0;
}
