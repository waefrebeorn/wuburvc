/* tt_harmony.c — probe: validate wubu_harmony_detect on (a) synthetic
 * two-tone harmony (A3 + C#4 = major third) and (b) a real album backing
 * vocal stem. Prints per-frame lead/harmony/gain at 100fps for the first
 * N frames so we can see note-flipping vs stability. */
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_harmony.h"

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
    *n_out = n / ch;
    *sr_out = sr;
    return x;
}

int main(int argc, char **argv) {
    float *x = NULL; int n = 0, sr = 16000;
    if (argc > 1) {
        x = load_pcm16(argv[1], &n, &sr);
        if (!x) { fprintf(stderr, "load fail %s\n", argv[1]); return 1; }
        printf("loaded %s: %d samples @%d\n", argv[1], n, sr);
    } else {
        /* synthetic: 2s of A3 (220Hz) + C#4 (277.18Hz) major third */
        n = sr * 2;
        x = (float *)malloc((size_t)n * 4);
        for (int i = 0; i < n; i++) {
            float t = (float)i / sr;
            x[i] = 0.5f * sinf(2.0f * (float)M_PI * 220.0f * t) +
                   0.35f * sinf(2.0f * (float)M_PI * 277.18f * t);
        }
        printf("synthetic: A3(220) + C#4(277.18) major third, 2s @16k\n");
    }

    int n_frames = n / 160;
    float *pin = (float *)calloc((size_t)n_frames, sizeof(float));
    float *pout = (float *)calloc((size_t)n_frames, sizeof(float));
    float *harm = (float *)calloc((size_t)n_frames, sizeof(float));
    float *gain = (float *)calloc((size_t)n_frames, sizeof(float));
    /* naive primary: use a simple zero-crossing-ish estimate as the
     * "existing contour" — for the synthetic we know 220. For real audio
     * leave 0 and let the detector do everything. */
    if (argc > 1) {
        /* leave pin=0 -> detector is standalone */
    } else {
        for (int j = 0; j < n_frames; j++) pin[j] = 220.0f;
    }
    int got = wubu_harmony_detect(x, n, sr, pin, n_frames, pout, harm, gain,
                                  n_frames, 1024, 160, 50.0f, 1100.0f);
    printf("frames=%d got=%d\n", n_frames, got);

    int shown = 0;
    for (int j = 0; j < n_frames && shown < 12; j++) {
        if (j % 4 == 0) {  /* every 40ms */
            printf("f=%4d lead=%7.2f harm=%7.2f gain=%.2f%s\n",
                   j, pout[j], harm[j], gain[j],
                   (harm[j] > 0) ? "  <-- HARMONY" : "");
            shown++;
        }
    }
    /* summary: count harmony frames */
    int hc = 0; double hsum = 0;
    for (int j = 0; j < n_frames; j++) if (harm[j] > 0) { hc++; hsum += harm[j]; }
    printf("harmony frames: %d/%d avg %.2f Hz\n", hc, n_frames, hc ? hsum / hc : 0);
    free(x); free(pin); free(pout); free(harm); free(gain);
    return 0;
}
