#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_harmony.h"

/* Scan a full harmony detect run: report octave-family harmony frames
 * (harm/lead within ±0.6 semitone of 2.0, 4.0, 0.5 — the "single singer
 * double octave swap" signature) and lead octave jumps. */
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
        if (!x) { fprintf(stderr, "load fail\n"); return 1; }
    } else { return 1; }
    printf("clip %s: %d samples @%d\n", argv[1], n, sr);
    int n_frames = n / 160;
    float *pin = (float *)calloc((size_t)n_frames, sizeof(float));
    /* Feed a realistic interpolated f0 contour like the CLI does (nsff0):
     * smooth 220Hz-ish with a few notes, all voiced — the RMVPE authority. */
    {
        float notes[] = {196.0f, 220.0f, 246.94f, 220.0f, 293.66f, 261.63f, 196.0f};
        for (int f = 0; f < n_frames; f++) {
            int ni = (f / 40) % 7;
            pin[f] = notes[ni];
        }
    }
    float *pout = (float *)calloc((size_t)n_frames, sizeof(float));
    float *harm = (float *)calloc((size_t)n_frames, sizeof(float));
    float *gain = (float *)calloc((size_t)n_frames, sizeof(float));
    int got = wubu_harmony_detect(x, n, sr, pin, n_frames, pout, harm, gain,
                                  n_frames, 1024, 160, 50.0f, 1100.0f);
    printf("frames=%d got=%d\n", n_frames, got);

    int oct = 0, double_oct = 0, sub_oct = 0, total_harm = 0;
    int lead_jump = 0;
    float prev_lead = 0;
    for (int f = 0; f < got; f++) {
        float L = pout[f], H = harm[f], G = gain[f];
        if (H > 0 && L > 40) {
            total_harm++;
            float r = H / L;
            float cents = 1200.0f * fabsf(log2f(r / 2.0f));
            if (cents < 60.0f) oct++;
            float cents4 = 1200.0f * fabsf(log2f(r / 4.0f));
            if (cents4 < 60.0f) double_oct++;
            float centsh = 1200.0f * fabsf(log2f(r / 0.5f));
            if (centsh < 60.0f) sub_oct++;
            if (f < 300 || (f % 50 == 0))
                printf("f=%4d lead=%7.2f harm=%7.2f gain=%.2f ratio=%.3f %s\n",
                       f, L, H, G, r,
                       (cents < 60 || cents4 < 60 || centsh < 60) ? "  <-- OCTAVE?" : "");
        }
        if (prev_lead > 40 && L > 40) {
            float cents = 1200.0f * fabsf(log2f(L / prev_lead));
            if (cents > 850.0f && cents < 1550.0f) lead_jump++;  /* full ~octave jump */
        }
        if (L > 40) prev_lead = L;
    }
    printf("harmony frames=%d | octave(2x)=%d double-oct(4x)=%d sub-oct(0.5x)=%d lead-octave-jumps=%d\n",
           total_harm, oct, double_oct, sub_oct, lead_jump);
    free(x); free(pin); free(pout); free(harm); free(gain);
    return 0;
}
