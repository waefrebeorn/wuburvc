#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_fft.h"

int main(void) {
    FILE *f = fopen("out/demo/sb2/kpop_autotune_harmony.wav", "rb");
    unsigned char hdr[44];
    fread(hdr, 1, 44, f);
    int sr = hdr[24] | (hdr[25] << 8) | (hdr[26] << 16) | (hdr[27] << 24);
    int n = hdr[40] | (hdr[41] << 8) | (hdr[42] << 16) | (hdr[43] << 24);
    n /= 2;
    short *pcm = malloc((size_t)n * 2);
    fread(pcm, 2, (size_t)n, f); fclose(f);
    float *x = malloc((size_t)n * 4);
    for (int i = 0; i < n; i++) x[i] = pcm[i] / 32768.0f;
    /* examine spectral peaks at 3 different times: 2s, 6s, 10s */
    int nfft = 4096; /* 4096 @32k = 128ms window, 7.8Hz bins */
    float *hann = malloc((size_t)nfft * 4);
    for (int i = 0; i < nfft; i++) hann[i] = 0.5f * (1 - cosf(2 * (float)M_PI * i / (float)(nfft - 1)));
    for (int t = 0; t < 3; t++) {
        int start = (2 + t * 4) * sr;
        if (start + nfft > n) break;
        WuBuCpx *spec = calloc((size_t)nfft, sizeof(WuBuCpx));
        for (int i = 0; i < nfft; i++) spec[i].re = x[start + i] * hann[i];
        wubu_fft(spec, nfft, 0);
        int nbins = nfft / 2;
        /* find peaks 50-1100 Hz, print top 8 with mag */
        typedef struct { int b; float m; } Pk;
        Pk pks[16]; int npk = 0;
        for (int b = 1; b < nbins; b++) {
            float m = sqrtf(spec[b].re * spec[b].re + spec[b].im * spec[b].im);
            if (m > 20.0f && m >= sqrtf(spec[b-1].re*spec[b-1].re+spec[b-1].im*spec[b-1].im)
                          && m >= sqrtf(spec[b+1].re*spec[b+1].re+spec[b+1].im*spec[b+1].im)) {
                float hz = (float)b * sr / nfft;
                if (hz < 50 || hz > 1100) continue;
                if (npk < 16) { pks[npk].b = b; pks[npk].m = m; npk++; }
            }
        }
        /* sort desc */
        for (int a = 1; a < npk; a++) { Pk k = pks[a]; int b2 = a-1;
            while (b2 >= 0 && pks[b2].m < k.m) { pks[b2+1] = pks[b2]; b2--; } pks[b2+1] = k; }
        printf("t=%ds peaks:", 2 + t * 4);
        for (int i = 0; i < npk && i < 8; i++)
            printf(" %.1fHz(%.0f)", (float)pks[i].b * sr / nfft, pks[i].m);
        printf("\n");
        free(spec);
    }
    free(pcm); free(x); free(hann);
    return 0;
}
