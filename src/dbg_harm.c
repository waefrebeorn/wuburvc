#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_fft.h"

int main(void) {
    int sr = 16000, n = sr * 2;
    float *x = (float *)malloc((size_t)n * 4);
    for (int i = 0; i < n; i++) {
        float t = (float)i / sr;
        x[i] = 0.5f * sinf(2.0f * (float)M_PI * 220.0f * t) +
               0.35f * sinf(2.0f * (float)M_PI * 277.18f * t);
    }
    int window = 1024, hop = 160, nfft = 1024;
    int start = 40 * hop;
    WuBuCpx *spec = (WuBuCpx *)calloc((size_t)nfft, sizeof(WuBuCpx));
    float *hann = (float *)malloc((size_t)window * 4);
    for (int i = 0; i < window; i++) hann[i] = 0.5f * (1 - cosf(2 * (float)M_PI * i / (float)(window - 1)));
    for (int i = 0; i < window; i++) { spec[i].re = x[start + i] * hann[i]; }
    wubu_fft(spec, nfft, 0);
    int nbins = nfft / 2 + 1;
    float *mag = (float *)malloc((size_t)nbins * 4);
    for (int b = 0; b < nbins; b++) mag[b] = sqrtf(spec[b].re * spec[b].re + spec[b].im * spec[b].im);
    float bin_hz = (float)sr / nfft;
    for (int k = 1; k <= 10; k++) {
        int b1 = (int)lroundf(220.0f * k / bin_hz);
        int b2 = (int)lroundf(277.18f * k / bin_hz);
        printf("harm k=%d 220->bin%d mag%.3f | 277->bin%d mag%.3f\n", k, b1, mag[b1], b2, mag[b2]);
    }
    /* what does the 277 peak look like vs 220 peak, both at their fundamental */
    printf("mag@220 bin: %.3f  mag@277 bin: %.3f  ratio %.2f\n",
           mag[(int)lroundf(220.0f / bin_hz)], mag[(int)lroundf(277.18f / bin_hz)],
           mag[(int)lroundf(277.18f / bin_hz)] / (mag[(int)lroundf(220.0f / bin_hz)] + 1e-9f));
    free(x); free(spec); free(hann); free(mag);
    return 0;
}
