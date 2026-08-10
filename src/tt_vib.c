#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_postproc.h"

/* Probe: vibrato-aware f0 smoothing. Synthetic contour: 220Hz carrier +
 * 5Hz vibrato (depth ±1.5 semitones) + random single-frame jitter noise.
 * Old smoothing killed the vibrato; new must keep ~5Hz modulation while
 * removing jitter. Measure vibrato energy via FFT of the contour. */
int main(void) {
    int n = 2000;   /* 20s @100fps */
    float *f0 = (float *)malloc((size_t)n * 4);
    unsigned long long rng = 42;
    for (int i = 0; i < n; i++) {
        float t = i / 100.0f;
        /* carrier 220 + vibrato 5Hz ±1.5% (≈26 cents) */
        float vib = 220.0f * (1.0f + 0.015f * sinf(2 * (float)M_PI * 5.0f * t));
        /* jitter: ±0.5% single-frame noise */
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        float jit = ((float)((rng >> 33) & 0xFFFF) / 32768.0f - 1.0f) * 0.005f;
        f0[i] = vib * (1.0f + jit);
    }
    /* FFT of original vibrato component (remove carrier) */
    float *orig_vib = (float *)malloc((size_t)n * 4);
    for (int i = 0; i < n; i++) orig_vib[i] = (f0[i] - 220.0f) / 220.0f;
    /* FFT magnitude at the 5Hz bin: contour is 100fps so 5Hz = bin
     * round(5/100 * nfft) */
    int nfft = 2048;
    int bin5 = (int)(5.0 / 100.0 * nfft + 0.5);
    double re5 = 0, im5 = 0;
    for (int i = 0; i < n && i < nfft; i++) {
        double ph = -2.0 * M_PI * bin5 * i / nfft;
        re5 += orig_vib[i] * cos(ph); im5 += orig_vib[i] * sin(ph);
    }
    double orig_amp5 = sqrt(re5 * re5 + im5 * im5) / nfft * 2;

    wubu_f0_smooth(f0, n, 0.8f);   /* heavy smoothing — old code would nuke vibrato */

    /* FFT of smoothed vibrato at 5Hz */
    re5 = 0; im5 = 0;
    for (int i = 0; i < n && i < nfft; i++) {
        float v = (f0[i] - 220.0f) / 220.0f;
        double ph = -2.0 * M_PI * bin5 * i / nfft;
        re5 += v * cos(ph); im5 += v * sin(ph);
    }
    double sm_amp5 = sqrt(re5 * re5 + im5 * im5) / nfft * 2;
    /* jitter removal: std of single-frame delta */
    double d1 = 0;
    for (int i = 1; i < n; i++) d1 += (f0[i] - f0[i-1]) * (f0[i] - f0[i-1]);
    d1 = sqrt(d1 / (n - 1));

    printf("vibrato 5Hz amplitude: orig=%.5f smoothed=%.5f (keep=%.0f%%)\n",
           orig_amp5, sm_amp5, 100.0 * sm_amp5 / orig_amp5);
    printf("frame-to-frame delta after smoothing: %.4f Hz (vibrato slope ~0.9 Hz)\n", d1);
    printf("%s\n", (sm_amp5 > 0.6 * orig_amp5 && d1 < 1.2)
                   ? "VIBRATO FIX PASS" : "VIBRATO FIX FAIL");
    free(f0); free(orig_vib);
    return 0;
}
