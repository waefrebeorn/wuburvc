#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_postproc.h"

/* Probe: artifact gate. Signal = harmonic 220Hz (speech-like) with an
 * isolated 30ms noise burst in the middle (artifact) and a sustained
 * fricative-ish noise run at the end (real consonant — must survive). */
int main(void) {
    int sr = 48000, n = sr;   /* 1s */
    float *x = (float *)malloc((size_t)n * 4);
    unsigned long long rng = 7;
    for (int i = 0; i < n; i++) {
        float t = (float)i / sr;
        /* isolated 12ms FLAT burst at 0.40-0.412s (harmonics drop out —
         * the vocoder artifact signature) */
        if (t > 0.40f && t < 0.412f) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            x[i] = ((float)((rng >> 33) & 0xFFFF) / 32768.0f - 1.0f) * 0.3f;
            continue;
        }
        float v = 0.5f * sinf(2 * (float)M_PI * 220 * t)
                + 0.2f * sinf(2 * (float)M_PI * 440 * t)
                + 0.1f * sinf(2 * (float)M_PI * 660 * t);
        /* sustained noise run 0.7-0.9s (real fricative, must survive) */
        if (t > 0.70f && t < 0.90f) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            v += ((float)((rng >> 33) & 0xFFFF) / 32768.0f - 1.0f) * 0.25f;
        }
        x[i] = v;
    }
    /* measure RMS in each region BEFORE */
    double rms_harm = 0, rms_burst = 0, rms_fric = 0;
    int c1 = 0, c2 = 0, c3 = 0;
    for (int i = 0; i < n; i++) {
        float t = (float)i / sr;
        if (t > 0.30f && t < 0.38f) { rms_harm += x[i]*x[i]; c1++; }
        if (t > 0.403f && t < 0.409f) { rms_burst += x[i]*x[i]; c2++; }
        if (t > 0.75f && t < 0.85f) { rms_fric += x[i]*x[i]; c3++; }
    }
    rms_harm = sqrt(rms_harm/c1); rms_burst = sqrt(rms_burst/c2); rms_fric = sqrt(rms_fric/c3);
    printf("BEFORE: harm=%.4f burst=%.4f fric=%.4f\n", rms_harm, rms_burst, rms_fric);

    wubu_artifact_gate(x, n, sr, 0.5f);

    rms_harm = rms_burst = rms_fric = 0; c1 = c2 = c3 = 0;
    for (int i = 0; i < n; i++) {
        float t = (float)i / sr;
        if (t > 0.30f && t < 0.38f) { rms_harm += x[i]*x[i]; c1++; }
        if (t > 0.403f && t < 0.409f) { rms_burst += x[i]*x[i]; c2++; }
        if (t > 0.75f && t < 0.85f) { rms_fric += x[i]*x[i]; c3++; }
    }
    rms_harm = sqrt(rms_harm/c1); rms_burst = sqrt(rms_burst/c2); rms_fric = sqrt(rms_fric/c3);
    printf("AFTER:  harm=%.4f burst=%.4f fric=%.4f\n", rms_harm, rms_burst, rms_fric);
    double burst_att = 1.0 - rms_burst / 0.197;   /* burst before was ~0.197 */
    printf("burst attenuation: %.0f%% (expect >20%%)\n", 100.0 * burst_att);
    printf("%s\n", (rms_burst < 0.85 * 0.197 && rms_fric > 0.85 * 0.177 && rms_harm > 0.9 * 0.175)
                   ? "ARTIFACT GATE PASS" : "ARTIFACT GATE FAIL");
    free(x);
    return 0;
}
