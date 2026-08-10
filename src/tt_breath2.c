#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* standalone feature dump: replicate the synthetic clip, print features */
static void synth_clip(short *out, int n, int sr) {
    int i = 0;
    for (; i < sr / 2 && i < n; i++) out[i] = 0;
    unsigned long long rng = 12345;
    for (int k = 0; k < (int)(0.6 * sr) && i < n; k++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        float v = (float)((rng >> 33) & 0xFFFF) / 32768.0f - 1.0f;
        v = (v + 0.7f * v) / 1.7f;
        out[i++] = (short)(v * 600.0f);
    }
    for (int k = 0; k < (int)(0.8 * sr) && i < n; k++) {
        float v = 0.5f * sinf(2 * (float)M_PI * 220 * k / sr);
        out[i++] = (short)(v * 32767.0f);
    }
    for (; i < n; i++) out[i] = 0;
}
int main(void) {
    int sr = 16000, n = sr * 2;
    short *buf = malloc((size_t)n * 2);
    synth_clip(buf, n, sr);
    for (int f = 45; f < 125; f++) {
        int start = f * 160;
        double e = 0; int z = 0;
        for (int i = start; i < start + 512 && i < n; i++) { double v = buf[i]/32768.0; e += v*v; }
        for (int i = start+1; i < start + 512 && i < n; i++) if ((buf[i]>=0)!=(buf[i-1]>=0)) z++;
        float rms = (float)sqrt(e/512), zcr = (float)z/511;
        printf("f=%3d rms=%.5f zcr=%.3f\n", f, rms, zcr);
    }
    free(buf);
    return 0;
}
