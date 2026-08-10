/* tt_breath.c — probe: validate wubu_breath_detect on synthetic and real.
 * Synthetic: 0.5s silence, 0.6s breath (low-RMS filtered noise), 0.8s
 * voiced 220Hz. Expect: silence -> SILENCE, breath -> BREATH, voiced -> VOICED.
 */
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_breath.h"
#include "wubu_audioio.h"

static void synth_clip(short *out, int n, int sr) {
    /* 0.5 s silence */
    int i = 0;
    for (; i < sr / 2 && i < n; i++) out[i] = 0;
    /* 0.6 s breath: low-RMS white noise, shaped (0.02 rms) */
    unsigned long long rng = 12345;
    for (int k = 0; k < (int)(0.6 * sr) && i < n; k++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        float v = (float)((rng >> 33) & 0xFFFF) / 32768.0f - 1.0f;
        /* soft lowpass: average of 4 samples = breath-like */
        v = (v + 0.7f * v) / 1.7f;
        out[i++] = (short)(v * 600.0f);   /* ~0.018 rms */
    }
    /* 0.8 s voiced 220Hz */
    for (int k = 0; k < (int)(0.8 * sr) && i < n; k++) {
        float v = 0.5f * sinf(2 * (float)M_PI * 220 * k / sr);
        out[i++] = (short)(v * 32767.0f);
    }
    for (; i < n; i++) out[i] = 0;
}

int main(int argc, char **argv) {
    int sr = 16000;
    if (argc > 1) {
        /* real clip path */
        WuBuAudio *a = wubu_audio_read(argv[1]);
        if (!a) { fprintf(stderr, "load fail %s\n", argv[1]); return 1; }
        printf("clip %s: %d samples @%d\n", argv[1], a->n, a->sr);
        /* convert to 16k mono s16 */
        int mn = a->n;
        int n16 = (int)((double)mn * 16000 / a->sr) + 1;
        float *r16 = (float *)malloc((size_t)n16 * sizeof(float));
        wubu_audio_resample(a->data, mn, a->sr, 16000, r16);
        short *pcm = (short *)malloc((size_t)n16 * sizeof(short));
        float *pf = (float *)malloc((size_t)n16 * sizeof(float));
        for (int i = 0; i < n16; i++) {
            float v = r16[i] * 32767.0f;
            if (v > 32767.0f) v = 32767.0f;
            if (v < -32768.0f) v = -32768.0f;
            pcm[i] = (short)v;
            pf[i] = r16[i];
        }
        int n = n16;
        printf("clip %s: %d samples @16000\n", argv[1], n);
        float *f0 = calloc((size_t)(n / 160 + 1), sizeof(float));
        int *cls = calloc((size_t)(n / 160 + 1), sizeof(int));
        float *gain = calloc((size_t)(n / 160 + 1), sizeof(float));
        WuBuBreathStats st;
        int nf = wubu_breath_detect(pf, n, f0, n / 160 + 1, cls, gain, &st);
        printf("frames=%d silence=%d breath=%d consonant=%d voiced=%d\n",
               nf, st.class_total[0], st.class_total[1], st.class_total[2], st.class_total[3]);
        printf("voiced_frac=%.3f breath_frac=%.3f events=%d\n",
               st.voiced_frac, st.breath_frac, st.n_events);
        for (int e = 0; e < st.n_events && e < 8; e++)
            printf("  evt[%d] %d..%d (%.0f ms) zcr=%.3f rms=%.4f\n", e,
                   st.events[e].start_frame, st.events[e].end_frame,
                   st.events[e].dur_ms, st.events[e].peak_zcr, st.events[e].peak_rms);
        wubu_audio_free(a);
        free(r16); free(pcm); free(pf); free(f0); free(cls); free(gain);
        return 0;
    }
    int n = sr * 2;
    short *buf = malloc((size_t)n * 2);
    synth_clip(buf, n, sr);
    float *pff = malloc((size_t)n * 4);
    for (int i = 0; i < n; i++) pff[i] = buf[i] / 32768.0f;
    float *f0 = calloc((size_t)(n / 160 + 1), sizeof(float));
    /* feed real f0: 220 Hz for the voiced region (frames 110..190) */
    for (int f = 110; f < 190 && f < n / 160 + 1; f++) f0[f] = 220.0f;
    int *cls = calloc((size_t)(n / 160 + 1), sizeof(int));
    float *gain = calloc((size_t)(n / 160 + 1), sizeof(float));
    WuBuBreathStats st;
    int nf = wubu_breath_detect(pff, n, f0, n / 160 + 1, cls, gain, &st);
    printf("SYNTH: frames=%d silence=%d breath=%d consonant=%d voiced=%d\n",
           nf, st.class_total[0], st.class_total[1], st.class_total[2], st.class_total[3]);
    /* check the 3 regions (expect ~50 silence, ~60 breath, ~80 voiced) */
    int sil_ok = 0, breath_ok = 0, voiced_ok = 0;
    for (int f = 0; f < nf; f++) {
        if (f < 48 && cls[f] == WUBU_BREATH_SILENCE) sil_ok++;
        if (f >= 52 && f < 110 && cls[f] == WUBU_BREATH_BREATH) breath_ok++;
        if (f >= 115 && cls[f] == WUBU_BREATH_VOICED) voiced_ok++;
    }
    printf("region checks: silence %d/48 breath %d/58 voiced %d/80\n",
           sil_ok, breath_ok, voiced_ok);
    printf("%s\n", (sil_ok > 35 && breath_ok > 45 && voiced_ok > 60)
                   ? "BREATH PROBE PASS" : "BREATH PROBE FAIL");
    free(buf); free(pff); free(f0); free(cls); free(gain);
    return 0;
}
