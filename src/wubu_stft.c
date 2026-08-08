/* wubu_stft.c — self-contained STFT magnitude + mel filterbank (C11).
 *
 * Matches torch.stft(audio, n_fft, hop_length, win_length=n_fft,
 * window=hann_periodic, center=True, return_complex=True) -> magnitude,
 * then mel = mel_basis @ magnitude (librosa htk basis, supplied as data).
 *
 * License: WaefreBeorn-UMV3
 */
#define _USE_MATH_DEFINES
#include "wubu_stft.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct WuBuStft {
    int n_fft;
    int hop;
    int n_bins;
    float *window;   /* periodic hann, n_fft */
    float *cos_tab;  /* cos(-2*pi*b*t/n_fft), n_bins*n_fft */
    float *sin_tab;  /* sin(-2*pi*b*t/n_fft), n_bins*n_fft */
};

WuBuStft *wubu_stft_create(int n_fft, int hop) {
    if (n_fft < 32 || hop < 1) return NULL;
    WuBuStft *s = (WuBuStft *)calloc(1, sizeof(WuBuStft));
    if (!s) return NULL;
    s->n_fft = n_fft;
    s->hop = hop;
    s->n_bins = n_fft / 2 + 1;
    s->window = (float *)malloc((size_t)n_fft * sizeof(float));
    s->cos_tab = (float *)malloc((size_t)s->n_bins * n_fft * sizeof(float));
    s->sin_tab = (float *)malloc((size_t)s->n_bins * n_fft * sizeof(float));
    if (!s->window || !s->cos_tab || !s->sin_tab) {
        wubu_stft_free(s);
        return NULL;
    }
    /* torch.hann_window = periodic Hann: 0.5*(1 - cos(2*pi*n/N)) */
    for (int i = 0; i < n_fft; i++)
        s->window[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)i / (float)n_fft);
    /* precompute DFT basis: exp(-j*2*pi*b*t/n) for b in [0, n_bins) */
    for (int b = 0; b < s->n_bins; b++) {
        for (int t = 0; t < n_fft; t++) {
            float ph = -2.0f * (float)M_PI * (float)b * (float)t / (float)n_fft;
            s->cos_tab[(size_t)b * n_fft + t] = cosf(ph);
            s->sin_tab[(size_t)b * n_fft + t] = sinf(ph);
        }
    }
    return s;
}

void wubu_stft_free(WuBuStft *s) {
    if (!s) return;
    free(s->window);
    free(s->cos_tab);
    free(s->sin_tab);
    free(s);
}

int wubu_stft_n_frames(const WuBuStft *s, int n_samples) {
    if (!s || n_samples < 1) return 0;
    int pad = s->n_fft / 2;
    int n_padded = n_samples + 2 * pad;
    return (n_padded - s->n_fft) / s->hop + 1;
}

int wubu_stft_magnitude(const WuBuStft *s, const float *pcm, int n_samples,
                        float *mag_out, int max_frames) {
    if (!s || !pcm || !mag_out) return -1;
    int pad = s->n_fft / 2;
    int n_padded = n_samples + 2 * pad;
    int T = (n_padded - s->n_fft) / s->hop + 1;
    if (T < 1) T = 1;
    if (T > max_frames) T = max_frames;

    /* zero-padded buffer (torch.stft center=True pads zeros) */
    float *buf = (float *)calloc((size_t)n_padded, sizeof(float));
    if (!buf) return -1;
    memcpy(buf + pad, pcm, (size_t)n_samples * sizeof(float));

    const int n_fft = s->n_fft;
    const int n_bins = s->n_bins;
    for (int f = 0; f < T; f++) {
        const float *x = buf + (size_t)f * s->hop;
        for (int b = 0; b < n_bins; b++) {
            float re = 0.0f, im = 0.0f;
            const float *ct = s->cos_tab + (size_t)b * n_fft;
            const float *st = s->sin_tab + (size_t)b * n_fft;
            for (int t = 0; t < n_fft; t++) {
                float v = x[t] * s->window[t];
                re += v * ct[t];
                im += v * st[t];
            }
            mag_out[(size_t)b * T + f] = sqrtf(re * re + im * im);
        }
    }
    free(buf);
    return T;
}

void wubu_mel_apply(const float *basis, int n_mels, int n_bins,
                    const float *mag, int T, float *mel_out) {
    if (!basis || !mag || !mel_out) return;
    for (int m = 0; m < n_mels; m++) {
        const float *brow = basis + (size_t)m * n_bins;
        float *mrow = mel_out + (size_t)m * T;
        for (int f = 0; f < T; f++) {
            float acc = 0.0f;
            for (int b = 0; b < n_bins; b++)
                acc += brow[b] * mag[(size_t)b * T + f];
            mrow[f] = acc;
        }
    }
}
