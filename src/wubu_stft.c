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
#include "wubu_fft.h"
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
    if (!s->window) {
        wubu_stft_free(s);
        return NULL;
    }
    /* torch.hann_window = periodic Hann: 0.5*(1 - cos(2*pi*n/N)) */
    for (int i = 0; i < n_fft; i++)
        s->window[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)i / (float)n_fft);
    return s;
}

void wubu_stft_free(WuBuStft *s) {
    if (!s) return;
    free(s->window);
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
    WuBuCpx *spec = (WuBuCpx *)malloc((size_t)n_fft * sizeof(WuBuCpx));
    if (!spec) { free(buf); return -1; }
    for (int f = 0; f < T; f++) {
        const float *x = buf + (size_t)f * s->hop;
        for (int t = 0; t < n_fft; t++) {
            spec[t].re = x[t] * s->window[t];
            spec[t].im = 0.0f;
        }
        wubu_fft(spec, n_fft, 0);
        for (int b = 0; b < n_bins; b++) {
            float re = spec[b].re, im = spec[b].im;
            mag_out[(size_t)b * T + f] = sqrtf(re * re + im * im);
        }
    }
    free(spec);
    free(buf);
    return T;
}

void wubu_mel_apply(const float *basis, int n_mels, int n_bins,
                    const float *mag, int T, float *mel_out) {
    if (!basis || !mag || !mel_out) return;
    /* each mel row is an independent dot product — parallel over rows */
#pragma omp parallel for schedule(static) if(n_mels >= 4)
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
