/* wubu_stft.h — self-contained STFT magnitude + mel filterbank (C11).
 *
 * Minimal include surface; no dependencies. Used by WuBuRMVPE for the mel
 * spectrogram front-end and reusable by any pitch/analysis module.
 *
 * License: WaefreBeorn-UMV3
 */
#ifndef WUBU_STFT_H
#define WUBU_STFT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WuBuStft WuBuStft;

/* Create an STFT analyzer. n_fft: FFT size (1024), hop: frame hop (160).
 * Matches torch.stft(center=True) semantics: n_fft/2 zero-pad each side. */
WuBuStft *wubu_stft_create(int n_fft, int hop);
void wubu_stft_free(WuBuStft *s);

/* Number of magnitude frames for n input samples (center=True padding). */
int wubu_stft_n_frames(const WuBuStft *s, int n_samples);

/* Magnitude spectrogram: pcm[n] -> mag[n_bins * T] col-major
 * (n_bins = n_fft/2 + 1). Returns T, or -1 on error. */
int wubu_stft_magnitude(const WuBuStft *s, const float *pcm, int n_samples,
                        float *mag_out, int max_frames);

/* Mel filterbank: mag[n_bins * T] -> mel[n_mels * T] col-major.
 * basis: [n_mels * n_bins] (row-major, e.g. librosa htk basis). */
void wubu_mel_apply(const float *basis, int n_mels, int n_bins,
                    const float *mag, int T, float *mel_out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_STFT_H */
