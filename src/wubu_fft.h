#ifndef WUBU_FFT_H
#define WUBU_FFT_H

/* wubu_fft.h — shared radix-2 complex FFT/IFFT.
 *
 * Used by wubu_stft (RMVPE mel), wubu_pitch (phase vocoder), and anything
 * else that needs a spectrum. Iterative Cooley-Tukey, in place.
 *
 * C11, minimal includes. License: WaefreBeorn-UMV3
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float re, im;
} WuBuCpx;

/* In-place radix-2 FFT (n must be a power of 2). inv=0 forward, inv=1
 * inverse (scaled by 1/n). */
void wubu_fft(WuBuCpx *a, int n, int inv);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_FFT_H */
